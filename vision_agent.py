"""
InstaWorker™ Vision Agent
=========================
On-device warehouse vision system using local Gemma/LLaVA models.

Pipeline:
  iPhone camera (via DroidCam) 
    → OpenCV frame capture
    → LLaVA vision model (local, on-device)
    → Agent decision loop
    → Auto inbound / sorting / restock alert
    → TTS voice announcement
    → Streamlit dashboard

Requirements:
  pip install opencv-python ollama streamlit pyttsx3 pillow numpy requests
  
  iPhone setup:
    1. Install DroidCam app on iPhone (App Store, free)
    2. Open DroidCam on iPhone, note the IP address shown
    3. Set DROIDCAM_IP below to that IP
"""

import cv2
import ollama
import streamlit as st
import threading
import time
import base64
import json
import queue
import numpy as np
from PIL import Image
from datetime import datetime
import io
import re

# ── Config ─────────────────────────────────────────────────────────
DROIDCAM_IP   = "172.22.51.94"    # iPhone DroidCam (backup)
DROIDCAM_PORT = 4747
CAMERA_URL    = None              # None = MacBook built-in camera

VISION_MODEL  = "llava"           # Local vision model (on-device)
AGENT_MODEL   = "gemma3"          # Local agent/reasoning model

SCAN_INTERVAL = 3                 # Seconds between scans
CONFIDENCE_THRESHOLD = 0.6        # Min confidence to act

# ── Warehouse Knowledge Base ────────────────────────────────────────
KNOWN_ITEMS = {
    "circuit_breaker_20a": {
        "sku": "CB-1201-A", "name": "Circuit Breaker 20A",
        "shelf": "A-03-02", "unit_price": 45.0,
        "keywords": ["circuit breaker", "breaker", "20a", "small breaker"],
    },
    "circuit_breaker_100a": {
        "sku": "CB-2200-D", "name": "Circuit Breaker 100A",
        "shelf": "A-04-03", "unit_price": 180.0,
        "keywords": ["circuit breaker", "100a", "large breaker"],
    },
    "safety_switch": {
        "sku": "SW-4400-B", "name": "Safety Switch 60A",
        "shelf": "A-05-01", "unit_price": 120.0,
        "keywords": ["safety switch", "switch", "60a"],
    },
    "transformer": {
        "sku": "TR-9900-C", "name": "Transformer 10KVA",
        "shelf": "C-01-01", "unit_price": 850.0,
        "keywords": ["transformer", "10kva", "large yellow"],
    },
    "panel_board": {
        "sku": "PNL-100-A", "name": "Panel Board 100A",
        "shelf": "B-02-01", "unit_price": 320.0,
        "keywords": ["panel", "panel board", "distribution panel"],
    },
}

INVENTORY = {
    "CB-1201-A": {"current_stock": 45, "min_stock": 20, "max_stock": 500},
    "CB-2200-D": {"current_stock": 12, "min_stock": 15, "max_stock": 200},
    "SW-4400-B": {"current_stock": 180, "min_stock": 10, "max_stock": 200},
    "TR-9900-C": {"current_stock": 3,  "min_stock": 5,  "max_stock": 20},
    "PNL-100-A": {"current_stock": 8,  "min_stock": 5,  "max_stock": 50},
}

# ── Agent Action Functions ──────────────────────────────────────────
AGENT_FUNCTIONS = {
    "auto_inbound": {
        "description": "Record item as received into inventory",
        "params": ["sku", "quantity", "shelf"],
    },
    "trigger_restock": {
        "description": "Flag item for urgent restock",
        "params": ["sku", "current_qty", "min_qty"],
    },
    "sort_to_shelf": {
        "description": "Direct robotic arm (simulated) to sort item to correct shelf",
        "params": ["sku", "target_shelf"],
    },
    "flag_unknown": {
        "description": "Flag unrecognized item for human review",
        "params": ["description", "image_id"],
    },
    "announce": {
        "description": "Make voice announcement",
        "params": ["message"],
    },
}

# ── TTS Voice ───────────────────────────────────────────────────────
def speak(text: str):
    """Text-to-speech announcement."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        print(f"[TTS] {text}")

# ── Vision: Capture Frame ────────────────────────────────────────────
def capture_frame(camera_url: str = None):
    """
    Capture one frame from DroidCam or MacBook built-in camera.
    Returns PIL Image or None.
    """
    # Try DroidCam if URL provided
    if camera_url:
        try:
            cap = cv2.VideoCapture(camera_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
        except Exception:
            pass

    # MacBook built-in camera (default)
    for cam_idx in [0, 1]:
        try:
            cap = cv2.VideoCapture(cam_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
        except Exception:
            continue

    return None

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string for Ollama."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

# ── Vision Model: Identify Item ──────────────────────────────────────
def identify_item(img: Image.Image) -> dict:
    """
    Send frame to local LLaVA model for item identification.
    Returns structured result.
    """
    known_items_str = "\n".join([
        f"- {v['name']} (SKU: {v['sku']}, keywords: {', '.join(v['keywords'])})"
        for v in KNOWN_ITEMS.values()
    ])

    prompt = f"""You are a warehouse item identification system.
Look at this image and identify any warehouse items you can see.

Known items in our inventory:
{known_items_str}

Respond in JSON format only:
{{
  "item_detected": true/false,
  "item_name": "name of item or null",
  "sku": "SKU code or null",
  "quantity_visible": number or null,
  "confidence": 0.0-1.0,
  "condition": "good/damaged/unknown",
  "notes": "any observations"
}}"""

    try:
        img_b64 = image_to_base64(img)
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }]
        )
        text = response["message"]["content"]
        # Extract JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        pass

    return {"item_detected": False, "confidence": 0, "notes": "Vision model error"}

# ── Agent: Decide Action ─────────────────────────────────────────────
def agent_decide(vision_result: dict, inventory: dict) -> dict:
    """
    Local Gemma agent decides what action to take based on vision result.
    Uses function calling pattern.
    """
    if not vision_result.get("item_detected"):
        return {"action": None, "reason": "No item detected"}

    sku = vision_result.get("sku")
    confidence = vision_result.get("confidence", 0)

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "action": "flag_unknown",
            "params": {
                "description": vision_result.get("notes", "Unrecognized item"),
                "image_id": datetime.now().strftime("%H%M%S"),
            },
            "reason": f"Low confidence: {confidence:.0%}",
        }

    # Build context for agent
    inv_status = ""
    if sku and sku in inventory:
        inv = inventory[sku]
        inv_status = f"Current stock: {inv['current_stock']}, Min: {inv['min_stock']}"

    prompt = f"""You are a warehouse agent. Based on this detection, decide the best action.

Detection result:
- Item: {vision_result.get('item_name')}
- SKU: {sku}
- Quantity visible: {vision_result.get('quantity_visible')}
- Confidence: {confidence:.0%}
- Condition: {vision_result.get('condition')}
- Notes: {vision_result.get('notes')}
- Inventory: {inv_status}

Available actions:
- auto_inbound: if item is being received/delivered
- trigger_restock: if stock is below minimum
- sort_to_shelf: if item needs to be placed on correct shelf
- flag_unknown: if item is unrecognized or damaged

Respond in JSON only:
{{
  "action": "action_name",
  "params": {{}},
  "reason": "brief explanation",
  "announce": "short voice announcement (max 15 words)"
}}"""

    try:
        response = ollama.chat(
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response["message"]["content"]
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    # Rule-based fallback
    if sku and sku in inventory:
        inv = inventory[sku]
        if inv["current_stock"] < inv["min_stock"]:
            return {
                "action": "trigger_restock",
                "params": {"sku": sku, "current_qty": inv["current_stock"], "min_qty": inv["min_stock"]},
                "reason": "Stock below minimum",
                "announce": f"Low stock alert for {vision_result.get('item_name')}",
            }
        else:
            shelf = next((v["shelf"] for v in KNOWN_ITEMS.values() if v["sku"] == sku), "Unknown")
            return {
                "action": "auto_inbound",
                "params": {"sku": sku, "quantity": 1, "shelf": shelf},
                "reason": "Item received, stock OK",
                "announce": f"Received {vision_result.get('item_name')}, directing to shelf {shelf}",
            }

    return {"action": "flag_unknown", "params": {}, "reason": "Unknown item"}

# ── Execute Action ───────────────────────────────────────────────────
def execute_action(action_result: dict, inventory: dict, event_log: list) -> str:
    """Execute the agent's chosen action and return status message."""
    action = action_result.get("action")
    params = action_result.get("params", {})
    announce_text = action_result.get("announce", "")

    timestamp = datetime.now().strftime("%H:%M:%S")
    status = ""

    if action == "auto_inbound":
        sku = params.get("sku", "")
        qty = params.get("quantity", 1)
        shelf = params.get("shelf", "")
        if sku in inventory:
            inventory[sku]["current_stock"] += qty
        status = f"✅ AUTO INBOUND: {sku} +{qty} units → Shelf {shelf}"
        announce_text = announce_text or f"Inbound complete. {sku} stored at shelf {shelf}."

    elif action == "trigger_restock":
        sku = params.get("sku", "")
        status = f"🚨 RESTOCK ALERT: {sku} — only {params.get('current_qty')} units left"
        announce_text = announce_text or f"Urgent restock needed for {sku}."

    elif action == "sort_to_shelf":
        sku = params.get("sku", "")
        shelf = params.get("target_shelf", "")
        status = f"🤖 ROBOTIC SORT: {sku} → Shelf {shelf} [SIMULATED]"
        announce_text = announce_text or f"Sorting {sku} to shelf {shelf}."

    elif action == "flag_unknown":
        desc = params.get("description", "Unknown item")
        status = f"⚠️ UNKNOWN ITEM: {desc} — flagged for human review"
        announce_text = announce_text or "Unknown item detected. Human review required."

    else:
        status = f"ℹ️ No action taken: {action_result.get('reason', '')}"

    # Log event
    event_log.append({
        "time": timestamp,
        "action": action,
        "status": status,
        "reason": action_result.get("reason", ""),
    })

    # Voice announcement in background thread
    if announce_text:
        threading.Thread(target=speak, args=(announce_text,), daemon=True).start()

    return status

# ── Main Agent Loop ──────────────────────────────────────────────────
def run_agent_loop(state: dict, stop_event: threading.Event):
    """
    Continuous agent loop running in background thread.
    state: shared dict with inventory, event_log, current_frame, last_result
    """
    while not stop_event.is_set():
        try:
            # 1. Capture frame
            cam_url = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video" if state.get("use_iphone") else None
            img = capture_frame(cam_url)
            if img is None:
                time.sleep(2)
                continue

            state["current_frame"] = img
            state["frame_time"] = datetime.now().strftime("%H:%M:%S")

            # 2. Vision: identify item
            vision_result = identify_item(img)
            state["last_vision"] = vision_result

            # 3. Agent: decide action
            if vision_result.get("item_detected"):
                action_result = agent_decide(vision_result, state["inventory"])
                state["last_action"] = action_result

                # 4. Execute action
                status = execute_action(action_result, state["inventory"], state["event_log"])
                state["last_status"] = status
                state["scan_count"] = state.get("scan_count", 0) + 1

        except Exception as e:
            state["last_status"] = f"Error: {str(e)}"

        time.sleep(SCAN_INTERVAL)


# ── Streamlit Dashboard ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="InstaWorker™ Vision Agent",
        layout="wide",
        page_icon="🤖"
    )

    st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f1f5f9; }
    .metric-card {
        background: #1e293b; border-radius: 12px;
        padding: 16px; margin: 4px; text-align: center;
    }
    .action-badge {
        padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Shared state ──
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {
            "inventory": {k: dict(v) for k, v in INVENTORY.items()},
            "event_log": [],
            "current_frame": None,
            "last_vision": {},
            "last_action": {},
            "last_status": "Waiting...",
            "frame_time": "--:--:--",
            "scan_count": 0,
            "use_iphone": True,
            "running": False,
        }
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = None
    if "agent_thread" not in st.session_state:
        st.session_state.agent_thread = None

    state = st.session_state.agent_state

    # ── Header ──
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px 0;'>
        <div style='font-size:48px;'>🤖</div>
        <h1 style='color:#f1f5f9;margin:4px 0;'>InstaWorker™ Vision Agent</h1>
        <p style='color:#64748b;'>On-Device · Gemma + LLaVA · Auto Inbound · Robotic Sorting</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar controls ──
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        # Camera IP
        ip = st.text_input("iPhone DroidCam IP", value=DROIDCAM_IP,
                           placeholder="192.168.x.x")
        state["use_iphone"] = st.toggle("Use iPhone Camera (DroidCam)", value=False)

        st.divider()
        st.markdown("**Models (on-device)**")
        st.caption(f"Vision: {VISION_MODEL}")
        st.caption(f"Agent: {AGENT_MODEL}")
        st.caption("✅ 100% local — no cloud")

        st.divider()

        # Start/Stop
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start", use_container_width=True, type="primary",
                         disabled=state["running"]):
                global CAMERA_URL
                CAMERA_URL = f"http://{ip}:{DROIDCAM_PORT}/video"
                stop_event = threading.Event()
                st.session_state.stop_event = stop_event
                t = threading.Thread(
                    target=run_agent_loop,
                    args=(state, stop_event),
                    daemon=True
                )
                t.start()
                st.session_state.agent_thread = t
                state["running"] = True
                st.rerun()

        with col2:
            if st.button("⏹ Stop", use_container_width=True,
                         disabled=not state["running"]):
                if st.session_state.stop_event:
                    st.session_state.stop_event.set()
                state["running"] = False
                st.rerun()

        st.divider()
        # Manual scan
        if st.button("📸 Manual Scan", use_container_width=True):
            with st.spinner("Scanning..."):
                cam_url = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video" if state["use_iphone"] else None
                img = capture_frame(cam_url)
                if img:
                    state["current_frame"] = img
                    vision_result = identify_item(img)
                    state["last_vision"] = vision_result
                    if vision_result.get("item_detected"):
                        action_result = agent_decide(vision_result, state["inventory"])
                        state["last_action"] = action_result
                        status = execute_action(action_result, state["inventory"], state["event_log"])
                        state["last_status"] = status
                    st.rerun()

        st.divider()
        # Voice test
        if st.button("🔊 Test Voice", use_container_width=True):
            threading.Thread(
                target=speak,
                args=("InstaWorker Vision Agent ready. Warehouse monitoring active.",),
                daemon=True
            ).start()

    # ── Status bar ──
    status_color = "#22c55e" if state["running"] else "#64748b"
    status_text = "🟢 AGENT RUNNING" if state["running"] else "⚫ STOPPED"
    st.markdown(
        f"<div style='background:#1e293b;border-radius:8px;padding:10px 16px;"
        f"border-left:4px solid {status_color};margin-bottom:16px;'>"
        f"<b style='color:{status_color};'>{status_text}</b>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;Scans: {state['scan_count']}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;Last scan: {state['frame_time']}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;{state['last_status']}"
        f"</div>",
        unsafe_allow_html=True
    )

    # ── Main layout ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Camera feed
        st.markdown("#### 📷 Live Camera Feed")
        cam_placeholder = st.empty()
        if state["current_frame"]:
            cam_placeholder.image(state["current_frame"], use_container_width=True)
        else:
            cam_placeholder.markdown("""
            <div style='background:#1e293b;border-radius:12px;height:300px;
                        display:flex;align-items:center;justify-content:center;
                        border:2px dashed #334155;'>
                <div style='text-align:center;color:#475569;'>
                    <div style='font-size:48px;'>📷</div>
                    <div>Camera feed will appear here</div>
                    <div style='font-size:12px;margin-top:8px;'>Press Start or Manual Scan</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Vision result
        st.markdown("#### 🔍 Vision Detection")
        v = state.get("last_vision", {})
        if v.get("item_detected"):
            conf = v.get("confidence", 0)
            conf_color = "#22c55e" if conf > 0.7 else ("#f59e0b" if conf > 0.4 else "#ef4444")
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:12px;padding:16px;'>
                <div style='font-size:20px;font-weight:700;color:#f1f5f9;'>
                    {v.get('item_name','Unknown')}
                </div>
                <div style='color:#64748b;font-size:14px;margin:4px 0;'>
                    SKU: <b style='color:#3b82f6;'>{v.get('sku','—')}</b>
                    &nbsp;|&nbsp; Qty: {v.get('quantity_visible','—')}
                    &nbsp;|&nbsp; Condition: {v.get('condition','—')}
                </div>
                <div style='margin-top:8px;'>
                    <span style='background:{conf_color}22;color:{conf_color};
                                 padding:2px 10px;border-radius:20px;font-size:13px;'>
                        Confidence: {conf:.0%}
                    </span>
                </div>
                <div style='color:#94a3b8;font-size:12px;margin-top:8px;'>
                    {v.get('notes','')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No item detected in current frame")

    with col_right:
        # Agent decision
        st.markdown("#### 🤖 Agent Decision")
        a = state.get("last_action", {})
        if a.get("action"):
            action_colors = {
                "auto_inbound": "#22c55e",
                "trigger_restock": "#ef4444",
                "sort_to_shelf": "#3b82f6",
                "flag_unknown": "#f59e0b",
                "announce": "#8b5cf6",
            }
            action_icons = {
                "auto_inbound": "📥",
                "trigger_restock": "🚨",
                "sort_to_shelf": "🤖",
                "flag_unknown": "⚠️",
            }
            color = action_colors.get(a["action"], "#64748b")
            icon = action_icons.get(a["action"], "ℹ️")
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:12px;padding:16px;
                        border-left:4px solid {color};'>
                <div style='font-size:18px;font-weight:700;color:{color};'>
                    {icon} {a['action'].replace('_',' ').upper()}
                </div>
                <div style='color:#94a3b8;font-size:13px;margin:8px 0;'>
                    {a.get('reason','')}
                </div>
                <div style='color:#f1f5f9;font-size:13px;'>
                    {json.dumps(a.get('params',{}), indent=2)}
                </div>
                <div style='margin-top:10px;color:#8b5cf6;font-size:12px;'>
                    🔊 "{a.get('announce','')}"
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Waiting for first detection...")

        # Inventory status
        st.markdown("#### 📦 Live Inventory")
        for sku, inv in state["inventory"].items():
            pct = inv["current_stock"] / inv["max_stock"]
            status = "🚨" if inv["current_stock"] < inv["min_stock"] else "✅"
            bar_color = "#ef4444" if inv["current_stock"] < inv["min_stock"] else "#22c55e"
            item_name = next((v["name"] for v in KNOWN_ITEMS.values() if v["sku"] == sku), sku)
            col_a, col_b, col_c = st.columns([3, 2, 1])
            col_a.write(f"**{sku}**")
            col_b.write(f"{inv['current_stock']} / {inv['max_stock']}")
            col_c.write(status)

    # ── Event Log ──
    st.markdown("---")
    st.markdown("#### 📋 Agent Event Log")
    if state["event_log"]:
        for event in reversed(state["event_log"][-15:]):
            icon = {"auto_inbound":"📥","trigger_restock":"🚨",
                    "sort_to_shelf":"🤖","flag_unknown":"⚠️"}.get(event["action"],"ℹ️")
            st.markdown(
                f"<div style='background:#1e293b;border-radius:8px;padding:8px 12px;"
                f"margin:4px 0;font-size:13px;'>"
                f"<span style='color:#64748b;'>{event['time']}</span>&nbsp;&nbsp;"
                f"{icon} {event['status']}"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No events yet. Start the agent or press Manual Scan.")

    # ── iPhone Setup Guide ──
    with st.expander("📱 iPhone Camera Setup (DroidCam)"):
        st.markdown("""
        **Step 1:** Install **DroidCam** from App Store (free)
        
        **Step 2:** Open DroidCam on iPhone — note the IP address shown on screen
        
        **Step 3:** Enter that IP in the sidebar above
        
        **Step 4:** Make sure iPhone and MacBook are on the **same WiFi network**
        
        **Step 5:** Press Start — camera feed will appear automatically
        
        **Tip:** Point iPhone at any warehouse item and watch the agent identify and process it!
        """)

    # Auto-refresh while running
    if state["running"]:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
