"""
InstaWorker™ Backend — Core Data, Logic & AI Engine
"""

import streamlit as st
from google import genai
from PIL import Image
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    _ollama = None
    OLLAMA_AVAILABLE = False
import json
import pandas as pd
from datetime import datetime
import urllib.request
import urllib.parse
import socket
import io
import hashlib
import pickle
import os

# ── Task State Machine ────────────────────────────────────────────
TASK_STATES = ["Loading", "Picked", "Dispatched", "In Transit", "Delivered"]
TASK_ICONS  = {"Loading":"🔵","Picked":"🟣","Dispatched":"🟡","In Transit":"🚛","Delivered":"✅"}

# ── RAG imports (graceful fallback if not installed) ──
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── Unified AI Engine — 3-Layer Fallback ─────────────────────────
def _rule_based_fallback(prompt: str) -> str:
    """
    Layer 3: Pure Python rule engine.
    No AI needed. Analyzes inventory data directly and returns advice.
    Used when both Gemini and Ollama are unavailable.
    """
    lines = ["[Offline Rule Engine - No AI available]\n"]
    critical, low, ok, over = [], [], [], []
    for sku in INVENTORY:
        r = calculate_reorder(sku)
        urgency = r.get("urgency","")
        if urgency == "URGENT":     critical.append(sku)
        elif urgency == "ORDER NOW": low.append(sku)
        elif urgency == "REDUCE":    over.append(sku)
        else:                        ok.append(sku)

    if critical:
        lines.append(f"CRITICAL — immediate reorder required: {', '.join(critical)}")
    if low:
        lines.append(f"LOW STOCK — order soon: {', '.join(low)}")
    if over:
        lines.append(f"OVERSTOCKED — skip next order: {', '.join(over)}")
    if ok:
        lines.append(f"Healthy: {', '.join(ok)}")

    total_val = sum(
        calculate_reorder(s)["recommended_qty"] * INVENTORY[s]["unit_price"]
        for s in INVENTORY if calculate_reorder(s)["recommended_qty"] > 0
    )
    lines.append(f"Total recommended reorder value: ${total_val:,.2f}")
    lines.append("Recommendation: Process critical items first, then low-stock items.")
    return "\n".join(lines)

def ai_call(prompt: str, image=None) -> str:
    """
    3-layer fallback AI engine:
    Layer 1 — Gemini (cloud, best quality)
    Layer 2 — Ollama Gemma3 (local, offline)
    Layer 3 — Rule engine (pure Python, always works)
    """
    mode = st.session_state.get("ai_mode", "gemini")
    ai_log = st.session_state.setdefault("ai_call_log", [])

    # ── Layer 1: Gemini ──
    if mode == "gemini":
        api_key = st.session_state.get("api_key", "")
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                contents = [image, prompt] if image else prompt
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
                ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": "Gemini ☁️", "status": "OK"})
                return resp.text
            except Exception as e:
                ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": "Gemini ☁️", "status": f"FAILED: {e}"})
                # Auto-fallback to Ollama
                st.toast("⚠️ Gemini unavailable — falling back to Ollama Gemma3", icon="⚠️")
                mode = "ollama"

    # ── Layer 2: Ollama Gemma3 ──
    if mode == "ollama" or (mode == "gemini" and not st.session_state.get("api_key","")):
        if OLLAMA_AVAILABLE and _ollama is not None:
            selected_model = st.session_state.get("ollama_model", "gemma3")
            model_label = f"{selected_model} 🔒"
            try:
                if image:
                    # Gemma3 supports multimodal — send image + text
                    import io as _io, base64 as _b64
                    buf = _io.BytesIO()
                    image.save(buf, format="JPEG", quality=85)
                    img_b64 = _b64.b64encode(buf.getvalue()).decode()
                    result = _ollama.chat(
                        model=selected_model,
                        messages=[{"role":"user","content": prompt, "images": [img_b64]}]
                    )["message"]["content"]
                    ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": model_label, "status": "OK (vision)"})
                    return result
                response = _ollama.chat(
                    model=selected_model,
                    messages=[{"role":"user","content": prompt}]
                )
                ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": model_label, "status": "OK"})
                return response["message"]["content"]
            except Exception as e:
                ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": model_label, "status": f"FAILED: {e}"})
                st.toast(f"⚠️ {selected_model} unavailable — using rule engine", icon="⚠️")
        else:
            ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": "Ollama 🔒", "status": "NOT INSTALLED"})

    # ── Layer 3: Rule engine ──
    ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": "Rule Engine ⚙️", "status": "OK"})
    return _rule_based_fallback(prompt)

# ══════════════════════════════════════════════════════════════════
# RAG / CRAG ENGINE — Multi-tenant, edge-native
# Each company has its own isolated FAISS vector store
# ══════════════════════════════════════════════════════════════════

RAG_DIR = os.path.expanduser("~/instaworker_rag")
os.makedirs(RAG_DIR, exist_ok=True)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None and RAG_AVAILABLE:
        try:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass
    return _embedding_model

def get_company_rag_path(company_id: str) -> str:
    return os.path.join(RAG_DIR, f"{company_id}")

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_rag_index(company_id: str, texts: list, sources: list):
    """Build FAISS index for a company from list of text chunks."""
    if not RAG_AVAILABLE:
        return False
    model = get_embedding_model()
    if not model:
        return False
    try:
        embeddings = model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        path = get_company_rag_path(company_id)
        os.makedirs(path, exist_ok=True)
        faiss.write_index(index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump({"texts": texts, "sources": sources}, f)
        return True
    except Exception as e:
        return False

def rag_search(company_id: str, query: str, top_k: int = 4) -> list:
    """Search RAG index, return top_k relevant chunks."""
    if not RAG_AVAILABLE:
        return []
    path = get_company_rag_path(company_id)
    index_path = os.path.join(path, "index.faiss")
    chunks_path = os.path.join(path, "chunks.pkl")
    if not os.path.exists(index_path):
        return []
    try:
        model = get_embedding_model()
        if not model:
            return []
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        q_emb = model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(data["texts"]):
                results.append({
                    "text": data["texts"][idx],
                    "source": data["sources"][idx],
                    "score": float(score),
                })
        return results
    except Exception:
        return []

def crag_search(company_id: str, query: str, threshold: float = 0.4) -> dict:
    """
    CRAG — Corrective RAG.
    If top result confidence < threshold, reformulate query and retry.
    Returns chunks + confidence assessment.
    """
    results = rag_search(company_id, query, top_k=4)
    if not results:
        return {"chunks": [], "confidence": "none", "used_correction": False}

    top_score = results[0]["score"] if results else 0

    # High confidence — use as-is
    if top_score >= threshold:
        return {"chunks": results, "confidence": "high", "used_correction": False}

    # Low confidence — try corrective search with expanded query
    corrected_query = f"{query} specifications details requirements"
    results2 = rag_search(company_id, corrected_query, top_k=4)

    if results2 and results2[0]["score"] > top_score:
        return {"chunks": results2, "confidence": "medium", "used_correction": True}

    # Still low — return what we have with warning
    return {"chunks": results, "confidence": "low", "used_correction": False}

def ai_call_with_rag(prompt: str, company_id: str = "default", image=None) -> dict:
    """
    RAG-enhanced AI call.
    1. Search company knowledge base
    2. Inject relevant context into prompt
    3. Call AI with enriched context
    Returns: {answer, chunks_used, confidence, used_correction}
    """
    crag_result = crag_search(company_id, prompt)
    chunks = crag_result["chunks"]

    if chunks:
        context = "\n\n".join([
            f"[Source: {c['source']}] {c['text']}"
            for c in chunks
            for c in chunks
        ])
        enriched_prompt = f"""You have access to the company's knowledge base.

RELEVANT COMPANY DOCUMENTS:
{context}

USER QUESTION: {prompt}

Answer based on the company documents above. If the documents don't contain enough information, say so clearly."""
    else:
        enriched_prompt = prompt

    answer = ai_call(enriched_prompt, image=image)
    return {
        "answer": answer,
        "chunks_used": chunks,
        "confidence": crag_result["confidence"],
        "used_correction": crag_result["used_correction"],
    }

# ── Sales Orders Data ──────────────────────────────────────────────


st.set_page_config(page_title="InstaWorker™ Pro", layout="wide", page_icon="🏭")

st.markdown("""
<style>
.stApp { background-color: #0f172a; color: #f1f5f9; }
.role-card {
    border-radius: 16px; padding: 24px; text-align: center;
    cursor: pointer; margin: 8px; transition: transform 0.2s;
    border: 2px solid transparent;
}
.role-card:hover { transform: scale(1.03); }
.manager-card { background: linear-gradient(135deg, #1e3a8a, #1d4ed8); }
.staff-card   { background: linear-gradient(135deg, #14532d, #15803d); }
.driver-card  { background: linear-gradient(135deg, #7c2d12, #c2410c); }
.metric-box {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 16px; margin: 6px 0;
}
.alert-red    { background:#450a0a; border-left:4px solid #ef4444; padding:12px; border-radius:8px; margin:6px 0; }
.alert-yellow { background:#422006; border-left:4px solid #f59e0b; padding:12px; border-radius:8px; margin:6px 0; }
.alert-green  { background:#052e16; border-left:4px solid #22c55e; padding:12px; border-radius:8px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ── Scanner HTML Component ─────────────────────────────────────────
BARCODE_SCANNER_HTML = """
<div id="scanner-container" style="background:#1e293b;border-radius:12px;padding:16px;">
  <div id="video-container" style="position:relative;width:100%;max-width:400px;margin:0 auto;">
    <video id="video" style="width:100%;border-radius:8px;border:2px solid #3b82f6;" autoplay playsinline></video>
    <canvas id="canvas" style="display:none;"></canvas>
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                width:200px;height:200px;border:3px solid #3b82f6;border-radius:8px;
                box-shadow:0 0 0 9999px rgba(0,0,0,0.4);pointer-events:none;"></div>
  </div>
  <p id="scan-status" style="color:#94a3b8;text-align:center;margin-top:8px;font-size:14px;">
    📷 Starting camera...
  </p>
  <p id="scan-result" style="color:#22c55e;text-align:center;font-size:18px;font-weight:bold;min-height:28px;"></p>
  <button onclick="stopScanner()" style="background:#ef4444;color:white;border:none;border-radius:8px;
          padding:8px 24px;cursor:pointer;display:block;margin:8px auto;">Stop Scanner</button>
</div>

<script src="https://cdn.jsdelivr.net/npm/jsqr@1.4.0/dist/jsQR.js"></script>
<script>
let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let stream = null;
let scanning = true;

async function startScanner() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = stream;
    document.getElementById('scan-status').textContent = '🔍 Scanning for barcode/QR code...';
    requestAnimationFrame(scan);
  } catch(e) {
    document.getElementById('scan-status').textContent = '❌ Camera access denied. Please allow camera.';
  }
}

function scan() {
  if (!scanning) return;
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const code = jsQR(imageData.data, imageData.width, imageData.height, {
      inversionAttempts: 'dontInvert'
    });
    if (code) {
      document.getElementById('scan-result').textContent = '✅ Scanned: ' + code.data;
      document.getElementById('scan-status').textContent = '✅ Barcode detected!';
      scanning = false;
      // Send to Streamlit
      window.parent.postMessage({type: 'streamlit:setComponentValue', value: code.data}, '*');
    }
  }
  requestAnimationFrame(scan);
}

function stopScanner() {
  scanning = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  document.getElementById('scan-status').textContent = 'Scanner stopped.';
}

startScanner();
</script>
"""

# ── Data ──────────────────────────────────────────────────────────
WAREHOUSE_CONFIG = {
    "total_capacity": 5000,
    "safety_stock_days": 14,
    "lead_time_days": 7,
}

INVENTORY = {
    "CB-1201-A": {
        "name": "Circuit Breaker 20A",
        "supplier": "Eaton Corporation",
        "contact": "john.smith@eaton.com",
        "current_stock": 45,
        "unit_price": 45.0,
        "unit_volume": 0.5,
        "min_order_qty": 50,
        "max_stock": 500,
        "shelf": "A-03-02",
        "barcode": "8901001234567",
        "sales_history": [28,32,25,35,30,28,33,27,31,29,34,26,30,32,28],
    },
    "SW-4400-B": {
        "name": "Safety Switch 60A",
        "supplier": "Eaton Corporation",
        "contact": "john.smith@eaton.com",
        "current_stock": 180,
        "unit_price": 120.0,
        "unit_volume": 1.2,
        "min_order_qty": 20,
        "max_stock": 200,
        "shelf": "A-05-01",
        "barcode": "8901001234568",
        "sales_history": [3,2,4,2,3,2,1,3,2,2,1,2,3,2,2],
    },
    "TR-9900-C": {
        "name": "Transformer 10KVA",
        "supplier": "Siemens Energy",
        "contact": "orders@siemens.com",
        "current_stock": 8,
        "unit_price": 850.0,
        "unit_volume": 8.0,
        "min_order_qty": 5,
        "max_stock": 30,
        "shelf": "C-01-01",
        "barcode": "8901001234569",
        "sales_history": [1,2,1,1,2,1,1,1,2,1,1,1,2,1,1],
    },
    "CB-2200-D": {
        "name": "Circuit Breaker 100A",
        "supplier": "Siemens Energy",
        "contact": "orders@siemens.com",
        "current_stock": 12,
        "unit_price": 95.0,
        "unit_volume": 0.8,
        "min_order_qty": 30,
        "max_stock": 300,
        "shelf": "A-04-03",
        "barcode": "8901001234570",
        "sales_history": [18,22,19,25,21,20,23,19,22,24,20,21,23,22,20],
    },
    "PNL-100-A": {
        "name": "Distribution Panel 100A",
        "supplier": "Schneider Electric",
        "contact": "supply@schneider.com",
        "current_stock": 3,
        "unit_price": 620.0,
        "unit_volume": 15.0,
        "min_order_qty": 2,
        "max_stock": 20,
        "shelf": "B-02-01",
        "barcode": "8901001234571",
        "sales_history": [1,0,1,1,0,1,0,1,1,0,1,0,1,1,0],
    },
}

PURCHASE_ORDERS = {
    "PO-2024-001": {
        "supplier": "Eaton Corporation",
        "contact": "john.smith@eaton.com",
        "status": "Pending",
        "items": {
            "CB-1201-A": {"qty": 200, "price": 45.0},
            "SW-4400-B": {"qty": 50, "price": 120.0},
        }
    },
    "PO-2024-002": {
        "supplier": "Siemens Energy",
        "contact": "orders@siemens.com",
        "status": "In Transit",
        "items": {
            "TR-9900-C": {"qty": 10, "price": 850.0},
            "CB-2200-D": {"qty": 100, "price": 95.0},
        }
    }
}

DELIVERIES = {
    "DEL-001": {
        "driver": "Mike Johnson",
        "destination": "Tesla Fremont Plant",
        "address": "45500 Fremont Blvd, Fremont, CA",
        "items": {"CB-1201-A": 50, "CB-2200-D": 30},
        "status": "Dispatched",
        "eta": "Today 14:30",
    },
    "DEL-002": {
        "driver": "Sarah Chen",
        "destination": "Google Data Center",
        "address": "1600 Amphitheatre Pkwy, Mountain View, CA",
        "items": {"TR-9900-C": 3, "PNL-100-A": 2},
        "status": "Loading",
        "eta": "Today 16:00",
    },
}

# ── Logic Functions ───────────────────────────────────────────────
def get_total_capacity_used():
    return sum(v["current_stock"] * v["unit_volume"] for v in INVENTORY.values())

def check_network() -> bool:
    """Check if internet is available. Cached for 60 seconds."""
    # Use cached result to avoid repeated timeouts
    cached = st.session_state.get("_net_cache", {})
    import time as _time
    if cached.get("time") and (_time.time() - cached["time"]) < 60:
        return cached["result"]
    try:
        socket.setdefaulttimeout(1)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
        s.close()
        st.session_state["_net_cache"] = {"result": True, "time": _time.time()}
        return True
    except Exception:
        st.session_state["_net_cache"] = {"result": False, "time": _time.time()}
        return False

def fetch_supply_chain_news(sku: str) -> str:
    """
    Search for supply chain news relevant to this SKU.
    Returns news summary or empty string if offline.
    """
    item = INVENTORY.get(sku, {})
    supplier = item.get("supplier", "")
    name = item.get("name", "")

    # Build search queries
    queries = [
        f"{supplier} supply chain disruption",
        f"{name} shortage trade war tariff",
        "semiconductor electrical component supply 2025",
    ]

    news_results = []
    for query in queries[:1]:  # limit to 1 search to save API calls
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = resp.read().decode("utf-8")
                # Extract first 3 titles from RSS
                import re
                titles = re.findall(r"<title><![CDATA[(.*?)]]></title>", data)[:3]
                news_results.extend(titles)
        except Exception:
            pass

    return " | ".join(news_results) if news_results else ""

def forecast_demand(sku: str, use_news: bool = True):
    """
    Demand forecast with optional web news signal.
    - Online: weighted moving avg + trend + AI news analysis
    - Offline: weighted moving avg + trend only
    """
    item = INVENTORY.get(sku)
    if not item:
        return {"daily_avg": 0, "trend_pct": 0, "demand_level": "UNKNOWN",
                "forecast_7d": 0, "forecast_14d": 0, "news_signal": 0, "news_summary": "", "online": False}

    h = item["sales_history"]
    n = len(h)
    weights = list(range(1, n+1))
    weighted_avg = sum(h[i]*weights[i] for i in range(n)) / sum(weights)
    recent = sum(h[-7:]) / 7
    older  = sum(h[:7]) / 7
    trend  = (recent - older) / older * 100 if older > 0 else 0
    base_daily = weighted_avg * (1 + trend/100 * 0.5)

    # ── News signal (online only) ──
    news_signal   = 0
    news_summary  = ""
    online        = False

    if use_news and check_network():
        online = True
        news_text = fetch_supply_chain_news(sku)
        if news_text and st.session_state.get("api_key"):
            try:
                prompt = f"""Analyze these news headlines for supply chain impact on {item["name"]} (supplier: {item["supplier"]}).
Headlines: {news_text}

Return JSON only:
{{"signal": <number from -30 to +30, positive=increase demand/reduce supply, negative=decrease>,
  "reason": "<one sentence explanation>",
  "risk": "high/medium/low"}}

If no relevant news, return {{"signal": 0, "reason": "No significant supply chain news", "risk": "low"}}"""

                result_text = ai_call(prompt)
                import re
                json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    news_signal  = float(parsed.get("signal", 0))
                    news_summary = parsed.get("reason", "")
            except Exception:
                pass

    # Apply news signal to forecast
    final_daily = base_daily * (1 + news_signal/100)
    final_daily = max(0, final_daily)

    level = "🔥 HIGH" if final_daily > 15 else ("📈 MEDIUM" if final_daily > 5 else "📉 LOW")
    if news_signal > 15:   level = "⚠️ " + level + " (supply risk)"
    if news_signal < -15:  level = level + " (demand drop)"

    return {
        "daily_avg":    round(final_daily, 1),
        "base_daily":   round(base_daily, 1),
        "trend_pct":    round(trend, 1),
        "news_signal":  round(news_signal, 1),
        "news_summary": news_summary,
        "demand_level": level,
        "forecast_7d":  round(final_daily * 7),
        "forecast_14d": round(final_daily * 14),
        "online":       online,
    }

def calculate_reorder(sku):
    item = INVENTORY.get(sku)
    if not item: return {}
    f = forecast_demand(sku)
    daily = f["daily_avg"]
    safety = daily * WAREHOUSE_CONFIG["safety_stock_days"]
    reorder_pt = safety + daily * WAREHOUSE_CONFIG["lead_time_days"]
    stock = item["current_stock"]
    days_left = round(stock / daily, 0) if daily > 0 else 999

    # EOQ
    annual = daily * 365
    if annual > 0 and item["unit_price"] > 0:
        eoq = ((2 * annual * 50) / (item["unit_price"] * 0.20)) ** 0.5
    else:
        eoq = item["min_order_qty"]

    available_cap = (WAREHOUSE_CONFIG["total_capacity"] - get_total_capacity_used()) / item["unit_volume"]
    base_qty = min(max(eoq, item["min_order_qty"]), item["max_stock"] - stock, available_cap)
    if f["trend_pct"] > 20: base_qty *= 1.2
    if f["trend_pct"] < -20 and stock > reorder_pt * 2: base_qty = 0
    final_qty = max(0, round(base_qty / item["min_order_qty"]) * item["min_order_qty"])

    if stock < safety * 0.5:  status, urgency = "🚨 CRITICAL", "URGENT"
    elif stock < reorder_pt:   status, urgency = "⚠️ LOW",      "ORDER NOW"
    elif stock > item["max_stock"] * 0.9: status, urgency = "📦 OVERSTOCK", "REDUCE"
    else:                      status, urgency = "✅ OK",        "MONITOR"

    return {
        "status": status, "urgency": urgency,
        "current_stock": stock, "safety_stock": round(safety),
        "reorder_point": round(reorder_pt), "recommended_qty": int(final_qty),
        "needs_reorder": stock <= reorder_pt, "days_left": days_left,
        "daily_demand": daily, "trend_pct": f["trend_pct"],
        "demand_level": f["demand_level"],
    }

def generate_reorder_email(sku, qty):
    item = INVENTORY[sku]
    r = calculate_reorder(sku)
    return f"""TO: {item['contact']}
SUBJECT: Replenishment Order — {sku} ({item['name']})

Dear {item['supplier']} Team,

We are placing a replenishment order:

  SKU:         {sku}
  Item:        {item['name']}
  Quantity:    {qty} units
  Unit Price:  ${item['unit_price']:.2f}
  Total:       ${qty * item['unit_price']:,.2f}

Current stock: {item['current_stock']} units ({r['days_left']:.0f} days remaining)
Demand trend:  {r['trend_pct']:+.0f}%

Please confirm and deliver within {WAREHOUSE_CONFIG['lead_time_days']} business days.

Best regards,
InstaWorker™ Procurement System"""

# ── Employee Credentials ──────────────────────────────────────────
EMPLOYEES = {
    "MGR001": {"name": "Claire Newton",   "role": "manager", "pin": "1234"},
    "MGR002": {"name": "David Zhang",     "role": "manager", "pin": "5678"},
    "DRV001": {"name": "Mike Johnson",    "role": "driver",  "pin": "3333"},
    "DRV002": {"name": "Sarah Chen",      "role": "driver",  "pin": "4444"},
    "SLS001": {"name": "David Park",        "role": "sales",   "pin": "5555"},
    "SLS002": {"name": "Emma Wilson",       "role": "sales",   "pin": "6666"},
}
