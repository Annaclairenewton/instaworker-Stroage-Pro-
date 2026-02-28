"""
InstaWorker™ Conveyor Belt + Camera Auto-Identification Engine
===============================================================
Simulates a smart warehouse conveyor belt system with:
  - Camera zones that auto-identify items via AI
  - Conveyor belt routing (inbound → storage / outbound → dispatch)
  - Order-matching: auto-route items to dispatch bay if order exists
  - Real-time conveyor state machine
"""

import json
import time
import threading
from datetime import datetime
from enum import Enum

# ── Conveyor Zone Definitions ─────────────────────────────────────
class Zone(str, Enum):
    INBOUND     = "inbound"       # Truck unloading area — items arrive here
    CAMERA_SCAN = "camera_scan"   # AI camera identifies item
    SORTING     = "sorting"       # Agent decides: storage or dispatch
    STORAGE     = "storage"       # Conveyor routes to shelf area
    DISPATCH    = "dispatch"      # Conveyor routes to dispatch bay for driver
    COMPLETED   = "completed"     # Item processed

class ConveyorStatus(str, Enum):
    IDLE     = "idle"
    RUNNING  = "running"
    PAUSED   = "paused"
    ERROR    = "error"

class ItemStatus(str, Enum):
    ON_BELT      = "on_belt"
    SCANNING     = "scanning"
    IDENTIFIED   = "identified"
    ROUTING      = "routing"
    STORED       = "stored"
    DISPATCHED   = "dispatched"
    UNKNOWN      = "unknown"
    ERROR        = "error"

# ── Zone config with icons ────────────────────────────────────────
ZONE_CONFIG = {
    Zone.INBOUND:     {"icon": "📥", "label": "Inbound Dock",    "color": "#3b82f6"},
    Zone.CAMERA_SCAN: {"icon": "📷", "label": "AI Camera Scan",  "color": "#8b5cf6"},
    Zone.SORTING:     {"icon": "🤖", "label": "Agent Sorting",   "color": "#f59e0b"},
    Zone.STORAGE:     {"icon": "📦", "label": "Storage Shelf",   "color": "#22c55e"},
    Zone.DISPATCH:    {"icon": "🚛", "label": "Dispatch Bay",    "color": "#ef4444"},
    Zone.COMPLETED:   {"icon": "✅", "label": "Completed",       "color": "#64748b"},
}

# ── Conveyor Belt Item ────────────────────────────────────────────
def create_belt_item(item_id: str, description: str = "Unknown item"):
    """Create a new item entering the conveyor belt."""
    return {
        "item_id": item_id,
        "description": description,
        "entered_at": datetime.now().strftime("%H:%M:%S"),
        "current_zone": Zone.INBOUND,
        "status": ItemStatus.ON_BELT,
        "sku": None,
        "identified_name": None,
        "confidence": 0.0,
        "target_shelf": None,
        "target_order": None,     # If matched to an order, route to dispatch
        "route": "storage",       # "storage" or "dispatch"
        "history": [
            {"time": datetime.now().strftime("%H:%M:%S"),
             "zone": Zone.INBOUND,
             "event": "Item entered conveyor belt"}
        ],
    }

# ── Conveyor Belt State ───────────────────────────────────────────
def create_conveyor_state():
    """Initialize conveyor belt system state."""
    return {
        "status": ConveyorStatus.IDLE,
        "speed": 1.0,               # 1.0 = normal, 0.5 = slow, 2.0 = fast
        "items_on_belt": [],         # List of belt items currently being processed
        "items_completed": [],       # Completed items log
        "total_processed": 0,
        "total_stored": 0,
        "total_dispatched": 0,
        "total_errors": 0,
        "camera_active": True,
        "belt_direction": "forward", # forward / reverse
        "event_log": [],
        "started_at": None,
    }

# ── Camera Identification (AI) ────────────────────────────────────
def camera_identify_item(item: dict, inventory: dict, ai_call_fn=None, api_key: str = "") -> dict:
    """
    Simulate camera + AI identification of item on belt.
    In real deployment: captures frame from camera → sends to Gemini/LLaVA.
    For demo: uses item description + AI to identify.
    Returns updated item dict with identification results.
    """
    item["current_zone"] = Zone.CAMERA_SCAN
    item["status"] = ItemStatus.SCANNING
    item["history"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "zone": Zone.CAMERA_SCAN,
        "event": "Camera scanning item..."
    })

    # Build inventory context for AI
    inv_context = {
        sku: {"name": data["name"], "shelf": data["shelf"], "barcode": data.get("barcode", "")}
        for sku, data in inventory.items()
    }

    prompt = f"""You are an AI camera system on a warehouse conveyor belt.
An item is passing through the camera zone.

Item description from sensor: "{item['description']}"

Available inventory items:
{json.dumps(inv_context, indent=2)}

Identify this item. Return JSON only:
{{
  "identified": true/false,
  "sku": "SKU code or null",
  "name": "item name",
  "confidence": 0.0-1.0,
  "target_shelf": "shelf location or null"
}}"""

    result = {"identified": False, "sku": None, "name": "Unknown", "confidence": 0.0, "target_shelf": None}

    # Try AI identification
    if ai_call_fn:
        try:
            ai_text = ai_call_fn(prompt)
            import re
            json_match = re.search(r"\{.*\}", ai_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                result = parsed
        except Exception:
            pass

    # Fallback: keyword matching
    if not result.get("identified"):
        desc_lower = item["description"].lower()
        for sku, data in inventory.items():
            name_lower = data["name"].lower()
            keywords = name_lower.split() + [sku.lower()]
            if any(kw in desc_lower for kw in keywords if len(kw) > 2):
                result = {
                    "identified": True,
                    "sku": sku,
                    "name": data["name"],
                    "confidence": 0.85,
                    "target_shelf": data["shelf"],
                }
                break

    # Update item with identification
    item["sku"] = result.get("sku")
    item["identified_name"] = result.get("name", "Unknown")
    item["confidence"] = result.get("confidence", 0.0)
    item["target_shelf"] = result.get("target_shelf")
    item["status"] = ItemStatus.IDENTIFIED if result.get("identified") else ItemStatus.UNKNOWN
    item["history"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "zone": Zone.CAMERA_SCAN,
        "event": f"Identified: {result.get('name', 'Unknown')} (SKU: {result.get('sku', '?')}) "
                 f"Confidence: {result.get('confidence', 0):.0%}"
    })

    return item


# ── Agent Sorting Decision ────────────────────────────────────────
def agent_sort_decision(item: dict, pending_orders: dict, inventory: dict) -> dict:
    """
    Agent decides where to route this item:
    - If item matches a pending order → route to DISPATCH bay
    - Otherwise → route to STORAGE shelf

    Returns updated item with routing decision.
    """
    item["current_zone"] = Zone.SORTING
    item["status"] = ItemStatus.ROUTING
    item["history"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "zone": Zone.SORTING,
        "event": "Agent analyzing routing..."
    })

    sku = item.get("sku")
    if not sku:
        item["route"] = "storage"
        item["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "zone": Zone.SORTING,
            "event": "⚠️ Unknown item — routing to storage for manual sorting"
        })
        return item

    # Check pending orders for this SKU
    matched_order = None
    for order_id, order in pending_orders.items():
        if order.get("status") in ["Loading", "Pending", "Confirmed"]:
            if sku in order.get("items", {}):
                matched_order = order_id
                break

    if matched_order:
        # Route to dispatch — order needs this item!
        item["route"] = "dispatch"
        item["target_order"] = matched_order
        order_dest = pending_orders[matched_order].get("destination",
                     pending_orders[matched_order].get("customer", "Unknown"))
        item["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "zone": Zone.SORTING,
            "event": f"🚛 ORDER MATCH! {matched_order} → Routing to DISPATCH bay "
                     f"(Destination: {order_dest})"
        })
    else:
        # Route to storage shelf
        item["route"] = "storage"
        item["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "zone": Zone.SORTING,
            "event": f"📦 No pending order — routing to STORAGE shelf {item.get('target_shelf', 'TBD')}"
        })

    return item


# ── Execute Routing ───────────────────────────────────────────────
def execute_routing(item: dict, inventory: dict, conveyor_state: dict) -> dict:
    """
    Move item to final destination zone.
    Updates inventory stock counts.
    """
    if item["route"] == "dispatch":
        item["current_zone"] = Zone.DISPATCH
        item["status"] = ItemStatus.DISPATCHED
        conveyor_state["total_dispatched"] += 1
        item["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "zone": Zone.DISPATCH,
            "event": f"✅ Item arrived at DISPATCH bay — ready for driver pickup "
                     f"(Order: {item.get('target_order', '?')})"
        })
    else:
        item["current_zone"] = Zone.STORAGE
        item["status"] = ItemStatus.STORED
        conveyor_state["total_stored"] += 1

        # Update inventory stock
        sku = item.get("sku")
        if sku and sku in inventory:
            inventory[sku]["current_stock"] += 1

        item["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "zone": Zone.STORAGE,
            "event": f"✅ Item stored at shelf {item.get('target_shelf', 'TBD')} — "
                     f"stock updated"
        })

    # Mark complete
    item["current_zone"] = Zone.COMPLETED
    conveyor_state["total_processed"] += 1

    return item


# ── Full Pipeline: Process One Item ───────────────────────────────
def process_belt_item(item: dict, inventory: dict, pending_orders: dict,
                      conveyor_state: dict, ai_call_fn=None) -> dict:
    """
    Run full conveyor pipeline for one item:
    INBOUND → CAMERA_SCAN → SORTING → STORAGE/DISPATCH → COMPLETED
    """
    # Step 1: Camera identification
    item = camera_identify_item(item, inventory, ai_call_fn)

    # Step 2: Agent sorting decision
    item = agent_sort_decision(item, pending_orders, inventory)

    # Step 3: Execute routing
    item = execute_routing(item, inventory, conveyor_state)

    # Log event
    route_icon = "🚛" if item["route"] == "dispatch" else "📦"
    conveyor_state["event_log"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "item_id": item["item_id"],
        "sku": item.get("sku", "?"),
        "name": item.get("identified_name", "Unknown"),
        "route": item["route"],
        "icon": route_icon,
        "shelf": item.get("target_shelf", "—"),
        "order": item.get("target_order", "—"),
        "confidence": item.get("confidence", 0),
    })

    # Move from active to completed
    if item in conveyor_state["items_on_belt"]:
        conveyor_state["items_on_belt"].remove(item)
    conveyor_state["items_completed"].append(item)

    return item


# ── Conveyor Belt Animation Data ──────────────────────────────────
def get_belt_visual_data(conveyor_state: dict) -> dict:
    """Generate data for the conveyor belt visualization."""
    zones = []
    for zone_id, config in ZONE_CONFIG.items():
        items_in_zone = [
            i for i in conveyor_state["items_on_belt"]
            if i["current_zone"] == zone_id
        ]
        zones.append({
            "zone_id": zone_id,
            "icon": config["icon"],
            "label": config["label"],
            "color": config["color"],
            "item_count": len(items_in_zone),
            "items": items_in_zone,
        })

    return {
        "zones": zones,
        "status": conveyor_state["status"],
        "speed": conveyor_state["speed"],
        "stats": {
            "processed": conveyor_state["total_processed"],
            "stored": conveyor_state["total_stored"],
            "dispatched": conveyor_state["total_dispatched"],
            "errors": conveyor_state["total_errors"],
            "on_belt": len(conveyor_state["items_on_belt"]),
        },
    }
