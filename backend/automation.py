"""
InstaWorker™ Smart Automation Modules
======================================
Module 1: Agent Auto-Order Generator + Driver Task Assignment
Module 2: Shelf Camera Auto-Detection + Email Reorder Alerts
Module 3: Voice TTS for Driver Alerts
"""

import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# MODULE 1: Agent Auto-Order + Driver Task
# ═══════════════════════════════════════════════════════════════════

def agent_auto_generate_orders(inventory: dict, existing_orders: dict,
                                deliveries: dict, drivers: list,
                                ai_call_fn=None) -> dict:
    """
    AI Agent analyzes inventory → auto-generates purchase orders for
    low-stock items → auto-creates delivery tasks for drivers.

    Returns dict with new_orders and new_tasks.
    """
    # Step 1: Find items needing reorder
    reorder_items = {}
    for sku, item in inventory.items():
        current = item["current_stock"]
        min_stock = item.get("min_stock", item.get("reorder_point", 20))
        max_stock = item.get("max_stock", item.get("max_capacity", 200))
        if current < min_stock:
            recommended = max_stock - current
            reorder_items[sku] = {
                "name": item["name"],
                "current_stock": current,
                "min_stock": min_stock,
                "recommended_qty": recommended,
                "supplier": item.get("supplier", "Default Supplier"),
                "unit_price": item.get("unit_price", 0),
                "shelf": item.get("shelf", "—"),
                "urgency": "CRITICAL" if current < min_stock * 0.5 else "LOW",
            }

    if not reorder_items:
        return {"new_orders": {}, "new_tasks": {}, "message": "✅ All stock levels healthy. No orders needed."}

    # Step 2: Generate purchase orders (grouped by supplier)
    supplier_groups = {}
    for sku, info in reorder_items.items():
        supplier = info["supplier"]
        if supplier not in supplier_groups:
            supplier_groups[supplier] = {}
        supplier_groups[supplier][sku] = info

    new_orders = {}
    order_counter = len(existing_orders) + 1
    for supplier, items in supplier_groups.items():
        order_id = f"PO-AUTO-{order_counter:03d}"
        total_value = sum(i["recommended_qty"] * i["unit_price"] for i in items.values())
        new_orders[order_id] = {
            "supplier": supplier,
            "items": {sku: i["recommended_qty"] for sku, i in items.items()},
            "item_details": items,
            "total_value": total_value,
            "status": "Auto-Generated",
            "created_at": datetime.now().strftime("%H:%M:%S"),
            "created_by": "AI Agent",
            "urgency": "CRITICAL" if any(i["urgency"] == "CRITICAL" for i in items.values()) else "LOW",
        }
        order_counter += 1

    # Step 3: Auto-assign delivery tasks to available drivers
    new_tasks = {}
    available_drivers = [d for d in drivers if d.get("available", True)]
    driver_idx = 0

    for order_id, order in new_orders.items():
        if not available_drivers:
            break
        driver = available_drivers[driver_idx % len(available_drivers)]
        task_id = f"TASK-AUTO-{datetime.now().strftime('%H%M%S')}-{driver_idx}"
        new_tasks[task_id] = {
            "driver": driver.get("name", "Unassigned"),
            "driver_id": driver.get("id", "DRV001"),
            "order_id": order_id,
            "supplier": order["supplier"],
            "items": order["items"],
            "type": "PICKUP",  # pickup from supplier
            "status": "Assigned",
            "assigned_at": datetime.now().strftime("%H:%M:%S"),
            "assigned_by": "AI Agent",
            "eta": "TBD",
            "priority": order["urgency"],
        }
        driver_idx += 1

    return {
        "new_orders": new_orders,
        "new_tasks": new_tasks,
        "reorder_items": reorder_items,
        "message": f"🤖 Agent generated {len(new_orders)} purchase order(s) "
                   f"and assigned {len(new_tasks)} driver task(s).",
    }


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: Shelf Camera Auto-Detection + Email Alert
# ═══════════════════════════════════════════════════════════════════

SHELF_CAMERAS = {
    "CAM-A": {"shelf_zone": "A", "shelves": ["A-03-02", "A-04-03", "A-05-01"], "status": "active"},
    "CAM-B": {"shelf_zone": "B", "shelves": ["B-02-01"], "status": "active"},
    "CAM-C": {"shelf_zone": "C", "shelves": ["C-01-01"], "status": "active"},
}

def shelf_camera_scan(inventory: dict, camera_id: str = None, ai_call_fn=None) -> dict:
    """
    Simulate shelf camera scanning stock levels.
    Detects: low stock, empty shelves, misplaced items.
    Returns scan results with alerts.
    """
    cameras = [camera_id] if camera_id else list(SHELF_CAMERAS.keys())
    results = []
    alerts = []

    for cam_id in cameras:
        cam = SHELF_CAMERAS.get(cam_id)
        if not cam or cam["status"] != "active":
            continue

        for shelf_loc in cam["shelves"]:
            # Find items on this shelf
            items_on_shelf = [
                (sku, item) for sku, item in inventory.items()
                if item.get("shelf") == shelf_loc
            ]

            for sku, item in items_on_shelf:
                current = item["current_stock"]
                min_stock = item.get("min_stock", item.get("reorder_point", 20))
                max_stock = item.get("max_stock", item.get("max_capacity", 200))
                fill_pct = current / max_stock * 100 if max_stock > 0 else 0

                status = "OK"
                alert_level = None
                if current == 0:
                    status = "EMPTY"
                    alert_level = "CRITICAL"
                elif current < min_stock * 0.5:
                    status = "CRITICAL"
                    alert_level = "CRITICAL"
                elif current < min_stock:
                    status = "LOW"
                    alert_level = "WARNING"

                scan_result = {
                    "camera": cam_id,
                    "shelf": shelf_loc,
                    "sku": sku,
                    "name": item["name"],
                    "current_stock": current,
                    "min_stock": min_stock,
                    "max_stock": max_stock,
                    "fill_pct": fill_pct,
                    "status": status,
                    "scanned_at": datetime.now().strftime("%H:%M:%S"),
                }
                results.append(scan_result)

                if alert_level:
                    alerts.append({
                        **scan_result,
                        "alert_level": alert_level,
                        "recommended_action": f"Reorder {max_stock - current} units of {sku}",
                    })

    return {
        "scan_results": results,
        "alerts": alerts,
        "total_scanned": len(results),
        "total_alerts": len(alerts),
        "critical_count": sum(1 for a in alerts if a["alert_level"] == "CRITICAL"),
    }


def generate_reorder_email_auto(alerts: list, inventory: dict) -> str:
    """Generate automated reorder email from shelf camera alerts."""
    if not alerts:
        return ""

    # Group by supplier
    supplier_items = {}
    for alert in alerts:
        sku = alert["sku"]
        item = inventory.get(sku, {})
        supplier = item.get("supplier", "Unknown Supplier")
        if supplier not in supplier_items:
            supplier_items[supplier] = []
        supplier_items[supplier].append(alert)

    emails = []
    for supplier, items in supplier_items.items():
        lines = []
        total_value = 0
        for a in items:
            item = inventory.get(a["sku"], {})
            qty = a.get("max_stock", 100) - a["current_stock"]
            price = item.get("unit_price", 0) * qty
            total_value += price
            lines.append(
                f"  • {a['sku']} — {a['name']}: "
                f"Current {a['current_stock']}/{a['max_stock']} → "
                f"Order {qty} units (${price:,.0f})"
            )

        email = f"""📧 AUTO-REORDER EMAIL
━━━━━━━━━━━━━━━━━━━━━━━━━━━
To: {supplier} (procurement)
From: InstaWorker™ Auto-Reorder System
Subject: 🚨 Urgent Reorder — {len(items)} item(s) below minimum

Dear {supplier} Team,

Our automated shelf monitoring system has detected the following items require immediate restocking:

{chr(10).join(lines)}

Total Order Value: ${total_value:,.0f}

Please confirm availability and expected delivery date.

Regards,
InstaWorker™ Automated Procurement
━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        emails.append(email)

    return "\n\n".join(emails)


# ═══════════════════════════════════════════════════════════════════
# MODULE 3: Voice TTS for Driver Alerts
# ═══════════════════════════════════════════════════════════════════

# Browser-based TTS HTML component (works without any server-side libs)
VOICE_TTS_HTML = """
<div id="tts-container" style="background:#1e293b;border-radius:12px;padding:16px;text-align:center;">
    <div style="font-size:32px;margin-bottom:8px;" id="tts-icon">🔊</div>
    <div style="color:#f1f5f9;font-weight:700;font-size:16px;" id="tts-status">Voice Ready</div>
    <div style="color:#64748b;font-size:12px;margin:4px 0;" id="tts-text"></div>
    <div style="margin-top:12px;">
        <select id="tts-voice" style="background:#334155;color:#f1f5f9;border:none;
                border-radius:8px;padding:8px 12px;font-size:13px;width:100%;">
        </select>
    </div>
    <div style="margin-top:8px;">
        <input type="range" id="tts-rate" min="0.5" max="2" step="0.1" value="1.0"
               style="width:100%;">
        <div style="color:#64748b;font-size:11px;">Speed: <span id="rate-val">1.0</span>x</div>
    </div>
</div>
<script>
const synth = window.speechSynthesis;
const voiceSelect = document.getElementById('tts-voice');
const rateSlider = document.getElementById('tts-rate');
const rateVal = document.getElementById('rate-val');
const statusEl = document.getElementById('tts-status');
const textEl = document.getElementById('tts-text');
const iconEl = document.getElementById('tts-icon');

// Load voices
function loadVoices() {
    const voices = synth.getVoices();
    voiceSelect.innerHTML = '';
    voices.forEach((v, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = v.name + ' (' + v.lang + ')';
        if (v.default) opt.selected = true;
        voiceSelect.appendChild(opt);
    });
}
loadVoices();
synth.onvoiceschanged = loadVoices;

rateSlider.oninput = () => { rateVal.textContent = rateSlider.value; };

// Global speak function — called from Streamlit
window.speakText = function(text) {
    if (!text) return;
    synth.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    const voices = synth.getVoices();
    utter.voice = voices[voiceSelect.value] || voices[0];
    utter.rate = parseFloat(rateSlider.value);
    utter.onstart = () => {
        iconEl.textContent = '🗣️';
        statusEl.textContent = 'Speaking...';
        textEl.textContent = text;
    };
    utter.onend = () => {
        iconEl.textContent = '🔊';
        statusEl.textContent = 'Voice Ready';
    };
    synth.speak(utter);
};

// Auto-speak if message passed via URL hash
if (window.location.hash) {
    const msg = decodeURIComponent(window.location.hash.substring(1));
    if (msg) setTimeout(() => window.speakText(msg), 500);
}
</script>
"""

def generate_driver_voice_alerts(deliveries: dict, driver_id: str) -> list:
    """Generate voice alert messages for a specific driver."""
    alerts = []
    for del_id, d in deliveries.items():
        if d.get("driver_id") != driver_id:
            continue

        status = d["status"]
        dest = d["destination"]

        if status == "Loading":
            alerts.append(f"Delivery {del_id} is being loaded for {dest}. Please stand by.")
        elif status == "Picked":
            items_str = ", ".join(
                f"{qty} units of {sku}" for sku, qty in d["items"].items()
            )
            alerts.append(f"Delivery {del_id} is ready. Items: {items_str}. Destination: {dest}.")
        elif status == "Dispatched":
            alerts.append(f"Attention driver. Delivery {del_id} dispatched. "
                          f"Please proceed to {dest}. Estimated arrival: {d.get('eta', 'unknown')}.")
        elif status == "In Transit":
            alerts.append(f"You are in transit to {dest}. ETA: {d.get('eta', 'unknown')}.")
        elif status == "Delivered":
            alerts.append(f"Delivery {del_id} to {dest} confirmed. Good job!")

    if not alerts:
        alerts.append("No active deliveries at this time.")

    return alerts


def generate_voice_status_summary(inventory: dict, deliveries: dict) -> str:
    """Generate a voice summary of warehouse status for managers."""
    critical = sum(1 for item in inventory.values()
                   if item["current_stock"] < item.get("min_stock", item.get("reorder_point", 20)))
    active_del = sum(1 for d in deliveries.values() if d["status"] not in ["Delivered"])
    total_items = len(inventory)

    return (f"Warehouse status report. "
            f"{total_items} items tracked. "
            f"{critical} items at critical stock level. "
            f"{active_del} active deliveries in progress. "
            f"{'Immediate reorder recommended.' if critical > 0 else 'All systems normal.'}")
