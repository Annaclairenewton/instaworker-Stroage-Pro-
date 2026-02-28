"""
InstaWorker™ Backend — Data & Business Logic
============================================
All warehouse data, inventory logic, demand forecasting,
reorder calculations, and AI engine live here.
Frontend imports from this module.
"""

import json
import math
import socket
import urllib.request
import urllib.parse
import re
from datetime import datetime

# ── Employee Credentials ──────────────────────────────────────────
EMPLOYEES = {
    "MGR001": {"name": "Claire Newton",    "role": "manager", "pin": "1234"},
    "MGR002": {"name": "David Zhang",      "role": "manager", "pin": "5678"},
    "STF001": {"name": "James Rodriguez",  "role": "staff",   "pin": "1111"},
    "STF002": {"name": "Aisha Patel",      "role": "staff",   "pin": "2222"},
    "DRV001": {"name": "Mike Johnson",     "role": "driver",  "pin": "3333"},
    "DRV002": {"name": "Sarah Chen",       "role": "driver",  "pin": "4444"},
    "SLS001": {"name": "David Park",       "role": "sales",   "pin": "5555"},
    "SLS002": {"name": "Emma Wilson",      "role": "sales",   "pin": "6666"},
}

# ── Inventory ─────────────────────────────────────────────────────
INVENTORY_DEFAULT = {
    "CB-1201-A": {
        "name": "Circuit Breaker 20A", "supplier": "Eaton Corporation",
        "contact": "john.smith@eaton.com", "current_stock": 45,
        "unit_price": 45.0, "unit_volume": 0.5, "min_order_qty": 50,
        "max_stock": 500, "shelf": "A-03-02", "barcode": "8901001234567",
        "sales_history": [18,22,19,25,21,20,23,19,22,24,20,21,23,22,20],
    },
    "SW-4400-B": {
        "name": "Safety Switch 60A", "supplier": "Eaton Corporation",
        "contact": "john.smith@eaton.com", "current_stock": 180,
        "unit_price": 120.0, "unit_volume": 1.2, "min_order_qty": 20,
        "max_stock": 200, "shelf": "A-05-01", "barcode": "8901001234568",
        "sales_history": [5,4,6,5,4,5,6,4,5,5,4,6,5,4,5],
    },
    "TR-9900-C": {
        "name": "Transformer 10KVA", "supplier": "ABB Ltd",
        "contact": "sales@abb.com", "current_stock": 3,
        "unit_price": 850.0, "unit_volume": 8.0, "min_order_qty": 2,
        "max_stock": 20, "shelf": "C-01-01", "barcode": "8901001234569",
        "sales_history": [1,2,1,1,2,1,1,2,1,1,2,1,1,2,1],
    },
    "CB-2200-D": {
        "name": "Circuit Breaker 100A", "supplier": "Schneider Electric",
        "contact": "orders@schneider.com", "current_stock": 12,
        "unit_price": 180.0, "unit_volume": 1.5, "min_order_qty": 10,
        "max_stock": 200, "shelf": "A-04-03", "barcode": "8901001234570",
        "sales_history": [22,25,28,24,26,23,27,25,24,26,25,28,23,26,24],
    },
    "PNL-100-A": {
        "name": "Panel Board 100A", "supplier": "Siemens AG",
        "contact": "supply@siemens.com", "current_stock": 8,
        "unit_price": 320.0, "unit_volume": 3.0, "min_order_qty": 5,
        "max_stock": 50, "shelf": "B-02-01", "barcode": "8901001234571",
        "sales_history": [3,4,3,4,3,4,3,4,3,4,3,4,3,4,3],
    },
}

WAREHOUSE_CONFIG = {
    "total_capacity": 5000,
    "safety_stock_days": 14,
    "lead_time_days": 7,
}

PURCHASE_ORDERS_DEFAULT = {
    "PO-2024-001": {
        "supplier": "Eaton Corporation", "contact": "john.smith@eaton.com",
        "items": {"CB-1201-A": 200, "SW-4400-B": 50},
        "status": "Confirmed", "expected_date": "2024-02-15",
        "total_value": 15000.0,
    },
    "PO-2024-002": {
        "supplier": "ABB Ltd", "contact": "sales@abb.com",
        "items": {"TR-9900-C": 5},
        "status": "Pending", "expected_date": "2024-02-20",
        "total_value": 4250.0,
    },
}

DELIVERIES_DEFAULT = {
    "DEL-001": {
        "driver": "Mike Johnson", "driver_id": "DRV001",
        "destination": "Tesla Fremont Plant",
        "address": "45500 Fremont Blvd, Fremont, CA",
        "items": {"CB-1201-A": 50, "CB-2200-D": 30},
        "status": "Dispatched", "eta": "Today 14:30",
        "history": [
            {"time": "08:00", "status": "Loading",    "by": "System"},
            {"time": "09:15", "status": "Picked",     "by": "STF001"},
            {"time": "09:30", "status": "Dispatched", "by": "STF001"},
        ],
    },
    "DEL-002": {
        "driver": "Sarah Chen", "driver_id": "DRV002",
        "destination": "Google Data Center",
        "address": "1600 Amphitheatre Pkwy, Mountain View, CA",
        "items": {"TR-9900-C": 3, "PNL-100-A": 2},
        "status": "Loading", "eta": "Today 16:00",
        "history": [{"time": "10:00", "status": "Loading", "by": "System"}],
    },
}

SALES_ORDERS_DEFAULT = {
    "SO-001": {
        "customer": "Tesla Fremont Plant", "contact": "procurement@tesla.com",
        "items": {"CB-1201-A": 50, "CB-2200-D": 30},
        "status": "Confirmed", "created_by": "SLS001",
        "created_at": "08:00", "delivery_id": "DEL-001",
        "notes": "Urgent — production line dependency",
    },
    "SO-002": {
        "customer": "Google Data Center", "contact": "ops@google.com",
        "items": {"TR-9900-C": 3, "PNL-100-A": 2},
        "status": "Pending", "created_by": "SLS002",
        "created_at": "09:30", "delivery_id": "DEL-002",
        "notes": "Standard delivery window",
    },
}

TASK_STATES = ["Loading", "Picked", "Dispatched", "In Transit", "Delivered"]
TASK_ICONS  = {"Loading":"🔵","Picked":"🟣","Dispatched":"🟡","In Transit":"🚛","Delivered":"✅"}

# ── Business Logic ────────────────────────────────────────────────

def check_network() -> bool:
    try:
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except Exception:
        return False

def get_total_capacity_used(inventory: dict) -> float:
    return sum(
        inventory[s]["current_stock"] * inventory[s]["unit_volume"]
        for s in inventory
    )

def forecast_demand(sku: str, inventory: dict) -> dict:
    item = inventory.get(sku)
    if not item:
        return {"daily_avg": 0, "trend_pct": 0, "demand_level": "UNKNOWN",
                "forecast_7d": 0, "forecast_14d": 0, "news_signal": 0,
                "news_summary": "", "online": False}
    h = item["sales_history"]
    n = len(h)
    weights = list(range(1, n+1))
    weighted_avg = sum(h[i]*weights[i] for i in range(n)) / sum(weights)
    recent = sum(h[-7:]) / 7
    older  = sum(h[:7]) / 7
    trend  = (recent - older) / older * 100 if older > 0 else 0
    daily  = max(0, weighted_avg * (1 + trend/100 * 0.5))
    level  = "🔥 HIGH" if daily > 15 else ("📈 MEDIUM" if daily > 5 else "📉 LOW")
    return {
        "daily_avg": round(daily, 1), "base_daily": round(weighted_avg, 1),
        "trend_pct": round(trend, 1), "news_signal": 0,
        "news_summary": "", "demand_level": level,
        "forecast_7d": round(daily*7), "forecast_14d": round(daily*14),
        "online": False,
    }

def calculate_reorder(sku: str, inventory: dict) -> dict:
    item = inventory.get(sku)
    if not item:
        return {}
    f = forecast_demand(sku, inventory)
    daily    = f["daily_avg"]
    safety   = daily * WAREHOUSE_CONFIG["safety_stock_days"]
    reorder_pt = safety + daily * WAREHOUSE_CONFIG["lead_time_days"]
    stock    = item["current_stock"]
    days_left = round(stock / daily, 0) if daily > 0 else 999
    annual   = daily * 365
    if annual > 0 and item["unit_price"] > 0:
        eoq = math.sqrt((2 * annual * 50) / (item["unit_price"] * 0.20))
    else:
        eoq = item["min_order_qty"]
    avail_cap = (WAREHOUSE_CONFIG["total_capacity"] - get_total_capacity_used(inventory)) / item["unit_volume"]
    base_qty = min(max(eoq, item["min_order_qty"]), item["max_stock"] - stock, avail_cap)
    if f["trend_pct"] > 20:  base_qty *= 1.2
    if f["trend_pct"] < -20 and stock > reorder_pt * 2: base_qty = 0
    final_qty = max(0, round(base_qty / item["min_order_qty"]) * item["min_order_qty"])

    if stock < safety * 0.5:          status, urgency = "🚨 CRITICAL", "URGENT"
    elif stock < reorder_pt:           status, urgency = "⚠️ LOW",      "ORDER NOW"
    elif stock > item["max_stock"]*0.9: status, urgency = "📦 OVERSTOCK","REDUCE"
    else:                              status, urgency = "✅ OK",        "MONITOR"

    return {
        "status": status, "urgency": urgency,
        "current_stock": stock, "safety_stock": round(safety),
        "reorder_point": round(reorder_pt), "recommended_qty": int(final_qty),
        "days_left": days_left, "needs_reorder": stock < reorder_pt,
    }

def advance_task(deliveries: dict, del_id: str, new_status: str, by: str):
    """Advance delivery to next state and record history."""
    deliveries[del_id]["status"] = new_status
    deliveries[del_id].setdefault("history", []).append({
        "time": datetime.now().strftime("%H:%M"),
        "status": new_status,
        "by": by,
    })
