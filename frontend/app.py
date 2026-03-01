"""
InstaWorker™ Frontend — Streamlit UI
All views: Manager / Driver / Sales
Imports backend from ../backend/core.py
"""

import sys, os
import streamlit as st
import pandas as pd
import json
import copy
import re
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from core import (
    EMPLOYEES, INVENTORY, PURCHASE_ORDERS, DELIVERIES,
    WAREHOUSE_CONFIG, TASK_STATES, TASK_ICONS,
    calculate_reorder, forecast_demand, check_network,
    fetch_supply_chain_news, generate_reorder_email,
    get_total_capacity_used, ai_call, ai_call_with_rag,
    crag_search, rag_search, build_rag_index, chunk_text,
    get_embedding_model, RAG_AVAILABLE, PDF_AVAILABLE,
    OLLAMA_AVAILABLE, RAG_DIR, get_company_rag_path,
    _rule_based_fallback, BARCODE_SCANNER_HTML,
)
from conveyor import (
    create_belt_item, create_conveyor_state, process_belt_item,
    get_belt_visual_data, Zone, ConveyorStatus, ZONE_CONFIG,
    camera_identify_item, agent_sort_decision, execute_routing,
)
from automation import (
    agent_auto_generate_orders, shelf_camera_scan,
    generate_reorder_email_auto, SHELF_CAMERAS,
    VOICE_TTS_HTML, generate_driver_voice_alerts,
    generate_voice_status_summary,
)


# ── Session State ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
# SESSION STATE — Single source of truth for all roles
# Persists across page refreshes within the same browser session
# ══════════════════════════════════════════════════════════════════

# ── Auth ──
if "role"          not in st.session_state: st.session_state.role          = None
if "employee_name" not in st.session_state: st.session_state.employee_name = ""
if "employee_id"   not in st.session_state: st.session_state.employee_id   = ""
if "login_error"   not in st.session_state: st.session_state.login_error   = ""
if "company_id"    not in st.session_state: st.session_state.company_id    = "default"
if "sales_orders"  not in st.session_state:
    st.session_state.sales_orders = {
        "SO-001": {
            "customer": "Tesla Fremont Plant",
            "contact": "procurement@tesla.com",
            "items": {"CB-1201-A": 50, "CB-2200-D": 30},
            "status": "Confirmed",
            "created_by": "SLS001",
            "created_at": "08:00",
            "delivery_id": "DEL-001",
            "notes": "Urgent — production line dependency",
        },
        "SO-002": {
            "customer": "Google Data Center",
            "contact": "ops@google.com",
            "items": {"TR-9900-C": 3, "PNL-100-A": 2},
            "status": "Pending",
            "created_by": "SLS002",
            "created_at": "09:30",
            "delivery_id": "DEL-002",
            "notes": "Standard delivery window",
        },
    }

# ── AI ──
if "api_key"                not in st.session_state: st.session_state.api_key                = ""
if "ai_mode"                not in st.session_state: st.session_state.ai_mode                = "gemini"
if "api_server_url"         not in st.session_state: st.session_state.api_server_url         = "http://127.0.0.1:8000"
if "ai_call_log"            not in st.session_state: st.session_state.ai_call_log            = []
if "cloud_tuned_model_id"   not in st.session_state: st.session_state.cloud_tuned_model_id   = ""

# ── Inventory — persists across refreshes ──
if "live_inventory" not in st.session_state:
    # Deep copy global INVENTORY into session state so edits persist
    import copy
    st.session_state.live_inventory = copy.deepcopy(INVENTORY)

# Always use session state inventory (not global)
INVENTORY = st.session_state.live_inventory

# ── Scan log ──
if "scan_log" not in st.session_state: st.session_state.scan_log = []

# ── Task State Machine ──
# States: Loading → Picked → Dispatched → In Transit → Delivered
# Transitions:
#   Manager/Staff: Loading → Picked
#   Staff:         Picked  → Dispatched (after confirming all items pulled)
#   Driver:        Dispatched → In Transit → Delivered
TASK_STATES  = ["Loading", "Picked", "Dispatched", "In Transit", "Delivered"]
TASK_ICONS   = {"Loading":"🔵","Picked":"🟣","Dispatched":"🟡","In Transit":"🚛","Delivered":"✅"}
TASK_OWNER   = {"Loading":"System","Picked":"System","Dispatched":"Manager","In Transit":"Driver","Delivered":"Driver"}

if "deliveries" not in st.session_state:
    st.session_state.deliveries = {
        "DEL-001": {
            "driver": "Mike Johnson", "driver_id": "DRV001",
            "destination": "Tesla Fremont Plant",
            "address": "45500 Fremont Blvd, Fremont, CA",
            "items": {"CB-1201-A": 50, "CB-2200-D": 30},
            "status": "Dispatched",
            "eta": "Today 14:30",
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
            "status": "Loading",
            "eta": "Today 16:00",
            "history": [
                {"time": "10:00", "status": "Loading", "by": "System"},
            ],
        },
    }

def advance_task(del_id: str, new_status: str, by: str):
    """Advance a delivery to next state and record history."""
    st.session_state.deliveries[del_id]["status"] = new_status
    st.session_state.deliveries[del_id].setdefault("history", []).append({
        "time": datetime.now().strftime("%H:%M"),
        "status": new_status,
        "by": by,
    })

# ═══════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ═══════════════════════════════════════════════════════════════════
if st.session_state.role is None:
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:

        # Header
        st.markdown("""
        <div style='text-align:center; padding:32px 0 24px 0;'>
            <div style='font-size:52px;'>🏭</div>
            <h1 style='color:#f1f5f9;font-size:36px;margin:8px 0;'>InstaWorker™ Pro</h1>
            <p style='color:#94a3b8;font-size:16px;'>Autonomous Warehouse Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        # Role cards — equal height via fixed-height HTML
        st.markdown("#### Select Your Role")
        st.markdown("""
        <div style='display:flex;gap:10px;margin:16px 0;'>
            <div style='flex:1;background:linear-gradient(135deg,#1e3a8a,#1d4ed8);
                        border-radius:16px;padding:20px 12px;text-align:center;height:150px;
                        display:flex;flex-direction:column;justify-content:center;'>
                <div style='font-size:32px;'>👔</div>
                <div style='color:white;font-weight:700;font-size:14px;margin:6px 0;'>Manager</div>
                <div style='color:#93c5fd;font-size:11px;'>Full access · Conveyor · Overrides</div>
            </div>
            <div style='flex:1;background:linear-gradient(135deg,#7c2d12,#c2410c);
                        border-radius:16px;padding:20px 12px;text-align:center;height:150px;
                        display:flex;flex-direction:column;justify-content:center;'>
                <div style='font-size:32px;'>🚛</div>
                <div style='color:white;font-weight:700;font-size:14px;margin:6px 0;'>Driver</div>
                <div style='color:#fdba74;font-size:11px;'>Deliveries</div>
            </div>
            <div style='flex:1;background:linear-gradient(135deg,#4a1d96,#7c3aed);
                        border-radius:16px;padding:20px 12px;text-align:center;height:150px;
                        display:flex;flex-direction:column;justify-content:center;'>
                <div style='font-size:32px;'>💼</div>
                <div style='color:white;font-weight:700;font-size:14px;margin:6px 0;'>Sales</div>
                <div style='color:#ddd6fe;font-size:11px;'>Orders · Customers · Corrections</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Login form
        st.markdown("#### 🔐 Employee Login")
        emp_id  = st.text_input("Employee ID", placeholder="e.g. MGR001 / DRV001 / SLS001").strip().upper()
        emp_pin = st.text_input("PIN", type="password", placeholder="4-digit PIN")
        api_key = st.text_input("Gemini API Key (for AI features)", type="password",
                                value=st.session_state.api_key)

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        if st.button("🔓 Login", use_container_width=True, type="primary"):
            if emp_id in EMPLOYEES and EMPLOYEES[emp_id]["pin"] == emp_pin:
                emp = EMPLOYEES[emp_id]
                st.session_state.role          = emp["role"]
                st.session_state.employee_name = emp["name"]
                st.session_state.employee_id   = emp_id
                st.session_state.api_key       = api_key
                st.session_state.login_error   = ""
                st.rerun()
            else:
                st.session_state.login_error = "❌ Invalid Employee ID or PIN. Please try again."
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#1e293b;border-radius:8px;padding:12px;font-size:12px;color:#64748b;'>
        <b style='color:#94a3b8;'>Demo credentials:</b><br>
        Manager: MGR001 / 1234 &nbsp;|&nbsp; Driver: DRV001 / 3333 &nbsp;|&nbsp; Sales: SLS001 / 5555
        </div>
        """, unsafe_allow_html=True)
        st.caption("Edge-Native · Data stays on premise · Google DeepMind × InstaLILY 2026")

    st.stop()

# ── Sidebar (logged in) ───────────────────────────────────────────
role_icons = {"manager": "👔", "driver": "🚛", "sales": "💼"}
role_labels = {"manager": "Manager", "driver": "Driver", "sales": "Sales"}

with st.sidebar:
    st.markdown(f"### {role_icons[st.session_state.role]} {role_labels[st.session_state.role]}")
    st.caption(f"👤 {st.session_state.employee_name}  ·  {st.session_state.employee_id}")
    if st.button("🔓 Logout", use_container_width=True):
        st.session_state.role = None
        st.session_state.employee_name = ""
        st.session_state.employee_id = ""
        st.rerun()
    st.divider()

    if st.session_state.role in ["manager", "sales"]:
        used = get_total_capacity_used()
        pct  = used / WAREHOUSE_CONFIG["total_capacity"] * 100
        st.metric("Warehouse Capacity", f"{pct:.0f}% used")
        st.progress(pct/100)

        critical = [sku for sku in INVENTORY if calculate_reorder(sku).get("status") == "🚨 CRITICAL"]
        if critical:
            st.error(f"🚨 {len(critical)} item(s) critical")
        st.divider()

    # ── AI Mode Toggle ──
    st.markdown("**🤖 AI Engine**")
    ai_mode = st.radio(
        "Mode",
        ["gemini", "api_server", "ollama"],
        format_func=lambda x: {"gemini": "☁️ Gemini", "api_server": "🌐 API Server", "ollama": "🔒 Ollama"}.get(x, x),
        index=["gemini", "api_server", "ollama"].index(st.session_state.ai_mode) if st.session_state.ai_mode in ("gemini", "api_server", "ollama") else 0,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state.ai_mode = ai_mode

    if ai_mode == "gemini":
        st.caption("☁️ Cloud · Requires internet")
    elif ai_mode == "api_server":
        url = st.text_input("API 地址", value=st.session_state.api_server_url, placeholder="http://127.0.0.1:8000", key="api_server_url_input")
        if url: st.session_state.api_server_url = url.strip().rstrip("/")
        st.caption("🌐 需先运行: python api_server.py")
    else:
        status_color = "🟢" if OLLAMA_AVAILABLE else "🔴"
        st.caption(f"{status_color} Offline · No data leaves device")

        # Model selector for Gemma family + custom (e.g. fine-tuned from zip)
        if "ollama_model" not in st.session_state:
            st.session_state.ollama_model = "gemma3"
        if "ollama_custom_model" not in st.session_state:
            st.session_state.ollama_custom_model = ""
        gemma_models = {
            "gemma3": "Gemma3 4B (Fast ⚡)",
            "gemma3:12b": "Gemma3 12B (Smart 🧠)",
            "gemma3:1b": "Gemma3 1B (Ultra-fast 🚀)",
            "gemma2:2b": "Gemma2 2B (Lightweight)",
        }
        selected_model = st.selectbox(
            "Local Model:",
            options=list(gemma_models.keys()),
            format_func=lambda x: gemma_models[x],
            index=list(gemma_models.keys()).index(st.session_state.ollama_model) if st.session_state.ollama_model in gemma_models else 0,
            key="model_selector"
        )
        custom_name = st.text_input(
            "Or use fine-tuned model (from zip):",
            value=st.session_state.ollama_custom_model or "",
            placeholder="e.g. warehouse-assistant (after ollama create)",
            key="ollama_custom_model_input"
        )
        if custom_name is not None:
            st.session_state.ollama_custom_model = (custom_name or "").strip()
        if st.session_state.ollama_custom_model:
            st.session_state.ollama_model = st.session_state.ollama_custom_model
            st.caption("Using custom model: " + st.session_state.ollama_custom_model)
        else:
            st.session_state.ollama_model = selected_model

        # Quick pull button (only for preset models)
        if st.button(f"📥 Pull {selected_model}", use_container_width=True, key="pull_model"):
            with st.spinner(f"Downloading {selected_model}..."):
                import subprocess
                result = subprocess.run(["ollama", "pull", selected_model],
                                       capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    st.success(f"✅ {selected_model} ready!")
                else:
                    st.error(f"❌ {result.stderr[:100]}")

    if st.session_state.role == "manager" and ai_mode == "gemini":
        new_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
        if new_key: st.session_state.api_key = new_key
        tuned_id = st.text_input("Cloud-trained model ID (optional)", value=st.session_state.cloud_tuned_model_id, placeholder="e.g. tunedModels/xxx or leave blank for default")
        if tuned_id is not None: st.session_state.cloud_tuned_model_id = tuned_id
        if st.session_state.cloud_tuned_model_id:
            st.caption("✅ Using cloud-trained model for best results. See docs/CLOUD_TRAINING.md")

    st.caption("InstaWorker™ Pro v5")
    st.caption("Edge-Native · Secure · Offline-Ready")
    st.divider()
    # Show last AI engine used
    if st.session_state.ai_call_log:
        last = st.session_state.ai_call_log[-1]
        st.caption(f"Last AI: {last['layer']} — {last['status'][:20]}")


# ═══════════════════════════════════════════════════════════════════
# MANAGER VIEW
# ═══════════════════════════════════════════════════════════════════
if st.session_state.role == "manager":
    st.title("👔 Manager Dashboard")

    # Network status banner
    net_ok = check_network()
    if net_ok:
        st.success("🌐 Online — Demand forecast includes real-time supply chain news signals", icon="🌐")
    else:
        st.warning("📴 Offline — Demand forecast based on historical data only (no news signals)", icon="📴")

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    critical_n = sum(1 for s in INVENTORY if calculate_reorder(s).get("status")=="🚨 CRITICAL")
    low_n      = sum(1 for s in INVENTORY if calculate_reorder(s).get("status")=="⚠️ LOW")
    reorder_val = sum(calculate_reorder(s)["recommended_qty"]*INVENTORY[s]["unit_price"]
                      for s in INVENTORY if calculate_reorder(s)["recommended_qty"]>0)
    c1.metric("Total SKUs", len(INVENTORY))
    c2.metric("🚨 Critical", critical_n)
    c3.metric("⚠️ Low Stock", low_n)
    c4.metric("Active Orders", len(PURCHASE_ORDERS))
    c5.metric("Reorder Value", f"${reorder_val:,.0f}")

    st.divider()

    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13,tab14 = st.tabs(["📦 Inventory","🚛 Orders","🚛 Deliveries","📋 Inbound","🔍 Find","📊 Analytics","🤖 AI Advisor","📥 Import","🏭 Conveyor","📲 Scan","📸 Photo ID","🤖 Auto-Order","📷 Shelf Cam","🔊 Voice"])

    # ── Inventory & Reorder ──
    with tab1:
        st.subheader("Real-Time Inventory with Smart Reorder")
        for sku, item in INVENTORY.items():
            r = calculate_reorder(sku)
            f = forecast_demand(sku)
            with st.expander(f"{r['status']}  {sku} — {item['name']}  |  Shelf: {item['shelf']}",
                             expanded=r["urgency"] in ["URGENT","ORDER NOW"]):
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Stock", r["current_stock"], f"{r['days_left']:.0f}d left")
                c2.metric("Safety Stock", r["safety_stock"])
                c3.metric("Reorder Point", r["reorder_point"])
                c4.metric("Daily Demand", r["daily_demand"], f"{r['trend_pct']:+.0f}%")
                c5.metric("Shelf", item["shelf"])

                st.progress(min(item["current_stock"]/item["max_stock"],1.0))
                st.caption(f"Demand: {r['demand_level']}  |  Supplier: {item['supplier']}  |  Barcode: {item['barcode']}")

                if r["recommended_qty"] > 0:
                    if st.button(f"📬 Generate Order for {r['recommended_qty']} units", key=f"ord_{sku}"):
                        st.code(generate_reorder_email(sku, r["recommended_qty"]), language="text")
                elif r["urgency"] == "REDUCE":
                    st.warning("⚠️ Overstocked — consider skipping next order")
                else:
                    st.success("✅ Stock healthy")

    # ── Purchase Orders ──
    with tab2:
        st.subheader("Purchase Orders")
        for po_id, po in PURCHASE_ORDERS.items():
            status_color = "🟡" if po["status"]=="Pending" else "🔵"
            with st.expander(f"{status_color} {po_id} — {po['supplier']} — {po['status']}"):
                st.write(f"**Contact:** {po['contact']}")
                total = sum(v["qty"]*v["price"] for v in po["items"].values())
                st.write(f"**Total Value:** ${total:,.2f}")
                items_df = pd.DataFrame([
                    {"SKU": k, "Qty": v["qty"], "Unit Price": f"${v['price']:.2f}",
                     "Total": f"${v['qty']*v['price']:,.2f}",
                     "Shelf": INVENTORY.get(k,{}).get("shelf","—")}
                    for k,v in po["items"].items()
                ])
                st.dataframe(items_df, use_container_width=True, hide_index=True)

    # ── Delivery Tracker ──
    with tab3:
        st.subheader("🚛 Delivery Tracker — All Shipments")
        for del_id, delivery in st.session_state.deliveries.items():
            status_icons = {"Loading": "🔵", "Dispatched": "🟡", "Delivered": "✅"}
            icon = status_icons.get(delivery["status"], "⚪")
            with st.expander(f"{icon} {del_id} — {delivery['destination']} — {delivery['status']} — ETA: {delivery['eta']}", expanded=True):
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    st.write(f"📍 **Destination:** {delivery['destination']}")
                    st.write(f"🗺️ **Address:** {delivery['address']}")
                    st.write(f"👤 **Driver:** {delivery['driver']}")
                    st.write(f"⏰ **ETA:** {delivery['eta']}")
                    st.markdown("**Items:**")
                    total_val = 0
                    for sku, qty in delivery["items"].items():
                        item = INVENTORY.get(sku, {})
                        val = qty * item.get("unit_price", 0)
                        total_val += val
                        st.write(f"  • {sku} — {item.get('name','Unknown')} × {qty}  |  Shelf: **{item.get('shelf','—')}**  |  ${val:,.0f}")
                    st.write(f"**Total Value: ${total_val:,.2f}**")
                with col_r:
                    # Manager can override status
                    new_status = st.selectbox("Update Status", ["Loading","Dispatched","Delivered"],
                        index=["Loading","Dispatched","Delivered"].index(delivery["status"]),
                        key=f"mgr_status_{del_id}")
                    if st.button("Update", key=f"mgr_upd_{del_id}", use_container_width=True):
                        st.session_state.deliveries[del_id]["status"] = new_status
                        st.success(f"Updated to {new_status}")
                        st.rerun()
                    st.divider()
                    encoded = delivery["address"].replace(" ", "+")
                    st.markdown(f"[🗺️ Open in Maps](https://maps.google.com?q={encoded})")

        st.divider()
        # Summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Deliveries", len(st.session_state.deliveries))
        c2.metric("In Transit", sum(1 for d in st.session_state.deliveries.values() if d["status"]=="Dispatched"))
        c3.metric("Delivered", sum(1 for d in st.session_state.deliveries.values() if d["status"]=="Delivered"))

    # ── Inbound Log ──
    with tab4:
        st.subheader("📋 Inbound / Outbound Log — All Activity")
        if st.session_state.scan_log:
            df_log = pd.DataFrame(st.session_state.scan_log)
            # Summary metrics
            in_count  = len([x for x in st.session_state.scan_log if x.get("operation")=="IN"])
            out_count = len([x for x in st.session_state.scan_log if x.get("operation")=="OUT"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Scans", len(st.session_state.scan_log))
            c2.metric("📥 Stock IN", in_count)
            c3.metric("📤 Stock OUT", out_count)
            st.divider()
            st.dataframe(df_log, use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Log", type="secondary"):
                st.session_state.scan_log = []
                st.rerun()
        else:
            st.info("No activity logged yet. Staff scan activity will appear here in real time.")
            st.markdown("**What this shows:**")
            st.write("Every time a staff member scans a barcode to receive or dispatch goods, it appears here with timestamp, SKU, quantity, shelf location, and staff name.")

    # ── Find Item ──
    with tab5:
        st.subheader("🔍 Find Item — Search by SKU, Name, or Shelf")
        search_query = st.text_input("Search:", placeholder="e.g. CB-1201-A  or  breaker  or  A-03")
        if search_query:
            q = search_query.strip().lower()
            results = {
                sku: item for sku, item in INVENTORY.items()
                if q in sku.lower()
                or q in item["name"].lower()
                or q in item["shelf"].lower()
                or q in item.get("barcode","").lower()
                or q in item.get("supplier","").lower()
            }
            if results:
                st.success(f"Found {len(results)} item(s)")
                for sku, item in results.items():
                    r = calculate_reorder(sku)
                    with st.expander(f"📦 {sku} — {item['name']}", expanded=True):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Shelf Location", item["shelf"])
                        c2.metric("Current Stock", item["current_stock"])
                        c3.metric("Status", r["status"])
                        c4.metric("Unit Price", f"${item['unit_price']:.2f}")
                        st.write(f"**Supplier:** {item['supplier']}  |  **Contact:** {item['contact']}")
                        st.write(f"**Barcode:** {item.get('barcode','—')}  |  **Min Order:** {item['min_order_qty']}  |  **Max Stock:** {item['max_stock']}")
            else:
                st.warning(f"No items found for '{search_query}'")
                st.markdown("**All available SKUs:**")
                for sku, item in INVENTORY.items():
                    st.caption(f"{sku} — {item['name']} — Shelf {item['shelf']}")
        else:
            st.markdown("#### All Items at a Glance")
            for sku, item in INVENTORY.items():
                r = calculate_reorder(sku)
                col_a, col_b, col_c, col_d, col_e = st.columns([2,3,2,2,2])
                col_a.write(f"**{sku}**")
                col_b.write(item["name"])
                col_c.write(f"📍 {item['shelf']}")
                col_d.write(f"Stock: {item['current_stock']}")
                col_e.write(r["status"])

    # ── Analytics ──
    with tab6:
        st.subheader("Demand Analytics & Forecast")
        rows = []
        for sku in INVENTORY:
            r = calculate_reorder(sku)
            f = forecast_demand(sku)
            news_col = f"{f['news_signal']:+.0f}%" if f.get("online") else "—"
            news_reason = f["news_summary"][:40] + "..." if f.get("news_summary") and len(f.get("news_summary","")) > 40 else f.get("news_summary","—")
            rows.append({
                "SKU": sku, "Name": INVENTORY[sku]["name"], "Shelf": INVENTORY[sku]["shelf"],
                "Stock": r["current_stock"], "Days Left": f"{r['days_left']:.0f}",
                "7D Forecast": f["forecast_7d"], "Trend": f"{f['trend_pct']:+.0f}%",
                "News Signal": news_col, "News Reason": news_reason,
                "Demand": f["demand_level"], "Status": r["status"],
                "Rec. Order": r["recommended_qty"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        sel = st.selectbox("View sales history:", list(INVENTORY.keys()))
        if sel:
            h = INVENTORY[sel]["sales_history"]
            chart_df = pd.DataFrame({"Day":[f"D-{14-i}" for i in range(15)],"Sales":h})
            st.bar_chart(chart_df.set_index("Day"))
            f2 = forecast_demand(sel)
            st.info(f"Avg: {f2['daily_avg']} units/day  |  Trend: {f2['trend_pct']:+.0f}%  |  {f2['demand_level']}")

    # ── AI Advisor ──
    with tab7:
        st.subheader("🤖 AI Inventory Advisor")
        if st.button("Run Full AI Analysis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                    summary = [{"sku":s,"stock":calculate_reorder(s)["current_stock"],
                                "days_left":calculate_reorder(s)["days_left"],
                                "status":calculate_reorder(s)["status"],
                                "trend":f"{forecast_demand(s)['trend_pct']:+.0f}%",
                                "rec_order":calculate_reorder(s)["recommended_qty"]}
                               for s in INVENTORY]
                    prompt = f"""You are an expert warehouse manager. Analyze this inventory data and give:
1. Top urgent items (max 3)
2. Overall health assessment
3. One strategic recommendation
Max 200 words. Be direct.
Data: {json.dumps(summary)}"""
                    result = ai_call(prompt)
                    st.markdown(result)

        st.divider()
        st.markdown("**Ask the AI anything:**")
        question = st.text_input("e.g. 'Which items will run out this week?'")
        if st.button("Ask") and question:
            context = json.dumps([{"sku":s,"name":INVENTORY[s]["name"],
                                   **calculate_reorder(s), **forecast_demand(s)} for s in INVENTORY])
            base_prompt = f"Warehouse inventory data: {context}\n\nQuestion: {question}\n\nAnswer concisely:"
            if RAG_AVAILABLE:
                with st.spinner("Searching knowledge base + AI..."):
                    rag_result = ai_call_with_rag(base_prompt, st.session_state.company_id)
                conf = rag_result["confidence"]
                conf_icon = {"high":"🟢","medium":"🟡","low":"🔴","none":"⚫"}.get(conf,"⚪")
                st.caption(f"RAG Confidence: {conf_icon} {conf}" +
                           (" — query corrected" if rag_result["used_correction"] else ""))
                st.markdown(rag_result["answer"])
                if rag_result["chunks_used"]:
                    with st.expander(f"📄 {len(rag_result['chunks_used'])} docs used"):
                        for c in rag_result["chunks_used"]:
                            st.caption(f"[{c['source']}] {c['text'][:150]}...")
            else:
                result = ai_call(base_prompt)
                st.markdown(result)

        st.divider()
        st.markdown("**🔁 AI Engine Call Log**")
        if st.session_state.ai_call_log:
            for entry in reversed(st.session_state.ai_call_log[-8:]):
                color = "#22c55e" if "OK" in entry["status"] else "#ef4444"
                st.markdown(
                    f"<span style='color:#94a3b8;font-size:12px;'>{entry['time']}</span> "
                    f"<span style='font-size:12px;'>{entry['layer']}</span> "
                    f"<span style='color:{color};font-size:12px;'>{entry['status']}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("No AI calls yet.")


    # ── Import Inventory (tab8) ──
    with tab8:
        st.subheader("📥 Import Inventory from Spreadsheet")
        st.info("Upload your existing inventory CSV or Excel file to update the system. New deployment? Start here.")

        col_l, col_r = st.columns([1,1])
        with col_l:
            st.markdown("#### Upload File")
            st.markdown("**Supported formats:** CSV, Excel (.xlsx)")
            st.markdown("**Required columns:**")
            st.code("sku, name, current_stock, unit_price, shelf\n(optional: supplier, contact, min_order_qty, max_stock)")

            uploaded_file = st.file_uploader("Choose file", type=["csv","xlsx","xls"])

            st.divider()
            st.markdown("**Or download template:**")
            template_csv = """sku,name,current_stock,unit_price,shelf,supplier,contact,min_order_qty,max_stock
CB-1201-A,Circuit Breaker 20A,45,45.0,A-03-02,Eaton Corporation,john@eaton.com,50,500
SW-4400-B,Safety Switch 60A,180,120.0,A-05-01,Eaton Corporation,john@eaton.com,20,200"""
            st.download_button("📄 Download CSV Template", template_csv, "inventory_template.csv", "text/csv")

            st.divider()
            st.markdown("**Export for training (no hardcoding):**")
            st.caption("Download current inventory as JSON. Then: python scripts/generate_training_data.py -o training_data.jsonl --inventory exported_inventory.json")
            export_json = json.dumps(INVENTORY, indent=2, ensure_ascii=False)
            st.download_button("📤 Export inventory JSON (for training)", export_json, "exported_inventory.json", "application/json", key="export_inv_json")

        with col_r:
            st.markdown("#### Preview & Import")
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    st.success(f"✅ Found {len(df)} items")
                    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

                    # Validate required columns
                    required = ["sku","name","current_stock","unit_price","shelf"]
                    missing = [c for c in required if c not in df.columns]

                    if missing:
                        st.error(f"Missing columns: {missing}")
                    else:
                        if st.button("✅ Import to System", use_container_width=True, type="primary"):
                            imported = 0
                            for _, row in df.iterrows():
                                sku = str(row["sku"]).strip().upper()
                                INVENTORY[sku] = {
                                    "name":          str(row.get("name", sku)),
                                    "supplier":      str(row.get("supplier", "Unknown")),
                                    "contact":       str(row.get("contact", "")),
                                    "current_stock": int(row.get("current_stock", 0)),
                                    "unit_price":    float(row.get("unit_price", 0)),
                                    "unit_volume":   float(row.get("unit_volume", 1.0)),
                                    "min_order_qty": int(row.get("min_order_qty", 10)),
                                    "max_stock":     int(row.get("max_stock", 1000)),
                                    "shelf":         str(row.get("shelf", "TBD")),
                                    "barcode":       str(row.get("barcode", "")),
                                    "sales_history": [int(row.get("current_stock",0)//30)] * 15,
                                }
                                imported += 1
                            st.success(f"✅ Imported {imported} items successfully!")
                            st.balloons()
                            st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            else:
                st.info("Upload a file to preview")

                st.markdown("**Current inventory:**")
                current_df = pd.DataFrame([
                    {"SKU": k, "Name": v["name"], "Stock": v["current_stock"],
                     "Shelf": v["shelf"], "Supplier": v["supplier"]}
                    for k,v in INVENTORY.items()
                ])
                st.dataframe(current_df, use_container_width=True, hide_index=True)

    # ── Conveyor Belt (tab9) ──
    with tab9:
        st.subheader("🏭 Smart Conveyor Belt System")
        st.info("📷 Camera auto-identifies items → 🤖 Agent routes to storage or dispatch → 🚛 Auto-fulfills orders")

        # Initialize conveyor state
        if "conveyor" not in st.session_state:
            st.session_state.conveyor = create_conveyor_state()
        conv = st.session_state.conveyor

        # ── Control Panel ──
        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns(5)
        with ctrl1:
            if conv["status"] != ConveyorStatus.RUNNING:
                if st.button("▶️ Start Belt", use_container_width=True, type="primary", key="conv_start"):
                    conv["status"] = ConveyorStatus.RUNNING
                    conv["started_at"] = datetime.now().strftime("%H:%M:%S")
                    st.rerun()
            else:
                if st.button("⏸️ Pause Belt", use_container_width=True, key="conv_pause"):
                    conv["status"] = ConveyorStatus.PAUSED
                    st.rerun()
        with ctrl2:
            if st.button("⏹️ Stop Belt", use_container_width=True, key="conv_stop"):
                conv["status"] = ConveyorStatus.IDLE
                st.rerun()
        with ctrl3:
            speed = st.select_slider("Speed", options=[0.5, 1.0, 1.5, 2.0],
                                      value=conv["speed"], format_func=lambda x: f"{x}x",
                                      key="conv_speed")
            conv["speed"] = speed
        with ctrl4:
            cam_label = "🟢 Camera ON" if conv["camera_active"] else "🔴 Camera OFF"
            if st.button(cam_label, use_container_width=True, key="conv_cam"):
                conv["camera_active"] = not conv["camera_active"]
                st.rerun()
        with ctrl5:
            belt_icons = {
                ConveyorStatus.RUNNING: ("🟢", "#22c55e", "RUNNING"),
                ConveyorStatus.PAUSED:  ("🟡", "#f59e0b", "PAUSED"),
                ConveyorStatus.IDLE:    ("⚫", "#64748b", "IDLE"),
                ConveyorStatus.ERROR:   ("🔴", "#ef4444", "ERROR"),
            }
            b_icon, b_color, b_text = belt_icons.get(conv["status"], ("⚫","#64748b","IDLE"))
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:12px;padding:12px;text-align:center;
                        border:2px solid {b_color};'>
                <div style='font-size:24px;'>{b_icon}</div>
                <div style='color:{b_color};font-weight:700;'>{b_text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Conveyor Belt Visualization ──
        st.markdown("#### 🏭 Conveyor Belt — Live View")

        belt_moving = conv["status"] == ConveyorStatus.RUNNING
        move_class = "moving" if belt_moving else ""

        # Count items per zone
        zone_counts = {}
        for zone_id in Zone:
            zone_counts[zone_id] = len([i for i in conv["items_on_belt"] if i["current_zone"] == zone_id])

        st.markdown(f"""
        <style>
        .conv-track {{ height:6px; border-radius:3px; margin:8px 0;
            background:linear-gradient(90deg, #1e293b 0%, #3b82f6 50%, #1e293b 100%);
            background-size:200% 100%; }}
        .conv-track.moving {{ animation: conv-move 1.5s linear infinite; }}
        @keyframes conv-move {{ 0%{{background-position:0% 0%;}} 100%{{background-position:200% 0%;}} }}
        .conv-zones {{ display:flex; align-items:center; justify-content:center; gap:0; padding:12px 0; flex-wrap:wrap; }}
        .conv-zone {{ background:#1e293b; border-radius:12px; padding:14px 10px; text-align:center;
            min-width:120px; border:2px solid #334155; transition:all 0.3s; }}
        .conv-zone.has-items {{ box-shadow:0 0 12px rgba(59,130,246,0.3); transform:scale(1.03); }}
        .conv-arrow {{ color:#3b82f6; font-size:22px; font-weight:700; margin:0 6px; }}
        .conv-fork {{ display:flex; gap:20px; margin-top:12px; justify-content:center; }}
        .conv-fork-branch {{ display:flex; align-items:center; gap:6px; }}
        </style>

        <div class='conv-track {move_class}'></div>
        <div class='conv-zones'>
            <!-- Inbound -->
            <div class='conv-zone {"has-items" if zone_counts.get(Zone.INBOUND,0) > 0 else ""}' style='border-color:#3b82f640;'>
                <div style='font-size:28px;'>📥</div>
                <div style='color:#f1f5f9;font-weight:700;font-size:12px;'>Inbound Dock</div>
                <div style='color:#3b82f6;font-size:11px;'>{zone_counts.get(Zone.INBOUND,0)} items</div>
            </div>
            <div class='conv-arrow'>→</div>
            <!-- Camera -->
            <div class='conv-zone {"has-items" if zone_counts.get(Zone.CAMERA_SCAN,0) > 0 else ""}' style='border-color:#8b5cf640;'>
                <div style='font-size:28px;'>📷</div>
                <div style='color:#f1f5f9;font-weight:700;font-size:12px;'>AI Camera Scan</div>
                <div style='color:#8b5cf6;font-size:11px;'>{zone_counts.get(Zone.CAMERA_SCAN,0)} items</div>
            </div>
            <div class='conv-arrow'>→</div>
            <!-- Sorting -->
            <div class='conv-zone {"has-items" if zone_counts.get(Zone.SORTING,0) > 0 else ""}' style='border-color:#f59e0b40;'>
                <div style='font-size:28px;'>🤖</div>
                <div style='color:#f1f5f9;font-weight:700;font-size:12px;'>Agent Sorting</div>
                <div style='color:#f59e0b;font-size:11px;'>{zone_counts.get(Zone.SORTING,0)} items</div>
            </div>
        </div>
        <div class='conv-track {move_class}'></div>

        <!-- Fork -->
        <div class='conv-fork'>
            <div class='conv-fork-branch'>
                <div class='conv-arrow'>↙</div>
                <div class='conv-zone' style='border-color:#22c55e40;'>
                    <div style='font-size:28px;'>📦</div>
                    <div style='color:#f1f5f9;font-weight:700;font-size:12px;'>Storage Shelf</div>
                    <div style='color:#22c55e;font-size:11px;'>{conv["total_stored"]} stored</div>
                </div>
            </div>
            <div class='conv-fork-branch'>
                <div class='conv-arrow'>↘</div>
                <div class='conv-zone' style='border-color:#ef444440;'>
                    <div style='font-size:28px;'>🚛</div>
                    <div style='color:#f1f5f9;font-weight:700;font-size:12px;'>Dispatch Bay</div>
                    <div style='color:#ef4444;font-size:11px;'>{conv["total_dispatched"]} dispatched</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── KPI ──
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📦 Processed", conv["total_processed"])
        m2.metric("📥 Stored", conv["total_stored"])
        m3.metric("🚛 Dispatched", conv["total_dispatched"])
        m4.metric("🔄 On Belt", len(conv["items_on_belt"]))
        m5.metric("❌ Errors", conv["total_errors"])

        st.divider()

        # ── Add Item to Belt ──
        col_input, col_preview = st.columns([1, 1])
        with col_input:
            st.markdown("#### 📥 Add Item to Conveyor Belt")

            input_mode = st.radio("Input:", ["Select from inventory", "Describe item (AI识别)"],
                                   horizontal=True, key="conv_input_mode")

            item_desc = ""
            conv_qty = 1
            if input_mode == "Select from inventory":
                sel_sku = st.selectbox("Select:", list(INVENTORY.keys()),
                    format_func=lambda x: f"{x} — {INVENTORY[x]['name']} (Shelf: {INVENTORY[x]['shelf']})",
                    key="conv_sel_sku")
                if sel_sku:
                    item_desc = f"{INVENTORY[sel_sku]['name']} SKU:{sel_sku}"
                conv_qty = st.number_input("Quantity:", 1, 100, 1, key="conv_qty_sel")
            else:
                item_desc = st.text_input("Describe the item:",
                    placeholder="e.g. Circuit breaker 20A Eaton", key="conv_desc")
                conv_qty = st.number_input("Quantity:", 1, 100, 1, key="conv_qty_desc")

            if item_desc and st.button("🏭 Send to Belt", use_container_width=True,
                                        type="primary", key="conv_send"):
                for i in range(conv_qty):
                    iid = f"BELT-{datetime.now().strftime('%H%M%S')}-{i+1}"
                    new_item = create_belt_item(iid, item_desc)
                    conv["items_on_belt"].append(new_item)
                st.success(f"✅ {conv_qty} item(s) on belt!")
                st.rerun()

        with col_preview:
            st.markdown("#### 🔄 Items on Belt")
            if conv["items_on_belt"]:
                for itm in conv["items_on_belt"]:
                    zcfg = ZONE_CONFIG.get(itm["current_zone"], {"icon":"?","label":"?","color":"#64748b"})
                    st.markdown(f"""
                    <div style='background:#1e293b;border-radius:8px;padding:10px;margin:4px 0;
                                border-left:4px solid {zcfg["color"]};'>
                        <b style='color:#f1f5f9;'>{itm["item_id"]}</b>
                        <span style='color:#64748b;'> — {itm["description"][:40]}</span><br>
                        <span style='color:{zcfg["color"]};font-size:12px;'>
                            {zcfg["icon"]} {zcfg["label"]} — {itm["status"]}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No items on belt. Add items to start.")

        st.divider()

        # ── Process ──
        if conv["items_on_belt"]:
            st.markdown("#### ⚡ Process Items")
            pcol1, pcol2 = st.columns(2)

            with pcol1:
                if st.button("🔄 Process Next Item", use_container_width=True,
                              type="primary", key="conv_next"):
                    item = conv["items_on_belt"][0]
                    pending = {}
                    pending.update(st.session_state.deliveries)
                    for so_id, so in st.session_state.sales_orders.items():
                        pending[so_id] = so

                    with st.spinner(f"🏭 Processing {item['item_id']}..."):
                        processed = process_belt_item(item, INVENTORY, pending, conv, ai_call_fn=ai_call)

                    route_txt = "🚛 DISPATCH" if processed["route"] == "dispatch" else "📦 STORAGE"
                    order_txt = f" — Order: {processed['target_order']}" if processed.get("target_order") else ""
                    st.success(f"✅ {processed.get('identified_name','?')} ({processed.get('sku','?')}) → {route_txt}{order_txt}")
                    st.rerun()

            with pcol2:
                if st.button("⚡ Process ALL", use_container_width=True, key="conv_all"):
                    pending = {}
                    pending.update(st.session_state.deliveries)
                    for so_id, so in st.session_state.sales_orders.items():
                        pending[so_id] = so

                    items_list = list(conv["items_on_belt"])
                    prog = st.progress(0)
                    for idx, itm in enumerate(items_list):
                        process_belt_item(itm, INVENTORY, pending, conv, ai_call_fn=ai_call)
                        prog.progress((idx+1)/len(items_list))
                    st.success(f"✅ Processed {len(items_list)} items!")
                    st.rerun()

        st.divider()

        # ── Event Log ──
        st.markdown("#### 📋 Conveyor Event Log")
        if conv["event_log"]:
            for evt in reversed(conv["event_log"][-15:]):
                r_color = "#ef4444" if evt["route"] == "dispatch" else "#22c55e"
                st.markdown(f"""
                <div style='background:#1e293b;border-radius:8px;padding:8px 12px;margin:3px 0;
                            font-size:13px;border-left:3px solid {r_color};'>
                    <span style='color:#64748b;'>{evt['time']}</span>&nbsp;
                    {evt['icon']}&nbsp;
                    <b style='color:#f1f5f9;'>{evt.get('name','?')}</b>
                    <span style='color:#64748b;'>({evt.get('sku','?')})</span>
                    → <span style='color:{r_color};font-weight:700;'>{evt['route'].upper()}</span>
                    <span style='color:#64748b;'> | Shelf: {evt.get('shelf','—')} | Order: {evt.get('order','—')}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No events yet. Process items to see activity.")

        # ── Completed Detail ──
        if conv["items_completed"]:
            with st.expander(f"📜 Completed ({len(conv['items_completed'])})"):
                for citm in reversed(conv["items_completed"][-10:]):
                    st.markdown(f"**{citm['item_id']}** — {citm.get('identified_name','?')} "
                                f"({citm.get('sku','?')}) → {citm['route'].upper()}")
                    for h in citm.get("history", []):
                        zcfg = ZONE_CONFIG.get(h.get("zone"), {"icon":""})
                        st.caption(f"  {zcfg.get('icon','')} {h['time']} — {h['event']}")

    # ── Scan In/Out (tab10) — moved from Staff ──
    with tab10:
        st.subheader("📲 Barcode / QR Scanner")
        st.info("💡 Scan items to record inbound/outbound stock movements")

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown("#### 📷 Live Camera Scanner")
            st.components.v1.html(BARCODE_SCANNER_HTML, height=420)
            st.divider()
            st.markdown("#### Or Enter Manually")
            manual_code = st.text_input("SKU / Barcode:", placeholder="e.g. CB-1201-A or 8901001234567", key="mgr_scan_input")

        st.divider()
        _, proc_col, _ = st.columns([1, 2, 1])
        with proc_col:
            st.markdown("#### ⚙️ Result")
            scanned_sku = None
            input_code = manual_code.strip().upper() if manual_code else ""
            if input_code:
                if input_code in INVENTORY:
                    scanned_sku = input_code
                else:
                    for sku, item in INVENTORY.items():
                        if item.get("barcode","") == input_code:
                            scanned_sku = sku
                            break

            if scanned_sku:
                item = INVENTORY[scanned_sku]
                r = calculate_reorder(scanned_sku)
                st.success(f"✅ Found: **{scanned_sku}**")
                st.write(f"**{item['name']}**")
                col_a, col_b = st.columns(2)
                col_a.metric("Current Stock", item["current_stock"])
                col_b.metric("Shelf Location", item["shelf"])
                st.write(f"Status: {r['status']}")
                st.divider()
                operation = st.radio("Operation:", ["📥 Stock IN (Receiving)", "📤 Stock OUT (Dispatch)"], key="mgr_scan_op")
                qty = st.number_input("Quantity:", min_value=1, max_value=9999, value=1, key="mgr_scan_qty")
                note = st.text_input("Note (optional):", placeholder="e.g. PO-2024-001", key="mgr_scan_note")
                if st.button("✅ Confirm", use_container_width=True, type="primary", key="mgr_scan_confirm"):
                    direction = "IN" if "IN" in operation else "OUT"
                    st.session_state.scan_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "sku": scanned_sku, "name": item["name"], "shelf": item["shelf"],
                        "operation": direction, "qty": qty, "note": note,
                        "by": st.session_state.employee_name,
                    })
                    if direction == "IN":
                        INVENTORY[scanned_sku]["current_stock"] += qty
                        st.success(f"✅ {qty} units received → Shelf {item['shelf']}")
                    else:
                        if qty <= item["current_stock"]:
                            INVENTORY[scanned_sku]["current_stock"] -= qty
                            st.success(f"✅ {qty} units dispatched from Shelf {item['shelf']}")
                        else:
                            st.error(f"❌ Insufficient stock! Only {item['current_stock']} available.")
                    r2 = calculate_reorder(scanned_sku)
                    if r2["needs_reorder"]:
                        st.warning(f"⚠️ Reorder triggered: recommend {r2['recommended_qty']} units")
            elif input_code:
                st.error(f"❌ '{input_code}' not found.")
                st.markdown("**Available SKUs:**")
                for sku in INVENTORY:
                    st.caption(f"{sku} — {INVENTORY[sku]['name']} — Shelf {INVENTORY[sku]['shelf']}")

    # ── AI Photo ID (tab11) — moved from Staff ──
    with tab11:
        st.subheader("📸 AI Part Identification")
        st.info("Take a photo of any part — AI will identify the SKU and shelf location")

        col_l, col_r = st.columns([1,1])
        with col_l:
            capture_method = st.radio("Input method:", ["Upload photo", "Use camera"], key="mgr_photo_method")
            if capture_method == "Upload photo":
                photo = st.file_uploader("Upload part photo", type=["jpg","png","jpeg"], key="mgr_photo_upload")
            else:
                photo = st.camera_input("Take a photo with your camera", key="mgr_photo_cam")

        with col_r:
            if photo and st.button("🤖 Identify Part", use_container_width=True, type="primary", key="mgr_photo_id"):
                if not st.session_state.api_key:
                    st.error("API key required for Gemini, or switch to Gemma3 offline mode")
                else:
                    with st.spinner("AI analyzing image..."):
                        from PIL import Image as PILImage
                        from google import genai
                        img = PILImage.open(photo)
                        inventory_info = json.dumps({
                            k: {"name": v["name"], "shelf": v["shelf"]}
                            for k,v in INVENTORY.items()
                        })
                        try:
                            result_text = ai_call(
                                f"Identify this electrical part. Available inventory: {inventory_info}\n"
                                f"Return JSON only: {{\"sku\":\"...\",\"name\":\"...\",\"shelf\":\"...\",\"confidence\":\"high/medium/low\",\"notes\":\"...\"}}",
                                image=img
                            )
                            import re
                            if "```" in result_text:
                                result_text = result_text.split("```")[1].replace("json","").strip()
                            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
                            if json_match:
                                result = json.loads(json_match.group())
                                sku = result.get("sku","")
                                if sku in INVENTORY:
                                    itm = INVENTORY[sku]
                                    r = calculate_reorder(sku)
                                    st.success(f"✅ Identified: **{sku}**")
                                    st.write(f"**{itm['name']}**")
                                    ca, cb, cc = st.columns(3)
                                    ca.metric("Shelf", itm["shelf"])
                                    cb.metric("Stock", itm["current_stock"])
                                    cc.metric("Status", r["status"])
                                    st.caption(f"Confidence: {result.get('confidence','—')} | {result.get('notes','')}")
                                else:
                                    st.warning(f"Part identified as: {result.get('name','unknown')} — not in inventory")
                                    st.json(result)
                            else:
                                st.write(result_text)
                        except Exception as e:
                            st.error(f"Error: {e}")

    # ── Agent Auto-Order (tab12) ──
    with tab12:
        st.subheader("🤖 Agent Auto-Order Generator")
        st.info("AI Agent: analyze inventory → generate purchase orders → assign driver tasks")

        col_al, col_ar = st.columns([1, 1])
        with col_al:
            st.markdown("#### 📊 Items Needing Reorder")
            crit_items = []
            for sku, item in INVENTORY.items():
                min_s = item.get("min_stock", item.get("reorder_point", 20))
                if item["current_stock"] < min_s:
                    urg = "🔴 CRITICAL" if item["current_stock"] < min_s * 0.5 else "🟡 LOW"
                    crit_items.append((sku, item, urg, min_s))
            if crit_items:
                for sku, item, urg, min_s in crit_items:
                    urg_color = "#ef4444" if "CRITICAL" in urg else "#f59e0b"
                    st.markdown(f"<div style='background:#1e293b;border-radius:8px;padding:10px;margin:4px 0;"
                                f"border-left:4px solid {urg_color};'>"
                                f"<b style='color:#f1f5f9;'>{sku}</b> — {item['name']}<br>"
                                f"<span style='color:#64748b;'>Stock: {item['current_stock']} / Min: {min_s}</span>"
                                f"<span style='float:right;'>{urg}</span></div>", unsafe_allow_html=True)
            else:
                st.success("✅ All stock levels healthy!")

        with col_ar:
            st.markdown("#### ⚡ Auto-Generate")
            if st.button("🤖 Run Agent — Generate Orders & Tasks", use_container_width=True,
                          type="primary", key="auto_gen_btn"):
                drivers_list = [
                    {"name": "Mike Johnson", "id": "DRV001", "available": True},
                    {"name": "Sarah Chen", "id": "DRV002", "available": True},
                ]
                result = agent_auto_generate_orders(
                    INVENTORY, {}, st.session_state.deliveries, drivers_list, ai_call_fn=ai_call
                )
                st.success(result["message"])
                if result["new_orders"]:
                    st.markdown("##### 📋 Generated Purchase Orders")
                    for oid, order in result["new_orders"].items():
                        urg_icon = "🔴" if order["urgency"] == "CRITICAL" else "🟡"
                        with st.expander(f"{urg_icon} {oid} — {order['supplier']} — ${order['total_value']:,.0f}"):
                            for s, q in order["items"].items():
                                st.write(f"  • {s}: {q} units")
                            st.caption(f"Created: {order['created_at']} by {order['created_by']}")
                if result["new_tasks"]:
                    st.markdown("##### 🚛 Auto-Assigned Driver Tasks")
                    for tid, task in result["new_tasks"].items():
                        st.markdown(f"<div style='background:#1e293b;border-radius:8px;padding:10px;margin:4px 0;"
                                    f"border-left:4px solid #3b82f6;'>"
                                    f"<b style='color:#f1f5f9;'>{tid}</b> — {task['type']}<br>"
                                    f"<span style='color:#64748b;'>Driver: {task['driver']} | Order: {task['order_id']}"
                                    f" | Priority: {task['priority']}</span></div>", unsafe_allow_html=True)

    # ── Shelf Camera (tab13) ──
    with tab13:
        st.subheader("📷 Shelf Camera Monitoring")
        st.info("Cameras auto-scan shelf stock → detect low stock → trigger reorder emails")

        st.markdown("#### 📹 Camera Status")
        cam_cols = st.columns(len(SHELF_CAMERAS))
        for idx, (cam_id, cam) in enumerate(SHELF_CAMERAS.items()):
            with cam_cols[idx]:
                cc = "#22c55e" if cam["status"] == "active" else "#ef4444"
                st.markdown(f"<div style='background:#1e293b;border-radius:12px;padding:14px;text-align:center;"
                            f"border:2px solid {cc};'>"
                            f"<div style='font-size:24px;'>📷</div>"
                            f"<div style='color:#f1f5f9;font-weight:700;'>{cam_id}</div>"
                            f"<div style='color:{cc};font-size:12px;'>Zone {cam['shelf_zone']} — {cam['status'].upper()}</div>"
                            f"<div style='color:#64748b;font-size:11px;'>Shelves: {', '.join(cam['shelves'])}</div>"
                            f"</div>", unsafe_allow_html=True)
        st.divider()

        sc1, sc2 = st.columns(2)
        with sc1:
            sel_cam = st.selectbox("Camera:", ["All Cameras"] + list(SHELF_CAMERAS.keys()), key="shelf_cam_sel")
        with sc2:
            do_scan = st.button("📷 Run Shelf Scan", use_container_width=True, type="primary", key="shelf_scan_btn")

        if do_scan:
            cam_p = None if sel_cam == "All Cameras" else sel_cam
            sr = shelf_camera_scan(INVENTORY, cam_p)
            if sr["total_alerts"] > 0:
                st.error(f"🚨 {sr['total_alerts']} alert(s)! ({sr['critical_count']} critical)")
            else:
                st.success("✅ All shelves OK!")
            st.markdown("#### 📊 Scan Results")
            for r in sr["scan_results"]:
                sc_map = {"OK": "#22c55e", "LOW": "#f59e0b", "CRITICAL": "#ef4444", "EMPTY": "#ef4444"}
                scolor = sc_map.get(r["status"], "#64748b")
                bw = min(r["fill_pct"], 100)
                st.markdown(f"<div style='background:#1e293b;border-radius:8px;padding:10px;margin:4px 0;'>"
                            f"<div style='display:flex;justify-content:space-between;'>"
                            f"<div><b style='color:#f1f5f9;'>{r['sku']}</b> — {r['name']}"
                            f" <span style='color:#64748b;font-size:12px;'>📍{r['shelf']} 📷{r['camera']}</span></div>"
                            f"<span style='background:{scolor}22;color:{scolor};padding:2px 10px;border-radius:12px;"
                            f"font-size:12px;font-weight:700;'>{r['status']}</span></div>"
                            f"<div style='background:#334155;border-radius:4px;height:8px;margin-top:8px;'>"
                            f"<div style='background:{scolor};border-radius:4px;height:8px;width:{bw}%;'></div></div>"
                            f"<div style='color:#64748b;font-size:11px;margin-top:4px;'>"
                            f"{r['current_stock']}/{r['max_stock']} ({r['fill_pct']:.0f}%)</div></div>",
                            unsafe_allow_html=True)
            if sr["alerts"]:
                st.divider()
                st.markdown("#### 📧 Auto-Generated Reorder Email")
                email_text = generate_reorder_email_auto(sr["alerts"], INVENTORY)
                st.code(email_text, language=None)
                if st.button("📤 Send Reorder Email", use_container_width=True, key="send_email_btn"):
                    st.success("✅ Reorder email sent! (simulated)")
                    st.balloons()

    # ── Voice (tab14) ──
    with tab14:
        st.subheader("🔊 Voice Alert System")
        st.info("Text-to-speech for warehouse announcements")
        st.components.v1.html(VOICE_TTS_HTML, height=220)
        st.divider()

        st.markdown("#### ⚡ Quick Announcements")
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            if st.button("📊 Status Report", use_container_width=True, key="v_status"):
                msg = generate_voice_status_summary(INVENTORY, st.session_state.deliveries)
                st.write(f"🗣️ {msg}")
                js_msg = msg.replace("'", "\\'").replace('"', '\\"')
                st.components.v1.html(f'<script>speechSynthesis.speak(new SpeechSynthesisUtterance("{js_msg}"));</script>', height=0)
        with vc2:
            if st.button("🚨 Critical Alerts", use_container_width=True, key="v_critical"):
                crit = [f"{s} {INVENTORY[s]['name']}" for s in INVENTORY
                        if INVENTORY[s]["current_stock"] < INVENTORY[s].get("min_stock", INVENTORY[s].get("reorder_point", 20))]
                msg = f"{len(crit)} critical items: {', '.join(crit[:3])}" if crit else "No critical alerts."
                st.write(f"🗣️ {msg}")
                js_msg = msg.replace("'", "\\'").replace('"', '\\"')
                st.components.v1.html(f'<script>speechSynthesis.speak(new SpeechSynthesisUtterance("{js_msg}"));</script>', height=0)
        with vc3:
            if st.button("🚛 Delivery Update", use_container_width=True, key="v_delivery"):
                active = sum(1 for d in st.session_state.deliveries.values() if d["status"] != "Delivered")
                msg = f"{active} active deliveries in progress."
                st.write(f"🗣️ {msg}")
                js_msg = msg.replace("'", "\\'").replace('"', '\\"')
                st.components.v1.html(f'<script>speechSynthesis.speak(new SpeechSynthesisUtterance("{js_msg}"));</script>', height=0)

        st.divider()
        st.markdown("#### 💬 Custom Announcement")
        custom_msg = st.text_input("Message:", placeholder="e.g. Truck arriving at dock 3", key="v_custom")
        if custom_msg and st.button("🔊 Speak", use_container_width=True, type="primary", key="v_speak"):
            js_msg = custom_msg.replace("'", "\\'").replace('"', '\\"')
            st.components.v1.html(f'<script>speechSynthesis.speak(new SpeechSynthesisUtterance("{js_msg}"));</script>', height=0)


# ═══════════════════════════════════════════════════════════════════
# DRIVER VIEW
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.role == "driver":
    st.title("🚛 Driver Portal")
    st.subheader("My Deliveries Today")

    for del_id, delivery in st.session_state.deliveries.items():
        status_icon = "🔵" if delivery["status"]=="Loading" else "🟡"
        with st.expander(f"{status_icon} {del_id} — {delivery['destination']} — ETA: {delivery['eta']}",
                         expanded=True):
            col_l, col_r = st.columns([2,1])
            with col_l:
                st.write(f"📍 **Destination:** {delivery['destination']}")
                st.write(f"🗺️ **Address:** {delivery['address']}")
                st.write(f"👤 **Driver:** {delivery['driver']}")
                st.write(f"⏰ **ETA:** {delivery['eta']}")
                st.write(f"**Status:** {delivery['status']}")

                st.markdown("**Items to deliver:**")
                for sku, qty in delivery["items"].items():
                    item = INVENTORY.get(sku, {})
                    shelf = item.get("shelf","—")
                    st.write(f"  • {sku} — {item.get('name','Unknown')} × {qty} units  |  Pick from shelf: **{shelf}**")

            with col_r:
                status = delivery["status"]
                if status == "Dispatched":
                    if st.button("🚛 Mark In Transit", key=f"transit_{del_id}", use_container_width=True, type="primary"):
                        advance_task(del_id, "In Transit", st.session_state.employee_id)
                        st.success("Marked as In Transit!")
                        st.rerun()
                elif status == "In Transit":
                    if st.button("✅ Confirm Delivered", key=f"confirm_{del_id}", use_container_width=True, type="primary"):
                        advance_task(del_id, "Delivered", st.session_state.employee_id)
                        st.success("✅ Delivery confirmed!")
                        st.balloons()
                        st.rerun()
                elif status == "Delivered":
                    st.success("✅ Delivered")
                elif status in ["Loading", "Picked"]:
                    st.warning(f"⏳ Waiting for automated system — Status: {status}")

                # Task history timeline
                st.divider()
                st.markdown("**Timeline:**")
                for h in delivery.get("history", []):
                    icon = TASK_ICONS.get(h["status"], "⚪")
                    st.caption(f"{icon} {h['time']} — {h['status']} by {h['by']}")

                st.divider()
                encoded = delivery["address"].replace(" ", "+")
                maps_url = f"https://maps.google.com?q={encoded}"
                st.link_button("🗺️ Open in Google Maps", maps_url, use_container_width=True)

    st.divider()
    st.markdown("#### 🔊 Voice Alerts")
    driver_alerts = generate_driver_voice_alerts(st.session_state.deliveries, st.session_state.employee_id)
    for i, alert in enumerate(driver_alerts):
        col_msg, col_btn = st.columns([3, 1])
        col_msg.write(f"🗣️ {alert}")
        if col_btn.button("🔊", key=f"drv_voice_{i}"):
            js_msg = alert.replace("'", "\\'").replace('"', '\\"')
            st.components.v1.html(f'<script>speechSynthesis.speak(new SpeechSynthesisUtterance("{js_msg}"));</script>', height=0)

    st.divider()
    st.markdown("#### 📞 Need Help?")
    c1, c2 = st.columns(2)
    c1.info("📞 Warehouse: +1-555-0100")
    c2.info("📞 Manager: +1-555-0101")

# ═══════════════════════════════════════════════════════════════════
# SALES VIEW
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.role == "sales":
    st.title("💼 Sales Portal")
    company_id = st.session_state.company_id

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 My Orders", "➕ New Order", "🧠 Knowledge Base (RAG)", "📊 Customer Analytics", "🔧 Stock Corrections"
    ])

    # ── My Orders ──
    with tab1:
        st.subheader("📋 Sales Orders")
        for so_id, order in st.session_state.sales_orders.items():
            status_colors = {"Confirmed": "🟢", "Pending": "🟡", "Cancelled": "🔴", "Delivered": "✅"}
            icon = status_colors.get(order["status"], "⚪")
            with st.expander(f"{icon} {so_id} — {order['customer']} — {order['status']}", expanded=True):
                col_l, col_r = st.columns([2,1])
                with col_l:
                    st.write(f"**Customer:** {order['customer']}")
                    st.write(f"**Contact:** {order['contact']}")
                    st.write(f"**Created by:** {order['created_by']}  at  {order['created_at']}")
                    st.write(f"**Notes:** {order.get('notes','—')}")
                    st.markdown("**Items:**")
                    total = 0
                    for sku, qty in order["items"].items():
                        item = INVENTORY.get(sku, {})
                        val = qty * item.get("unit_price", 0)
                        total += val
                        st.write(f"  • {sku} — {item.get('name','?')} × {qty}  →  ${val:,.2f}")
                    st.write(f"**Order Total: ${total:,.2f}**")
                with col_r:
                    # Link to delivery
                    if order.get("delivery_id"):
                        del_status = st.session_state.deliveries.get(
                            order["delivery_id"], {}).get("status", "—")
                        del_icon = TASK_ICONS.get(del_status, "⚪")
                        st.metric("Delivery Status", f"{del_icon} {del_status}")
                    # Update order status
                    new_status = st.selectbox("Order Status",
                        ["Pending","Confirmed","Cancelled","Delivered"],
                        index=["Pending","Confirmed","Cancelled","Delivered"].index(order["status"]),
                        key=f"so_status_{so_id}")
                    if st.button("Update", key=f"so_upd_{so_id}", use_container_width=True):
                        st.session_state.sales_orders[so_id]["status"] = new_status
                        st.success("Updated!")
                        st.rerun()

    # ── New Order ──
    with tab2:
        st.subheader("➕ Create New Sales Order")
        col_l, col_r = st.columns([1,1])
        with col_l:
            customer   = st.text_input("Customer Name", placeholder="e.g. Tesla Fremont Plant")
            contact    = st.text_input("Contact Email", placeholder="e.g. procurement@tesla.com")
            notes      = st.text_input("Notes (optional)")
            st.markdown("**Select Items:**")
            selected_items = {}
            for sku, item in INVENTORY.items():
                r = calculate_reorder(sku)
                col_a, col_b, col_c = st.columns([2,2,2])
                col_a.write(f"**{sku}**")
                col_b.write(item["name"])
                qty = col_c.number_input(
                    f"Qty", min_value=0, max_value=item["current_stock"],
                    value=0, key=f"so_qty_{sku}", label_visibility="collapsed")
                if qty > 0:
                    selected_items[sku] = qty
        with col_r:
            st.markdown("**Order Preview:**")
            if selected_items:
                total = 0
                for sku, qty in selected_items.items():
                    item = INVENTORY[sku]
                    val = qty * item["unit_price"]
                    total += val
                    st.write(f"• {sku} × {qty} = ${val:,.2f}")
                st.metric("Total Value", f"${total:,.2f}")
                st.divider()
                # Check stock availability
                warnings = []
                for sku, qty in selected_items.items():
                    stock = INVENTORY[sku]["current_stock"]
                    if qty > stock:
                        warnings.append(f"⚠️ {sku}: need {qty}, only {stock} in stock")
                if warnings:
                    for w in warnings:
                        st.warning(w)
                if customer and selected_items:
                    if st.button("✅ Create Order", use_container_width=True, type="primary"):
                        so_id = f"SO-{str(len(st.session_state.sales_orders)+1).zfill(3)}"
                        st.session_state.sales_orders[so_id] = {
                            "customer": customer,
                            "contact": contact,
                            "items": selected_items,
                            "status": "Confirmed",
                            "created_by": st.session_state.employee_id,
                            "created_at": datetime.now().strftime("%H:%M"),
                            "delivery_id": None,
                            "notes": notes,
                        }
                        # Auto-create delivery order
                        del_id = f"DEL-{str(len(st.session_state.deliveries)+1).zfill(3)}"
                        st.session_state.deliveries[del_id] = {
                            "driver": "Unassigned",
                            "driver_id": None,
                            "destination": customer,
                            "address": contact,
                            "items": selected_items,
                            "status": "Loading",
                            "eta": "TBD",
                            "history": [{"time": datetime.now().strftime("%H:%M"),
                                        "status": "Loading", "by": st.session_state.employee_id}],
                        }
                        st.session_state.sales_orders[so_id]["delivery_id"] = del_id
                        st.success(f"✅ Order {so_id} created! Delivery {del_id} auto-generated.")
                        st.balloons()
                        st.rerun()
            else:
                st.info("Select items on the left to preview order")

    # ── Knowledge Base (RAG) ──
    with tab3:
        st.subheader("🧠 Company Knowledge Base — RAG")
        st.info("Upload company documents to enable AI-powered Q&A. Each company has its own isolated knowledge base.")

        col_l, col_r = st.columns([1,1])
        with col_l:
            st.markdown("#### Upload Documents")
            company_name = st.text_input("Company ID / Name", value=company_id,
                                          placeholder="e.g. tesla, google, default")
            if company_name:
                st.session_state.company_id = company_name.lower().replace(" ","_")

            uploaded_docs = st.file_uploader(
                "Upload PDFs, TXT, or CSV files",
                type=["pdf","txt","csv"],
                accept_multiple_files=True
            )

            if uploaded_docs and st.button("📥 Index Documents", use_container_width=True, type="primary"):
                if not RAG_AVAILABLE:
                    st.error("Install dependencies: pip install faiss-cpu sentence-transformers pypdf2")
                else:
                    all_chunks, all_sources = [], []
                    progress = st.progress(0)
                    for i, doc in enumerate(uploaded_docs):
                        text = ""
                        if doc.name.endswith(".pdf"):
                            if PDF_AVAILABLE:
                                try:
                                    reader = PyPDF2.PdfReader(doc)
                                    text = " ".join(p.extract_text() or "" for p in reader.pages)
                                except Exception:
                                    text = ""
                            else:
                                st.warning(f"pypdf2 not installed, skipping {doc.name}")
                        elif doc.name.endswith(".csv"):
                            import csv, io as _io
                            text = doc.read().decode("utf-8", errors="ignore")
                        else:
                            text = doc.read().decode("utf-8", errors="ignore")

                        if text.strip():
                            chunks = chunk_text(text)
                            all_chunks.extend(chunks)
                            all_sources.extend([doc.name] * len(chunks))
                        progress.progress((i+1)/len(uploaded_docs))

                    if all_chunks:
                        with st.spinner(f"Building vector index for {len(all_chunks)} chunks..."):
                            ok = build_rag_index(st.session_state.company_id, all_chunks, all_sources)
                        if ok:
                            st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(uploaded_docs)} files!")
                        else:
                            st.error("Failed to build index. Check dependencies.")

        with col_r:
            st.markdown("#### Ask Your Documents (CRAG)")
            rag_query = st.text_input("Ask anything about your company docs:",
                                       placeholder="e.g. What are our payment terms with Eaton?")
            if st.button("🔍 Search + Ask AI", use_container_width=True, type="primary") and rag_query:
                if not RAG_AVAILABLE:
                    st.error("Install: pip install faiss-cpu sentence-transformers")
                else:
                    with st.spinner("Searching knowledge base..."):
                        result = ai_call_with_rag(rag_query, st.session_state.company_id)
                    # Show confidence badge
                    conf_colors = {"high":"🟢","medium":"🟡","low":"🔴","none":"⚫"}
                    conf = result["confidence"]
                    st.markdown(f"**Confidence:** {conf_colors.get(conf,'⚪')} {conf.upper()}"
                                + (" *(query was corrected)*" if result["used_correction"] else ""))
                    st.markdown("**Answer:**")
                    st.markdown(result["answer"])
                    if result["chunks_used"]:
                        with st.expander(f"📄 {len(result['chunks_used'])} source chunks used"):
                            for c in result["chunks_used"]:
                                st.caption(f"[{c['source']}] score: {c['score']:.2f}")
                                st.text(c["text"][:200] + "...")

            st.divider()
            # Check if index exists
            path = get_company_rag_path(st.session_state.company_id)
            if os.path.exists(os.path.join(path, "index.faiss")):
                with open(os.path.join(path, "chunks.pkl"), "rb") as f:
                    data = pickle.load(f)
                st.success(f"✅ Index loaded: {len(data['texts'])} chunks")
                st.caption(f"Sources: {', '.join(set(data['sources']))}")
            else:
                st.info("No index yet. Upload documents to get started.")

    # ── Customer Analytics ──
    with tab4:
        st.subheader("📊 Customer Analytics")
        # Order summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Orders", len(st.session_state.sales_orders))
        c2.metric("Confirmed", sum(1 for o in st.session_state.sales_orders.values() if o["status"]=="Confirmed"))
        c3.metric("Pending", sum(1 for o in st.session_state.sales_orders.values() if o["status"]=="Pending"))
        total_rev = sum(
            sum(INVENTORY.get(sku,{}).get("unit_price",0)*qty for sku,qty in o["items"].items())
            for o in st.session_state.sales_orders.values()
        )
        c4.metric("Total Revenue", f"${total_rev:,.2f}")
        st.divider()
        # Orders table
        rows = []
        for so_id, order in st.session_state.sales_orders.items():
            total = sum(INVENTORY.get(s,{}).get("unit_price",0)*q for s,q in order["items"].items())
            rows.append({
                "Order ID": so_id,
                "Customer": order["customer"],
                "Status": order["status"],
                "Items": len(order["items"]),
                "Value": f"${total:,.2f}",
                "Created": order["created_at"],
                "Delivery": order.get("delivery_id","—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Also update manager AI advisor to use RAG
        st.divider()
        st.markdown("#### 🧠 Ask AI about this customer")
        cust_query = st.text_input("e.g. What did Tesla order last time?")
        if st.button("Ask", key="sales_ask") and cust_query:
            context = json.dumps([{
                "order_id": k, "customer": v["customer"],
                "items": v["items"], "status": v["status"], "value": sum(
                    INVENTORY.get(s,{}).get("unit_price",0)*q for s,q in v["items"].items())
            } for k,v in st.session_state.sales_orders.items()])
            result = ai_call(f"Sales orders: {context} Question: {cust_query} Answer concisely:")
            st.markdown(result)

    # ── Stock Corrections (tab5) — Sales can adjust stock ──
    with tab5:
        st.subheader("🔧 Stock Corrections")
        st.info("Correct stock levels when discrepancies are found (e.g. customer returns, miscounts)")

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("#### Adjust Stock")
            corr_sku = st.selectbox("Select Item:", list(INVENTORY.keys()),
                format_func=lambda x: f"{x} — {INVENTORY[x]['name']} (Stock: {INVENTORY[x]['current_stock']})",
                key="sales_corr_sku")

            if corr_sku:
                item = INVENTORY[corr_sku]
                st.write(f"**{item['name']}** — Shelf: {item['shelf']}")
                st.metric("Current Stock", item["current_stock"])

                corr_type = st.radio("Correction type:",
                    ["📥 Add stock (return/found)", "📤 Remove stock (damaged/lost)"],
                    key="sales_corr_type")
                corr_qty = st.number_input("Quantity:", min_value=1, max_value=999, value=1, key="sales_corr_qty")
                corr_reason = st.text_input("Reason:", placeholder="e.g. Customer return, Damaged in transit", key="sales_corr_reason")

                if st.button("✅ Apply Correction", use_container_width=True, type="primary", key="sales_corr_apply"):
                    if "ADD" in corr_type.upper() or "Add" in corr_type:
                        INVENTORY[corr_sku]["current_stock"] += corr_qty
                        direction = "IN"
                        st.success(f"✅ +{corr_qty} units added to {corr_sku}")
                    else:
                        if corr_qty <= item["current_stock"]:
                            INVENTORY[corr_sku]["current_stock"] -= corr_qty
                            direction = "OUT"
                            st.success(f"✅ -{corr_qty} units removed from {corr_sku}")
                        else:
                            st.error(f"❌ Cannot remove {corr_qty}, only {item['current_stock']} in stock")
                            direction = None

                    if direction:
                        st.session_state.scan_log.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "sku": corr_sku, "name": item["name"], "shelf": item["shelf"],
                            "operation": direction, "qty": corr_qty,
                            "note": f"[Sales Correction] {corr_reason}",
                            "by": st.session_state.employee_name,
                        })
                        st.rerun()

        with col_r:
            st.markdown("#### Recent Corrections")
            corrections = [x for x in st.session_state.scan_log if "Sales Correction" in x.get("note","")]
            if corrections:
                for c in reversed(corrections[-10:]):
                    icon = "📥" if c["operation"] == "IN" else "📤"
                    st.markdown(f"""
                    <div style='background:#1e293b;border-radius:8px;padding:10px;margin:4px 0;font-size:13px;'>
                        <span style='color:#64748b;'>{c['time']}</span>&nbsp;
                        {icon} <b>{c['sku']}</b> — {c['operation']} {c['qty']} units<br>
                        <span style='color:#94a3b8;'>{c['note']} — by {c['by']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No corrections yet.")


st.divider()
st.caption("InstaWorker™ Pro v7  ·  2026 Hackathon · San Francisco  ·  Powered by Google DeepMind × InstaLILY  ·  Edge-Native 🔒  ·  Automated Warehouse")
