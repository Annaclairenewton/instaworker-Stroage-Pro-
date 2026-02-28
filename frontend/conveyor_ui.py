"""
InstaWorker™ Conveyor Belt UI Module
=====================================
New tab for Manager and Staff views.
Paste this into the appropriate tab in frontend/app.py

Features:
  - Real-time conveyor belt visualization (animated)
  - Camera identification panel
  - Auto-routing: storage vs dispatch
  - Conveyor control panel (start/stop/speed)
  - Event log
"""

# ════════════════════════════════════════════════════════════════════
# CONVEYOR BELT TAB — Add this to Manager or Staff view
# ════════════════════════════════════════════════════════════════════

# NOTE: This is the code for the conveyor tab content.
# Integration instructions are in INTEGRATION_GUIDE.md

def render_conveyor_tab(st, INVENTORY, ai_call, conveyor_state, deliveries, sales_orders):
    """Render the full conveyor belt tab UI."""
    from backend.conveyor import (
        create_belt_item, create_conveyor_state, process_belt_item,
        get_belt_visual_data, Zone, ConveyorStatus, ZONE_CONFIG
    )
    from datetime import datetime
    import json

    st.subheader("🏭 Smart Conveyor Belt System")
    st.info("📷 Camera auto-identifies items → 🤖 Agent routes to storage or dispatch → 🚛 Auto-fulfills orders")

    # ── Initialize conveyor state in session ──
    if "conveyor" not in st.session_state:
        st.session_state.conveyor = create_conveyor_state()
    conv = st.session_state.conveyor

    # ── Control Panel ──
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns(5)
    with ctrl_col1:
        if conv["status"] != ConveyorStatus.RUNNING:
            if st.button("▶️ Start Belt", use_container_width=True, type="primary"):
                conv["status"] = ConveyorStatus.RUNNING
                conv["started_at"] = datetime.now().strftime("%H:%M:%S")
                st.rerun()
        else:
            if st.button("⏸️ Pause Belt", use_container_width=True):
                conv["status"] = ConveyorStatus.PAUSED
                st.rerun()
    with ctrl_col2:
        if st.button("⏹️ Stop Belt", use_container_width=True):
            conv["status"] = ConveyorStatus.IDLE
            st.rerun()
    with ctrl_col3:
        speed = st.select_slider("Speed", options=[0.5, 1.0, 1.5, 2.0],
                                  value=conv["speed"], format_func=lambda x: f"{x}x")
        conv["speed"] = speed
    with ctrl_col4:
        cam_status = "🟢 Active" if conv["camera_active"] else "🔴 Off"
        if st.button(f"📷 Camera: {cam_status}", use_container_width=True):
            conv["camera_active"] = not conv["camera_active"]
            st.rerun()
    with ctrl_col5:
        belt_status_colors = {
            ConveyorStatus.RUNNING: ("🟢", "#22c55e"),
            ConveyorStatus.PAUSED: ("🟡", "#f59e0b"),
            ConveyorStatus.IDLE: ("⚫", "#64748b"),
            ConveyorStatus.ERROR: ("🔴", "#ef4444"),
        }
        icon, color = belt_status_colors.get(conv["status"], ("⚫", "#64748b"))
        st.markdown(f"""
        <div style='background:#1e293b;border-radius:12px;padding:12px;text-align:center;
                    border:2px solid {color};'>
            <div style='font-size:24px;'>{icon}</div>
            <div style='color:{color};font-weight:700;font-size:14px;'>{conv['status'].upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Conveyor Belt Visualization (Animated) ──
    st.markdown("#### 🏭 Conveyor Belt — Live View")

    # Build zone visualization
    zone_html_parts = []
    for zone_id, config in ZONE_CONFIG.items():
        items_here = [i for i in conv["items_on_belt"] if i["current_zone"] == zone_id]
        count = len(items_here)
        item_dots = "".join([f"<span style='color:{config['color']};'>●</span> " for _ in items_here])

        active_class = "belt-zone-active" if count > 0 else ""
        zone_html_parts.append(f"""
        <div class='belt-zone {active_class}' style='border-color:{config["color"]}40;'>
            <div style='font-size:28px;'>{config["icon"]}</div>
            <div style='color:#f1f5f9;font-weight:700;font-size:12px;margin:4px 0;'>
                {config["label"]}
            </div>
            <div style='color:{config["color"]};font-size:11px;'>
                {count} item{"s" if count != 1 else ""}
            </div>
            <div style='min-height:16px;'>{item_dots}</div>
        </div>
        """)

    # Arrow connectors
    arrow = "<div class='belt-arrow'>→</div>"
    # Split at sorting for the fork
    zones_html = arrow.join(zone_html_parts[:3])  # Inbound → Camera → Sorting

    belt_running_class = "belt-running" if conv["status"] == ConveyorStatus.RUNNING else ""

    st.markdown(f"""
    <style>
    .belt-container {{
        display:flex; align-items:center; justify-content:center;
        gap:0; padding:16px 0; overflow-x:auto;
    }}
    .belt-zone {{
        background:#1e293b; border-radius:12px; padding:14px 10px;
        text-align:center; min-width:110px; border:2px solid #334155;
        transition: all 0.3s;
    }}
    .belt-zone-active {{
        box-shadow: 0 0 12px rgba(59,130,246,0.3);
        transform: scale(1.02);
    }}
    .belt-arrow {{
        color:#3b82f6; font-size:24px; font-weight:700; margin:0 4px;
        animation: {belt_running_class} 1s infinite;
    }}
    @keyframes belt-running {{
        0%,100% {{ opacity:1; }}
        50% {{ opacity:0.3; }}
    }}
    .belt-track {{
        height:6px; background:linear-gradient(90deg, #1e293b 0%, #3b82f6 50%, #1e293b 100%);
        border-radius:3px; margin:8px 0;
        background-size: 200% 100%;
    }}
    .belt-track.moving {{
        animation: belt-move 2s linear infinite;
    }}
    @keyframes belt-move {{
        0% {{ background-position: 0% 0%; }}
        100% {{ background-position: 200% 0%; }}
    }}
    .fork-container {{
        display:flex; gap:16px; margin-top:8px;
    }}
    .fork-branch {{
        flex:1; display:flex; align-items:center; gap:4px;
        justify-content:center;
    }}
    </style>

    <div class='belt-track {"moving" if conv["status"] == "running" else ""}'></div>
    <div class='belt-container'>
        {zones_html}
    </div>
    <div class='belt-track {"moving" if conv["status"] == "running" else ""}'></div>

    <!-- Fork: Storage vs Dispatch -->
    <div class='fork-container'>
        <div class='fork-branch'>
            <div class='belt-arrow'>↙</div>
            {zone_html_parts[3]}
        </div>
        <div class='fork-branch'>
            <div class='belt-arrow'>↘</div>
            {zone_html_parts[4]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── KPI Metrics ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📦 Total Processed", conv["total_processed"])
    m2.metric("📥 Stored", conv["total_stored"])
    m3.metric("🚛 Dispatched", conv["total_dispatched"])
    m4.metric("🔄 On Belt", len(conv["items_on_belt"]))
    m5.metric("❌ Errors", conv["total_errors"])

    st.divider()

    # ── Add Item to Belt (Simulate inbound) ──
    col_input, col_preview = st.columns([1, 1])

    with col_input:
        st.markdown("#### 📥 Add Item to Conveyor Belt")
        st.caption("Simulate an item arriving at the inbound dock")

        input_method = st.radio("Input method:",
                                ["Select from inventory", "Describe item (AI识别)", "Upload photo"],
                                horizontal=True, key="conv_input_method")

        item_desc = ""
        if input_method == "Select from inventory":
            selected_sku = st.selectbox(
                "Select item:",
                options=list(INVENTORY.keys()),
                format_func=lambda x: f"{x} — {INVENTORY[x]['name']} (Shelf: {INVENTORY[x]['shelf']})"
            )
            if selected_sku:
                item_desc = f"{INVENTORY[selected_sku]['name']} SKU:{selected_sku}"
                qty = st.number_input("Quantity:", min_value=1, max_value=100, value=1, key="conv_qty")

        elif input_method == "Describe item (AI识别)":
            item_desc = st.text_input("Describe the item on the belt:",
                                       placeholder="e.g. Small circuit breaker, 20 amp, Eaton brand")
            qty = st.number_input("Quantity:", min_value=1, max_value=100, value=1, key="conv_qty_ai")

        else:  # Upload photo
            photo = st.file_uploader("Upload item photo", type=["jpg", "png", "jpeg"], key="conv_photo")
            if photo:
                st.image(photo, width=200)
                item_desc = st.text_input("Description (optional):", placeholder="What does this look like?")
            qty = 1

        if item_desc and st.button("🏭 Send to Conveyor Belt", use_container_width=True, type="primary"):
            for i in range(qty if 'qty' in dir() else 1):
                item_id = f"BELT-{datetime.now().strftime('%H%M%S')}-{i+1}"
                new_item = create_belt_item(item_id, item_desc)
                conv["items_on_belt"].append(new_item)

            st.success(f"✅ {qty} item(s) placed on conveyor belt!")
            st.rerun()

    with col_preview:
        st.markdown("#### 🔄 Items Currently on Belt")
        if conv["items_on_belt"]:
            for item in conv["items_on_belt"]:
                zone_cfg = ZONE_CONFIG.get(item["current_zone"], {})
                st.markdown(f"""
                <div style='background:#1e293b;border-radius:8px;padding:10px;margin:6px 0;
                            border-left:4px solid {zone_cfg.get("color","#334155")}'>
                    <b style='color:#f1f5f9;'>{item["item_id"]}</b>
                    <span style='color:#64748b;'> — {item["description"][:40]}</span><br>
                    <span style='color:{zone_cfg.get("color","#64748b")};font-size:12px;'>
                        {zone_cfg.get("icon","")} {zone_cfg.get("label","")} — {item["status"]}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No items on belt. Add items above to start processing.")

    st.divider()

    # ── Process Belt (Run pipeline) ──
    if conv["items_on_belt"]:
        st.markdown("#### ⚡ Process Items on Belt")
        col_process, col_auto = st.columns(2)
        with col_process:
            if st.button("🔄 Process Next Item", use_container_width=True, type="primary"):
                item = conv["items_on_belt"][0]
                # Build pending orders from deliveries + sales orders
                pending = {}
                if hasattr(st.session_state, 'deliveries'):
                    pending.update(st.session_state.deliveries)
                if hasattr(st.session_state, 'sales_orders'):
                    for so_id, so in st.session_state.sales_orders.items():
                        pending[so_id] = so

                with st.spinner(f"🏭 Processing {item['item_id']}..."):
                    processed = process_belt_item(
                        item, INVENTORY, pending, conv,
                        ai_call_fn=ai_call
                    )

                route_icon = "🚛 → DISPATCH" if processed["route"] == "dispatch" else "📦 → STORAGE"
                st.success(
                    f"✅ {processed['identified_name']} ({processed.get('sku','?')}) "
                    f"— {route_icon}"
                    f"{' — Order: ' + processed['target_order'] if processed.get('target_order') else ''}"
                    f" — Shelf: {processed.get('target_shelf', '—')}"
                )
                st.rerun()

        with col_auto:
            if st.button("⚡ Process ALL Items", use_container_width=True):
                pending = {}
                if hasattr(st.session_state, 'deliveries'):
                    pending.update(st.session_state.deliveries)
                if hasattr(st.session_state, 'sales_orders'):
                    for so_id, so in st.session_state.sales_orders.items():
                        pending[so_id] = so

                items_to_process = list(conv["items_on_belt"])
                progress_bar = st.progress(0)
                for idx, item in enumerate(items_to_process):
                    processed = process_belt_item(
                        item, INVENTORY, pending, conv,
                        ai_call_fn=ai_call
                    )
                    progress_bar.progress((idx + 1) / len(items_to_process))

                st.success(f"✅ Processed {len(items_to_process)} items!")
                st.rerun()

    st.divider()

    # ── Recent Activity Log ──
    st.markdown("#### 📋 Conveyor Event Log")
    if conv["event_log"]:
        for event in reversed(conv["event_log"][-15:]):
            route_color = "#ef4444" if event["route"] == "dispatch" else "#22c55e"
            conf_pct = f"{event['confidence']:.0%}" if event.get("confidence") else "—"
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:8px;padding:8px 12px;margin:4px 0;
                        font-size:13px; border-left:3px solid {route_color};'>
                <span style='color:#64748b;'>{event['time']}</span>&nbsp;
                {event['icon']}&nbsp;
                <b style='color:#f1f5f9;'>{event.get('name','?')}</b>
                <span style='color:#64748b;'>({event.get('sku','?')})</span>
                → <span style='color:{route_color};font-weight:700;'>
                    {event['route'].upper()}</span>
                &nbsp;
                <span style='color:#64748b;'>
                    Shelf: {event.get('shelf','—')} |
                    Order: {event.get('order','—')} |
                    Conf: {conf_pct}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No events yet. Process items on the belt to see activity.")

    # ── Completed Items Detail ──
    if conv["items_completed"]:
        with st.expander(f"📜 Completed Items ({len(conv['items_completed'])})"):
            for item in reversed(conv["items_completed"][-10:]):
                st.markdown(f"**{item['item_id']}** — {item.get('identified_name', '?')} "
                            f"({item.get('sku', '?')}) → {item['route'].upper()}")
                for h in item.get("history", []):
                    zone_cfg = ZONE_CONFIG.get(h.get("zone"), {})
                    st.caption(f"  {zone_cfg.get('icon','')} {h['time']} — {h['event']}")
