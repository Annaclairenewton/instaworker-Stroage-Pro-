"""
Patch script: Adds 3 new features to app.py
1. Agent Auto-Order tab (Manager)
2. Shelf Camera tab (Manager)  
3. Voice TTS (Driver + Manager)
Run: python3 patch_features.py
"""
import os

APP_PATH = os.path.join(os.path.dirname(__file__), "frontend", "app.py")

text = open(APP_PATH).read()

# ── 1. Add automation import ──
old_import = """from conveyor import ("""
new_import = """from automation import (
    agent_auto_generate_orders, shelf_camera_scan,
    generate_reorder_email_auto, SHELF_CAMERAS,
    VOICE_TTS_HTML, generate_driver_voice_alerts,
    generate_voice_status_summary,
)
from conveyor import ("""

if "from automation import" not in text:
    text = text.replace(old_import, new_import)
    print("✅ Added automation imports")
else:
    print("⏭️ Automation imports already present")

# ── 2. Expand Manager tabs: add Agent Orders, Shelf Camera ──
old_tabs = '["📦 Inventory & Reorder","🚛 Purchase Orders","🚛 Delivery Tracker","📋 Inbound Log","🔍 Find Item","📊 Analytics","🤖 AI Advisor","📥 Import Inventory","🏭 Conveyor Belt","📲 Scan In/Out","📸 AI Photo ID"]'
new_tabs = '["📦 Inventory","🚛 Purchase Orders","🚛 Deliveries","📋 Inbound Log","🔍 Find Item","📊 Analytics","🤖 AI Advisor","📥 Import","🏭 Conveyor","📲 Scan","📸 Photo ID","🤖 Auto Orders","📷 Shelf Cam","🔊 Voice"]'

if "Auto Orders" not in text:
    # Also update tab variable names
    text = text.replace(
        "tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11 = st.tabs(" + old_tabs + ")",
        "tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13,tab14 = st.tabs(" + new_tabs + ")"
    )
    print("✅ Added 3 new Manager tabs")
else:
    print("⏭️ Manager tabs already updated")

# ── 3. Add tab12 (Auto Orders), tab13 (Shelf Cam), tab14 (Voice) before DRIVER VIEW ──
driver_marker = """# ═══════════════════════════════════════════════════════════════════
# DRIVER VIEW
# ═══════════════════════════════════════════════════════════════════"""

new_tabs_code = """
    # ── Agent Auto-Order Generator (tab12) ──
    with tab12:
        st.subheader("🤖 Agent Auto-Order Generator")
        st.info("AI Agent analyzes inventory → auto-generates purchase orders → assigns driver tasks")

        col_auto_l, col_auto_r = st.columns([1, 1])
        with col_auto_l:
            st.markdown("#### 📊 Current Stock Alerts")
            critical_items = []
            for sku, item in INVENTORY.items():
                min_s = item.get("min_stock", item.get("reorder_point", 20))
                if item["current_stock"] < min_s:
                    urgency = "🔴 CRITICAL" if item["current_stock"] < min_s * 0.5 else "🟡 LOW"
                    critical_items.append((sku, item, urgency))

            if critical_items:
                for sku, item, urgency in critical_items:
                    st.markdown(f"""
                    <div style='background:#1e293b;border-radius:8px;padding:10px;margin:4px 0;
                                border-left:4px solid {"#ef4444" if "CRITICAL" in urgency else "#f59e0b"};'>
                        <b style='color:#f1f5f9;'>{sku}</b> — {item['name']}<br>
                        <span style='color:#64748b;'>Stock: {item['current_stock']} | Min: {item.get('min_stock', item.get('reorder_point','?'))}</span>
                        <span style='float:right;'>{urgency}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ All stock levels healthy!")

        with col_auto_r:
            st.markdown("#### ⚡ Auto-Generate")
            if st.button("🤖 Run Agent — Generate Orders & Tasks", use_container_width=True,
                          type="primary", key="auto_gen_btn"):
                drivers = [
                    {"name": "Mike Johnson", "id": "DRV001", "available": True},
                    {"name": "Sarah Chen", "id": "DRV002", "available": True},
                ]
                result = agent_auto_generate_orders(
                    INVENTORY, st.session_state.get("purchase_orders", {}),
                    st.session_state.deliveries, drivers, ai_call_fn=ai_call
                )

                st.success(result["message"])

                if result["new_orders"]:
                    st.markdown("##### 📋 Generated Purchase Orders")
                    for oid, order in result["new_orders"].items():
                        with st.expander(f"{'🔴' if order['urgency']=='CRITICAL' else '🟡'} {oid} — {order['supplier']} — ${order['total_value']:,.0f}"):
                            for sku, qty in order["items"].items():
                                st.write(f"  • {sku}: {qty} units")
                            st.caption(f"Created: {order['created_at']} by {order['created_by']}")

                if result["new_tasks"]:
                    st.markdown("##### 🚛 Auto-Assigned Driver Tasks")
                    for tid, task in result["new_tasks"].items():
                        st.markdown(f"""
                        <div style='background:#1e293b;border-radius:8px;padding:10px;margin:4px 0;
                                    border-left:4px solid #3b82f6;'>
                            <b style='color:#f1f5f9;'>{tid}</b> — {task['type']}<br>
                            <span style='color:#64748b;'>Driver: {task['driver']} | Order: {task['order_id']} | Priority: {task['priority']}</span>
                        </div>
                        """, unsafe_allow_html=True)

    # ── Shelf Camera Auto-Detection (tab13) ──
    with tab13:
        st.subheader("📷 Shelf Camera Monitoring")
        st.info("Cameras auto-scan shelf stock levels → detect low stock → trigger reorder emails")

        # Camera status
        st.markdown("#### 📹 Camera Status")
        cam_cols = st.columns(len(SHELF_CAMERAS))
        for idx, (cam_id, cam) in enumerate(SHELF_CAMERAS.items()):
            with cam_cols[idx]:
                color = "#22c55e" if cam["status"] == "active" else "#ef4444"
                st.markdown(f\"\"\"
                <div style='background:#1e293b;border-radius:12px;padding:14px;text-align:center;
                            border:2px solid {color};'>
                    <div style='font-size:24px;'>📷</div>
                    <div style='color:#f1f5f9;font-weight:700;'>{cam_id}</div>
                    <div style='color:{color};font-size:12px;'>Zone {cam["shelf_zone"]} — {cam["status"].upper()}</div>
                    <div style='color:#64748b;font-size:11px;'>Shelves: {", ".join(cam["shelves"])}</div>
                </div>
                \"\"\", unsafe_allow_html=True)

        st.divider()

        # Scan controls
        scan_col1, scan_col2 = st.columns(2)
        with scan_col1:
            selected_cam = st.selectbox("Select Camera:", ["All Cameras"] + list(SHELF_CAMERAS.keys()), key="shelf_cam_sel")
        with scan_col2:
            scan_btn = st.button("📷 Run Shelf Scan", use_container_width=True, type="primary", key="shelf_scan_btn")

        if scan_btn:
            cam_param = None if selected_cam == "All Cameras" else selected_cam
            scan_result = shelf_camera_scan(INVENTORY, cam_param)

            # Show results
            if scan_result["total_alerts"] > 0:
                st.error(f"🚨 {scan_result['total_alerts']} alert(s) detected! ({scan_result['critical_count']} critical)")
            else:
                st.success("✅ All shelves OK!")

            # Scan results table
            st.markdown("#### 📊 Scan Results")
            for r in scan_result["scan_results"]:
                status_colors = {"OK": "#22c55e", "LOW": "#f59e0b", "CRITICAL": "#ef4444", "EMPTY": "#ef4444"}
                sc = status_colors.get(r["status"], "#64748b")
                bar_width = min(r["fill_pct"], 100)
                st.markdown(f\"\"\"
                <div style='background:#1e293b;border-radius:8px;padding:10px;margin:4px 0;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div>
                            <b style='color:#f1f5f9;'>{r["sku"]}</b> — {r["name"]}
                            <span style='color:#64748b;font-size:12px;'> | 📍{r["shelf"]} | 📷{r["camera"]}</span>
                        </div>
                        <span style='background:{sc}22;color:{sc};padding:2px 10px;border-radius:12px;
                                     font-size:12px;font-weight:700;'>{r["status"]}</span>
                    </div>
                    <div style='background:#334155;border-radius:4px;height:8px;margin-top:8px;'>
                        <div style='background:{sc};border-radius:4px;height:8px;width:{bar_width}%;'></div>
                    </div>
                    <div style='color:#64748b;font-size:11px;margin-top:4px;'>
                        {r["current_stock"]}/{r["max_stock"]} units ({r["fill_pct"]:.0f}%)
                    </div>
                </div>
                \"\"\", unsafe_allow_html=True)

            # Auto-generate email if alerts
            if scan_result["alerts"]:
                st.divider()
                st.markdown("#### 📧 Auto-Generated Reorder Email")
                email = generate_reorder_email_auto(scan_result["alerts"], INVENTORY)
                st.code(email, language=None)

                if st.button("📤 Send Reorder Email", use_container_width=True, key="send_email_btn"):
                    st.success("✅ Reorder email sent to supplier(s)! (simulated)")
                    st.balloons()

    # ── Voice Alerts (tab14) ──
    with tab14:
        st.subheader("🔊 Voice Alert System")
        st.info("Text-to-speech announcements for warehouse operations")

        # TTS Component
        st.components.v1.html(VOICE_TTS_HTML, height=220)

        st.divider()

        # Quick voice buttons
        st.markdown("#### ⚡ Quick Announcements")
        v_col1, v_col2, v_col3 = st.columns(3)

        with v_col1:
            if st.button("📊 Warehouse Status", use_container_width=True, key="voice_status"):
                msg = generate_voice_status_summary(INVENTORY, st.session_state.deliveries)
                st.markdown(f"**🗣️ Speaking:** {msg}")
                st.components.v1.html(f\"\"\"<script>
                    const s = new SpeechSynthesisUtterance("{msg}");
                    s.rate = 1.0;
                    window.speechSynthesis.speak(s);
                </script>\"\"\", height=0)

        with v_col2:
            if st.button("🚨 Critical Alerts", use_container_width=True, key="voice_critical"):
                critical = [f"{sku} {item['name']}" for sku, item in INVENTORY.items()
                           if item["current_stock"] < item.get("min_stock", item.get("reorder_point", 20))]
                if critical:
                    msg = f"Critical stock alert. {len(critical)} items need reorder: {', '.join(critical[:3])}"
                else:
                    msg = "No critical alerts. All stock levels healthy."
                st.markdown(f"**🗣️ Speaking:** {msg}")
                safe_msg = msg.replace("'", "\\\\'")
                st.components.v1.html(f\"\"\"<script>
                    const s = new SpeechSynthesisUtterance('{safe_msg}');
                    s.rate = 1.0;
                    window.speechSynthesis.speak(s);
                </script>\"\"\", height=0)

        with v_col3:
            if st.button("🚛 Delivery Update", use_container_width=True, key="voice_delivery"):
                active = sum(1 for d in st.session_state.deliveries.values() if d["status"] not in ["Delivered"])
                msg = f"{active} active deliveries in progress."
                st.markdown(f"**🗣️ Speaking:** {msg}")
                st.components.v1.html(f\"\"\"<script>
                    const s = new SpeechSynthesisUtterance("{msg}");
                    s.rate = 1.0;
                    window.speechSynthesis.speak(s);
                </script>\"\"\", height=0)

        st.divider()
        st.markdown("#### 💬 Custom Announcement")
        custom_msg = st.text_input("Type message to announce:", placeholder="e.g. Delivery truck arriving at dock 3", key="voice_custom")
        if custom_msg and st.button("🔊 Speak", use_container_width=True, type="primary", key="voice_speak"):
            safe = custom_msg.replace("'", "\\\\'").replace('"', '\\\\"')
            st.components.v1.html(f\"\"\"<script>
                const s = new SpeechSynthesisUtterance("{safe}");
                s.rate = 1.0;
                window.speechSynthesis.speak(s);
            </script>\"\"\", height=0)


""" + driver_marker

if "Auto Orders" not in text:
    text = text.replace(driver_marker, new_tabs_code)
    print("✅ Added tab12, tab13, tab14 code")
else:
    print("⏭️ Tab code already present")

# ── 4. Add Voice to Driver view ──
old_driver_help = """    st.divider()
    st.markdown("#### 📞 Need Help?")
    c1, c2 = st.columns(2)
    c1.info("📞 Warehouse: +1-555-0100")
    c2.info("📞 Manager: +1-555-0101")"""

new_driver_help = """    st.divider()
    st.markdown("#### 🔊 Voice Alerts")
    driver_alerts = generate_driver_voice_alerts(st.session_state.deliveries, st.session_state.employee_id)
    for i, alert in enumerate(driver_alerts):
        col_msg, col_btn = st.columns([3, 1])
        col_msg.write(f"🗣️ {alert}")
        if col_btn.button("🔊", key=f"drv_voice_{i}"):
            safe = alert.replace("'", "\\\\'").replace('"', '\\\\"')
            st.components.v1.html(f\"\"\"<script>
                const s = new SpeechSynthesisUtterance("{safe}");
                s.rate = 1.0;
                window.speechSynthesis.speak(s);
            </script>\"\"\", height=0)

    st.divider()
    st.markdown("#### 📞 Need Help?")
    c1, c2 = st.columns(2)
    c1.info("📞 Warehouse: +1-555-0100")
    c2.info("📞 Manager: +1-555-0101")"""

if "Voice Alerts" not in text:
    text = text.replace(old_driver_help, new_driver_help)
    print("✅ Added Voice to Driver view")
else:
    print("⏭️ Driver voice already present")

# ── Save ──
open(APP_PATH, 'w').write(text)
print("\n🎉 All 3 features patched successfully!")
print("Run: streamlit run frontend/app.py")
