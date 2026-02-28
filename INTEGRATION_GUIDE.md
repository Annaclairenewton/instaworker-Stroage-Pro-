# InstaWorker™ v7 — 传送带 + Camera 自动识别系统 集成指南

## 新增文件
1. `backend/conveyor.py` — 传送带引擎核心（状态机、Camera识别、Agent路由）
2. `frontend/conveyor_ui.py` — 传送带控制面板UI

## 集成步骤

### 第1步：复制新文件
把以下文件复制到你的项目：
- `backend/conveyor.py` → 放到 `backend/` 目录
- `frontend/conveyor_ui.py` → 放到 `frontend/` 目录

### 第2步：修改 `frontend/app.py`

在文件顶部 import 区域添加：
```python
from frontend.conveyor_ui import render_conveyor_tab
```

在 **Manager View** 的 tabs 那行（约第240行），添加传送带tab：

找到这行：
```python
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([...])
```

改为：
```python
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 = st.tabs([
    "📦 Inventory & Reorder","🚛 Purchase Orders","🚛 Delivery Tracker",
    "📋 Inbound Log","🔍 Find Item","📊 Analytics","🤖 AI Advisor",
    "📥 Import Inventory","🏭 Conveyor Belt"
])
```

然后在 `with tab8:` 代码块之后添加：
```python
with tab9:
    render_conveyor_tab(st, INVENTORY, ai_call, 
                        st.session_state.get("conveyor", None),
                        st.session_state.deliveries,
                        st.session_state.sales_orders)
```

### 第3步：运行
```bash
cd ~/Desktop/"instaworker 2"
streamlit run run.py
```

## 功能说明

### 🏭 传送带系统
- **5个区域**：入库口 → Camera扫描 → Agent分拣 → 存储/发货
- **自动识别**：Camera + AI 自动识别货物SKU
- **智能路由**：有匹配订单 → 直接送到发货区；无订单 → 送到对应货架
- **实时可视化**：动画展示传送带状态和货物流向

### 📷 Camera 识别
- 支持从库存选择、文字描述AI识别、上传照片
- AI自动匹配SKU和货架位置
- 回退机制：AI不可用时使用关键词匹配

### 🤖 Agent 自动路由
- 检查所有待处理订单
- 匹配到订单 → 自动路由到发货区
- 无匹配 → 路由到存储货架
- 自动更新库存数量
