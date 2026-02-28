"""
InstaWorker™ Pro v4 — Complete Warehouse Management System
Features:
- Role-based access: Manager / Warehouse Staff / Driver
- Barcode/QR scanning via phone camera (browser-based)
- Photo capture for AI part identification
- Real-time inventory with shelf locations
- Smart reorder engine (demand forecasting + capacity constraints)
- Delivery anomaly detection
- Driver delivery management
- Gemini Cloud vs Ollama phi3 (fully offline) toggle
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
    Layer 2 — Ollama phi3 (local, offline)
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
                st.toast("⚠️ Gemini unavailable — falling back to Ollama phi3", icon="⚠️")
                mode = "ollama"

    # ── Layer 2: Ollama phi3 ──
    if mode == "ollama" or (mode == "gemini" and not st.session_state.get("api_key","")):
        if OLLAMA_AVAILABLE and _ollama is not None:
            try:
                if image:
                    result = _ollama.chat(
                        model="phi3",
                        messages=[{"role":"user","content": prompt}]
                    )["message"]["content"]
                    ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": "phi3 🔒", "status": "OK (text only)"})
                    return "[Image analysis not supported offline] " + result
                response = _ollama.chat(
                    model="phi3",
                    messages=[{"role":"user","content": prompt}]
                )
                ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": "phi3 🔒", "status": "OK"})
                return response["message"]["content"]
            except Exception as e:
                ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": "phi3 🔒", "status": f"FAILED: {e}"})
                st.toast("⚠️ Ollama unavailable — using rule engine", icon="⚠️")
        else:
            ai_log.append({"time": datetime.now().strftime("%H:%M:%S"), "layer": "phi3 🔒", "status": "NOT INSTALLED"})

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
    """Check if internet is available."""
    try:
        socket.setdefaulttimeout(2)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except Exception:
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
    "STF001": {"name": "James Rodriguez", "role": "staff",   "pin": "1111"},
    "STF002": {"name": "Aisha Patel",     "role": "staff",   "pin": "2222"},
    "DRV001": {"name": "Mike Johnson",    "role": "driver",  "pin": "3333"},
    "DRV002": {"name": "Sarah Chen",      "role": "driver",  "pin": "4444"},
    "SLS001": {"name": "David Park",        "role": "sales",   "pin": "5555"},
    "SLS002": {"name": "Emma Wilson",       "role": "sales",   "pin": "6666"},
}

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
if "api_key"      not in st.session_state: st.session_state.api_key      = ""
if "ai_mode"      not in st.session_state: st.session_state.ai_mode      = "gemini"
if "ai_call_log"  not in st.session_state: st.session_state.ai_call_log  = []

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
TASK_OWNER   = {"Loading":"Staff","Picked":"Staff","Dispatched":"Driver","In Transit":"Driver","Delivered":"Driver"}

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
                <div style='color:#93c5fd;font-size:11px;'>Full access</div>
            </div>
            <div style='flex:1;background:linear-gradient(135deg,#14532d,#15803d);
                        border-radius:16px;padding:20px 12px;text-align:center;height:150px;
                        display:flex;flex-direction:column;justify-content:center;'>
                <div style='font-size:32px;'>📦</div>
                <div style='color:white;font-weight:700;font-size:14px;margin:6px 0;'>Warehouse Staff</div>
                <div style='color:#86efac;font-size:11px;'>Scan in/out</div>
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
                <div style='color:#ddd6fe;font-size:11px;'>Orders · Customers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Login form
        st.markdown("#### 🔐 Employee Login")
        emp_id  = st.text_input("Employee ID", placeholder="e.g. MGR001 / STF001 / DRV001").strip().upper()
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
        Manager: MGR001 / 1234 &nbsp;|&nbsp; Staff: STF001 / 1111 &nbsp;|&nbsp; Driver: DRV001 / 3333 &nbsp;|&nbsp; Sales: SLS001 / 5555
        </div>
        """, unsafe_allow_html=True)
        st.caption("Edge-Native · Data stays on premise · Google DeepMind × InstaLILY 2026")

    st.stop()

# ── Sidebar (logged in) ───────────────────────────────────────────
role_icons = {"manager": "👔", "staff": "📦", "driver": "🚛", "sales": "💼"}
role_labels = {"manager": "Manager", "staff": "Warehouse Staff", "driver": "Driver", "sales": "Sales"}

with st.sidebar:
    st.markdown(f"### {role_icons[st.session_state.role]} {role_labels[st.session_state.role]}")
    st.caption(f"👤 {st.session_state.employee_name}  ·  {st.session_state.employee_id}")
    if st.button("🔓 Logout", use_container_width=True):
        st.session_state.role = None
        st.session_state.employee_name = ""
        st.session_state.employee_id = ""
        st.rerun()
    st.divider()

    if st.session_state.role in ["manager", "staff"]:
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
        ["gemini", "ollama"],
        format_func=lambda x: "☁️ Gemini (Cloud)" if x == "gemini" else "🔒 phi3 (Offline)",
        index=0 if st.session_state.ai_mode == "gemini" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state.ai_mode = ai_mode

    if ai_mode == "gemini":
        st.caption("☁️ Cloud · Requires internet")
    else:
        status_color = "🟢" if OLLAMA_AVAILABLE else "🔴"
        st.caption(f"{status_color} Offline · No data leaves device")

    if st.session_state.role == "manager" and ai_mode == "gemini":
        new_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
        if new_key: st.session_state.api_key = new_key

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

    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(["📦 Inventory & Reorder","🚛 Purchase Orders","🚛 Delivery Tracker","📋 Inbound Log","🔍 Find Item","📊 Analytics","🤖 AI Advisor","📥 Import Inventory"])

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
        st.subheader("📋 Inbound / Outbound Log — All Staff Activity")
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


# ═══════════════════════════════════════════════════════════════════
# STAFF VIEW
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.role == "staff":
    st.title("📦 Warehouse Staff Terminal")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📲 Scan In/Out", "📸 AI Photo ID", "📋 My Scan Log", "🚛 Outbound Prep", "🔍 Find Item"])

    # ── Scan In/Out ──
    with tab1:
        st.subheader("Barcode / QR Scanner")
        st.info("💡 Open on your phone → point camera at barcode or QR code")

        # Centre the scanner
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown("#### 📷 Live Camera Scanner")
            st.components.v1.html(BARCODE_SCANNER_HTML, height=420)
            st.divider()
            st.markdown("#### Or Enter Manually")
            manual_code = st.text_input("SKU / Barcode:", placeholder="e.g. CB-1201-A or 8901001234567")

        st.divider()
        st.markdown("#### Process Scan")
        col_l, col_r = st.columns([1,1])
        with col_l:
            pass  # spacer
        with col_r:
            pass  # spacer

        # Process below scanner centred
        _, proc_col, _ = st.columns([1, 2, 1])
        with proc_col:
            st.markdown("#### ⚙️ Result")

            # Resolve barcode to SKU
            scanned_sku = None
            input_code = manual_code.strip().upper() if manual_code else ""

            if input_code:
                if input_code in INVENTORY:
                    scanned_sku = input_code
                else:
                    # Try barcode lookup
                    for sku, item in INVENTORY.items():
                        if item["barcode"] == input_code:
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
                operation = st.radio("Operation:", ["📥 Stock IN (Receiving)", "📤 Stock OUT (Dispatch)"])
                qty = st.number_input("Quantity:", min_value=1, max_value=9999, value=1)
                note = st.text_input("Note (optional):", placeholder="e.g. PO-2024-001")

                if st.button("✅ Confirm", use_container_width=True, type="primary"):
                    direction = "IN" if "IN" in operation else "OUT"
                    log_entry = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "sku": scanned_sku,
                        "name": item["name"],
                        "shelf": item["shelf"],
                        "operation": direction,
                        "qty": qty,
                        "note": note,
                        "by": "Staff",
                    }
                    st.session_state.scan_log.append(log_entry)

                    if direction == "IN":
                        INVENTORY[scanned_sku]["current_stock"] += qty
                        st.success(f"✅ {qty} units received → Shelf {item['shelf']}")
                    else:
                        if qty <= item["current_stock"]:
                            INVENTORY[scanned_sku]["current_stock"] -= qty
                            st.success(f"✅ {qty} units dispatched from Shelf {item['shelf']}")
                        else:
                            st.error(f"❌ Insufficient stock! Only {item['current_stock']} available.")

                    # Check reorder after update
                    r2 = calculate_reorder(scanned_sku)
                    if r2["needs_reorder"]:
                        st.warning(f"⚠️ Reorder triggered: recommend {r2['recommended_qty']} units")

            elif input_code:
                st.error(f"❌ '{input_code}' not found. Check SKU or barcode.")
                st.markdown("**Available SKUs:**")
                for sku in INVENTORY:
                    item = INVENTORY[sku]
                    st.caption(f"{sku} — {item['name']} — Shelf {item['shelf']} — Barcode: {item['barcode']}")

    # ── AI Photo ID ──
    with tab2:
        st.subheader("📸 AI Part Identification")
        st.info("Take a photo of the part — AI will identify the SKU and shelf location")

        col_l, col_r = st.columns([1,1])
        with col_l:
            capture_method = st.radio("Input method:", ["Upload photo", "Use camera"])

            if capture_method == "Upload photo":
                photo = st.file_uploader("Upload part photo", type=["jpg","png","jpeg"])
            else:
                photo = st.camera_input("Take a photo with your camera")

        with col_r:
            if photo and st.button("🤖 Identify Part", use_container_width=True, type="primary"):
                if not st.session_state.api_key:
                    st.error("API key required")
                else:
                    with st.spinner("AI analyzing image..."):
                        img = Image.open(photo)
                        client = genai.Client(api_key=st.session_state.api_key)

                        inventory_info = json.dumps({
                            k: {"name": v["name"], "shelf": v["shelf"]}
                            for k,v in INVENTORY.items()
                        })

                        resp = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[img, f"""Identify this electrical part.
Available inventory: {inventory_info}
Return JSON only: {{"sku":"...","name":"...","shelf":"...","confidence":"high/medium/low","notes":"..."}}"""]
                        )

                        try:
                            result_text = resp.text.strip()
                            if "```" in result_text:
                                result_text = result_text.split("```")[1].replace("json","").strip()
                            result = json.loads(result_text)

                            sku = result.get("sku","")
                            if sku in INVENTORY:
                                item = INVENTORY[sku]
                                r = calculate_reorder(sku)
                                st.success(f"✅ Identified: **{sku}**")
                                st.write(f"**{item['name']}**")
                                col_a, col_b, col_c = st.columns(3)
                                col_a.metric("Shelf", item["shelf"])
                                col_b.metric("Stock", item["current_stock"])
                                col_c.metric("Status", r["status"])
                                st.caption(f"Confidence: {result.get('confidence','—')} | {result.get('notes','')}")
                            else:
                                st.warning(f"Part identified as: {result.get('name','unknown')} — not in inventory")
                                st.json(result)
                        except:
                            st.write(resp_text)

    # ── Scan Log ──
    with tab3:
        st.subheader("My Scan Log — Today")
        if st.session_state.scan_log:
            df = pd.DataFrame(st.session_state.scan_log)
            st.dataframe(df, use_container_width=True, hide_index=True)
            if st.button("Clear Log"):
                st.session_state.scan_log = []
                st.rerun()
        else:
            st.info("No scans yet. Use the scanner tab to start.")


    # ── Outbound Prep (配合司机) ──
    with tab4:
        st.subheader("🚛 Outbound Prep — Prepare Goods for Driver")
        st.info("Pick items from shelves and mark them as ready before the driver arrives.")

        for del_id, delivery in st.session_state.deliveries.items():
            if delivery["status"] in ["Loading", "Dispatched"]:
                status_icon = "🔵" if delivery["status"] == "Loading" else "🟡"
                with st.expander(
                    f"{status_icon} {del_id} — {delivery['destination']} — Driver: {delivery['driver']} — ETA: {delivery['eta']}",
                    expanded=(delivery["status"] == "Loading")
                ):
                    st.markdown("**Items to pick:**")
                    all_picked = True
                    for sku, qty in delivery["items"].items():
                        item = INVENTORY.get(sku, {})
                        stock = item.get("current_stock", 0)
                        shelf = item.get("shelf", "—")
                        enough = stock >= qty

                        col_a, col_b, col_c, col_d = st.columns([2, 3, 2, 2])
                        col_a.write(f"**{sku}**")
                        col_b.write(f"{item.get('name','—')}")
                        col_c.write(f"📍 Shelf **{shelf}**")
                        if enough:
                            col_d.success(f"✅ {qty} units ready")
                        else:
                            col_d.error(f"❌ Need {qty}, only {stock}")
                            all_picked = False

                    st.divider()
                    col_btn, col_info = st.columns([1, 2])
                    with col_btn:
                        status = delivery["status"]
                    if status == "Loading":
                            if st.button("✅ Mark All Picked — Ready for Driver",
                                         key=f"staff_ready_{del_id}",
                                         use_container_width=True,
                                         type="primary",
                                         disabled=not all_picked):
                                # Log outbound scans
                                for sku, qty in delivery["items"].items():
                                    st.session_state.scan_log.append({
                                        "time": datetime.now().strftime("%H:%M:%S"),
                                        "sku": sku,
                                        "name": INVENTORY.get(sku, {}).get("name", ""),
                                        "shelf": INVENTORY.get(sku, {}).get("shelf", ""),
                                        "operation": "OUT",
                                        "qty": qty,
                                        "note": f"Outbound {del_id} → {delivery['destination']}",
                                        "by": st.session_state.employee_name,
                                    })
                                    if sku in INVENTORY:
                                        INVENTORY[sku]["current_stock"] = max(0, INVENTORY[sku]["current_stock"] - qty)
                                advance_task(del_id, "Picked", st.session_state.employee_id)
                                st.success(f"✅ {del_id} marked Picked! Driver can now collect.")
                                st.rerun()
                    elif status == "Picked":
                            if st.button("📦 Hand to Driver — Mark Dispatched",
                                         key=f"staff_dispatch_{del_id}",
                                         use_container_width=True,
                                         type="primary"):
                                advance_task(del_id, "Dispatched", st.session_state.employee_id)
                                st.success(f"✅ {del_id} dispatched!")
                                st.rerun()
                    else:
                            st.success(f"✅ {status}")

                    # Show task timeline
                    st.markdown("**Timeline:**")
                    for h in delivery.get("history", []):
                        icon = TASK_ICONS.get(h["status"], "⚪")
                        st.caption(f"{icon} {h['time']} — {h['status']} by {h['by']}")
                    with col_info:
                        st.write(f"🗺️ Delivering to: {delivery['address']}")

        # If no active deliveries
        active = [d for d in st.session_state.deliveries.values() if d["status"] in ["Loading","Dispatched"]]
        if not active:
            st.success("✅ All deliveries dispatched for today!")

    # ── Find Item ──
    with tab5:
        st.subheader("🔍 Find Item — Shelf Location Search")
        search_q = st.text_input("Search SKU, item name, or shelf:", placeholder="e.g. CB-1201  or  breaker  or  A-03")

        if search_q:
            q = search_q.strip().lower()
            results = {
                sku: item for sku, item in INVENTORY.items()
                if q in sku.lower()
                or q in item["name"].lower()
                or q in item["shelf"].lower()
                or q in item.get("barcode", "").lower()
            }
            if results:
                st.success(f"✅ Found {len(results)} item(s)")
                for sku, item in results.items():
                    r = calculate_reorder(sku)
                    st.markdown(f"""
                    <div style='background:#1e293b;border-radius:12px;padding:16px;margin:8px 0;
                                border-left:4px solid #3b82f6;'>
                        <div style='font-size:22px;font-weight:700;color:#f1f5f9;'>{sku} — {item["name"]}</div>
                        <div style='font-size:28px;font-weight:800;color:#3b82f6;margin:8px 0;'>
                            📍 Shelf: {item["shelf"]}
                        </div>
                        <div style='color:#94a3b8;font-size:14px;'>
                            Stock: {item["current_stock"]} units &nbsp;|&nbsp;
                            Status: {r["status"]} &nbsp;|&nbsp;
                            Barcode: {item.get("barcode","—")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"No items found for '{search_q}'")
        else:
            st.markdown("#### All Shelf Locations")
            for sku, item in INVENTORY.items():
                r = calculate_reorder(sku)
                col_a, col_b, col_c, col_d = st.columns([2, 3, 2, 2])
                col_a.write(f"**{sku}**")
                col_b.write(item["name"])
                col_c.markdown(f"📍 **{item['shelf']}**")
                col_d.write(f"{item['current_stock']} units  {r['status']}")


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
                    st.warning(f"⏳ Waiting for warehouse staff — Status: {status}")

                # Task history timeline
                st.divider()
                st.markdown("**Timeline:**")
                for h in delivery.get("history", []):
                    icon = TASK_ICONS.get(h["status"], "⚪")
                    st.caption(f"{icon} {h['time']} — {h['status']} by {h['by']}")

                st.divider()
                encoded = delivery["address"].replace(" ", "+")
                maps_url = f"https://maps.google.com?q={encoded}"
                st.markdown(
                    f'<a href="{maps_url}" target="_blank" style="display:block;text-align:center;'
                    f'background:#1e293b;color:#3b82f6;padding:10px;border-radius:8px;'
                    f'text-decoration:none;border:1px solid #334155;font-weight:600;">'
                    f'🗺️ Open in Google Maps</a>',
                    unsafe_allow_html=True
                )

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

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 My Orders", "➕ New Order", "🧠 Knowledge Base (RAG)", "📊 Customer Analytics"
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


st.divider()
st.caption("InstaWorker™ Pro v6  ·  2026 Hackathon · San Francisco  ·  Powered by Google DeepMind × InstaLILY  ·  Edge-Native 🔒")
