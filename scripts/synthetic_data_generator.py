"""
Synthetic Data Generator — build diverse JSONL training data by randomizing
inventory (current_stock, sales_history), then calling calculate_reorder and
forecast_demand to fill the same Reasoning + Action + Result template.

Usage:
  python scripts/synthetic_data_generator.py -o synthetic_training.jsonl
  python scripts/synthetic_data_generator.py -o synthetic_training.jsonl --samples 20 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from data import INVENTORY_DEFAULT, calculate_reorder, forecast_demand

# Prompt templates per action (English only)
ACTION_PROMPTS = {
    "stock_lookup": [
        "What is the current stock for {sku}?",
        "How many {name} do we have?",
        "Current quantity for {sku}?",
        "Stock level for {sku}?",
        "Do we have {name} in stock? How many?",
        "Inventory count for {sku}.",
    ],
    "reorder_check": [
        "Should we reorder {sku}?",
        "Do we need to order more {name}?",
        "Is {sku} below reorder point?",
        "Recommend reorder for {sku}.",
        "Should I place an order for {name}?",
        "Reorder suggestion for {sku}?",
        "Are we low on {name}?",
    ],
    "shelf_location": [
        "Where is {sku} stored?",
        "Location of {name} in the warehouse?",
        "Which shelf is {sku} on?",
        "Where can I find {name}?",
        "Shelf location for {sku}.",
    ],
    "supplier_info": [
        "Who supplies {sku}?",
        "Vendor for {name}?",
        "Supplier of {sku}?",
        "Who do we order {name} from?",
    ],
    "demand_forecast": [
        "What is the demand forecast for {sku}?",
        "Demand trend for {name}?",
        "Sales trend for {sku}?",
        "How is {name} selling?",
    ],
    "recommend_next_purchase": [
        "What should we buy next?",
        "Recommend products to purchase next.",
        "Which items should we order next? Consider sales and time.",
        "What do you recommend we stock up on? Consider demand and lead time.",
        "Next purchase recommendations based on sales and inventory.",
        "Prioritize what to reorder — consider volume and time.",
    ],
}


def _response_with_reasoning(reasoning: str, action: str, result: str) -> str:
    """Format: Reasoning + Action + Result (English only)."""
    return f"Reasoning: {reasoning} Action: {action}. Result: {result}"


def _build_recommend_next_purchase(inventory: dict) -> tuple[str, str]:
    """Build one response for recommend next purchase (reasoning, result)."""
    candidates = []
    for sku, item in inventory.items():
        ro = calculate_reorder(sku, inventory)
        fd = forecast_demand(sku, inventory)
        if ro.get("recommended_qty", 0) <= 0:
            continue
        urgency_order = {"URGENT": 0, "ORDER NOW": 1, "MONITOR": 2}
        rank = urgency_order.get(ro.get("urgency", ""), 3)
        candidates.append({
            "sku": sku,
            "name": item.get("name", sku),
            "rec_qty": ro["recommended_qty"],
            "status": ro.get("status", ""),
            "days_left": ro.get("days_left", 999),
            "trend": fd.get("trend_pct", 0),
            "demand_level": fd.get("demand_level", ""),
            "rank": rank,
        })
    candidates.sort(key=lambda x: (x["rank"], -x["days_left"], -x["trend"]))
    if not candidates:
        reasoning = "Considered reorder status, demand trend, and days of stock. No items currently need reorder."
        result = "No products need to be purchased right now; inventory is above reorder points."
        return reasoning, result
    top = candidates[:5]
    reasoning = (
        "Consider: (1) reorder urgency and status, (2) sales/demand trend and volume, "
        "(3) days of stock left, (4) lead time. Ranked items that need reorder by urgency, then by days left and demand trend."
    )
    lines = [f"{i+1}. {x['sku']} ({x['name']}) — {x['status']}, recommend {x['rec_qty']} units, ~{x['days_left']:.0f}d left, demand {x['demand_level']}, trend {x['trend']:+.1f}%." for i, x in enumerate(top)]
    result = "Recommended to purchase next: " + " ".join(lines)
    return reasoning, result


def make_synthetic_inventory(base_inventory: dict, rng: random.Random) -> dict:
    """
    Clone base_inventory and assign random current_stock and sales_history per SKU.
    Other fields (name, shelf, supplier, min_order_qty, max_stock, etc.) are unchanged.
    """
    inv = copy.deepcopy(base_inventory)
    for sku, item in inv.items():
        max_stock = max(1, item.get("max_stock", 500))
        min_order = max(1, item.get("min_order_qty", 10))
        # current_stock: 0 to max_stock (sometimes overstock)
        item["current_stock"] = rng.randint(0, int(max_stock * 1.1))
        # sales_history: 15 days of non-negative ints (daily sales)
        cap = max(1, max_stock // 10)
        item["sales_history"] = [rng.randint(0, cap) for _ in range(15)]
    return inv


def build_pairs_for_inventory(inventory: dict, skus: list[str], templates_per_action: int, rng: random.Random) -> list[dict]:
    """
    For the given inventory, build prompt/response pairs for each SKU in skus.
    Uses calculate_reorder and forecast_demand so results match the randomized data.
    templates_per_action: max number of prompt templates to use per action (for diversity).
    """
    pairs = []
    for sku in skus:
        if sku not in inventory:
            continue
        item = inventory[sku]
        name = item.get("name", sku)
        stock = item.get("current_stock", 0)
        shelf = item.get("shelf", "")
        supplier = item.get("supplier", "")
        ro = calculate_reorder(sku, inventory)
        fd = forecast_demand(sku, inventory)
        rec_qty = ro.get("recommended_qty", 0)
        status = ro.get("status", "")
        days_left = ro.get("days_left", 0)
        safety = ro.get("safety_stock", 0)
        reorder_pt = ro.get("reorder_point", 0)
        trend = fd.get("trend_pct", 0)
        demand_level = fd.get("demand_level", "")
        daily_avg = fd.get("daily_avg", 0)

        # Stock lookup
        reasoning_stock = f"User asked about current quantity for {sku}; need to look up stock and shelf."
        result_stock = f"{sku} ({name}) — current stock {stock} units, shelf {shelf}."
        response_stock = _response_with_reasoning(reasoning_stock, "stock lookup", result_stock)
        for template in rng.sample(ACTION_PROMPTS["stock_lookup"], min(templates_per_action, len(ACTION_PROMPTS["stock_lookup"]))):
            pairs.append({"prompt": template.format(sku=sku, name=name), "response": response_stock})

        # Reorder check
        if rec_qty > 0:
            reasoning_reorder = f"User asked whether to reorder {sku}; checked reorder point and safety stock — below threshold."
            result_reorder = f"Recommend reorder — {sku} ({name}) is {status}. Recommended quantity: {rec_qty} units. Safety stock {safety}, reorder point {reorder_pt}, ~{days_left} days left."
        else:
            reasoning_reorder = f"User asked whether to reorder {sku}; checked reorder logic — stock above reorder point."
            result_reorder = f"No reorder needed — {sku} ({name}) is {status}. Current stock {stock} is above reorder point."
        response_reorder = _response_with_reasoning(reasoning_reorder, "reorder check", result_reorder)
        for template in rng.sample(ACTION_PROMPTS["reorder_check"], min(templates_per_action, len(ACTION_PROMPTS["reorder_check"]))):
            pairs.append({"prompt": template.format(sku=sku, name=name), "response": response_reorder})

        # Shelf location
        reasoning_shelf = f"User asked where {sku} is stored; looked up shelf location."
        result_shelf = f"{sku} ({name}) is stored at shelf {shelf}."
        response_shelf = _response_with_reasoning(reasoning_shelf, "location lookup", result_shelf)
        for template in rng.sample(ACTION_PROMPTS["shelf_location"], min(templates_per_action, len(ACTION_PROMPTS["shelf_location"]))):
            pairs.append({"prompt": template.format(sku=sku, name=name), "response": response_shelf})

        # Supplier
        if supplier:
            reasoning_supplier = f"User asked who supplies {sku}; looked up supplier/vendor."
            result_supplier = f"{sku} ({name}) is supplied by {supplier}."
            response_supplier = _response_with_reasoning(reasoning_supplier, "supplier lookup", result_supplier)
            for template in rng.sample(ACTION_PROMPTS["supplier_info"], min(templates_per_action, len(ACTION_PROMPTS["supplier_info"]))):
                pairs.append({"prompt": template.format(sku=sku, name=name), "response": response_supplier})

        # Demand forecast
        reasoning_demand = f"User asked about demand/sales for {sku}; used demand forecast (trend and daily average)."
        result_demand = f"{sku} ({name}) — demand level {demand_level}, daily avg {daily_avg}, trend {trend:+.1f}%."
        response_demand = _response_with_reasoning(reasoning_demand, "demand forecast", result_demand)
        for template in rng.sample(ACTION_PROMPTS["demand_forecast"], min(templates_per_action, len(ACTION_PROMPTS["demand_forecast"]))):
            pairs.append({"prompt": template.format(sku=sku, name=name), "response": response_demand})

    # One "recommend next purchase" per synthetic inventory
    reasoning_rec, result_rec = _build_recommend_next_purchase(inventory)
    response_recommend = _response_with_reasoning(reasoning_rec, "recommend next purchase", result_rec)
    template = rng.choice(ACTION_PROMPTS["recommend_next_purchase"])
    pairs.append({"prompt": template, "response": response_recommend})

    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Synthetic data generator: randomize INVENTORY (current_stock, sales_history), "
        "call calculate_reorder/forecast_demand, fill JSONL template."
    )
    ap.add_argument("-o", "--output", default="synthetic_training.jsonl", help="Output JSONL path")
    ap.add_argument("--samples", type=int, default=10, help="Number of synthetic inventory samples (default 10)")
    ap.add_argument("--templates", type=int, default=2, help="Max prompt templates per action per SKU (default 2)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--inventory", default=None, help="Optional: path to base inventory JSON (default: INVENTORY_DEFAULT)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = INVENTORY_DEFAULT
    if args.inventory and os.path.isfile(args.inventory):
        with open(args.inventory, encoding="utf-8") as f:
            base = json.load(f)
        if not isinstance(base, dict):
            base = INVENTORY_DEFAULT
        print(f"Using base inventory from {args.inventory} ({len(base)} SKUs)")
    skus = list(base.keys())
    if not skus:
        print("No SKUs in inventory.")
        return

    all_pairs = []
    for i in range(args.samples):
        synth_inv = make_synthetic_inventory(base, rng)
        # Optionally use a random subset of SKUs per sample for more variety
        n_skus = rng.randint(max(1, len(skus) - 1), len(skus)) if len(skus) > 1 else 1
        chosen = rng.sample(skus, n_skus)
        pairs = build_pairs_for_inventory(synth_inv, chosen, args.templates, rng)
        all_pairs.extend(pairs)
    rng.shuffle(all_pairs)

    def to_contents_row(prompt: str, response: str) -> dict:
        """Format one example as contents (user + model) for chat/fine-tuning."""
        return {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "model", "parts": [{"text": response}]},
            ]
        }

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(__file__), "..", out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            row = to_contents_row(p["prompt"], p["response"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_pairs)} synthetic examples to {out_path}")


if __name__ == "__main__":
    main()
