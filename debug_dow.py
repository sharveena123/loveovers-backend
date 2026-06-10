"""Debug: verify per-weekday features + predictions for each saved model."""
import os
os.environ["FOOD_CLASSIFIER_WARMUP"] = "0"

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import app as A

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

print("=" * 100)
print("SAVED MODELS OVERVIEW")
print("=" * 100)
for cafe_id, cafe in A.cafe_models.items():
    meta = cafe["metadata"]
    dow_means = meta.get("item_dow_means")
    print(f"{cafe_id:14s} | {meta.get('cafe_name','?'):25s} | trained_at={meta.get('trained_at','?'):26s} "
          f"| has item_dow_means={bool(dow_means)} | has item_dow_lag_avgs={bool(meta.get('item_dow_lag_avgs'))}")

print()
for cafe_id, cafe in A.cafe_models.items():
    meta = cafe["metadata"]
    model = cafe["model"]
    encoders = cafe["encoders"]
    items = meta.get("items", [])
    if not items:
        continue
    item = items[0]
    dow_means = meta.get("item_dow_means", {}).get(item, {})

    print("=" * 100)
    print(f"CAFE {cafe_id} ({meta.get('cafe_name')}) — item: {item}")
    print(f"  item_dow_means for this item: { {k: round(v,1) for k,v in dow_means.items()} if dow_means else 'EMPTY (falls back to item_avg!)'}")
    print(f"  top feature importance: { {k: round(v,3) for k,v in list(meta.get('feature_importance', {}).items())[:6]} }")

    # next Monday
    base = datetime.now()
    while base.weekday() != 0:
        base += timedelta(days=1)

    rows = []
    for i, day_name in enumerate(DAYS):
        d = base + timedelta(days=i)
        dow = d.weekday()
        feats = A._build_item_features(
            item, dow, 1 if dow >= 5 else 0, d.month, d.day,
            d.isocalendar()[1], 1 if d.day <= 3 else 0, 1 if d.day >= 28 else 0,
            (d.month - 1) // 3 + 1,
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12),
            np.sin(2 * np.pi * d.day / 31), np.cos(2 * np.pi * d.day / 31),
            meta, encoders, price=0,
        )
        X = pd.DataFrame([{c: feats.get(c, 0) for c in meta["feature_columns"]}])
        raw = model.predict(X)[0]
        if meta.get("log_transformed"):
            raw = np.expm1(raw)
        rows.append({
            "day": day_name, "dow": dow,
            "lag_1": round(feats["sold_qty_lag_1"], 1),
            "lag_7": round(feats["sold_qty_lag_7"], 1),
            "lag_14": round(feats["sold_qty_lag_14"], 1),
            "roll_7": round(feats["sold_qty_roll_7"], 2),
            "item_dow_avg": round(feats["item_dow_avg"], 1),
            "dow_sin": round(feats["dow_sin"], 3),
            "pred": round(float(raw), 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()
