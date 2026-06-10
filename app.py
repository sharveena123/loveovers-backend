from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import joblib
import os
import uuid
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import database as db

import requests

OLLAMA_AVAILABLE = False
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

load_dotenv()

app = Flask(__name__)
CORS(app)

def check_ollama():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False

if check_ollama():
    OLLAMA_AVAILABLE = True
    print(f"✅ Ollama available at {OLLAMA_URL}")
    print(f"   Using model: {OLLAMA_MODEL}")
else:
    print(f"⚠️ Ollama not running at {OLLAMA_URL}")
    print("   To use Ollama:")
    print("   1. Install from: https://ollama.ai")
    print("   2. Run: ollama serve")
    print("   3. Pull a model: ollama pull mistral")
    print("   Falling back to rule-based assessment only.")

cafe_models = {}
pending_assessments = {}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ASSESSMENTS_DIR = os.path.join(os.path.dirname(__file__), "assessments")
os.makedirs(ASSESSMENTS_DIR, exist_ok=True)

def save_assessment_to_disk(assessment_id, assessment_data):
    try:
        path = os.path.join(ASSESSMENTS_DIR, f"{assessment_id}.json")
        with open(path, "w") as f:
            json.dump(assessment_data, f, indent=2, default=str)
        print(f"✅ Assessment saved: {path}")
    except Exception as e:
        print(f"❌ Error saving assessment: {e}")

def load_assessment_from_disk(assessment_id):
    try:
        path = os.path.join(ASSESSMENTS_DIR, f"{assessment_id}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"❌ Error loading assessment: {e}")
    return None

def save_model_to_disk(cafe_id, cafe_data):
    try:
        model_path = os.path.join(MODELS_DIR, f"{cafe_id}.joblib")
        joblib.dump(cafe_data, model_path)
        print(f"✅ Model saved: {model_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

def load_model_from_disk(cafe_id):
    try:
        model_path = os.path.join(MODELS_DIR, f"{cafe_id}.joblib")
        if os.path.exists(model_path):
            cafe_data = joblib.load(model_path)
            print(f"✅ Model loaded: {model_path}")
            return cafe_data
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    return None

def load_all_models():
    global cafe_models
    try:
        if os.path.exists(MODELS_DIR):
            for filename in os.listdir(MODELS_DIR):
                if filename.endswith(".joblib"):
                    cafe_id = filename[:-7]
                    cafe_data = load_model_from_disk(cafe_id)
                    if cafe_data:
                        cafe_models[cafe_id] = cafe_data
                        meta = cafe_data["metadata"]
                        if not db.get_cafe(cafe_id):
                            db.upsert_cafe(
                                cafe_id,
                                meta.get("cafe_name", "Unknown Cafe"),
                                column_mapping=meta.get("mapping"),
                                items=meta.get("items"),
                                metrics={
                                    "mae": meta.get("mae"),
                                    "r2": meta.get("r2"),
                                    "cv_mae": meta.get("cv_mae"),
                                    "cv_r2": meta.get("cv_r2"),
                                    "training_rows": meta.get("training_rows"),
                                    "items": meta.get("items"),
                                },
                                trained=True,
                            )
            print(f"✅ Loaded {len(cafe_models)} saved models (synced to database)")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_all_models()


def _warm_up_food_classifier():
    if os.getenv("FOOD_CLASSIFIER_WARMUP", "1") == "0":
        return
    try:
        import food_classifier as fc
        fc.warm_up()
    except ImportError:
        print("Food classifier: pip install torch torchvision pillow")
    except Exception as e:
        print(f"Food classifier warmup failed: {e}")


_warm_up_food_classifier()


def get_cafe_model(cafe_id):
    if cafe_id in cafe_models:
        return cafe_models[cafe_id]
    loaded = load_model_from_disk(cafe_id)
    if loaded:
        cafe_models[cafe_id] = loaded
        return loaded
    return None


def train_model_from_standardized_df(cafe_id, cafe_name, df_std, mapping, persist_sales=True):
    if df_std is None or len(df_std) == 0:
        raise ValueError("Dataset is empty — no sales rows available for training.")
    if df_std["item"].nunique() == 0:
        raise ValueError("Dataset not suitable — no item names found.")

    if persist_sales:
        db.save_sales_dataframe(cafe_id, df_std, source="upload")

    df_feat = engineer_features(df_std.copy())
    df_feat = df_feat.dropna(subset=["sold_qty_lag_1"]).reset_index(drop=True)

    if len(df_feat) < 20:
        raise ValueError("Need at least 20 rows with historical lags after feature engineering.")

    item_encoder = LabelEncoder()
    df_feat["item_encoded"] = item_encoder.fit_transform(df_feat["item"])

    feature_cols = [
        "item_encoded",
        "is_weekend", "month", "day_of_month", "week_of_year",
        "is_month_start", "is_month_end", "quarter", "day_of_week_num",
        "sold_qty_lag_1", "sold_qty_lag_2", "sold_qty_lag_3",
        "sold_qty_lag_7", "sold_qty_lag_14", "sold_qty_lag_21", "sold_qty_lag_28",
        "sold_qty_roll_3", "sold_qty_roll_7", "sold_qty_roll_14", "sold_qty_roll_30",
        "sold_qty_roll_std_7", "sold_qty_roll_std_14",
        "sold_qty_roll_max_7", "sold_qty_roll_min_7",
        "sold_qty_ewm_3", "sold_qty_ewm_7", "sold_qty_ewm_14",
        "sold_qty_expanding_mean", "sold_qty_trend_7v30",
        "days_since_last_sale",
        "item_avg_sales", "item_std_sales", "item_max_sales", "item_median_sales",
        "item_dow_avg",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "day_sin", "day_cos",
    ]
    if "price" in df_feat.columns and df_feat["price"].sum() > 0:
        feature_cols.append("price")

    available_features = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[available_features]
    y_raw = df_feat["sold_qty"].astype(float)
    y = np.log1p(y_raw)

    n_splits = min(5, max(2, len(X) // 30))
    xgb_params = dict(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
        tree_method="hist",
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores, cv_r2_scores = [], []
    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        y_val_raw_cv = y_raw.iloc[val_idx]
        model_cv = xgb.XGBRegressor(**xgb_params)
        model_cv.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            verbose=False,
        )
        preds_raw = np.expm1(model_cv.predict(X_val_cv)).clip(min=0)
        cv_scores.append(mean_absolute_error(y_val_raw_cv, preds_raw))
        cv_r2_scores.append(r2_score(y_val_raw_cv, preds_raw))

    cv_mae = float(np.mean(cv_scores))
    cv_r2 = float(np.mean(cv_r2_scores))

    split_idx = int(len(X) * 0.90)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    y_val_raw = y_raw.iloc[split_idx:]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = np.expm1(model.predict(X_val)).clip(min=0)
    mae = mean_absolute_error(y_val_raw, y_pred)
    r2 = r2_score(y_val_raw, y_pred)

    importance = dict(zip(available_features, model.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

    item_stats_df = df_feat.groupby("item")["sold_qty"].agg(
        mean="mean", std="std", max="max", min="min"
    ).reset_index()
    item_stats = item_stats_df.set_index("item").to_dict("index")
    items = list(df_feat["item"].unique())

    # ── FIX: Build per-item per-DOW averages for realistic lag inputs in batch-predict ──
    dow_means_df = df_feat.groupby(["item", "day_of_week_num"])["sold_qty"].mean().reset_index()
    item_dow_means = {}
    for _, row in dow_means_df.iterrows():
        item_dow_means.setdefault(row["item"], {})[int(row["day_of_week_num"])] = float(row["sold_qty"])

    # ── FIX: Build per-item per-DOW rolling/lag averages (last 4 weeks of each DOW) ──
    # We compute realistic lag values per DOW: what was typically sold on that same DOW
    # in prior weeks (lag_7, lag_14, lag_21, lag_28 are all same-DOW lags in weekly data)
    item_dow_lag_avgs = {}
    for item in items:
        item_dow_lag_avgs[item] = {}
        item_df = df_feat[df_feat["item"] == item].sort_values("date")
        for dow in range(7):
            dow_rows = item_df[item_df["day_of_week_num"] == dow]["sold_qty"].values
            if len(dow_rows) >= 2:
                avg = float(np.mean(dow_rows))
                std = float(np.std(dow_rows)) if len(dow_rows) > 1 else 1.0
                mx = float(np.max(dow_rows))
                mn = float(np.min(dow_rows))
                # use last few observations for lag estimates
                recent = dow_rows[-4:] if len(dow_rows) >= 4 else dow_rows
                item_dow_lag_avgs[item][dow] = {
                    "avg": avg, "std": std, "max": mx, "min": mn,
                    "lag_1": float(recent[-1]),
                    "lag_2": float(recent[-2]) if len(recent) >= 2 else avg,
                    "lag_3": float(recent[-3]) if len(recent) >= 3 else avg,
                    "roll_3": float(np.mean(recent[-3:])) if len(recent) >= 3 else avg,
                    "roll_7": avg,
                    "ewm": float(pd.Series(dow_rows).ewm(span=7, min_periods=1).mean().iloc[-1]),
                }
            else:
                fallback = float(item_stats.get(item, {}).get("mean", 5))
                item_dow_lag_avgs[item][dow] = {
                    "avg": fallback, "std": 1.0, "max": fallback * 2, "min": fallback * 0.5,
                    "lag_1": fallback, "lag_2": fallback, "lag_3": fallback,
                    "roll_3": fallback, "roll_7": fallback, "ewm": fallback,
                }

    metrics = {
        "mae": mae,
        "r2": r2,
        "cv_mae": cv_mae,
        "cv_r2": cv_r2,
        "training_rows": len(df_feat),
        "items": items,
    }

    cafe_models[cafe_id] = {
        "model": model,
        "encoders": {"item": item_encoder, "features": available_features},
        "metadata": {
            "cafe_name": cafe_name,
            "cafe_id": cafe_id,
            "training_rows": len(df_feat),
            "items": items,
            "item_stats": item_stats,
            "item_dow_means": item_dow_means,          # ── FIX: added ──
            "item_dow_lag_avgs": item_dow_lag_avgs,    # ── FIX: added ──
            "mae": mae,
            "r2": r2,
            "cv_mae": cv_mae,
            "cv_r2": cv_r2,
            "feature_columns": available_features,
            "feature_importance": importance_sorted,
            "mapping": mapping,
            "log_transformed": True,
            "trained_at": datetime.now().isoformat(),
        },
    }
    save_model_to_disk(cafe_id, cafe_models[cafe_id])
    db.upsert_cafe(
        cafe_id, cafe_name,
        column_mapping=mapping,
        items=items,
        metrics=metrics,
        trained=True,
    )
    return metrics, importance_sorted, items


# ============================================================
# LAYER 1: RULE-BASED COLUMN MAPPING
# ============================================================

COLUMN_ALIASES = {
    "date": [
        "date", "order_date", "transaction_date", "sale_date", "created_at",
        "datetime", "timestamp", "day", "date_time", "orderdate", "saledate",
        "txn_date", "dt", "record_date", "entry_date"
    ],
    "item": [
        "item", "product", "product_name", "menu_item", "food_item",
        "dish", "bakery_item", "name", "item_name", "productname", "fooditem",
        "prod", "menu", "food", "goods", "article", "sku", "variant"
    ],
    "sold_qty": [
        "sold_qty", "sales", "qty_sold", "units_sold", "quantity_sold",
        "sold", "units", "qty", "quantity", "amount_sold", "sale_qty", "volume",
        "sales_qty", "unit_sold", "pieces_sold", "items_sold", "count_sold",
        "txn_qty", "order_qty", "purchase_qty", "demand"
    ],
    "produced_qty": [
        "produced_qty", "stock", "baked_qty", "production", "made",
        "prepared", "baked", "inventory", "quantity_produced", "produce_qty",
        "manufactured", "created", "batch_size", "yield", "output",
        "supply", "stock_level", "on_hand", "available"
    ],
    "price": [
        "price", "unit_price", "cost", "selling_price", "price_rm", "amount",
        "value", "price_per_unit", "retail_price", "sale_price",
        "revenue", "sales_amount", "total_price", "item_price", "menu_price"
    ],
    "day_of_week": [
        "day_of_week", "weekday", "day", "week_day", "dow",
        "weekday_name", "day_name", "calendar_day"
    ]
}

REQUIRED_CORE = ["date", "item", "sold_qty"]
OPTIONAL_FEATURES = ["produced_qty", "price", "day_of_week"]
ALL_STANDARD_FIELDS = REQUIRED_CORE + OPTIONAL_FEATURES

def layer1_rule_based(columns):
    mapping = {}
    unmapped = []
    used_standards = set()

    for col in columns:
        col_clean = col.lower().strip().replace(" ", "_").replace("-", "_")
        matched = False

        for standard_name, aliases in COLUMN_ALIASES.items():
            if standard_name in used_standards:
                continue

            if col_clean in aliases:
                mapping[standard_name] = col
                used_standards.add(standard_name)
                matched = True
                print(f"   📋 Rule match: '{col}' → {standard_name} (exact)")
                break

            if not matched:
                for alias in aliases:
                    if len(alias) >= 4 and len(col_clean) >= 4:
                        if alias in col_clean and len(alias) >= len(col_clean) * 0.6:
                            mapping[standard_name] = col
                            used_standards.add(standard_name)
                            matched = True
                            print(f"   📋 Rule match: '{col}' → {standard_name} (substring: {alias})")
                            break

            if matched:
                break

        if not matched:
            unmapped.append(col)
            print(f"   ❓ Unmapped: '{col}' — will send to Ollama")

    print(f"📊 Layer 1 (Rule-based): Mapped {len(mapping)} columns, {len(unmapped)} unmapped")
    return mapping, unmapped

# ============================================================
# LAYER 2: LLM SEMANTIC MATCHING (Ollama)
# ============================================================

def layer2_llm_fallback(unmapped_columns, existing_mapping, cafe_name=""):
    if not unmapped_columns:
        print("✓ Layer 2 (Ollama): Skipped - no unmapped columns")
        return {}

    if not OLLAMA_AVAILABLE:
        print("⚠️ Layer 2 (Ollama): Skipped - Ollama not available")
        return {}

    try:
        standard_options = ["date", "item", "sold_qty", "produced_qty", "price", "day_of_week", "unknown"]

        prompt = f"""Map CSV columns to standard fields for cafe sales data.

Columns to map: {json.dumps(unmapped_columns)}

Standard fields:
- date: sale date/timestamp
- item: product/item name
- sold_qty: units sold to customers
- produced_qty: units produced/baked
- price: price per unit
- day_of_week: day of week
- unknown: unknown/other

Return only JSON. Example: {{"column_name": "date", "another_column": "sold_qty"}}
For each column in the list, return the most likely standard field. If uncertain, use "unknown"."""

        print(f"🤖 Layer 2 (Ollama): Assessing {len(unmapped_columns)} columns with {OLLAMA_MODEL}...")
        print(f"   Columns: {unmapped_columns}")

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            },
            timeout=(10, 300)
        )

        if response.status_code != 200:
            print(f"⚠️ Ollama returned {response.status_code}: {response.text[:200]}")
            return {}

        response_data = response.json()
        response_text = response_data.get("response", "").strip()
        print(f"   Ollama response: {response_text[:300]}...")

        try:
            llm_mapping = json.loads(response_text)
        except json.JSONDecodeError:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                start = response_text.find("{")
                end = response_text.rfind("}")
                if start != -1 and end != -1:
                    json_str = response_text[start:end+1]
                else:
                    print(f"❌ Could not parse Ollama response as JSON")
                    return {}
            llm_mapping = json.loads(json_str)

        print(f"   Parsed mapping: {llm_mapping}")

        validated = {}
        for col, standard in llm_mapping.items():
            if standard in standard_options and col in unmapped_columns:
                validated[col] = standard
                print(f"   ✅ Ollama mapped: '{col}' → {standard}")

        print(f"✅ Layer 2 (Ollama): Mapped {len(validated)}/{len(unmapped_columns)} columns")
        return validated

    except requests.exceptions.ConnectionError:
        print(f"❌ Layer 2 (Ollama): Could not connect to {OLLAMA_URL}")
        return {}
    except Exception as e:
        print(f"❌ Layer 2 (Ollama): Failed - {type(e).__name__}: {e}")
        return {}

# ============================================================
# LAYER 3: HUMAN CONFIRMATION UI
# ============================================================

def layer3_human_confirmation(mapping, unmapped_after_llm):
    needs_confirmation = []

    for col in unmapped_after_llm:
        suggestion = "unknown"
        col_lower = col.lower()

        if any(x in col_lower for x in ["date", "time", "day"]):
            suggestion = "date"
        elif any(x in col_lower for x in ["item", "product", "name", "menu"]):
            suggestion = "item"
        elif any(x in col_lower for x in ["sold", "sale", "qty", "quantity", "unit"]):
            suggestion = "sold_qty"
        elif any(x in col_lower for x in ["produce", "made", "bake", "stock", "inventory"]):
            suggestion = "produced_qty"
        elif any(x in col_lower for x in ["price", "cost", "amount", "revenue"]):
            suggestion = "price"

        needs_confirmation.append({
            "column": col,
            "suggested_mapping": suggestion,
            "confidence": "low",
            "options": ALL_STANDARD_FIELDS + ["unknown"]
        })

    return needs_confirmation

# ============================================================
# DATA STANDARDIZATION
# ============================================================

def standardize_dataset(df, mapping, user_corrections=None):
    standardized = pd.DataFrame()

    for standard, original in mapping.items():
        if original in df.columns:
            standardized[standard] = df[original].copy()

    if user_corrections:
        for original, standard in user_corrections.items():
            if original in df.columns and standard != "unknown":
                standardized[standard] = df[original].copy()

    if len(standardized) == 0:
        raise ValueError("Dataset is empty — no data rows found after applying the column mapping.")

    missing = [c for c in ("date", "item", "sold_qty") if c not in standardized.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns after mapping: {', '.join(missing)}.")

    # Coerce types and drop rows that cannot be used for training
    total_rows = len(standardized)
    standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce")
    standardized["sold_qty"] = pd.to_numeric(standardized["sold_qty"], errors="coerce")
    if "produced_qty" in standardized.columns:
        standardized["produced_qty"] = pd.to_numeric(standardized["produced_qty"], errors="coerce")
    if "price" in standardized.columns:
        standardized["price"] = pd.to_numeric(standardized["price"], errors="coerce")
    standardized["item"] = standardized["item"].astype(str).str.strip()

    standardized = standardized.dropna(subset=["date", "sold_qty"])
    standardized = standardized[standardized["item"].ne("") & (standardized["item"].str.lower() != "nan")]
    standardized = standardized[standardized["sold_qty"] >= 0]
    standardized = standardized.reset_index(drop=True)

    if len(standardized) == 0:
        raise ValueError(
            f"Dataset not suitable: none of the {total_rows} rows are usable "
            "(dates unparseable, sales values non-numeric/negative, or item names missing)."
        )

    dropped = total_rows - len(standardized)
    if dropped > 0:
        print(f"⚠️  standardize_dataset: dropped {dropped}/{total_rows} unusable rows")

    if "produced_qty" not in standardized.columns:
        standardized["produced_qty"] = (standardized["sold_qty"] * 1.12).astype(int)
    else:
        standardized["produced_qty"] = (
            standardized["produced_qty"].fillna(standardized["sold_qty"] * 1.12).astype(int)
        )

    if "price" not in standardized.columns:
        standardized["price"] = 0.0

    if "day_of_week" not in standardized.columns:
        if "date" in standardized.columns:
            standardized["day_of_week"] = pd.to_datetime(standardized["date"]).dt.day_name()
        else:
            standardized["day_of_week"] = "Unknown"

    standardized["surplus_qty"] = (standardized["produced_qty"] - standardized["sold_qty"]).clip(lower=0)

    return standardized

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["item", "date"]).reset_index(drop=True)

    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week_num"] = df["date"].dt.dayofweek

    df["sold_qty"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.clip(upper=x.quantile(0.99)) if len(x) > 10 else x
    )

    df["price"] = df["price"].fillna(df["price"].median() if df["price"].sum() > 0 else 0)

    df = df.sort_values(["item", "date"]).reset_index(drop=True)

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"sold_qty_lag_{lag}"] = df.groupby("item")["sold_qty"].shift(lag)

    for window in [3, 7, 14, 30]:
        df[f"sold_qty_roll_{window}"] = df.groupby("item")["sold_qty"].transform(
            lambda x, w=window: x.shift(1).rolling(window=w, min_periods=1).mean()
        )

    df["sold_qty_roll_std_7"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
    )
    df["sold_qty_roll_std_14"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=2).std()
    )
    df["sold_qty_roll_max_7"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).max()
    )
    df["sold_qty_roll_min_7"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).min()
    )

    df["sold_qty_ewm_3"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
    )
    df["sold_qty_ewm_7"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).ewm(span=7, min_periods=1).mean()
    )
    df["sold_qty_ewm_14"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).ewm(span=14, min_periods=1).mean()
    )

    df["sold_qty_expanding_mean"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    df["sold_qty_trend_7v30"] = df["sold_qty_roll_7"] - df["sold_qty_roll_30"]

    dow_item_mean = df.groupby(["item", "day_of_week_num"])["sold_qty"].mean().reset_index()
    dow_item_mean.rename(columns={"sold_qty": "item_dow_avg"}, inplace=True)
    df = df.merge(dow_item_mean, on=["item", "day_of_week_num"], how="left")

    df["days_since_last_sale"] = df.groupby("item")["date"].diff().dt.days

    item_stats = df.groupby("item")["sold_qty"].agg(
        mean="mean", std="std", max="max", median="median"
    ).reset_index()
    df = df.merge(item_stats, on="item", how="left")
    df.rename(columns={
        "mean": "item_avg_sales",
        "std": "item_std_sales",
        "max": "item_max_sales",
        "median": "item_median_sales"
    }, inplace=True)

    df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

    lag_cols = [
        "sold_qty_lag_1", "sold_qty_lag_2", "sold_qty_lag_3",
        "sold_qty_lag_7", "sold_qty_lag_14", "sold_qty_lag_21", "sold_qty_lag_28",
        "sold_qty_roll_3", "sold_qty_roll_7", "sold_qty_roll_14", "sold_qty_roll_30",
        "sold_qty_roll_std_7", "sold_qty_roll_std_14",
        "sold_qty_roll_max_7", "sold_qty_roll_min_7",
        "sold_qty_ewm_3", "sold_qty_ewm_7", "sold_qty_ewm_14",
        "sold_qty_expanding_mean", "sold_qty_trend_7v30",
        "days_since_last_sale"
    ]
    for col in lag_cols:
        df[col] = df[col].fillna(df["item_dow_avg"]).fillna(df["item_avg_sales"]).fillna(0)

    df["item_std_sales"] = df["item_std_sales"].fillna(0)

    return df


# ============================================================
# HELPER: Build DOW-aware feature dict for a single item
# ============================================================

def _build_item_features(item, day_of_week_num, is_weekend, month, day_of_month,
                          week_of_year, is_month_start, is_month_end, quarter,
                          dow_sin, dow_cos, month_sin, month_cos, day_sin, day_cos,
                          meta, encoders, price=0):
    """
    Build the full feature dict for one item using DOW-aware lag averages.

    KEY INSIGHT — what the model learned during training:
      lag_1  = yesterday's sales       → prev_dow avg
      lag_2  = 2 days ago              → prev_dow-1 avg
      lag_3  = 3 days ago              → prev_dow-2 avg
      lag_7  = 7 days ago (same DOW)   → same_dow avg
      lag_14 = 14 days ago (same DOW)  → same_dow avg
      lag_21 = 21 days ago (same DOW)  → same_dow avg
      lag_28 = 28 days ago (same DOW)  → same_dow avg
      roll_3 = mean(lag_1, lag_2, lag_3) → mean of last 3 days
      roll_7 = mean of last 7 days     → mean of 7 preceding DOW avgs
      roll_14 = mean of last 14 days   → overall item avg
      roll_30 = mean of last 30 days   → overall item avg

    This ensures Mon/Tue/Wed/Thu/Fri each receive different lag inputs,
    which is what drives differentiated predictions.
    """
    item_stats = meta.get("item_stats", {}).get(item, {})
    item_avg    = item_stats.get("mean", 5)
    item_std    = item_stats.get("std") or 2
    item_max    = item_stats.get("max", item_avg * 2)
    item_median = item_stats.get("median", item_avg)

    # Per-DOW average lookup (0=Mon … 6=Sun)
    dow_means = meta.get("item_dow_means", {}).get(item, {})

    def _dow_avg(d):
        """Return the historical avg for this item on day-of-week d, fall back to item_avg."""
        return dow_means.get(d, dow_means.get(str(d), item_avg))

    # ── Correct lag assignments matching what the model saw during training ──
    lag_1  = _dow_avg((day_of_week_num - 1) % 7)   # yesterday
    lag_2  = _dow_avg((day_of_week_num - 2) % 7)   # 2 days ago
    lag_3  = _dow_avg((day_of_week_num - 3) % 7)   # 3 days ago
    lag_7  = _dow_avg(day_of_week_num)              # same DOW last week
    lag_14 = _dow_avg(day_of_week_num)              # same DOW 2 weeks ago
    lag_21 = _dow_avg(day_of_week_num)              # same DOW 3 weeks ago
    lag_28 = _dow_avg(day_of_week_num)              # same DOW 4 weeks ago

    # Rolling windows: mean of the N preceding days in the DOW cycle
    roll_3  = sum(_dow_avg((day_of_week_num - i) % 7) for i in range(1, 4)) / 3
    roll_7  = sum(_dow_avg((day_of_week_num - i) % 7) for i in range(1, 8)) / 7
    roll_14 = item_avg   # 14-day avg smooths to overall mean
    roll_30 = item_avg   # 30-day avg smooths to overall mean

    # Std / max / min — use same-DOW stats if available, else item-wide stats
    dow_lag_data = meta.get("item_dow_lag_avgs", {}).get(item, {}).get(day_of_week_num, {})
    dow_std = dow_lag_data.get("std", item_std)
    dow_max = dow_lag_data.get("max", item_max)
    dow_min = dow_lag_data.get("min", max(0, item_avg * 0.3))

    # EWM: exponentially weighted mean of the last several same-DOW observations
    ewm_val = dow_lag_data.get("ewm", lag_7)

    # item_dow_avg feature: historical avg for this item on today's DOW
    item_dow_avg_val = _dow_avg(day_of_week_num)

    item_enc = encoders["item"].transform([item])[0] if item in encoders["item"].classes_ else 0

    features = {
        "item_encoded":          item_enc,
        "is_weekend":            is_weekend,
        "month":                 month,
        "day_of_month":          day_of_month,
        "week_of_year":          week_of_year,
        "is_month_start":        is_month_start,
        "is_month_end":          is_month_end,
        "quarter":               quarter,
        "day_of_week_num":       day_of_week_num,
        "sold_qty_lag_1":        lag_1,
        "sold_qty_lag_2":        lag_2,
        "sold_qty_lag_3":        lag_3,
        "sold_qty_lag_7":        lag_7,
        "sold_qty_lag_14":       lag_14,
        "sold_qty_lag_21":       lag_21,
        "sold_qty_lag_28":       lag_28,
        "sold_qty_roll_3":       roll_3,
        "sold_qty_roll_7":       roll_7,
        "sold_qty_roll_14":      roll_14,
        "sold_qty_roll_30":      roll_30,
        "sold_qty_roll_std_7":   dow_std,
        "sold_qty_roll_std_14":  item_std,
        "sold_qty_roll_max_7":   dow_max,
        "sold_qty_roll_min_7":   dow_min,
        "sold_qty_ewm_3":        ewm_val,
        "sold_qty_ewm_7":        ewm_val,
        "sold_qty_ewm_14":       ewm_val,
        "sold_qty_expanding_mean": item_avg,
        "sold_qty_trend_7v30":   roll_7 - roll_30,
        "days_since_last_sale":  1,
        "item_avg_sales":        item_avg,
        "item_std_sales":        item_std,
        "item_max_sales":        item_max,
        "item_median_sales":     item_median,
        "item_dow_avg":          item_dow_avg_val,
        "dow_sin":               dow_sin,
        "dow_cos":               dow_cos,
        "month_sin":             month_sin,
        "month_cos":             month_cos,
        "day_sin":               day_sin,
        "day_cos":               day_cos,
    }

    if "price" in meta["feature_columns"]:
        features["price"] = price

    return features


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/")
def home():
    return {
        "message": "Multi-Cafe AI Surplus Prediction API (XGBoost v4.4 - DOW-aware predictions)",
        "layers": [
            "Layer 1: Rule-based matching (fast, free)",
            "Layer 2: Ollama LLM semantic fallback (local, free)",
            "Layer 3: Human confirmation (safe + editable)"
        ],
        "model": "XGBoost with DOW-aware lag features (no weather/discount)",
        "ollama_available": OLLAMA_AVAILABLE,
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "features": [
            "AI Dataset Assessment",
            "Auto Column Mapping (via Local Ollama)",
            "User-Editable AI Mappings",
            "Per-Cafe Model Training",
            "DOW-aware lag feature engineering (fixes flat weekday predictions)",
            "Time-Series Feature Engineering",
            "Walk-Forward Validation",
            "SQLite persistence (survives restart)",
            "Daily manual sales entry",
            "Retrain from full sales history",
            "Food image classification — POST /classify-food"
        ]
    }


@app.route("/classify-food/status", methods=["GET"])
def classify_food_status():
    try:
        import food_classifier as fc
        return jsonify(fc.classifier_status())
    except ImportError as e:
        return jsonify({
            "error": "Food classifier dependencies not installed",
            "detail": str(e),
            "install": "pip install torch torchvision pillow",
        }), 503


def _read_classify_food_image_bytes():
    import base64
    if request.is_json:
        data = request.get_json(silent=True) or {}
        b64 = data.get("imageBase64") or data.get("image")
        if b64:
            if isinstance(b64, str) and "," in b64:
                b64 = b64.split(",", 1)[1]
            return base64.b64decode(b64)
    for field in ("file", "image", "photo"):
        if field in request.files:
            raw = request.files[field].read()
            if raw:
                return raw
    for key in request.files:
        raw = request.files[key].read()
        if raw:
            return raw
    return None


@app.route("/classify-food", methods=["POST"])
def classify_food():
    try:
        import time
        import food_classifier as fc
        t0 = time.perf_counter()
        image_bytes = _read_classify_food_image_bytes()
        if not image_bytes:
            return jsonify({"error": "No image uploaded", "hint": "multipart field file/image/photo, or JSON { imageBase64 }"}), 400
        max_bytes = int(os.getenv("FOOD_IMAGE_MAX_BYTES", 20 * 1024 * 1024))
        if len(image_bytes) > max_bytes:
            return jsonify({"error": f"Image too large ({len(image_bytes) // 1024} KB)."}), 400
        original_kb = len(image_bytes) // 1024
        if original_kb > 1500:
            image_bytes = fc.compress_image_bytes(image_bytes)
        if request.args.get("debug") in ("1", "true", "yes"):
            result = fc.classify_image_bytes_detailed(image_bytes)
        else:
            result = fc.classify_image_bytes(image_bytes)
        ms = (time.perf_counter() - t0) * 1000
        resp = jsonify(result)
        resp.headers["X-Processing-Ms"] = str(int(ms))
        return resp
    except ImportError:
        return jsonify({"error": "Food classifier not available", "install": "pip install torch torchvision pillow"}), 503
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/assess", methods=["POST"])
def assess():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        cafe_name = request.form.get("cafe_name", "Unknown Cafe")

        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            return jsonify({"error": "Uploaded CSV is empty — no columns or rows found."}), 400
        except pd.errors.ParserError as e:
            return jsonify({"error": "Could not parse CSV file.", "detail": str(e)}), 400
        if len(df) == 0:
            return jsonify({
                "error": "Uploaded CSV has headers but no data rows.",
                "columns": list(df.columns),
            }), 400
        columns = list(df.columns)
        print(f"\n📁 Assessing dataset: {cafe_name}")
        print(f"   Columns: {columns}")

        mapping, unmapped = layer1_rule_based(columns)

        llm_mapping = {}
        if unmapped and OLLAMA_AVAILABLE:
            llm_mapping = layer2_llm_fallback(unmapped, mapping, cafe_name)
            for col, standard in llm_mapping.items():
                if standard != "unknown" and standard not in mapping:
                    mapping[standard] = col
                    if col in unmapped:
                        unmapped.remove(col)

        needs_confirmation = layer3_human_confirmation(mapping, unmapped)

        total_cols = len(columns)
        mapped_count = len(mapping)
        llm_count = len(llm_mapping)

        if mapped_count == total_cols:
            confidence = "high"
        elif mapped_count + llm_count >= total_cols * 0.7:
            confidence = "medium"
        else:
            confidence = "low"

        missing_required = [req for req in REQUIRED_CORE if req not in mapping]
        missing_optional = [opt for opt in OPTIONAL_FEATURES if opt not in mapping]

        issues = []
        if "date" in mapping:
            try:
                pd.to_datetime(df[mapping["date"]])
            except:
                issues.append(f"Cannot parse dates in '{mapping['date']}'")
        if "sold_qty" in mapping:
            sold_col = mapping["sold_qty"]
            null_pct = (df[sold_col].isnull().sum() / len(df)) * 100
            if null_pct > 0:
                issues.append(f"{null_pct:.1f}% missing values in sales data")
            if (df[sold_col] < 0).any():
                issues.append("Negative sales values detected")

        suggestions = []
        if missing_optional:
            suggestions.append(f"Missing optional: {', '.join(missing_optional)}")
        if "produced_qty" not in mapping:
            suggestions.append("No production data - will estimate from sales")

        editable_mapping = []
        used_originals = set()

        for standard, original in mapping.items():
            source = "llm" if original in llm_mapping else "rule"
            editable_mapping.append({
                "original_column": original,
                "current_mapping": standard,
                "ai_suggested_mapping": standard,
                "source": source,
                "confidence": "high" if source == "rule" else "medium",
                "editable": True,
                "options": ALL_STANDARD_FIELDS + ["unknown"]
            })
            used_originals.add(original)

        for col in unmapped:
            suggestion = "unknown"
            col_lower = col.lower()
            if any(x in col_lower for x in ["date", "time", "day"]):
                suggestion = "date"
            elif any(x in col_lower for x in ["item", "product", "name", "menu"]):
                suggestion = "item"
            elif any(x in col_lower for x in ["sold", "sale", "qty", "quantity", "unit"]):
                suggestion = "sold_qty"
            elif any(x in col_lower for x in ["produce", "made", "bake", "stock", "inventory"]):
                suggestion = "produced_qty"
            elif any(x in col_lower for x in ["price", "cost", "amount", "revenue"]):
                suggestion = "price"

            editable_mapping.append({
                "original_column": col,
                "current_mapping": suggestion,
                "ai_suggested_mapping": suggestion,
                "source": "unmapped",
                "confidence": "low",
                "editable": True,
                "options": ALL_STANDARD_FIELDS + ["unknown"]
            })
            used_originals.add(col)

        assessment_id = str(uuid.uuid4())[:12]

        diagnostic = []
        if mapped_count - llm_count > 0:
            diagnostic.append(f"Layer 1 (Rule): {mapped_count - llm_count} columns")
        if llm_count > 0:
            diagnostic.append(f"Layer 2 (Ollama): {llm_count} columns")
        if len(unmapped) > 0:
            diagnostic.append(f"Layer 3 (Manual): {len(unmapped)} columns need review")
        diagnostic_msg = " → ".join(diagnostic) if diagnostic else "All columns need review"

        assessment_data = {
            "assessment_id": assessment_id,
            "cafe_name": cafe_name,
            "created_at": datetime.now().isoformat(),
            "total_rows": len(df),
            "total_columns": total_cols,
            "original_columns": columns,
            "editable_mapping": editable_mapping,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "data_quality_issues": issues,
            "suggestions": suggestions,
            "usable": len(missing_required) == 0,
            "confidence": confidence,
            "layer_breakdown": {
                "rule_based_mapped": mapped_count - llm_count,
                "llm_mapped": llm_count,
                "needs_confirmation": len(unmapped)
            },
            "diagnostic": diagnostic_msg,
            "ai_engine": "ollama" if OLLAMA_AVAILABLE else "rule-based-only"
        }

        pending_assessments[assessment_id] = assessment_data
        save_assessment_to_disk(assessment_id, assessment_data)

        return jsonify({
            **assessment_data,
            "message": "Review and edit mappings, then submit to /train with assessment_id"
        })

    except Exception as e:
        import traceback
        print(f"❌ Assessment error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/assessment/<assessment_id>", methods=["GET"])
def get_assessment(assessment_id):
    if assessment_id in pending_assessments:
        return jsonify(pending_assessments[assessment_id])
    loaded = load_assessment_from_disk(assessment_id)
    if loaded:
        pending_assessments[assessment_id] = loaded
        return jsonify(loaded)
    return jsonify({"error": "Assessment not found"}), 404


@app.route("/assessment/<assessment_id>/update", methods=["POST"])
def update_assessment(assessment_id):
    try:
        if assessment_id not in pending_assessments:
            loaded = load_assessment_from_disk(assessment_id)
            if loaded:
                pending_assessments[assessment_id] = loaded
            else:
                return jsonify({"error": "Assessment not found"}), 404

        data = request.json
        mapping_changes = data.get("mapping_changes", {})
        assessment = pending_assessments[assessment_id]
        editable_mapping = assessment["editable_mapping"]

        applied_changes = []
        rejected_changes = []

        for original_col, new_standard in mapping_changes.items():
            if new_standard not in ALL_STANDARD_FIELDS + ["unknown"]:
                rejected_changes.append({"column": original_col, "reason": f"Invalid standard field: {new_standard}"})
                continue
            found = False
            for entry in editable_mapping:
                if entry["original_column"] == original_col:
                    old_mapping = entry["current_mapping"]
                    entry["current_mapping"] = new_standard
                    entry["user_modified"] = True
                    applied_changes.append({"column": original_col, "old_mapping": old_mapping, "new_mapping": new_standard})
                    found = True
                    break
            if not found:
                rejected_changes.append({"column": original_col, "reason": "Column not found in assessment"})

        current_mappings = {entry["current_mapping"]: entry["original_column"]
                           for entry in editable_mapping if entry["current_mapping"] != "unknown"}

        missing_required = [req for req in REQUIRED_CORE if req not in current_mappings]
        missing_optional = [opt for opt in OPTIONAL_FEATURES if opt not in current_mappings]

        assessment["missing_required"] = missing_required
        assessment["missing_optional"] = missing_optional
        assessment["usable"] = len(missing_required) == 0
        assessment["last_modified"] = datetime.now().isoformat()
        assessment["user_changes_history"] = assessment.get("user_changes_history", []) + [{
            "timestamp": datetime.now().isoformat(),
            "changes": applied_changes
        }]
        save_assessment_to_disk(assessment_id, assessment)

        return jsonify({
            "assessment_id": assessment_id,
            "applied_changes": applied_changes,
            "rejected_changes": rejected_changes,
            "current_mapping": current_mappings,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "usable": len(missing_required) == 0,
            "editable_mapping": editable_mapping,
            "message": f"Updated {len(applied_changes)} mappings. {len(rejected_changes)} rejected."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train():
    try:
        assessment_id = request.form.get("assessment_id")
        cafe_name = request.form.get("cafe_name", "Unknown Cafe")
        cafe_id = request.form.get("cafe_id", str(uuid.uuid4())[:8])

        if assessment_id:
            if assessment_id not in pending_assessments:
                loaded = load_assessment_from_disk(assessment_id)
                if loaded:
                    pending_assessments[assessment_id] = loaded
                else:
                    return jsonify({"error": "Assessment not found"}), 404
            assessment = pending_assessments[assessment_id]
            cafe_name = assessment.get("cafe_name", cafe_name)
            user_corrections = {}
            for entry in assessment["editable_mapping"]:
                original = entry["original_column"]
                standard = entry["current_mapping"]
                if standard != "unknown":
                    user_corrections[original] = standard
        else:
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded and no assessment_id provided"}), 400
            user_corrections = {}
            if request.form.get("user_corrections"):
                user_corrections = json.loads(request.form.get("user_corrections"))

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400
        file = request.files["file"]
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            return jsonify({"error": "Uploaded CSV is empty — no columns or rows found."}), 400
        except pd.errors.ParserError as e:
            return jsonify({"error": "Could not parse CSV file.", "detail": str(e)}), 400
        if len(df) == 0:
            return jsonify({
                "error": "Uploaded CSV has headers but no data rows.",
                "columns": list(df.columns),
            }), 400

        mapping = {}
        for original, standard in user_corrections.items():
            if standard != "unknown":
                mapping[standard] = original

        missing_required = [req for req in REQUIRED_CORE if req not in mapping]
        if missing_required:
            return jsonify({
                "error": "Dataset unusable",
                "missing_required": missing_required,
                "message": f"Required columns missing: {', '.join(missing_required)}."
            }), 400

        try:
            df_std = standardize_dataset(df, mapping)
        except ValueError as e:
            return jsonify({"error": "Dataset not suitable for training.", "detail": str(e)}), 400

        db.upsert_cafe(cafe_id, cafe_name, column_mapping=mapping, assessment_id=assessment_id)

        try:
            metrics, importance_sorted, items = train_model_from_standardized_df(cafe_id, cafe_name, df_std, mapping)
        except ValueError as e:
            return jsonify({"error": "Dataset not suitable for training.", "detail": str(e)}), 400
        mae, r2, cv_mae, cv_r2 = metrics["mae"], metrics["r2"], metrics["cv_mae"], metrics["cv_r2"]
        summary = db.sales_summary(cafe_id)

        return jsonify({
            "cafe_id": cafe_id,
            "cafe_name": cafe_name,
            "status": "trained",
            "model": "XGBoost (log-target, DOW-aware lags)",
            "rows_used": metrics["training_rows"],
            "items": items,
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "cv_mae": round(cv_mae, 2),
            "cv_r2": round(cv_r2, 4),
            "accuracy_pct": round(max(0, r2) * 100, 1),
            "confidence": "high" if r2 >= 0.7 else ("medium" if r2 >= 0.4 else "low"),
            "detected_mapping": mapping,
            "top_features": importance_sorted,
            "persisted": True,
            "dataset_summary": summary,
            "message": f"Model trained for {cafe_name}. Data persists across restarts.",
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        cafe_id = data.get("cafe_id")
        if not cafe_id:
            return jsonify({"error": "cafe_id is required.", "hint": "GET /cafes lists all saved cafes."}), 400
        cafe_data = get_cafe_model(cafe_id)
        if not cafe_data:
            return jsonify({"error": "Cafe not found. Train model first.", "hint": "GET /cafes lists all saved cafes."}), 404

        model = cafe_data["model"]
        encoders = cafe_data["encoders"]
        meta = cafe_data["metadata"]

        items = meta.get("items") or []
        if not items:
            return jsonify({
                "error": "Model has no trained items — training dataset was empty or unsuitable.",
                "hint": f"Retrain with valid data: POST /cafe/{cafe_id}/retrain",
            }), 409

        item = data.get("item")
        if not item:
            return jsonify({"error": "item is required.", "available_items": items}), 400
        if item not in items:
            return jsonify({
                "error": f"Unknown item '{item}' — not present in this cafe's training data.",
                "available_items": items,
            }), 404

        day = data.get("day_of_week", "Saturday")
        price = data.get("price", 0)
        produced = data.get("produced_qty")

        try:
            date_obj = datetime.strptime(data.get("date", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d")
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid date '{data.get('date')}'. Expected format: YYYY-MM-DD."}), 400
        month = date_obj.month
        day_of_month = date_obj.day
        week_of_year = date_obj.isocalendar()[1]
        is_month_start = 1 if day_of_month <= 3 else 0
        is_month_end = 1 if day_of_month >= 28 else 0
        quarter = (month - 1) // 3 + 1
        day_of_week_num = date_obj.weekday()
        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0

        dow_sin = np.sin(2 * np.pi * day_of_week_num / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week_num / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day_of_month / 31)
        day_cos = np.cos(2 * np.pi * day_of_month / 31)

        # ── FIX: Use the shared helper so /predict also benefits from DOW-aware lags ──
        # Allow caller to override individual lags if they have real recent data
        features = _build_item_features(
            item, day_of_week_num, is_weekend, month, day_of_month,
            week_of_year, is_month_start, is_month_end, quarter,
            dow_sin, dow_cos, month_sin, month_cos, day_sin, day_cos,
            meta, encoders, price=price
        )

        # If caller explicitly supplied lag values, honour them (real data beats averages)
        lag_overrides = [
            "sold_qty_lag_1", "sold_qty_lag_2", "sold_qty_lag_3",
            "sold_qty_lag_7", "sold_qty_lag_14", "sold_qty_lag_21", "sold_qty_lag_28",
            "sold_qty_roll_3", "sold_qty_roll_7", "sold_qty_roll_14", "sold_qty_roll_30",
            "sold_qty_roll_std_7", "sold_qty_roll_std_14",
            "sold_qty_roll_max_7", "sold_qty_roll_min_7",
            "sold_qty_ewm_3", "sold_qty_ewm_7", "sold_qty_ewm_14",
            "sold_qty_expanding_mean", "item_avg_sales", "item_std_sales",
            "item_max_sales", "item_median_sales", "item_dow_avg",
        ]
        for key in lag_overrides:
            if key in data:
                features[key] = data[key]

        X = pd.DataFrame([{col: features.get(col, 0) for col in meta["feature_columns"]}])
        raw_pred = model.predict(X)[0]
        if meta.get("log_transformed", False):
            raw_pred = np.expm1(raw_pred)
        predicted_sales = int(max(0, round(raw_pred)))

        buffer = 3 if is_weekend else 2
        recommended = predicted_sales + buffer
        actual_produced = produced if produced else recommended
        surplus = max(0, actual_produced - predicted_sales)
        base_revenue = predicted_sales * price

        response = {
            "cafe_id": cafe_id,
            "cafe_name": meta["cafe_name"],
            "item": item,
            "day_of_week": day,
            "predicted_sales": predicted_sales,
            "recommended_production": recommended,
            "produced_qty": actual_produced,
            "expected_surplus": surplus,
            "surplus_rate": round(surplus / actual_produced * 100, 1) if actual_produced > 0 else 0,
            "price_rm": price,
            "revenue_rm": round(base_revenue, 2),
            "is_weekend": bool(is_weekend),
            "model_accuracy": meta["r2"],
            "cv_mae": meta.get("cv_mae", meta["mae"]),
            "training_size": meta["training_rows"]
        }
        if not meta.get("item_dow_means"):
            response["warning"] = (
                "Model was trained with an older version and lacks per-weekday averages — "
                f"predictions may be flat across weekdays. Retrain: POST /cafe/{cafe_id}/retrain"
            )
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    try:
        import sys
        print(f"\n🔴 BATCH-PREDICT CALLED", file=sys.stderr, flush=True)
        data = request.get_json(silent=True) or {}
        cafe_id = data.get("cafe_id")
        if not cafe_id:
            return jsonify({"error": "cafe_id is required.", "hint": "GET /cafes lists all saved cafes."}), 400
        cafe_data = get_cafe_model(cafe_id)
        if not cafe_data:
            return jsonify({"error": "Cafe not found. Train model first."}), 404

        model = cafe_data["model"]
        encoders = cafe_data["encoders"]
        meta = cafe_data["metadata"]

        items = meta.get("items") or []
        if not items:
            return jsonify({
                "error": "Model has no trained items — training dataset was empty or unsuitable.",
                "hint": f"Retrain with valid data: POST /cafe/{cafe_id}/retrain",
            }), 409

        day = data.get("day_of_week", "Saturday")
        date_str = data.get("date", datetime.now().strftime("%Y-%m-%d"))
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD."}), 400

        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
        month = date_obj.month
        day_of_month = date_obj.day
        week_of_year = date_obj.isocalendar()[1]
        is_month_start = 1 if day_of_month <= 3 else 0
        is_month_end = 1 if day_of_month >= 28 else 0
        quarter = (month - 1) // 3 + 1
        day_of_week_num = date_obj.weekday()

        dow_sin = np.sin(2 * np.pi * day_of_week_num / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week_num / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day_of_month / 31)
        day_cos = np.cos(2 * np.pi * day_of_month / 31)

        predictions = []
        for item in items:
            # ── FIX: use DOW-aware helper instead of flat item_avg for all lags ──
            features = _build_item_features(
                item, day_of_week_num, is_weekend, month, day_of_month,
                week_of_year, is_month_start, is_month_end, quarter,
                dow_sin, dow_cos, month_sin, month_cos, day_sin, day_cos,
                meta, encoders, price=0
            )

            X = pd.DataFrame([{col: features.get(col, 0) for col in meta["feature_columns"]}])
            raw_pred = model.predict(X)[0]
            if meta.get("log_transformed", False):
                raw_pred = np.expm1(raw_pred)
            pred = int(max(0, round(raw_pred)))
            buffer = 3 if is_weekend else 2
            rec = pred + buffer

            predictions.append({
                "item": item,
                "predicted_sales": pred,
                "recommended_production": rec,
                "expected_surplus": rec - pred
            })

        response = {
            "cafe_id": cafe_id,
            "cafe_name": meta["cafe_name"],
            "day": day,
            "date": date_str,
            "predictions": predictions,
            "total_predicted_sales": sum(p["predicted_sales"] for p in predictions),
            "total_recommended_production": sum(p["recommended_production"] for p in predictions),
            "total_expected_surplus": sum(p["expected_surplus"] for p in predictions)
        }
        if not meta.get("item_dow_means"):
            response["warning"] = (
                "Model was trained with an older version and lacks per-weekday averages — "
                f"predictions may be flat across weekdays. Retrain: POST /cafe/{cafe_id}/retrain"
            )
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        with open("error_log.txt", "a") as f:
            f.write(f"\n❌ {datetime.now()}\n{traceback.format_exc()}\n")
        return jsonify({"error": str(e)}), 500


@app.route("/cafes")
def list_cafes():
    cafes_out = []
    seen = set()
    for row in db.list_cafes():
        cid = row["cafe_id"]
        seen.add(cid)
        metrics = row.get("metrics") or {}
        model_loaded = get_cafe_model(cid) is not None
        summary = db.sales_summary(cid)
        cafes_out.append({
            "cafe_id": cid,
            "cafe_name": row["cafe_name"],
            "items": row.get("items") or metrics.get("items", []),
            "training_rows": summary.get("total_rows") or metrics.get("training_rows", 0),
            "r2": metrics.get("r2"),
            "cv_mae": metrics.get("cv_mae"),
            "model_loaded": model_loaded,
            "last_trained_at": row.get("last_trained_at"),
            "dataset_days": summary.get("total_days", 0),
            "manual_entries": summary.get("manual_rows", 0),
            "model": "XGBoost",
        })
    for cid in cafe_models:
        if cid not in seen:
            meta = cafe_models[cid]["metadata"]
            cafes_out.append({
                "cafe_id": cid,
                "cafe_name": meta["cafe_name"],
                "items": meta["items"],
                "training_rows": meta["training_rows"],
                "r2": meta["r2"],
                "cv_mae": meta.get("cv_mae", meta["mae"]),
                "model_loaded": True,
                "model": "XGBoost",
            })
    return jsonify({"cafes": cafes_out, "persisted": True})


@app.route("/cafe/<cafe_id>")
def get_cafe_info(cafe_id):
    cafe_row = db.get_cafe(cafe_id)
    cafe_data = get_cafe_model(cafe_id)
    if not cafe_row and not cafe_data:
        return jsonify({"error": "Cafe not found"}), 404
    summary = db.sales_summary(cafe_id)
    if cafe_data:
        meta = cafe_data["metadata"]
        return jsonify({
            "cafe_id": cafe_id,
            "cafe_name": meta["cafe_name"],
            "items": meta["items"],
            "training_rows": meta["training_rows"],
            "mae": meta["mae"],
            "r2": meta["r2"],
            "cv_mae": meta.get("cv_mae", meta["mae"]),
            "accuracy_pct": round(max(0, meta["r2"]) * 100, 1),
            "trained_at": meta["trained_at"],
            "detected_mapping": meta["mapping"],
            "top_features": meta.get("feature_importance", {}),
            "model": "XGBoost",
            "model_loaded": True,
            "persisted": True,
            "dataset_summary": summary,
        })
    metrics = cafe_row.get("metrics") or {}
    return jsonify({
        "cafe_id": cafe_id,
        "cafe_name": cafe_row["cafe_name"],
        "items": cafe_row.get("items", []),
        "training_rows": summary.get("total_rows", 0),
        "r2": metrics.get("r2"),
        "cv_mae": metrics.get("cv_mae"),
        "trained_at": cafe_row.get("last_trained_at"),
        "detected_mapping": cafe_row.get("column_mapping"),
        "model_loaded": False,
        "persisted": True,
        "dataset_summary": summary,
        "message": "Sales data saved. POST /cafe/<id>/retrain to load the model.",
    })


@app.route("/cafe/<cafe_id>/dataset", methods=["GET"])
def get_cafe_dataset(cafe_id):
    if not db.get_cafe(cafe_id) and not get_cafe_model(cafe_id):
        return jsonify({"error": "Cafe not found"}), 404
    summary = db.sales_summary(cafe_id)
    recent = db.get_recent_sales(cafe_id, limit=30)
    cafe_row = db.get_cafe(cafe_id) or {}
    return jsonify({
        "cafe_id": cafe_id,
        "cafe_name": cafe_row.get("cafe_name", "Unknown"),
        "summary": summary,
        "recent_sales": recent,
        "items": cafe_row.get("items", []),
    })


@app.route("/cafe/<cafe_id>/daily-sales", methods=["POST"])
def record_daily_sales(cafe_id):
    try:
        if not db.get_cafe(cafe_id) and not get_cafe_model(cafe_id):
            return jsonify({"error": "Cafe not found. Train with /train first."}), 404
        data = request.json or {}
        sale_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
        entries = data.get("entries", [])
        if not entries:
            return jsonify({"error": "entries required — list of {item, sold_qty}"}), 400
        result = db.add_daily_sales(
            cafe_id, sale_date, entries,
            day_of_week=data.get("day_of_week"),
            weather=data.get("weather", "Unknown"),
            default_discount=float(data.get("discount_pct", 0)),
        )
        response = {
            "cafe_id": cafe_id,
            "date": sale_date,
            "saved_count": result["saved"],
            "entries": result["entries"],
            "dataset_summary": db.sales_summary(cafe_id),
            "message": "Daily sales saved to database.",
        }
        if data.get("retrain", False):
            response["retrain"] = _retrain_cafe_from_db(cafe_id)
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _retrain_cafe_from_db(cafe_id: str) -> dict:
    cafe_row = db.get_cafe(cafe_id)
    if not cafe_row:
        raise ValueError("Cafe not found in database")
    df = db.load_sales_dataframe(cafe_id)
    if len(df) < 20:
        raise ValueError(f"Need at least 20 sales rows to retrain (have {len(df)}).")
    if "surplus_qty" not in df.columns:
        df["surplus_qty"] = (df["produced_qty"] - df["sold_qty"]).clip(lower=0)
    mapping = cafe_row.get("column_mapping") or {}
    metrics, importance_sorted, items = train_model_from_standardized_df(
        cafe_id, cafe_row["cafe_name"], df, mapping, persist_sales=False
    )
    return {
        "status": "retrained",
        "rows_used": metrics["training_rows"],
        "r2": round(metrics["r2"], 4),
        "accuracy_pct": round(max(0, metrics["r2"]) * 100, 1),
        "mae": round(metrics["mae"], 2),
        "items": items,
        "top_features": importance_sorted,
    }


@app.route("/cafe/<cafe_id>/retrain", methods=["POST"])
def retrain_cafe(cafe_id):
    try:
        result = _retrain_cafe_from_db(cafe_id)
        return jsonify({
            **result,
            "cafe_id": cafe_id,
            "dataset_summary": db.sales_summary(cafe_id),
            "message": "Model retrained from full database history.",
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)