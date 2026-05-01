from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib
import os
import uuid
import json
from datetime import datetime

# Try to import Gemini, fallback to rule-only if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini not installed. Install with: pip install google-generativeai")

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Gemini (free tier - 1000 requests/day)
if GEMINI_AVAILABLE:
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    else:
        GEMINI_AVAILABLE = False
        print("⚠️ No GEMINI_API_KEY found. Set it as environment variable.")

# In-memory storage
cafe_models = {}

# ============================================================
# LAYER 1: RULE-BASED COLUMN MAPPING (Fast, Free, Deterministic)
# ============================================================

COLUMN_ALIASES = {
    "date": [
        "date", "order_date", "transaction_date", "sale_date", "created_at", 
        "datetime", "timestamp", "day", "date_time", "orderdate", "saledate",
        "txn_date", "orderdate", "saledate", "dt", "record_date", "entry_date"
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
    "discount_pct": [
        "discount", "discount_pct", "discount_percent", "disc", 
        "discount_rate", "sale_discount", "promo", "promotion", "markdown",
        "rebate", "concession", "price_reduction", "special_offer"
    ],
    "weather": [
        "weather", "condition", "climate", "forecast", "temp", 
        "temperature", "rain", "sunny", "precipitation", "humidity",
        "wthr", "meteo", "outlook", "skies"
    ],
    "day_of_week": [
        "day_of_week", "weekday", "day", "week_day", "dow",
        "weekday_name", "day_name", "calendar_day"
    ]
}

REQUIRED_CORE = ["date", "item", "sold_qty"]
OPTIONAL_FEATURES = ["produced_qty", "price", "discount_pct", "weather", "day_of_week"]

def layer1_rule_based(columns):
    """
    Layer 1: Fast rule-based matching
    Returns: (mapping_dict, unmapped_columns)
    """
    mapping = {}
    unmapped = []
    used_standards = set()

    for col in columns:
        col_clean = col.lower().strip().replace(" ", "_").replace("-", "_")
        matched = False

        for standard_name, aliases in COLUMN_ALIASES.items():
            if standard_name in used_standards:
                continue

            # Exact match
            if col_clean in aliases:
                mapping[standard_name] = col
                used_standards.add(standard_name)
                matched = True
                break

            # Partial match (substring)
            for alias in aliases:
                if alias in col_clean or col_clean in alias:
                    if len(alias) >= 3:  # Avoid short false matches
                        mapping[standard_name] = col
                        used_standards.add(standard_name)
                        matched = True
                        break

            if matched:
                break

        if not matched:
            unmapped.append(col)

    return mapping, unmapped

# ============================================================
# LAYER 2: LLM SEMANTIC MATCHING (Smart Fallback)
# ============================================================

def layer2_llm_fallback(unmapped_columns, existing_mapping, cafe_name=""):
    """
    Layer 2: Gemini API for semantic understanding
    Returns: {original_column: standard_column}
    """
    if not GEMINI_AVAILABLE or not unmapped_columns:
        return {}

    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️ No Gemini API key, skipping LLM fallback")
        return {}

    try:
        standard_options = ["date", "item", "sold_qty", "produced_qty", "price", "discount_pct", "weather", "day_of_week", "unknown"]

        prompt = f"""You are a data preprocessing assistant for a cafe/bakery sales system.

Cafe name: {cafe_name}

Already mapped columns: {json.dumps(existing_mapping, indent=2)}
Unmapped columns that need interpretation: {json.dumps(unmapped_columns)}

Standard schema fields:
- date: When the sale happened
- item: Name of the product/food item
- sold_qty: How many units were sold to customers
- produced_qty: How many units were baked/made
- price: Price per unit in local currency
- discount_pct: Percentage discount applied (0-100)
- weather: Weather condition that day (Sunny/Cloudy/Rainy)
- day_of_week: Day name (Monday-Sunday)
- unknown: Column doesn't match any standard field

For each unmapped column, determine the MOST LIKELY standard field based on:
1. Column name semantics
2. Cafe context (bakery/cafe)
3. Common data patterns

Return ONLY a JSON object. No explanations.
Format: {{"original_column_name": "standard_field_name"}}

If truly unclear, map to "unknown".
"""

        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text

        llm_mapping = json.loads(json_str)

        # Validate mappings
        validated = {}
        for col, standard in llm_mapping.items():
            if standard in standard_options and col in unmapped_columns:
                validated[col] = standard

        print(f"🤖 LLM mapped {len(validated)} columns: {validated}")
        return validated

    except Exception as e:
        print(f"⚠️ LLM fallback failed: {e}")
        return {}

# ============================================================
# LAYER 3: HUMAN CONFIRMATION UI (Safety Net)
# ============================================================

def layer3_human_confirmation(mapping, unmapped_after_llm):
    """
    Layer 3: Flag columns that need user review
    Returns: list of columns with suggestions
    """
    needs_confirmation = []

    for col in unmapped_after_llm:
        # Try to suggest best guess based on name patterns
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
        elif any(x in col_lower for x in ["discount", "promo", "markdown", "off"]):
            suggestion = "discount_pct"
        elif any(x in col_lower for x in ["weather", "condition", "temp", "rain"]):
            suggestion = "weather"

        needs_confirmation.append({
            "column": col,
            "suggested_mapping": suggestion,
            "confidence": "low",
            "options": ["date", "item", "sold_qty", "produced_qty", "price", "discount_pct", "weather", "day_of_week", "unknown"]
        })

    return needs_confirmation

# ============================================================
# DATA STANDARDIZATION
# ============================================================

def standardize_dataset(df, mapping, user_corrections=None):
    """
    Transform any cafe's data into standard format
    user_corrections: {original_col: standard_col} from UI
    """
    standardized = pd.DataFrame()

    # Apply base mapping
    for standard, original in mapping.items():
        if original in df.columns:
            standardized[standard] = df[original].copy()

    # Apply user corrections if provided
    if user_corrections:
        for original, standard in user_corrections.items():
            if original in df.columns and standard != "unknown":
                standardized[standard] = df[original].copy()

    # Smart defaults for missing optional columns
    if "produced_qty" not in standardized.columns:
        standardized["produced_qty"] = (standardized["sold_qty"] * 1.12).astype(int)

    if "price" not in standardized.columns:
        standardized["price"] = 0.0

    if "discount_pct" not in standardized.columns:
        standardized["discount_pct"] = 0

    if "weather" not in standardized.columns:
        standardized["weather"] = "Unknown"

    if "day_of_week" not in standardized.columns:
        if "date" in standardized.columns:
            standardized["day_of_week"] = pd.to_datetime(standardized["date"]).dt.day_name()
        else:
            standardized["day_of_week"] = "Unknown"

    # Calculate surplus
    standardized["surplus_qty"] = (standardized["produced_qty"] - standardized["sold_qty"]).clip(lower=0)

    return standardized

def engineer_features(df):
    """Add ML features"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    df["month"] = df["date"].dt.month
    df["price"] = df["price"].fillna(df["price"].median() if df["price"].sum() > 0 else 0)
    return df

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/")
def home():
    return {
        "message": "Multi-Cafe AI Surplus Prediction API (Hybrid v3.0)",
        "layers": [
            "Layer 1: Rule-based matching (fast, free)",
            "Layer 2: LLM semantic fallback (smart)",
            "Layer 3: Human confirmation (safe)"
        ],
        "gemini_available": GEMINI_AVAILABLE,
        "features": ["AI Dataset Assessment", "Auto Column Mapping", "Per-Cafe Model Training", "Discount-aware Predictions"]
    }

@app.route("/assess", methods=["POST"])
def assess():
    """
    Full 3-layer assessment endpoint
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        cafe_name = request.form.get("cafe_name", "Unknown Cafe")

        df = pd.read_csv(file)
        columns = list(df.columns)

        # Layer 1: Rule-based
        mapping, unmapped = layer1_rule_based(columns)

        # Layer 2: LLM fallback
        llm_mapping = {}
        if unmapped and GEMINI_AVAILABLE:
            llm_mapping = layer2_llm_fallback(unmapped, mapping, cafe_name)
            # Add LLM results to mapping
            for col, standard in llm_mapping.items():
                if standard != "unknown" and standard not in mapping:
                    mapping[standard] = col
                    if col in unmapped:
                        unmapped.remove(col)

        # Layer 3: Human confirmation needed?
        needs_confirmation = layer3_human_confirmation(mapping, unmapped)

        # Calculate confidence
        total_cols = len(columns)
        mapped_count = len(mapping)
        llm_count = len(llm_mapping)

        if mapped_count == total_cols:
            confidence = "high"
        elif mapped_count + llm_count >= total_cols * 0.7:
            confidence = "medium"
        else:
            confidence = "low"

        # Check required columns
        missing_required = [req for req in REQUIRED_CORE if req not in mapping]
        missing_optional = [opt for opt in OPTIONAL_FEATURES if opt not in mapping]

        # Data quality checks
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
        if "weather" not in mapping:
            suggestions.append("Weather missing - consider weather API integration")

        return jsonify({
            "cafe_name": cafe_name,
            "total_rows": len(df),
            "total_columns": total_cols,
            "original_columns": columns,
            "detected_mapping": mapping,
            "llm_assisted_mapping": llm_mapping,
            "needs_confirmation": needs_confirmation,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "data_quality_issues": issues,
            "suggestions": suggestions,
            "usable": len(missing_required) == 0,
            "confidence": confidence,
            "layer_breakdown": {
                "rule_based_mapped": mapped_count - llm_count,
                "llm_mapped": llm_count,
                "needs_confirmation": len(needs_confirmation)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    """
    Train per-cafe model with optional user corrections
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        cafe_name = request.form.get("cafe_name", "Unknown Cafe")
        cafe_id = request.form.get("cafe_id", str(uuid.uuid4())[:8])

        # Optional: user corrections from UI
        user_corrections = {}
        if request.form.get("user_corrections"):
            user_corrections = json.loads(request.form.get("user_corrections"))

        df = pd.read_csv(file)

        # Re-run assessment to get mapping
        columns = list(df.columns)
        mapping, unmapped = layer1_rule_based(columns)

        if unmapped and GEMINI_AVAILABLE:
            llm_mapping = layer2_llm_fallback(unmapped, mapping, cafe_name)
            for col, standard in llm_mapping.items():
                if standard != "unknown" and standard not in mapping:
                    mapping[standard] = col

        # Check required
        missing_required = [req for req in REQUIRED_CORE if req not in mapping]
        if missing_required:
            return jsonify({
                "error": "Dataset unusable",
                "missing_required": missing_required,
                "message": f"Required columns missing: {', '.join(missing_required)}"
            }), 400

        # Standardize
        df_std = standardize_dataset(df, mapping, user_corrections)
        df_std = engineer_features(df_std)

        # Encode
        item_encoder = LabelEncoder()
        weather_encoder = LabelEncoder()

        df_std["item_encoded"] = item_encoder.fit_transform(df_std["item"])

        if df_std["weather"].nunique() > 1:
            df_std["weather_encoded"] = weather_encoder.fit_transform(df_std["weather"])
        else:
            df_std["weather_encoded"] = 0

        # Features (NO DISCOUNT in training)
        feature_cols = ["item_encoded", "is_weekend", "weather_encoded"]
        if "price" in df_std.columns and df_std["price"].sum() > 0:
            feature_cols.append("price")

        X = df_std[feature_cols]
        y = df_std["sold_qty"]

        # Train
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Evaluate
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Store
        cafe_models[cafe_id] = {
            "model": model,
            "encoders": {
                "item": item_encoder,
                "weather": weather_encoder,
                "features": feature_cols
            },
            "metadata": {
                "cafe_name": cafe_name,
                "cafe_id": cafe_id,
                "training_rows": len(df_std),
                "items": list(df_std["item"].unique()),
                "mae": mae,
                "r2": r2,
                "feature_columns": feature_cols,
                "mapping": mapping,
                "trained_at": datetime.now().isoformat()
            }
        }

        return jsonify({
            "cafe_id": cafe_id,
            "cafe_name": cafe_name,
            "status": "trained",
            "rows_used": len(df_std),
            "items": list(df_std["item"].unique()),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "confidence": "high",
            "detected_mapping": mapping,
            "message": f"Model trained successfully for {cafe_name}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Predict with optional discount (app-only)"""
    try:
        data = request.json
        cafe_id = data.get("cafe_id")

        if not cafe_id or cafe_id not in cafe_models:
            return jsonify({"error": "Cafe not found. Train model first."}), 404

        cafe_data = cafe_models[cafe_id]
        model = cafe_data["model"]
        encoders = cafe_data["encoders"]
        meta = cafe_data["metadata"]

        item = data.get("item")
        day = data.get("day_of_week", "Saturday")
        weather = data.get("weather", "Sunny")
        price = data.get("price", 0)
        produced = data.get("produced_qty")
        discount_pct = data.get("discount_pct", 0)

        # Encode
        item_enc = encoders["item"].transform([item])[0] if item in encoders["item"].classes_ else 0
        weather_enc = encoders["weather"].transform([weather])[0] if weather in encoders["weather"].classes_ else 0

        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0

        features = {
            "item_encoded": item_enc,
            "is_weekend": is_weekend,
            "weather_encoded": weather_enc
        }

        if "price" in meta["feature_columns"]:
            features["price"] = price

        X = pd.DataFrame([features])
        X = X[meta["feature_columns"]]

        # Base prediction (natural demand)
        base_predicted_sales = int(model.predict(X)[0])

        # Apply discount for "what if" scenario
        discount_boost = 1 + (discount_pct / 100) * 1.5
        predicted_sales = int(base_predicted_sales * discount_boost)

        buffer = 3 if is_weekend else 2
        recommended = predicted_sales + buffer

        actual_produced = produced if produced else recommended
        surplus = max(0, actual_produced - predicted_sales)

        base_revenue = base_predicted_sales * price
        discounted_revenue = predicted_sales * price * (1 - discount_pct / 100)

        return jsonify({
            "cafe_id": cafe_id,
            "cafe_name": meta["cafe_name"],
            "item": item,
            "day_of_week": day,
            "weather": weather,
            "discount_pct": discount_pct,
            "base_predicted_sales": base_predicted_sales,
            "predicted_sales": predicted_sales,
            "recommended_production": recommended,
            "produced_qty": actual_produced,
            "expected_surplus": surplus,
            "surplus_rate": round(surplus / actual_produced * 100, 1) if actual_produced > 0 else 0,
            "price_rm": price,
            "base_revenue_rm": round(base_revenue, 2),
            "discounted_revenue_rm": round(discounted_revenue, 2),
            "revenue_impact": round(discounted_revenue - base_revenue, 2),
            "is_weekend": bool(is_weekend),
            "model_accuracy": meta["r2"],
            "training_size": meta["training_rows"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Predict all items for a day"""
    try:
        data = request.json
        cafe_id = data.get("cafe_id")

        if not cafe_id or cafe_id not in cafe_models:
            return jsonify({"error": "Cafe not found"}), 404

        cafe_data = cafe_models[cafe_id]
        model = cafe_data["model"]
        encoders = cafe_data["encoders"]
        meta = cafe_data["metadata"]

        day = data.get("day_of_week", "Saturday")
        weather = data.get("weather", "Sunny")
        discount_pct = data.get("discount_pct", 0)

        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
        discount_boost = 1 + (discount_pct / 100) * 1.5

        predictions = []
        for item in meta["items"]:
            item_enc = encoders["item"].transform([item])[0] if item in encoders["item"].classes_ else 0
            weather_enc = encoders["weather"].transform([weather])[0] if weather in encoders["weather"].classes_ else 0

            features = {
                "item_encoded": item_enc,
                "is_weekend": is_weekend,
                "weather_encoded": weather_enc
            }

            if "price" in meta["feature_columns"]:
                features["price"] = 0

            X = pd.DataFrame([features])
            X = X[meta["feature_columns"]]

            base_pred = int(model.predict(X)[0])
            pred = int(base_pred * discount_boost)
            buffer = 3 if is_weekend else 2
            rec = pred + buffer

            predictions.append({
                "item": item,
                "base_predicted_sales": base_pred,
                "predicted_sales": pred,
                "recommended_production": rec,
                "expected_surplus": rec - pred
            })

        return jsonify({
            "cafe_id": cafe_id,
            "cafe_name": meta["cafe_name"],
            "day": day,
            "weather": weather,
            "discount_pct": discount_pct,
            "predictions": predictions,
            "total_base_sales": sum(p["base_predicted_sales"] for p in predictions),
            "total_predicted_sales": sum(p["predicted_sales"] for p in predictions),
            "total_recommended_production": sum(p["recommended_production"] for p in predictions),
            "total_expected_surplus": sum(p["expected_surplus"] for p in predictions)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/cafes")
def list_cafes():
    return jsonify({
        "cafes": [
            {
                "cafe_id": cid,
                "cafe_name": data["metadata"]["cafe_name"],
                "items": data["metadata"]["items"],
                "training_rows": data["metadata"]["training_rows"],
                "r2": data["metadata"]["r2"]
            }
            for cid, data in cafe_models.items()
        ]
    })

@app.route("/cafe/<cafe_id>")
def get_cafe_info(cafe_id):
    if cafe_id not in cafe_models:
        return jsonify({"error": "Cafe not found"}), 404

    meta = cafe_models[cafe_id]["metadata"]
    return jsonify({
        "cafe_id": cafe_id,
        "cafe_name": meta["cafe_name"],
        "items": meta["items"],
        "training_rows": meta["training_rows"],
        "mae": meta["mae"],
        "r2": meta["r2"],
        "trained_at": meta["trained_at"],
        "detected_mapping": meta["mapping"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)