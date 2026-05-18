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

# Ollama configuration (local LLM - no API key needed)
import requests

OLLAMA_AVAILABLE = False
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Check if Ollama is available
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

# In-memory storage
cafe_models = {}
# Store pending assessments for user modification
pending_assessments = {}

# Models persistence
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Assessments persistence (for user modification across sessions)
ASSESSMENTS_DIR = os.path.join(os.path.dirname(__file__), "assessments")
os.makedirs(ASSESSMENTS_DIR, exist_ok=True)

def save_assessment_to_disk(assessment_id, assessment_data):
    """Save assessment to disk for persistence"""
    try:
        path = os.path.join(ASSESSMENTS_DIR, f"{assessment_id}.json")
        with open(path, "w") as f:
            json.dump(assessment_data, f, indent=2, default=str)
        print(f"✅ Assessment saved: {path}")
    except Exception as e:
        print(f"❌ Error saving assessment: {e}")

def load_assessment_from_disk(assessment_id):
    """Load assessment from disk"""
    try:
        path = os.path.join(ASSESSMENTS_DIR, f"{assessment_id}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"❌ Error loading assessment: {e}")
    return None

def save_model_to_disk(cafe_id, cafe_data):
    """Save trained model to disk"""
    try:
        model_path = os.path.join(MODELS_DIR, f"{cafe_id}.joblib")
        joblib.dump(cafe_data, model_path)
        print(f"✅ Model saved: {model_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

def load_model_from_disk(cafe_id):
    """Load trained model from disk"""
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
    """Load all saved models on startup"""
    global cafe_models
    try:
        if os.path.exists(MODELS_DIR):
            for filename in os.listdir(MODELS_DIR):
                if filename.endswith(".joblib"):
                    cafe_id = filename[:-7]
                    cafe_data = load_model_from_disk(cafe_id)
                    if cafe_data:
                        cafe_models[cafe_id] = cafe_data
            print(f"✅ Loaded {len(cafe_models)} saved models")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_all_models()

# ============================================================
# LAYER 1: RULE-BASED COLUMN MAPPING (CONSERVATIVE)
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
ALL_STANDARD_FIELDS = REQUIRED_CORE + OPTIONAL_FEATURES

def layer1_rule_based(columns):
    """
    Layer 1: Conservative rule-based matching.
    Only maps columns with EXACT or very close matches.
    Deliberately leaves ambiguous columns for Ollama.
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

            # EXACT MATCH: column name exactly matches an alias
            if col_clean in aliases:
                mapping[standard_name] = col
                used_standards.add(standard_name)
                matched = True
                print(f"   📋 Rule match: '{col}' → {standard_name} (exact)")
                break

            # HIGH-CONFIDENCE SUBSTRING: only for substantial aliases (4+ chars)
            # that make up most of the column name (>60%)
            if not matched:
                for alias in aliases:
                    if len(alias) >= 4 and len(col_clean) >= 4:
                        # Only match if alias is a substantial portion
                        # e.g., "quantity_sold" contains "sold" but that's only 4/15 chars
                        # We want "sold_qty" to match "qty" (3/8 = 37.5% — too low)
                        # But "qty" to match "qty" is 100%
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
    """Layer 2: Ollama local LLM for semantic understanding"""

    if not unmapped_columns:
        print("✓ Layer 2 (Ollama): Skipped - no unmapped columns")
        return {}

    if not OLLAMA_AVAILABLE:
        print("⚠️ Layer 2 (Ollama): Skipped - Ollama not available")
        return {}

    try:
        standard_options = ["date", "item", "sold_qty", "produced_qty", "price", "discount_pct", "weather", "day_of_week", "unknown"]

        prompt = f"""Map CSV columns to standard fields for cafe sales data.

Columns to map: {json.dumps(unmapped_columns)}

Standard fields:
- date: sale date/timestamp
- item: product/item name
- sold_qty: units sold to customers
- produced_qty: units produced/baked
- price: price per unit
- discount_pct: discount percentage
- weather: weather condition
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
                    "num_predict": 300,  # Reduced from 500 for faster generation
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            },
            timeout=(10, 300)  # (connect_timeout, read_timeout) - 300 seconds for Mistral
        )

        if response.status_code != 200:
            print(f"⚠️ Ollama returned {response.status_code}: {response.text[:200]}")
            return {}

        response_data = response.json()
        response_text = response_data.get("response", "").strip()
        print(f"   Ollama response: {response_text[:300]}...")

        # Parse JSON response
        try:
            llm_mapping = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON object pattern
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
        print(f"   Make sure to run: ollama serve")
        return {}
    except Exception as e:
        print(f"❌ Layer 2 (Ollama): Failed - {type(e).__name__}: {e}")
        return {}

# ============================================================
# LAYER 3: HUMAN CONFIRMATION UI (Safety Net)
# ============================================================

def layer3_human_confirmation(mapping, unmapped_after_llm):
    """Layer 3: Flag columns that need user review"""
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
    """Transform any cafe's data into standard format"""
    standardized = pd.DataFrame()

    for standard, original in mapping.items():
        if original in df.columns:
            standardized[standard] = df[original].copy()

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

    standardized["surplus_qty"] = (standardized["produced_qty"] - standardized["sold_qty"]).clip(lower=0)

    return standardized

# ============================================================
# TIME-SERIES FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """Add ML features with time-series awareness"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["item", "date"]).reset_index(drop=True)

    # Basic features
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["quarter"] = df["date"].dt.quarter

    # Price fill
    df["price"] = df["price"].fillna(df["price"].median() if df["price"].sum() > 0 else 0)

    # Time-series lag features
    df = df.sort_values(["item", "date"]).reset_index(drop=True)

    df["sold_qty_lag_1"] = df.groupby("item")["sold_qty"].shift(1)
    df["sold_qty_lag_7"] = df.groupby("item")["sold_qty"].shift(7)
    df["sold_qty_lag_14"] = df.groupby("item")["sold_qty"].shift(14)

    df["sold_qty_roll_7"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df["sold_qty_roll_14"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).mean()
    )
    df["sold_qty_roll_30"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    )

    df["sold_qty_roll_std_7"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=2).std()
    )

    df["sold_qty_expanding_mean"] = df.groupby("item")["sold_qty"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    df["days_since_last_sale"] = df.groupby("item")["date"].diff().dt.days

    # Item-level statistics
    item_stats = df.groupby("item")["sold_qty"].agg(mean="mean", std="std", max="max").reset_index()
    df = df.merge(item_stats, on="item", how="left")
    df.rename(columns={"mean": "item_avg_sales", "std": "item_std_sales", "max": "item_max_sales"}, inplace=True)

    # Weather encoding
    weather_map = {"Sunny": 2, "Cloudy": 1, "Rainy": 0, "Unknown": 1}
    df["weather_score"] = df["weather"].map(weather_map).fillna(1)

    # Cyclical encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Discount features
    df["has_discount"] = (df["discount_pct"] > 0).astype(int)
    df["discount_bucket"] = pd.cut(df["discount_pct"], bins=[-1, 0, 10, 20, 50, 100], labels=[0, 1, 2, 3, 4]).astype(int)

    # Fill NaNs for lags
    lag_cols = ["sold_qty_lag_1", "sold_qty_lag_7", "sold_qty_lag_14",
                "sold_qty_roll_7", "sold_qty_roll_14", "sold_qty_roll_30",
                "sold_qty_roll_std_7", "sold_qty_expanding_mean", "days_since_last_sale"]
    for col in lag_cols:
        df[col] = df[col].fillna(df["item_avg_sales"])

    return df

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/")
def home():
    return {
        "message": "Multi-Cafe AI Surplus Prediction API (XGBoost v4.2 - Ollama Local LLM)",
        "layers": [
            "Layer 1: Rule-based matching (fast, free)",
            "Layer 2: Ollama LLM semantic fallback (local, free)",
            "Layer 3: Human confirmation (safe + editable)"
        ],
        "model": "XGBoost with time-series features",
        "ollama_available": OLLAMA_AVAILABLE,
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "features": [
            "AI Dataset Assessment",
            "Auto Column Mapping (via Local Ollama)",
            "User-Editable AI Mappings",
            "Per-Cafe Model Training",
            "Time-Series Feature Engineering",
            "Walk-Forward Validation",
            "Discount-aware Predictions"
        ]
    }

@app.route("/assess", methods=["POST"])
def assess():
    """
    Full 3-layer assessment endpoint.
    Returns editable mapping that user can modify before training.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        cafe_name = request.form.get("cafe_name", "Unknown Cafe")

        df = pd.read_csv(file)
        columns = list(df.columns)
        print(f"\n📁 Assessing dataset: {cafe_name}")
        print(f"   Columns: {columns}")

        # Layer 1: Rule-based
        mapping, unmapped = layer1_rule_based(columns)

        # Layer 2: LLM fallback
        llm_mapping = {}
        if unmapped and OLLAMA_AVAILABLE:
            llm_mapping = layer2_llm_fallback(unmapped, mapping, cafe_name)
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

        # BUILD EDITABLE MAPPING STRUCTURE
        editable_mapping = []
        used_originals = set()

        # First, add all AI-mapped columns (rule-based + LLM)
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

        # Add unmapped columns that need user decision
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
            elif any(x in col_lower for x in ["discount", "promo", "markdown", "off"]):
                suggestion = "discount_pct"
            elif any(x in col_lower for x in ["weather", "condition", "temp", "rain"]):
                suggestion = "weather"

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

        # Store assessment for later modification
        assessment_id = str(uuid.uuid4())[:12]

        # Build diagnostic message
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

        print(f"\n📋 Assessment Complete: {diagnostic_msg}")
        print(f"   Assessment ID: {assessment_id}")
        print(f"   Usable: {assessment_data['usable']}")
        print(f"   AI Engine: {assessment_data['ai_engine']}\n")

        return jsonify({
            "assessment_id": assessment_id,
            "cafe_name": cafe_name,
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
            "ai_engine": "ollama" if OLLAMA_AVAILABLE else "rule-based-only",
            "message": "Review and edit mappings, then submit to /train with assessment_id"
        })

    except Exception as e:
        import traceback
        print(f"❌ Assessment error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/assessment/<assessment_id>", methods=["GET"])
def get_assessment(assessment_id):
    """Retrieve a stored assessment for UI display"""
    if assessment_id in pending_assessments:
        return jsonify(pending_assessments[assessment_id])

    # Try loading from disk
    loaded = load_assessment_from_disk(assessment_id)
    if loaded:
        pending_assessments[assessment_id] = loaded
        return jsonify(loaded)

    return jsonify({"error": "Assessment not found"}), 404

@app.route("/assessment/<assessment_id>/update", methods=["POST"])
def update_assessment(assessment_id):
    """
    User updates mapping decisions.
    Body: {"mapping_changes": {"original_column_name": "new_standard_field"}}
    """
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

        # Apply user changes
        applied_changes = []
        rejected_changes = []

        for original_col, new_standard in mapping_changes.items():
            # Validate the standard field
            if new_standard not in ALL_STANDARD_FIELDS + ["unknown"]:
                rejected_changes.append({
                    "column": original_col,
                    "reason": f"Invalid standard field: {new_standard}"
                })
                continue

            # Find and update the mapping entry
            found = False
            for entry in editable_mapping:
                if entry["original_column"] == original_col:
                    old_mapping = entry["current_mapping"]
                    entry["current_mapping"] = new_standard
                    entry["user_modified"] = True
                    applied_changes.append({
                        "column": original_col,
                        "old_mapping": old_mapping,
                        "new_mapping": new_standard
                    })
                    found = True
                    break

            if not found:
                rejected_changes.append({
                    "column": original_col,
                    "reason": "Column not found in assessment"
                })

        # Recompute missing required/optional based on new mappings
        current_mappings = {entry["current_mapping"]: entry["original_column"]
                           for entry in editable_mapping
                           if entry["current_mapping"] != "unknown"}

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

        # Save updated assessment
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
    """
    Train per-cafe model using assessment_id (with user-modified mappings).
    Supports both: assessment_id reference OR direct file upload with user_corrections.
    """
    try:
        assessment_id = request.form.get("assessment_id")
        cafe_name = request.form.get("cafe_name", "Unknown Cafe")
        cafe_id = request.form.get("cafe_id", str(uuid.uuid4())[:8])

        # Load assessment if provided
        if assessment_id:
            if assessment_id not in pending_assessments:
                loaded = load_assessment_from_disk(assessment_id)
                if loaded:
                    pending_assessments[assessment_id] = loaded
                else:
                    return jsonify({"error": "Assessment not found"}), 404

            assessment = pending_assessments[assessment_id]
            cafe_name = assessment.get("cafe_name", cafe_name)

            # Build final mapping from editable_mapping
            user_corrections = {}
            for entry in assessment["editable_mapping"]:
                original = entry["original_column"]
                standard = entry["current_mapping"]
                if standard != "unknown":
                    user_corrections[original] = standard
        else:
            # Fallback: direct file upload with old-style user_corrections
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded and no assessment_id provided"}), 400
            user_corrections = {}
            if request.form.get("user_corrections"):
                user_corrections = json.loads(request.form.get("user_corrections"))

        file = request.files["file"]
        df = pd.read_csv(file)

        # Build mapping from user_corrections (which now includes ALL mappings)
        mapping = {}
        for original, standard in user_corrections.items():
            if standard != "unknown":
                mapping[standard] = original

        # Check required
        missing_required = [req for req in REQUIRED_CORE if req not in mapping]
        if missing_required:
            return jsonify({
                "error": "Dataset unusable",
                "missing_required": missing_required,
                "message": f"Required columns missing: {', '.join(missing_required)}. Please update mapping via /assessment/{assessment_id}/update"
            }), 400

        # Standardize
        df_std = standardize_dataset(df, mapping)
        df_std = engineer_features(df_std)

        # Drop rows where we can't compute lags
        df_std = df_std.dropna(subset=["sold_qty_lag_1"]).reset_index(drop=True)

        if len(df_std) < 20:
            return jsonify({
                "error": "Not enough data after feature engineering",
                "message": "Need at least 20 rows with historical lags."
            }), 400

        # Encode categorical
        item_encoder = LabelEncoder()
        df_std["item_encoded"] = item_encoder.fit_transform(df_std["item"])

        # Feature columns
        feature_cols = [
            "item_encoded",
            "is_weekend", "month", "day_of_month", "week_of_year",
            "is_month_start", "is_month_end", "quarter",
            "sold_qty_lag_1", "sold_qty_lag_7", "sold_qty_lag_14",
            "sold_qty_roll_7", "sold_qty_roll_14", "sold_qty_roll_30",
            "sold_qty_roll_std_7", "sold_qty_expanding_mean",
            "days_since_last_sale",
            "item_avg_sales", "item_std_sales", "item_max_sales",
            "weather_score",
            "dow_sin", "dow_cos", "month_sin", "month_cos",
            "has_discount", "discount_bucket"
        ]

        if "price" in df_std.columns and df_std["price"].sum() > 0:
            feature_cols.append("price")

        available_features = [c for c in feature_cols if c in df_std.columns]

        X = df_std[available_features]
        y = df_std["sold_qty"]

        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

            model_cv = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=30,
                eval_metric="mae"
            )
            model_cv.fit(
                X_train_cv, y_train_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                verbose=False
            )

            preds = model_cv.predict(X_val_cv)
            cv_scores.append(mean_absolute_error(y_val_cv, preds))

        cv_mae = np.mean(cv_scores)

        # Final model
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="mae"
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        importance = dict(zip(available_features, model.feature_importances_.tolist()))
        importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

        # Calculate item statistics for batch prediction
        item_stats_df = df_std.groupby("item")["sold_qty"].agg(mean="mean", std="std", max="max", min="min").reset_index()
        item_stats = item_stats_df.set_index("item").to_dict("index")

        # Store
        cafe_models[cafe_id] = {
            "model": model,
            "encoders": {
                "item": item_encoder,
                "features": available_features
            },
            "metadata": {
                "cafe_name": cafe_name,
                "cafe_id": cafe_id,
                "training_rows": len(df_std),
                "items": list(df_std["item"].unique()),
                "item_stats": item_stats,
                "mae": mae,
                "r2": r2,
                "cv_mae": cv_mae,
                "feature_columns": available_features,
                "feature_importance": importance_sorted,
                "mapping": mapping,
                "trained_at": datetime.now().isoformat()
            }
        }

        save_model_to_disk(cafe_id, cafe_models[cafe_id])

        return jsonify({
            "cafe_id": cafe_id,
            "cafe_name": cafe_name,
            "status": "trained",
            "model": "XGBoost",
            "rows_used": len(df_std),
            "items": list(df_std["item"].unique()),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "cv_mae": round(cv_mae, 2),
            "confidence": "high",
            "detected_mapping": mapping,
            "top_features": importance_sorted,
            "message": f"XGBoost model trained successfully for {cafe_name}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Predict with optional discount"""
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

        # Historical lags
        lag_1 = data.get("sold_qty_lag_1", meta.get("item_avg_sales", 5))
        lag_7 = data.get("sold_qty_lag_7", lag_1)
        lag_14 = data.get("sold_qty_lag_14", lag_1)
        roll_7 = data.get("sold_qty_roll_7", lag_1)
        roll_14 = data.get("sold_qty_roll_14", lag_1)
        roll_30 = data.get("sold_qty_roll_30", lag_1)
        roll_std_7 = data.get("sold_qty_roll_std_7", 2)
        expanding_mean = data.get("sold_qty_expanding_mean", lag_1)
        item_avg = data.get("item_avg_sales", lag_1)
        item_std = data.get("item_std_sales", 2)
        item_max = data.get("item_max_sales", lag_1 * 2)

        item_enc = encoders["item"].transform([item])[0] if item in encoders["item"].classes_ else 0

        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
        date_obj = datetime.strptime(data.get("date", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d")
        month = date_obj.month
        day_of_month = date_obj.day
        week_of_year = date_obj.isocalendar()[1]
        is_month_start = 1 if day_of_month <= 3 else 0
        is_month_end = 1 if day_of_month >= 28 else 0
        quarter = (month - 1) // 3 + 1

        weather_map = {"Sunny": 2, "Cloudy": 1, "Rainy": 0, "Unknown": 1}
        weather_score = weather_map.get(weather, 1)

        dow_sin = np.sin(2 * np.pi * date_obj.weekday() / 7)
        dow_cos = np.cos(2 * np.pi * date_obj.weekday() / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        has_discount = 1 if discount_pct > 0 else 0
        discount_bucket = pd.cut([discount_pct], bins=[-1, 0, 10, 20, 50, 100], labels=[0, 1, 2, 3, 4])[0]

        features = {
            "item_encoded": item_enc,
            "is_weekend": is_weekend,
            "month": month,
            "day_of_month": day_of_month,
            "week_of_year": week_of_year,
            "is_month_start": is_month_start,
            "is_month_end": is_month_end,
            "quarter": quarter,
            "sold_qty_lag_1": lag_1,
            "sold_qty_lag_7": lag_7,
            "sold_qty_lag_14": lag_14,
            "sold_qty_roll_7": roll_7,
            "sold_qty_roll_14": roll_14,
            "sold_qty_roll_30": roll_30,
            "sold_qty_roll_std_7": roll_std_7,
            "sold_qty_expanding_mean": expanding_mean,
            "days_since_last_sale": 1,
            "item_avg_sales": item_avg,
            "item_std_sales": item_std,
            "item_max_sales": item_max,
            "weather_score": weather_score,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "has_discount": has_discount,
            "discount_bucket": int(discount_bucket)
        }

        if "price" in meta["feature_columns"]:
            features["price"] = price

        # Create DataFrame with exactly the columns needed
        X = pd.DataFrame([{col: features[col] for col in meta["feature_columns"]}])

        base_predicted_sales = int(model.predict(X)[0])
        base_predicted_sales = max(0, base_predicted_sales)

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
            "cv_mae": meta.get("cv_mae", meta["mae"]),
            "training_size": meta["training_rows"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Predict all items for a day - construct features without rolling calculations"""
    try:
        import sys
        print(f"\n🔴 BATCH-PREDICT CALLED", file=sys.stderr, flush=True)
        
        data = request.json
        cafe_id = data.get("cafe_id")
        
        print(f"   cafe_id={cafe_id}", file=sys.stderr, flush=True)

        if not cafe_id or cafe_id not in cafe_models:
            print(f"   ❌ Cafe not found", file=sys.stderr, flush=True)
            return jsonify({"error": "Cafe not found"}), 404

        cafe_data = cafe_models[cafe_id]
        model = cafe_data["model"]
        encoders = cafe_data["encoders"]
        meta = cafe_data["metadata"]
        
        print(f"   Loaded cafe metadata. Items: {meta.get('items', [])}", file=sys.stderr, flush=True)
        print(f"   Feature columns: {meta.get('feature_columns', [])}", file=sys.stderr, flush=True)

        day = data.get("day_of_week", "Saturday")
        weather = data.get("weather", "Sunny")
        discount_pct = data.get("discount_pct", 0)
        date_str = data.get("date", datetime.now().strftime("%Y-%m-%d"))

        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
        discount_boost = 1 + (discount_pct / 100) * 1.5
        
        # Parse date features
        month = date_obj.month
        day_of_month = date_obj.day
        week_of_year = date_obj.isocalendar()[1]
        is_month_start = 1 if day_of_month <= 3 else 0
        is_month_end = 1 if day_of_month >= 28 else 0
        quarter = (month - 1) // 3 + 1
        
        # Weather encoding
        weather_map = {"Sunny": 2, "Cloudy": 1, "Rainy": 0, "Unknown": 1}
        weather_score = weather_map.get(weather, 1)
        
        # Cyclical encoding
        dow_sin = np.sin(2 * np.pi * date_obj.weekday() / 7)
        dow_cos = np.cos(2 * np.pi * date_obj.weekday() / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Discount features
        has_discount = 1 if discount_pct > 0 else 0
        discount_bucket = pd.cut([discount_pct], bins=[-1, 0, 10, 20, 50, 100], labels=[0, 1, 2, 3, 4])[0]
        
        predictions = []
        
        for item in meta["items"]:
            # Get item's historical stats for feature defaults
            item_stats = meta.get("item_stats", {}).get(item, {})
            item_avg = item_stats.get("mean", 5)
            item_std = item_stats.get("std", 2)
            item_max = item_stats.get("max", item_avg * 2)
            
            # Encode item
            item_enc = encoders["item"].transform([item])[0] if item in encoders["item"].classes_ else 0
            
            # Initialize features with ALL required columns set to 0
            features = {col: 0 for col in meta["feature_columns"]}
            
            # Now override with actual values
            features.update({
                "item_encoded": item_enc,
                "is_weekend": is_weekend,
                "month": month,
                "day_of_month": day_of_month,
                "week_of_year": week_of_year,
                "is_month_start": is_month_start,
                "is_month_end": is_month_end,
                "quarter": quarter,
                "sold_qty_lag_1": item_avg,
                "sold_qty_lag_7": item_avg,
                "sold_qty_lag_14": item_avg,
                "sold_qty_roll_7": item_avg,
                "sold_qty_roll_14": item_avg,
                "sold_qty_roll_30": item_avg,
                "sold_qty_roll_std_7": item_std,
                "sold_qty_expanding_mean": item_avg,
                "days_since_last_sale": 1,
                "item_avg_sales": item_avg,
                "item_std_sales": item_std,
                "item_max_sales": item_max,
                "weather_score": weather_score,
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "has_discount": has_discount,
                "discount_bucket": int(discount_bucket)
            })
            
            # Ensure price column exists if required
            if "price" in meta["feature_columns"]:
                features["price"] = 0
            
            # Create DataFrame with exactly the columns needed
            X = pd.DataFrame([{col: features[col] for col in meta["feature_columns"]}])
            
            # Make prediction
            base_pred = int(model.predict(X)[0])
            base_pred = max(0, base_pred)
            
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
        error_msg = str(e)
        print(f"❌ Batch predict error: {error_msg}")
        import traceback
        tb = traceback.format_exc()
        print(tb)
        
        # Also write to file for debugging
        with open("error_log.txt", "a") as f:
            f.write(f"\n❌ {datetime.now()}\n{tb}\n")
        
        return jsonify({"error": error_msg}), 500

@app.route("/cafes")
def list_cafes():
    return jsonify({
        "cafes": [
            {
                "cafe_id": cid,
                "cafe_name": data["metadata"]["cafe_name"],
                "items": data["metadata"]["items"],
                "training_rows": data["metadata"]["training_rows"],
                "r2": data["metadata"]["r2"],
                "cv_mae": data["metadata"].get("cv_mae", data["metadata"]["mae"]),
                "model": "XGBoost"
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
        "cv_mae": meta.get("cv_mae", meta["mae"]),
        "trained_at": meta["trained_at"],
        "detected_mapping": meta["mapping"],
        "top_features": meta.get("feature_importance", {}),
        "model": "XGBoost"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)