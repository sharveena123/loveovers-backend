"""
Realistic synthetic cafe dataset generator.

Designed so a properly-tuned XGBoost model can reach R² ~ 0.75-0.85, which is
what real retail forecasting models actually achieve on real POS data.

Signal sources (in order of importance):
  1. Strong day-of-week pattern  (weekends ~2x weekdays for most items)
  2. Lag-1 / lag-7 autocorrelation  (yesterday predicts today)
  3. Weather effect  (rainy = -20%, sunny = +10%)
  4. Discount uplift (discounts boost sales ~linearly)
  5. Monthly trend (gradual growth)
  6. Holiday spikes (specific dates)
  7. Light Gaussian noise (CV ~ 0.15, not 0.55)

Run with:
  python generate_realistic_data.py
This writes cafe_data_realistic.csv to the current folder.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# Same column names as your existing dataset, so the AI mapping still works
OUT_COLS = ["txn_dt", "menu_article", "units_moved", "batch_yield",
            "unit_cost_rm", "markdown_rate", "meteo", "calendar_dow"]

ITEMS = {
    # name             baseline  weekend_mult  rainy_mult  sunny_mult  price  trend_per_day
    "Croissant":         (30,    1.6,          0.85,       1.10,       3.50,  0.05),
    "Baguette":          (25,    1.3,          0.90,       1.05,       2.80,  0.03),
    "Danish":            (22,    1.5,          0.80,       1.15,       4.20,  0.04),
    "Muffin":            (18,    1.7,          0.95,       1.10,       3.00,  0.02),
    "Scone":             (15,    1.4,          0.90,       1.05,       3.20,  0.02),
    "Pain_au_Chocolat":  (28,    1.6,          0.85,       1.12,       4.50,  0.05),
    "Eclair":            (14,    1.8,          0.75,       1.20,       5.00,  0.03),
    "Macaron":           (13,    2.0,          0.70,       1.25,       2.50,  0.02),
    "Tart":              (17,    1.7,          0.80,       1.15,       4.80,  0.03),
    "Cinnamon_Roll":     (16,    1.5,          0.95,       1.05,       3.80,  0.03),
}

WEATHER_OPTIONS = ["Sunny", "Cloudy", "Rainy"]
WEATHER_PROBS = [0.55, 0.30, 0.15]

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# Public holidays / busy spikes (Jan-Jun 2024)
HOLIDAY_DATES = {
    "2024-01-01": 1.4,   # New Year
    "2024-02-14": 1.6,   # Valentine's
    "2024-02-10": 1.5,   # CNY
    "2024-03-08": 1.3,
    "2024-03-31": 1.7,   # Easter
    "2024-04-10": 1.4,   # Hari Raya
    "2024-05-01": 1.3,   # Labour Day
    "2024-05-12": 1.5,   # Mother's Day
}


def generate(start_date="2024-01-01", n_days=180):
    """Generate realistic cafe sales for n_days * len(ITEMS) rows."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []

    # Pre-generate one weather sequence shared across all items per day
    weather_by_day = np.random.choice(
        WEATHER_OPTIONS, size=n_days, p=WEATHER_PROBS
    )

    for item, (base, wkd_mult, rain_mult, sun_mult, price, trend) in ITEMS.items():
        # Persistent state for autocorrelation: yesterday's residual
        prev_residual = 0.0

        for d in range(n_days):
            date = start + timedelta(days=d)
            date_str = date.strftime("%Y-%m-%d")
            dow_name = DAY_NAMES[date.weekday()]
            is_weekend = date.weekday() >= 5
            weather = str(weather_by_day[d])

            # 1. Baseline + linear trend
            expected = base + trend * d

            # 2. Day-of-week multiplier
            if is_weekend:
                expected *= wkd_mult
            elif date.weekday() == 4:  # Friday
                expected *= 1.15
            elif date.weekday() == 0:  # Monday quiet
                expected *= 0.85

            # 3. Weather effect
            if weather == "Rainy":
                expected *= rain_mult
            elif weather == "Sunny":
                expected *= sun_mult

            # 4. Holiday spike
            if date_str in HOLIDAY_DATES:
                expected *= HOLIDAY_DATES[date_str]

            # 5. Random discount (15% of days have a discount)
            if np.random.rand() < 0.15:
                discount = int(np.random.choice([10, 15, 20, 25, 30]))
                discount_uplift = 1 + (discount / 100) * 1.5
                expected *= discount_uplift
            else:
                discount = 0

            # 6. Autocorrelation: carry 60% of yesterday's residual into today
            #    Real POS data shows lag-1 autocorr ~0.5-0.6
            expected += 0.6 * prev_residual

            # 7. Light Gaussian noise (CV ~ 0.10 — clean POS data)
            noise = np.random.normal(0, expected * 0.10)
            sold = max(0, expected + noise)

            # Track residual for next day's autocorrelation
            prev_residual = sold - expected

            sold_int = int(round(sold))
            # Bakery typically overproduces ~10-20%
            produced = int(round(sold * np.random.uniform(1.08, 1.22)))

            rows.append({
                "txn_dt": date_str,
                "menu_article": item,
                "units_moved": sold_int,
                "batch_yield": produced,
                "unit_cost_rm": price,
                "markdown_rate": discount,
                "meteo": weather,
                "calendar_dow": dow_name,
            })

    df = pd.DataFrame(rows, columns=OUT_COLS)
    return df


def verify(df):
    """Quick R² ceiling check on the generated data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["txn_dt"])
    df["dow_n"] = df["date"].dt.dayofweek
    df["disc_bucket"] = pd.cut(
        df["markdown_rate"], bins=[-1, 0, 10, 20, 100], labels=[0, 1, 2, 3]
    )

    y = df["units_moved"]
    ss_tot = ((y - y.mean()) ** 2).sum()

    # Lookup ceilings
    for label, groupers in [
        ("item only", ["menu_article"]),
        ("item x dow", ["menu_article", "dow_n"]),
        ("item x dow x weather", ["menu_article", "dow_n", "meteo"]),
        ("item x dow x weather x disc",
         ["menu_article", "dow_n", "meteo", "disc_bucket"]),
    ]:
        pred = df.groupby(groupers, observed=True)["units_moved"].transform("mean")
        r2 = 1 - ((y - pred) ** 2).sum() / ss_tot
        print(f"  R2 ceiling [{label:30s}]: {r2:.4f}")

    # Lag-1 autocorrelation
    df = df.sort_values(["menu_article", "date"])
    acfs = [
        df[df["menu_article"] == it]["units_moved"]
        .reset_index(drop=True).autocorr(lag=1)
        for it in df["menu_article"].unique()
    ]
    print(f"  Avg lag-1 autocorrelation:                : {np.mean(acfs):+.3f}")

    # CV per item
    cvs = df.groupby("menu_article")["units_moved"].apply(
        lambda x: x.std() / x.mean()
    )
    print(f"  Avg coefficient of variation              : {cvs.mean():.3f}")


if __name__ == "__main__":
    print("Generating realistic synthetic cafe dataset (180 days x 10 items)...")
    df = generate(n_days=180)
    out = "cafe_data_realistic.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}  ({len(df)} rows)\n")
    print("=== INTRINSIC PREDICTABILITY OF GENERATED DATA ===")
    verify(df)
    print("\nGoal: avg lag-1 > 0.4, CV < 0.25, R2 ceiling > 0.80")
    print("If those hold, XGBoost should reach R2 ~ 0.75-0.85.")
