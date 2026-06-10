"""
Loveovers cafe — realistic synthetic POS dataset.

Open days only: Thursday, Friday, Saturday, Sunday.
Same CSV schema as the backend (txn_dt, menu_article, units_moved, ...).

Run:
  python generate_loveovers_cafe_data.py

Output:
  cafe_data_loveovers.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(2026)

OUT_COLS = [
    "txn_dt", "menu_article", "units_moved", "batch_yield",
    "unit_cost_rm", "markdown_rate", "meteo", "calendar_dow",
]

# weekday(): Mon=0 ... Sun=6  ->  open Thu=3, Fri=4, Sat=5, Sun=6
OPEN_WEEKDAYS = {3, 4, 5, 6}

DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

# Per open-day multiplier (Thu-Sun only)
DOW_MULT = {
    3: 0.88,   # Thursday — quieter start to the weekend
    4: 1.12,   # Friday — after-work crowd
    5: 1.48,   # Saturday — peak
    6: 1.38,   # Sunday — strong, slightly below Sat
}

WEATHER_OPTIONS = ["Sunny", "Cloudy", "Rainy"]
WEATHER_PROBS = [0.52, 0.32, 0.16]

# name, baseline_units, price_rm, rainy_mult, sunny_mult, sat_sun_extra, trend_per_open_day
MENU = {
    "Chicken Focaccia": (
        26, 18.0, 0.88, 1.08, 1.0, 0.06,
    ),
    "Beef Focaccia": (
        22, 19.0, 0.90, 1.06, 1.0, 0.05,
    ),
    "Earl Grey Cinnamon Bun": (
        20, 14.0, 0.92, 1.10, 1.12, 0.07,
    ),
    "Dubai Chocolate Kunafa Brownie": (
        32, 22.0, 0.82, 1.12, 1.25, 0.09,
    ),
    "Carrot Cake": (
        16, 16.0, 0.90, 1.05, 1.0, 0.04,
    ),
    "Butter Croissant": (
        38, 8.0, 0.93, 1.04, 1.0, 0.05,
    ),
    "Almond Croissant": (
        24, 10.0, 0.91, 1.06, 1.08, 0.05,
    ),
    "Lemon Tart": (
        14, 12.0, 0.88, 1.08, 1.05, 0.03,
    ),
    "Seasonal Special": (
        12, 15.0, 0.90, 1.05, 1.15, 0.02,
    ),
}

# Holidays that fall on Thu-Sun get a footfall boost
HOLIDAY_MULT = {
    "2025-02-13": 1.35,  # Valentine's eve (Thu)
    "2025-02-14": 1.55,  # Valentine's (Fri)
    "2025-01-30": 1.40,  # CNY weekend lead-up
    "2025-03-29": 1.30,  # Easter weekend
    "2025-05-01": 1.25,  # Labour Day (Thu)
    "2025-05-11": 1.35,  # Mother's Day (Sun)
    "2025-12-25": 1.20,  # Christmas (Thu)
    "2025-12-26": 1.45,  # Boxing weekend (Fri)
    "2026-02-12": 1.35,
    "2026-02-13": 1.55,
    "2026-02-14": 1.50,
    "2026-05-10": 1.35,
}


def open_dates_between(start: datetime, end: datetime) -> list[datetime]:
    """All calendar dates in [start, end] that are Thu-Sun."""
    days = []
    d = start
    while d <= end:
        if d.weekday() in OPEN_WEEKDAYS:
            days.append(d)
        d += timedelta(days=1)
    return days


def generate(
    start_date: str = "2025-01-02",
    end_date: str = "2026-05-17",
) -> pd.DataFrame:
    """
    Generate one row per (open_date, menu_item).
    ~70 open weeks x 4 days x 9 items ≈ 2,500 rows.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    open_days = open_dates_between(start, end)
    n_open = len(open_days)

    weather_seq = np.random.choice(WEATHER_OPTIONS, size=n_open, p=WEATHER_PROBS)

    rows = []
    open_day_index = {d.date(): i for i, d in enumerate(open_days)}

    for item, (base, price, rain_m, sun_m, wkd_extra, trend) in MENU.items():
        prev_residual = 0.0
        prev_open_idx = -1

        for od, date in enumerate(open_days):
            wd = date.weekday()
            date_str = date.strftime("%Y-%m-%d")
            weather = str(weather_seq[od])

            expected = base + trend * od

            # Day-of-week (within Thu-Sun pattern)
            expected *= DOW_MULT[wd]
            if wd in (5, 6):
                expected *= wkd_extra

            # Weather
            if weather == "Rainy":
                expected *= rain_m
            elif weather == "Sunny":
                expected *= sun_m

            # Holiday spike
            if date_str in HOLIDAY_MULT:
                expected *= HOLIDAY_MULT[date_str]

            # Random markdown (~12% of open days)
            if np.random.rand() < 0.12:
                discount = int(np.random.choice([10, 15, 20, 25]))
                expected *= 1 + (discount / 100) * 1.4
            else:
                discount = 0

            # Same-day-last-week effect: if previous open day was 4 days ago (Fri->Thu)
            # or 1 day ago, blend autocorr from last open day
            if prev_open_idx >= 0:
                gap = od - prev_open_idx
                # Stronger carry when gap is 1 (consecutive open days); still useful at gap 7
                carry = 0.55 if gap == 1 else (0.45 if gap <= 4 else 0.35)
                expected += carry * prev_residual

            noise = np.random.normal(0, expected * 0.10)
            sold = max(0, expected + noise)
            prev_residual = sold - expected
            prev_open_idx = od

            sold_int = int(round(sold))
            produced = int(round(sold * np.random.uniform(1.10, 1.22)))

            rows.append({
                "txn_dt": date_str,
                "menu_article": item,
                "units_moved": sold_int,
                "batch_yield": produced,
                "unit_cost_rm": price,
                "markdown_rate": discount,
                "meteo": weather,
                "calendar_dow": DAY_NAMES[wd],
            })

    return pd.DataFrame(rows, columns=OUT_COLS)


def verify(df: pd.DataFrame) -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["txn_dt"])
    df["dow_n"] = df["date"].dt.dayofweek

    print(f"\nRows: {len(df)}")
    print(f"Items: {df['menu_article'].nunique()}")
    print(f"Open days: {df['txn_dt'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Days of week present: {sorted(df['calendar_dow'].unique())}")

    y = df["units_moved"]
    ss_tot = ((y - y.mean()) ** 2).sum()

    pred = df.groupby(
        ["menu_article", "dow_n", "meteo"], observed=True
    )["units_moved"].transform("mean")
    r2 = 1 - ((y - pred) ** 2).sum() / ss_tot
    print(f"R2 ceiling (item x dow x weather): {r2:.4f}")

    df = df.sort_values(["menu_article", "date"])
    acfs = [
        df[df["menu_article"] == it]["units_moved"]
        .reset_index(drop=True).autocorr(lag=1)
        for it in df["menu_article"].unique()
    ]
    print(f"Avg lag-1 autocorr (consecutive open days): {np.mean(acfs):+.3f}")

    print("\nAvg units sold by day:")
    for dow in ["Thursday", "Friday", "Saturday", "Sunday"]:
        sub = df[df["calendar_dow"] == dow]["units_moved"]
        print(f"  {dow:10s}  mean={sub.mean():.1f}  total={sub.sum()}")


if __name__ == "__main__":
    print("Generating Loveovers cafe dataset (Thu-Sun only)...")
    df = generate()
    out = "cafe_data_loveovers.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    verify(df)
