"""
Fetch South Korea macro series from FRED, clean them, and store them in CSV + SQLite.

Setup:
    pip install pandas requests
    export FRED_API_KEY="your_key"   # free key: https://fred.stlouisfed.org/docs/api/api_key.html
Run:
    python fetch_fred_korea.py
"""
import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import requests

load_dotenv()
API_URL = "https://api.stlouisfed.org/fred/series/observations"
API_KEY = os.environ.get("FRED_API_KEY")

# name -> FRED series id  (verify IDs on fred.stlouisfed.org if one fails)
SERIES = {
    "inflation_yoy": "KORCPALTT01CTGYM",             # Consumer Price Index, all items (monthly)
    "unemployment": "LRHUTTTTKRM156S",    # Unemployment rate, 15+ (monthly)
    "policy_rate": "IRSTCI01KRM156N",     # Call money / interbank rate (monthly)
    "real_gdp": "NGDPRSAXDCKRQ",          # Real GDP, seasonally adjusted (quarterly)
}

START = "2000-01-01"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def fetch_series(series_id: str) -> pd.Series:
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": START,
    }
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
    return df.set_index("date")["value"]


def main():
    if not API_KEY:
        raise SystemExit("Set the FRED_API_KEY environment variable first.")

    frames = {}
    for name, sid in SERIES.items():
        try:
            s = fetch_series(sid)
            s.to_csv(DATA_DIR / f"{name}_raw.csv", header=[name])
            frames[name] = s
            print(f"OK   {name:<13} {sid:<18} {s.index.min().date()} -> {s.index.max().date()}")
        except Exception as e:
            print(f"FAIL {name:<13} {sid:<18} {e}")

    # Monthly dataset: quarterly GDP is forward-filled to months
    monthly = pd.DataFrame({k: v for k, v in frames.items() if k != "real_gdp"})
    if "real_gdp" in frames:
        monthly["real_gdp"] = frames["real_gdp"].resample("MS").ffill()

    # Derived indicators
    if "cpi" in monthly:
        monthly["inflation_yoy"] = monthly["cpi"].pct_change(12) * 100
    if "real_gdp" in monthly:
        monthly["gdp_growth_yoy"] = monthly["real_gdp"].pct_change(12) * 100

    monthly = monthly.dropna(how="all")
    monthly.index.name = "date"
    monthly.to_csv(DATA_DIR / "korea_macro_monthly.csv")

    with sqlite3.connect(DATA_DIR / "korea_macro.db") as conn:
        monthly.reset_index().to_sql("macro_monthly", conn, if_exists="replace", index=False)

    print(f"\nSaved {len(monthly)} rows -> data/korea_macro_monthly.csv and data/korea_macro.db")


if __name__ == "__main__":
    main()
