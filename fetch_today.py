import os
import pandas as pd
from datetime import date, timedelta
from sku_data import fetch_sku_ads_data

cache_path = "sku_ads_cache.csv"
yesterday = date.today() - timedelta(days=1)
MARCH_FLOOR = date(2026, 3, 1)

# Determine start date
# Priority 1: manual override via INPUT_START_DATE env var
# Priority 2: day after last date in existing CSV
# Priority 3: March 1 2026 hard floor
manual = os.environ.get("INPUT_START_DATE", "").strip()
if manual:
    start_date = date.fromisoformat(manual)
    print(f"Manual override: fetching from {start_date}")
elif os.path.exists(cache_path):
    existing = pd.read_csv(cache_path, parse_dates=["Date"])
    last_date = existing["Date"].max().date()
    start_date = last_date + timedelta(days=1)
    if start_date < MARCH_FLOOR:
        start_date = MARCH_FLOOR
    print(f"CSV last date: {last_date} -> fetching from {start_date}")
else:
    start_date = MARCH_FLOOR
    print(f"No cache found -> fetching from {start_date}")

end_date = yesterday

if start_date > end_date:
    print(f"Already up to date ({start_date} > {end_date}) -- nothing to fetch.")
    exit(0)

print(f"Fetching: {start_date} -> {end_date}")

new_df = fetch_sku_ads_data(
    start_date.strftime("%Y-%m-%d"),
    end_date.strftime("%Y-%m-%d"),
    market="BOTH",
    max_workers=3,
)

if new_df.empty:
    print("No data returned -- nothing to write.")
    exit(0)

print(f"Fetched {len(new_df):,} rows")

if os.path.exists(cache_path):
    existing = pd.read_csv(cache_path, parse_dates=["Date"])
    fetched_dates = new_df["Date"].dt.date.unique()
    existing = existing[~existing["Date"].dt.date.isin(fetched_dates)]
    combined = pd.concat([existing, new_df], ignore_index=True)
    print(f"Existing: {len(existing):,}  +  New: {len(new_df):,}  =  Total: {len(combined):,}")
else:
    combined = new_df

combined = combined.sort_values(["Date", "Market", "Parent_SKU"]).reset_index(drop=True)
combined.to_csv(cache_path, index=False)
print(f"Saved -> {cache_path}")
