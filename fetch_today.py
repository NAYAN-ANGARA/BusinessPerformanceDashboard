"""
Daily SKU Ads fetcher — stores data in Supabase (PostgreSQL).
Run by GitHub Actions every day at 9 AM IST.
"""
import os
import sys
import pandas as pd
import requests
from datetime import date, timedelta
from sku_data import fetch_sku_ads_data

# ── Supabase connection ────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
TABLE        = "sku_ads_cache"
MARCH_FLOOR  = date(2026, 3, 1)

SB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "resolution=merge-duplicates",
}


def sb_get(query: str) -> list:
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{TABLE}?{query}",
        headers={**SB_HEADERS, "Prefer": "count=none"},
    )
    r.raise_for_status()
    return r.json()


def sb_upsert(rows: list):
    batch_size = 500
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{TABLE}",
            headers=SB_HEADERS,
            json=batch,
        )
        if r.status_code not in (200, 201):
            print(f"Upsert error {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
        total += len(batch)
        print(f"  Upserted batch {i // batch_size + 1} — {total:,} rows so far")


# ── Determine fetch range ──────────────────────────────────────────────────────
yesterday = date.today() - timedelta(days=1)

manual = os.environ.get("INPUT_START_DATE", "").strip()
if manual:
    start_date = date.fromisoformat(manual)
    print(f"Manual override: fetching from {start_date}")
else:
    try:
        # Use lowercase column name — table columns are all lowercase
        rows = sb_get("select=date&order=date.desc&limit=1")
        if rows:
            last = date.fromisoformat(rows[0]["date"])
            start_date = last + timedelta(days=1)
            if start_date < MARCH_FLOOR:
                start_date = MARCH_FLOOR
            print(f"Supabase last date: {last} → fetching from {start_date}")
        else:
            start_date = MARCH_FLOOR
            print(f"Supabase table is empty → fetching from {start_date}")
    except Exception as e:
        start_date = MARCH_FLOOR
        print(f"Could not query Supabase ({e}) → defaulting to {start_date}")

end_date = yesterday

if start_date > end_date:
    print(f"Already up to date ({start_date} > {end_date}) — nothing to fetch.")
    sys.exit(0)

print(f"Fetching: {start_date} → {end_date}")

# ── Fetch from Amazon Ads ──────────────────────────────────────────────────────
new_df = fetch_sku_ads_data(
    start_date.strftime("%Y-%m-%d"),
    end_date.strftime("%Y-%m-%d"),
    market="BOTH",
    max_workers=3,
)

if new_df.empty:
    print("No data returned from Amazon Ads — nothing to write.")
    sys.exit(0)

print(f"Fetched {len(new_df):,} rows from Amazon Ads")

# ── Rename to lowercase to match Supabase table columns ───────────────────────
new_df = new_df.copy()
new_df["Date"] = new_df["Date"].dt.strftime("%Y-%m-%d")

new_df = new_df.rename(columns={
    "Date": "date", "Market": "market", "Parent_SKU": "parent_sku",
    "SKU": "sku", "ASIN": "asin", "Impressions": "impressions",
    "Clicks": "clicks", "Spend": "spend", "Ad_Sales": "ad_sales",
    "Ad_Orders": "ad_orders", "CTR": "ctr", "CPC": "cpc", "ACOS": "acos",
})

keep = ["date", "market", "parent_sku", "sku", "asin",
        "impressions", "clicks", "spend", "ad_sales", "ad_orders",
        "ctr", "cpc", "acos"]
rows = new_df[[c for c in keep if c in new_df.columns]].to_dict(orient="records")

print(f"Upserting {len(rows):,} rows into Supabase table '{TABLE}'...")
sb_upsert(rows)
print(f"Done ✅  {len(rows):,} rows saved to Supabase.")
