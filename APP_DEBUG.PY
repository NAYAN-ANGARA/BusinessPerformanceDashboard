import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Business Performance Dashboard - DEBUG",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 DEBUG MODE - Finding Discrepancies")

# ---------------- LOAD DATA ----------------
@st.cache_data(show_spinner=True, ttl=300)
def load_data():
    try:
        return load_all_sheets(
            "secret-envoy-486405-j3-03851d061385.json",
            "USA - DB for Marketplace Dashboard"
        )
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

if st.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

data = load_data()

if data is None:
    st.stop()

st.write("## Step 1: Available Sheets")
st.write(list(data.keys()))

# ---------------- LOAD ALL SHEETS ----------------
sales = data["Sales_data"].copy()
st.write(f"✅ Sales_data loaded: {sales.shape}")

# Load all spend sheets
spend_sheets = []
for sheet_name in data.keys():
    if 'spend' in sheet_name.lower():
        st.write(f"Found spend sheet: {sheet_name}")
        spend_sheets.append((sheet_name, data[sheet_name].copy()))

# ---------------- NORMALIZE FUNCTION ----------------
def normalize_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

# ---------------- NORMALIZE SALES ----------------
st.write("## Step 2: Sales Data Analysis")

sales = normalize_columns(sales)
st.write("Columns:", list(sales.columns))

# Show raw data sample
st.write("### Raw Sales Sample (first 5 rows):")
st.dataframe(sales.head())

# Check for data type issues
st.write("### Column Data Types:")
st.write(sales.dtypes)

# ---------------- CLEAN CHANNEL NAMES ----------------
st.write("## Step 3: Channel Standardization")
st.write("### BEFORE cleaning:")
st.write("Unique channels:", sorted(sales["channel"].dropna().astype(str).unique()))

# Clean channels
sales["channel"] = (
    sales["channel"]
    .astype(str)
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)
# Standardize eBay variations
sales["channel"] = sales["channel"].str.replace(r'_ebay$', '_eBay', case=False, regex=True)
sales["channel"] = sales["channel"].str.replace(r'_Ebay$', '_eBay', case=False, regex=True)

st.write("### AFTER cleaning:")
st.write("Unique channels:", sorted(sales["channel"].dropna().unique()))

# ---------------- TYPE CAST SALES ----------------
st.write("## Step 4: Type Casting")

# Date
sales["purchased_on"] = pd.to_datetime(
    sales["purchased_on"],
    format="mixed",
    errors="coerce"
)
rows_before = len(sales)
sales = sales.dropna(subset=["purchased_on"])
rows_after = len(sales)
st.write(f"Date conversion: {rows_before} → {rows_after} rows (dropped {rows_before - rows_after} invalid dates)")

# Orders
sales["no_of_orders"] = pd.to_numeric(
    sales["no_of_orders"],
    errors="coerce"
).fillna(0)

# Revenue
st.write("### Revenue Analysis:")
st.write("Available columns with 'price' in name:", [col for col in sales.columns if 'price' in col.lower()])

# Check if discounted_price exists
if "discounted_price" in sales.columns:
    st.write(f"Sample raw values from 'discounted_price':")
    st.write(sales["discounted_price"].head(10))
    st.write(f"Data type: {sales['discounted_price'].dtype}")
    
    # Convert - handle dollar signs and commas
    sales["discounted_price"] = (
        sales["discounted_price"]
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    sales["discounted_price"] = pd.to_numeric(
        sales["discounted_price"],
        errors="coerce"
    ).fillna(0)
    
    st.write("After conversion:")
    st.write(sales["discounted_price"].head(10))
    st.write(f"Non-zero revenue rows: {(sales['discounted_price'] > 0).sum()}")
    
    # CRITICAL: Revenue = discounted_price × no_of_orders (this is what BI uses!)
    sales["revenue"] = sales["discounted_price"] * sales["no_of_orders"]
    
    st.write(f"Total revenue (discounted_price only): ${sales['discounted_price'].sum():,.2f}")
    st.write(f"Total revenue (discounted_price × orders): ${sales['revenue'].sum():,.2f}")
else:
    st.error("❌ discounted_price column not found!")
    st.write("Available columns:", list(sales.columns))

# Commission
st.write("### Commission Analysis:")
st.write("Available columns with 'commission' in name:", [col for col in sales.columns if 'commission' in col.lower()])

commission_col = None
for col in ["selling_commission", "commission", "seller_commission", "marketplace_commission", "selling_commision"]:
    if col in sales.columns:
        commission_col = col
        st.write(f"✅ Found commission column: '{commission_col}'")
        break

if commission_col:
    # Show sample values BEFORE conversion
    st.write(f"Sample raw values from '{commission_col}':")
    st.write(sales[commission_col].head(10))
    
    # Check data type
    st.write(f"Data type: {sales[commission_col].dtype}")
    
    # Convert - handle dollar signs and commas
    sales["selling_commission"] = (
        sales[commission_col]
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    sales["selling_commission"] = pd.to_numeric(
        sales["selling_commission"],
        errors="coerce"
    ).fillna(0)
    
    # Show after conversion
    st.write("After conversion:")
    st.write(sales["selling_commission"].head(10))
    st.write(f"Non-zero commission rows: {(sales['selling_commission'] > 0).sum()}")
    st.write(f"Total commission (all data): ${sales['selling_commission'].sum():,.2f}")
else:
    st.warning("⚠️ No commission column found")
    st.write("All available columns:", list(sales.columns))
    sales["selling_commission"] = 0

st.write("### After Type Casting:")
st.write(f"- Total rows: {len(sales):,}")
st.write(f"- Date range: {sales['purchased_on'].min()} to {sales['purchased_on'].max()}")

# Check different revenue possibilities
st.write("### 🔍 Revenue Calculation Analysis:")
if "price" in sales.columns:
    sales["price"] = pd.to_numeric(sales["price"], errors="coerce").fillna(0)
    revenue_from_price = sales["price"].sum()
    st.write(f"Total from 'price' column: ${revenue_from_price:,.2f}")

if "discounted_price" in sales.columns:
    revenue_from_discounted = sales["discounted_price"].sum()
    st.write(f"Total from 'discounted_price' column: ${revenue_from_discounted:,.2f}")

if "price" in sales.columns and "no_of_orders" in sales.columns:
    revenue_from_price_times_orders = (sales["price"] * sales["no_of_orders"]).sum()
    st.write(f"Total from price × no_of_orders: ${revenue_from_price_times_orders:,.2f}")

if "discounted_price" in sales.columns and "no_of_orders" in sales.columns:
    revenue_from_discounted_times_orders = (sales["discounted_price"] * sales["no_of_orders"]).sum()
    st.write(f"Total from discounted_price × no_of_orders: ${revenue_from_discounted_times_orders:,.2f}")

# ---------------- SPEND DATA ----------------
st.write("## Step 5: Spend Data Analysis")

all_spend_data = []
for sheet_name, sheet_data in spend_sheets:
    st.write(f"### Processing: {sheet_name}")
    st.write(f"Shape: {sheet_data.shape}")
    st.write(f"Columns (raw): {list(sheet_data.columns)}")
    
    # Normalize
    sheet_data = normalize_columns(sheet_data)
    st.write(f"Columns (normalized): {list(sheet_data.columns)}")
    
    # Show sample
    st.write("Sample:")
    st.dataframe(sheet_data.head(3))
    
    all_spend_data.append(sheet_data)

# Combine all spend data
if all_spend_data:
    channel_spend = pd.concat(all_spend_data, ignore_index=True)
    st.write(f"✅ Combined spend data: {channel_spend.shape}")
    
    # Clean channel names in spend data
    if "channel" in channel_spend.columns:
        st.write("### Spend Channels BEFORE cleaning:")
        st.write(sorted(channel_spend["channel"].dropna().astype(str).unique()))
        
        channel_spend["channel"] = (
            channel_spend["channel"]
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
        channel_spend["channel"] = channel_spend["channel"].str.replace(r'_ebay$', '_eBay', case=False, regex=True)
        channel_spend["channel"] = channel_spend["channel"].str.replace(r'_Ebay$', '_eBay', case=False, regex=True)
        
        st.write("### Spend Channels AFTER cleaning:")
        st.write(sorted(channel_spend["channel"].dropna().unique()))
    
    # Find spend column
    st.write("### Looking for spend column...")
    ad_spend_column = None
    possible_names = ["spend", "ad_spend", "adspend", "cost", "ad_cost", "advertising_spend"]
    for col_name in possible_names:
        if col_name in channel_spend.columns:
            ad_spend_column = col_name
            st.write(f"✅ Found: {col_name}")
            break
    
    if ad_spend_column:
        # Convert date
        channel_spend["date"] = pd.to_datetime(
            channel_spend["date"],
            format="mixed",
            errors="coerce"
        )
        channel_spend = channel_spend.dropna(subset=["date"])
        
        # Convert spend
        channel_spend["ad_spend"] = pd.to_numeric(
            channel_spend[ad_spend_column],
            errors="coerce"
        ).fillna(0)
        
        st.write(f"Date range: {channel_spend['date'].min()} to {channel_spend['date'].max()}")
    else:
        st.error(f"❌ No spend column found! Available: {list(channel_spend.columns)}")
        channel_spend["ad_spend"] = 0
else:
    st.error("❌ No spend data found!")
    channel_spend = pd.DataFrame(columns=["date", "channel", "ad_spend"])

# ---------------- CALCULATE TOTALS (NO FILTERS) ----------------
st.write("## Step 6: UNFILTERED Totals (All Data)")

total_orders_all = sales["no_of_orders"].sum()
total_revenue_all = sales["revenue"].sum()  # Changed from discounted_price to revenue
total_commission_all = sales["selling_commission"].sum()
total_spend_all = channel_spend["ad_spend"].sum() if len(channel_spend) > 0 and "ad_spend" in channel_spend.columns else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Orders (ALL)", f"{total_orders_all:,.0f}")
with col2:
    st.metric("Revenue (ALL)", f"${total_revenue_all:,.0f}")
with col3:
    st.metric("Commission (ALL)", f"${total_commission_all:,.0f}")
with col4:
    st.metric("Ad Spend (ALL)", f"${total_spend_all:,.0f}")

# By channel breakdown
st.write("### Breakdown by Channel (ALL DATA):")
channel_all = sales.groupby("channel").agg({
    "no_of_orders": "sum",
    "revenue": "sum",  # Changed from discounted_price
    "selling_commission": "sum"
}).reset_index()

if len(channel_spend) > 0 and "ad_spend" in channel_spend.columns:
    spend_by_channel = channel_spend.groupby("channel").agg({
        "ad_spend": "sum"
    }).reset_index()
    channel_all = channel_all.merge(spend_by_channel, on="channel", how="left")
    channel_all["ad_spend"] = channel_all["ad_spend"].fillna(0)
else:
    channel_all["ad_spend"] = 0

channel_all["net_earning"] = (
    channel_all["revenue"] -  # Changed from discounted_price
    channel_all["ad_spend"] - 
    channel_all["selling_commission"]
)

st.dataframe(channel_all)

# ---------------- FILTER BY DATE ----------------
st.write("## Step 7: Apply Date Filter")

# Date filter
date_col1, date_col2 = st.columns(2)
with date_col1:
    start_date_input = st.date_input(
        "Start Date",
        value=pd.to_datetime("2026-01-01").date()
    )
with date_col2:
    end_date_input = st.date_input(
        "End Date",
        value=pd.to_datetime("2026-02-02").date()
    )

start_date = pd.to_datetime(start_date_input)
end_date = pd.to_datetime(end_date_input)

st.write(f"Filter: {start_date.date()} to {end_date.date()}")

# Filter sales
sales_filtered = sales[
    sales["purchased_on"].between(start_date, end_date)
].copy()

st.write(f"Filtered rows: {len(sales_filtered):,} out of {len(sales):,}")

# Filter spend
spend_filtered = channel_spend[
    channel_spend["date"].between(start_date, end_date)
].copy() if len(channel_spend) > 0 and "date" in channel_spend.columns else pd.DataFrame()

st.write(f"Filtered spend rows: {len(spend_filtered):,} out of {len(channel_spend):,}")

# ---------------- CALCULATE FILTERED TOTALS ----------------
st.write("## Step 8: FILTERED Totals")

total_orders_filt = sales_filtered["no_of_orders"].sum()
total_revenue_filt = sales_filtered["revenue"].sum()  # Changed from discounted_price
total_commission_filt = sales_filtered["selling_commission"].sum()

# Calculate ad spend using the BI-correct method (daily join)
sales_daily = (
    sales_filtered
    .groupby(
        [pd.Grouper(key="purchased_on", freq="D"), "channel"],
        as_index=False
    )
    .agg({"revenue": "sum"})  # Changed from discounted_price
)

if len(spend_filtered) > 0:
    spend_daily = (
        spend_filtered
        .groupby(["date", "channel"], as_index=False)
        .agg({"ad_spend": "sum"})
    )
    
    merged = sales_daily.merge(
        spend_daily,
        left_on=["purchased_on", "channel"],
        right_on=["date", "channel"],
        how="left"
    )
    
    total_spend_filt = merged["ad_spend"].sum(skipna=True)
else:
    total_spend_filt = 0

net_earning_filt = total_revenue_filt - total_spend_filt - total_commission_filt

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Orders", f"{total_orders_filt:,.0f}")
with col2:
    st.metric("Revenue", f"${total_revenue_filt:,.0f}")
with col3:
    st.metric("Commission", f"${total_commission_filt:,.0f}")
with col4:
    st.metric("Ad Spend", f"${total_spend_filt:,.0f}")
with col5:
    st.metric("Net Earning", f"${net_earning_filt:,.0f}")

# Breakdown by channel
st.write("### Breakdown by Channel (FILTERED):")
channel_filt = sales_filtered.groupby("channel").agg({
    "no_of_orders": "sum",
    "revenue": "sum",  # Changed from discounted_price
    "selling_commission": "sum"
}).reset_index()

if len(spend_filtered) > 0:
    spend_by_channel_filt = spend_filtered.groupby("channel").agg({
        "ad_spend": "sum"
    }).reset_index()
    channel_filt = channel_filt.merge(spend_by_channel_filt, on="channel", how="left")
    channel_filt["ad_spend"] = channel_filt["ad_spend"].fillna(0)
else:
    channel_filt["ad_spend"] = 0

channel_filt["net_earning"] = (
    channel_filt["revenue"] -  # Changed from discounted_price
    channel_filt["ad_spend"] - 
    channel_filt["selling_commission"]
)

st.dataframe(channel_filt)

# ---------------- COMPARISON WITH YOUR BI ----------------
st.write("## Step 9: Compare with YOUR BI Numbers")

st.write("### Enter your BI numbers for the SAME date range:")

bi_col1, bi_col2, bi_col3 = st.columns(3)
with bi_col1:
    bi_orders = st.number_input("BI Orders", value=606, step=1)
    bi_revenue = st.number_input("BI Revenue", value=346100.0, step=0.01)
with bi_col2:
    bi_commission = st.number_input("BI Commission", value=38900.0, step=0.01)
    bi_spend = st.number_input("BI Ad Spend", value=94040.0, step=0.01)
with bi_col3:
    bi_net = st.number_input("BI Net Earning", value=81640.0, step=0.01)

st.write("### 🔍 Comparison:")

comparison_data = {
    "Metric": ["Orders", "Revenue", "Commission", "Ad Spend", "Net Earning"],
    "This Dashboard": [
        f"{total_orders_filt:,.0f}",
        f"${total_revenue_filt:,.2f}",
        f"${total_commission_filt:,.2f}",
        f"${total_spend_filt:,.2f}",
        f"${net_earning_filt:,.2f}"
    ],
    "Your BI": [
        f"{bi_orders:,.0f}",
        f"${bi_revenue:,.2f}",
        f"${bi_commission:,.2f}",
        f"${bi_spend:,.2f}",
        f"${bi_net:,.2f}"
    ],
    "Difference": [
        f"{total_orders_filt - bi_orders:,.0f}",
        f"${total_revenue_filt - bi_revenue:,.2f}",
        f"${total_commission_filt - bi_commission:,.2f}",
        f"${total_spend_filt - bi_spend:,.2f}",
        f"${net_earning_filt - bi_net:,.2f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

# ---------------- EXPORT RAW DATA ----------------
st.write("## Step 10: Export for Manual Verification")

if st.button("📥 Export Filtered Sales Data to CSV"):
    csv = sales_filtered.to_csv(index=False)
    st.download_button(
        label="Download Filtered Sales CSV",
        data=csv,
        file_name=f"debug_sales_{start_date.date()}_{end_date.date()}.csv",
        mime="text/csv"
    )

if len(spend_filtered) > 0:
    if st.button("📥 Export Filtered Spend Data to CSV"):
        csv_spend = spend_filtered.to_csv(index=False)
        st.download_button(
            label="Download Filtered Spend CSV",
            data=csv_spend,
            file_name=f"debug_spend_{start_date.date()}_{end_date.date()}.csv",
            mime="text/csv"
        )