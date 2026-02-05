import streamlit as st
import pandas as pd
from gsheets import load_all_sheets

st.title("🔍 January 2026 Data Audit - Finding Discrepancies")

# Load data
@st.cache_data
def load_data():
    return load_all_sheets(
        "secret-envoy-486405-j3-03851d061385.json",
        "USA - DB for Marketplace Dashboard"
    )

data = load_data()
sales = data["Sales_data"].copy()

# Normalize columns
def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df

sales = normalize_columns(sales)

st.write("## Step 1: Raw Data Analysis")
st.write(f"Total rows in Sales_data: {len(sales):,}")
st.write("Columns:", list(sales.columns))

# Show sample of raw data
st.write("### Sample Raw Data (first 5 rows):")
st.dataframe(sales.head())

# Convert date
st.write("## Step 2: Date Conversion")
st.write(f"Date column before conversion (sample): {sales['purchased_on'].head().tolist()}")

sales["purchased_on"] = pd.to_datetime(sales["purchased_on"], format="mixed", errors="coerce")
invalid_dates = sales["purchased_on"].isna().sum()
st.write(f"⚠️ Invalid dates found: {invalid_dates}")

sales_before_drop = len(sales)
sales = sales.dropna(subset=["purchased_on"])
sales_after_drop = len(sales)
st.write(f"Rows dropped due to invalid dates: {sales_before_drop - sales_after_drop}")

# Show date range
st.write(f"Date range in data: {sales['purchased_on'].min()} to {sales['purchased_on'].max()}")

# Filter January 2026
st.write("## Step 3: January 2026 Data")
jan_2026_start = pd.to_datetime("2026-01-01")
jan_2026_end = pd.to_datetime("2026-01-31")

sales_jan = sales[
    (sales["purchased_on"] >= jan_2026_start) & 
    (sales["purchased_on"] <= jan_2026_end)
].copy()

st.write(f"### Rows in January 2026: {len(sales_jan):,}")

if len(sales_jan) == 0:
    st.error("❌ NO DATA FOUND FOR JANUARY 2026!")
    st.write("Checking what dates ARE present:")
    st.write(sales.groupby(sales["purchased_on"].dt.to_period("M")).size())
    st.stop()

# Show daily breakdown
st.write("### Daily Breakdown:")
daily = sales_jan.groupby(sales_jan["purchased_on"].dt.date).size().reset_index()
daily.columns = ["Date", "Row Count"]
st.dataframe(daily)

# Clean channel names
st.write("## Step 4: Channel Cleaning")
st.write("### Channels BEFORE cleaning:")
st.write(sorted(sales_jan["channel"].dropna().unique()))

sales_jan["channel"] = (
    sales_jan["channel"]
    .astype(str)
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)
sales_jan["channel"] = sales_jan["channel"].str.replace(r'_ebay$', '_eBay', case=False, regex=True)
sales_jan["channel"] = sales_jan["channel"].str.replace(r'_Ebay$', '_eBay', case=False, regex=True)

st.write("### Channels AFTER cleaning:")
st.write(sorted(sales_jan["channel"].dropna().unique()))

# Clean numeric fields
st.write("## Step 5: Numeric Field Conversion")

# Orders
st.write("### no_of_orders:")
st.write(f"Sample raw values: {sales_jan['no_of_orders'].head(10).tolist()}")
st.write(f"Data type: {sales_jan['no_of_orders'].dtype}")

sales_jan["no_of_orders"] = pd.to_numeric(sales_jan["no_of_orders"], errors="coerce").fillna(0)
st.write(f"After conversion - Total orders: {sales_jan['no_of_orders'].sum():,.0f}")

# Discounted Price
st.write("### discounted_price:")
st.write(f"Sample raw values: {sales_jan['discounted_price'].head(10).tolist()}")
st.write(f"Data type: {sales_jan['discounted_price'].dtype}")

sales_jan["discounted_price"] = (
    sales_jan["discounted_price"].astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
sales_jan["discounted_price"] = pd.to_numeric(sales_jan["discounted_price"], errors="coerce").fillna(0)

st.write(f"After conversion - Sum of discounted_price: ${sales_jan['discounted_price'].sum():,.2f}")

# Revenue calculation
sales_jan["revenue"] = sales_jan["discounted_price"] * sales_jan["no_of_orders"]
st.write(f"**Calculated Revenue (price × orders): ${sales_jan['revenue'].sum():,.2f}**")

# Commission
st.write("### selling_commission:")
commission_col = None
for col in ["selling_commission", "commission", "seller_commission"]:
    if col in sales_jan.columns:
        commission_col = col
        st.write(f"Found commission column: '{commission_col}'")
        break

if commission_col:
    st.write(f"Sample raw values: {sales_jan[commission_col].head(10).tolist()}")
    
    sales_jan["selling_commission"] = (
        sales_jan[commission_col].astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    sales_jan["selling_commission"] = pd.to_numeric(sales_jan["selling_commission"], errors="coerce").fillna(0)
    st.write(f"After conversion - Total commission: ${sales_jan['selling_commission'].sum():,.2f}")
else:
    sales_jan["selling_commission"] = 0
    st.write("No commission column found!")

# Load spend data
st.write("## Step 6: Ad Spend Data")

spend_sheets = []
for sheet_name in data.keys():
    if 'spend' in sheet_name.lower():
        st.write(f"Found spend sheet: {sheet_name}")
        spend_sheets.append(data[sheet_name].copy())

if spend_sheets:
    channel_spend = pd.concat(spend_sheets, ignore_index=True)
    channel_spend = normalize_columns(channel_spend)
    
    st.write(f"Total spend rows: {len(channel_spend):,}")
    st.write("Spend columns:", list(channel_spend.columns))
    
    # Clean channels in spend
    if "channel" in channel_spend.columns:
        channel_spend["channel"] = (
            channel_spend["channel"].astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
        channel_spend["channel"] = channel_spend["channel"].str.replace(r'_ebay$', '_eBay', case=False, regex=True)
        channel_spend["channel"] = channel_spend["channel"].str.replace(r'_Ebay$', '_eBay', case=False, regex=True)
    
    # Convert date
    channel_spend["date"] = pd.to_datetime(channel_spend["date"], format="mixed", errors="coerce")
    channel_spend = channel_spend.dropna(subset=["date"])
    
    # Find ad spend column
    ad_spend_col = None
    for col in ["spend", "ad_spend", "adspend", "cost"]:
        if col in channel_spend.columns:
            ad_spend_col = col
            st.write(f"Found ad spend column: '{col}'")
            break
    
    if ad_spend_col:
        channel_spend["ad_spend"] = pd.to_numeric(channel_spend[ad_spend_col], errors="coerce").fillna(0)
        
        # Filter to January 2026
        spend_jan = channel_spend[
            (channel_spend["date"] >= jan_2026_start) & 
            (channel_spend["date"] <= jan_2026_end)
        ]
        
        st.write(f"Spend rows in January 2026: {len(spend_jan):,}")
        st.write(f"Total ad spend (simple sum): ${spend_jan['ad_spend'].sum():,.2f}")
        
        # Show by channel
        st.write("### Ad Spend by Channel:")
        spend_by_ch = spend_jan.groupby("channel")["ad_spend"].sum().reset_index()
        st.dataframe(spend_by_ch)
        
        # BI-Correct method (daily join)
        st.write("### Using BI-Correct Method (Daily Join):")
        sales_daily = (
            sales_jan.groupby([pd.Grouper(key="purchased_on", freq="D"), "channel"], as_index=False)
            .agg({"revenue": "sum"})
        )
        
        spend_daily = (
            spend_jan.groupby(["date", "channel"], as_index=False)
            .agg({"ad_spend": "sum"})
        )
        
        merged = sales_daily.merge(
            spend_daily,
            left_on=["purchased_on", "channel"],
            right_on=["date", "channel"],
            how="left"
        )
        
        ad_spend_total = merged["ad_spend"].sum(skipna=True)
        st.write(f"**Total ad spend (BI method): ${ad_spend_total:,.2f}**")
else:
    ad_spend_total = 0
    st.write("No spend data found!")

# Final calculations
st.write("## Step 7: Final January 2026 Totals")

SAFE_MARGIN = 0.62

orders_total = sales_jan["no_of_orders"].sum()
revenue_total = sales_jan["revenue"].sum()
commission_total = sales_jan["selling_commission"].sum()
net_earning_total = (revenue_total * SAFE_MARGIN) - ad_spend_total - commission_total

st.write("### 📊 STREAMLIT CALCULATIONS:")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Orders", f"{orders_total:,.0f}")
    st.metric("Revenue", f"${revenue_total:,.2f}")
with col2:
    st.metric("Commission", f"${commission_total:,.2f}")
    st.metric("Ad Spend", f"${ad_spend_total:,.2f}")
with col3:
    st.metric("Net Earning", f"${net_earning_total:,.2f}")
    st.metric("AOV", f"${revenue_total/orders_total:,.2f}" if orders_total > 0 else "$0.00")

st.write("---")

# Breakdown by channel
st.write("## Step 8: Breakdown by Channel")

summary = (
    sales_jan.groupby("channel")
    .agg({
        "no_of_orders": "sum",
        "revenue": "sum",
        "selling_commission": "sum"
    })
    .reset_index()
)

if len(spend_jan) > 0:
    spend_summary = spend_jan.groupby("channel")["ad_spend"].sum().reset_index()
    summary = summary.merge(spend_summary, on="channel", how="left")
    summary["ad_spend"] = summary["ad_spend"].fillna(0)
else:
    summary["ad_spend"] = 0

summary["net_earning"] = (summary["revenue"] * SAFE_MARGIN) - summary["ad_spend"] - summary["selling_commission"]
summary["aov"] = summary["revenue"] / summary["no_of_orders"]

st.dataframe(summary.style.format({
    "no_of_orders": "{:.0f}",
    "revenue": "${:,.2f}",
    "selling_commission": "${:,.2f}",
    "ad_spend": "${:,.2f}",
    "net_earning": "${:,.2f}",
    "aov": "${:,.2f}"
}))

# Input BI values
st.write("---")
st.write("## Step 9: 🎯 Compare with YOUR BI")

st.write("### Enter your BI values for January 2026:")

col1, col2, col3 = st.columns(3)
with col1:
    bi_orders = st.number_input("BI Orders", value=0, step=1)
    bi_revenue = st.number_input("BI Revenue", value=0.0, step=0.01)
with col2:
    bi_commission = st.number_input("BI Commission", value=0.0, step=0.01)
    bi_ad_spend = st.number_input("BI Ad Spend", value=0.0, step=0.01)
with col3:
    bi_net_earning = st.number_input("BI Net Earning", value=0.0, step=0.01)

if bi_orders > 0 or bi_revenue > 0:
    st.write("### 🔍 Comparison:")
    
    comparison = pd.DataFrame({
        "Metric": ["Orders", "Revenue", "Commission", "Ad Spend", "Net Earning"],
        "Streamlit": [
            f"{orders_total:,.0f}",
            f"${revenue_total:,.2f}",
            f"${commission_total:,.2f}",
            f"${ad_spend_total:,.2f}",
            f"${net_earning_total:,.2f}"
        ],
        "Your BI": [
            f"{bi_orders:,.0f}",
            f"${bi_revenue:,.2f}",
            f"${bi_commission:,.2f}",
            f"${bi_ad_spend:,.2f}",
            f"${bi_net_earning:,.2f}"
        ],
        "Difference": [
            f"{orders_total - bi_orders:,.0f}",
            f"${revenue_total - bi_revenue:,.2f}",
            f"${commission_total - bi_commission:,.2f}",
            f"${ad_spend_total - bi_ad_spend:,.2f}",
            f"${net_earning_total - bi_net_earning:,.2f}"
        ],
        "% Diff": [
            f"{((orders_total - bi_orders) / bi_orders * 100):.1f}%" if bi_orders > 0 else "N/A",
            f"{((revenue_total - bi_revenue) / bi_revenue * 100):.1f}%" if bi_revenue > 0 else "N/A",
            f"{((commission_total - bi_commission) / bi_commission * 100):.1f}%" if bi_commission > 0 else "N/A",
            f"{((ad_spend_total - bi_ad_spend) / bi_ad_spend * 100):.1f}%" if bi_ad_spend > 0 else "N/A",
            f"{((net_earning_total - bi_net_earning) / bi_net_earning * 100):.1f}%" if bi_net_earning > 0 else "N/A"
        ]
    })
    
    st.dataframe(comparison, use_container_width=True)

# Export January data
st.write("---")
st.write("## Step 10: Export January 2026 Data")

if st.button("📥 Export January 2026 Sales to CSV"):
    csv = sales_jan.to_csv(index=False)
    st.download_button(
        label="Download January Sales Data",
        data=csv,
        file_name="january_2026_sales.csv",
        mime="text/csv"
    )

st.write("---")
st.write("## 🔍 Potential Issues to Check:")

st.write("""
1. **Date Format Issues:**
   - Is BI using inclusive dates (Jan 1 - Jan 31)?
   - Or exclusive (Jan 1 - Feb 1)?
   
2. **Revenue Calculation:**
   - BI might be using different formula
   - Check if BI multiplies discounted_price × no_of_orders
   
3. **Commission:**
   - Are there missing commission values?
   - Check if BI calculates commission differently
   
4. **Ad Spend:**
   - BI might not use daily join method
   - Simple sum vs daily matching could differ
   
5. **Data Filtering:**
   - Does BI filter by channel or type?
   - Check for any hidden filters in BI
   
6. **Return Status:**
   - Does BI exclude returns/refunds?
   - Check 'return_status' column
   
7. **Duplicate Rows:**
   - Check if same order_id appears multiple times
   - BI might be deduplicating
""")
