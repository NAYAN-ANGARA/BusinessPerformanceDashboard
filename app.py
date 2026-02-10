import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets

# Configure Plotly to work offline (fixes network access issues)
import plotly.io as pio
pio.renderers.default = "browser"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Business Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add mobile viewport meta tag for proper scaling
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

# ---------------- CONSTANTS ----------------
SAFE_MARGIN = 0.62  # 62% margin

# ---------------- CSS ----------------
st.markdown("""
<style>
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.kpi-title { 
    font-size: 13px; 
    color: rgba(255,255,255,0.9);
    margin-bottom: 8px;
    font-weight: 500;
}
.kpi-value { 
    font-size: 32px; 
    font-weight: 700; 
    margin-bottom: 4px;
}
.kpi-change {
    font-size: 12px;
    margin-top: 4px;
    font-weight: 600;
}
.positive { color: #10b981; }
.negative { color: #ef4444; }

/* Mobile Responsive Styles */
@media only screen and (max-width: 768px) {
    .kpi-value {
        font-size: 24px !important;
    }
    .kpi-title {
        font-size: 12px !important;
    }
    .kpi-card {
        padding: 12px !important;
        margin-bottom: 8px !important;
    }
}
</style>
""", unsafe_allow_html=True)

def kpi(title, value, change=None):
    change_html = ""
    if change is not None:
        color = "positive" if change >= 0 else "negative"
        arrow = "↑" if change >= 0 else "↓"
        change_html = f'<div class="kpi-change {color}">{arrow} {abs(change):.1f}%</div>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

def multiselect_with_all(label, options):
    ALL = "All"
    opts = [ALL] + sorted(list(options))
    selected = st.sidebar.multiselect(label, opts, default=[ALL])
    if ALL in selected or not selected:
        return list(options)
    return selected

# ---------------- LOAD DATA ----------------
@st.cache_data(show_spinner=True, ttl=300)
def load_data():
    """Load data from both spreadsheets using the same credentials."""
    all_data = {}
    
    credentials_file = "secret-envoy-486405-j3-03851d061385.json"
    
    # Spreadsheet 1: USA - DB for Marketplace Dashboard
    try:
        data1 = load_all_sheets(
            credentials_file,
            "USA - DB for Marketplace Dashboard"
        )
        if data1:
            for sheet_name, df in data1.items():
                all_data[f"USA_{sheet_name}"] = df
    except Exception as e:
        st.warning(f"⚠️ Could not load 'USA - DB for Marketplace Dashboard': {str(e)}")
    
    # Spreadsheet 2: IB Marketplace
    try:
        data2 = load_all_sheets(
            credentials_file,
            "IB - Database for Marketplace Dashboard"
        )
        if data2:
            for sheet_name, df in data2.items():
                all_data[f"IB_{sheet_name}"] = df
    except Exception as e:
        st.warning(f"⚠️ Could not load 'IB Marketplace': {str(e)}")
    
    if not all_data:
        st.error("❌ No data loaded from any spreadsheet. Please check your configuration.")
        return None
    
    return all_data

# ---------------- HEADER ----------------
col1, col2 = st.columns([6, 1])
with col1:
    st.title("📊 Business Performance Dashboard")
with col2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

data = load_data()

if data is None:
    st.stop()

# ---------------- GET REQUIRED SHEETS ----------------
# Collect all sales data from both spreadsheets
sales_sheets = []
sales_sheet_keywords = ["sales_data", "sales data", "sales"]

for sheet_name, df in data.items():
    sheet_lower = sheet_name.lower()
    if any(keyword in sheet_lower for keyword in sales_sheet_keywords):
        if not df.empty and len(df) > 0:
            sales_sheets.append(df.copy())

if not sales_sheets:
    st.error("❌ No sales data sheets found")
    st.info(f"Available sheets: {', '.join(data.keys())}")
    st.stop()

# Combine all sales data
try:
    sales = pd.concat(sales_sheets, ignore_index=True)
except Exception as e:
    st.error(f"❌ Error combining sales data: {str(e)}")
    st.stop()

# ---------------- LOAD SPEND SHEETS ----------------
spend_sheets = []
spend_sheet_info = []

st.write("### 🔍 DEBUG: Sheet Detection")
st.write("**All available sheets:**", sorted(list(data.keys())))

# FIXED: Better detection for spend sheets
for sheet_name in data.keys():
    sheet_lower = sheet_name.lower()
    
    # For USA: ONLY use channel_spend_data (skip generic spend_data)
    if 'usa' in sheet_lower:
        if 'channel_spend_data' in sheet_lower:
            df = data[sheet_name].copy()
            if not df.empty and len(df) > 0:
                st.success(f"✓ Loading {sheet_name}: {len(df)} rows, columns: {list(df.columns)[:5]}")
                spend_sheets.append(df)
                spend_sheet_info.append({'name': sheet_name, 'rows': len(df), 'columns': list(df.columns)})
    
    # For IB: Look for spend or marketplace sheets  
    elif 'ib' in sheet_lower:
        if any(keyword in sheet_lower for keyword in ['spend', 'channel_spend']):
            df = data[sheet_name].copy()
            if not df.empty and len(df) > 0:
                st.success(f"✓ Loading {sheet_name}: {len(df)} rows, columns: {list(df.columns)[:5]}")
                spend_sheets.append(df)
                spend_sheet_info.append({'name': sheet_name, 'rows': len(df), 'columns': list(df.columns)})

st.write(f"**Total spend sheets loaded:** {len(spend_sheets)}")

# Combine spend sheets
if spend_sheets:
    channel_spend = pd.concat(spend_sheets, ignore_index=True)
    st.write(f"**Combined spend data:** {len(channel_spend):,} rows")
else:
    channel_spend = pd.DataFrame(columns=["date", "channel", "spend"])
    st.error("❌ No spend sheets were loaded!")

# ---------------- NORMALIZE COLUMNS ----------------
def normalize_columns(df):
    """Normalize column names to lowercase with underscores."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

sales = normalize_columns(sales)

# Clean channel names in sales
if "channel" in sales.columns:
    sales["channel"] = (
        sales["channel"]
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )
    sales["channel"] = sales["channel"].str.replace(r'_ebay$', '_eBay', case=False, regex=True)

# Normalize spend data
if len(channel_spend) > 0:
    st.write("### 🔍 DEBUG: Before Normalization")
    st.write(f"Columns before: {list(channel_spend.columns)}")
    
    channel_spend = normalize_columns(channel_spend)
    
    st.write(f"Columns after normalization: {list(channel_spend.columns)}")
    
    # Find spend column (handles different naming)
    spend_col = None
    for col in ["ad_spend", "spend", "advertising_spend", "cost"]:
        if col in channel_spend.columns:
            spend_col = col
            st.success(f"✓ Found spend column: '{col}'")
            break
    
    if spend_col and spend_col != "ad_spend":
        channel_spend["ad_spend"] = channel_spend[spend_col]
        st.info(f"Renamed '{spend_col}' to 'ad_spend'")
    elif not spend_col:
        st.error("❌ No spend column found!")
        st.write("Available columns:", list(channel_spend.columns))
        channel_spend["ad_spend"] = 0
    
    # Show sample of spend data
    st.write("### 📊 Sample Spend Data (first 5 rows)")
    st.dataframe(channel_spend.head())
    
    # Clean channel names in spend
    if "channel" in channel_spend.columns:
        channel_spend["channel"] = (
            channel_spend["channel"]
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
        channel_spend["channel"] = channel_spend["channel"].str.replace(r'_ebay$', '_eBay', case=False, regex=True)

# ---------------- TYPE CAST SALES ----------------
required_sales_cols = ["purchased_on", "no_of_orders", "discounted_price", "channel"]
missing_cols = [col for col in required_sales_cols if col not in sales.columns]

if missing_cols:
    st.error(f"❌ Missing required columns in sales data: {', '.join(missing_cols)}")
    st.stop()

sales["purchased_on"] = pd.to_datetime(sales["purchased_on"], format="mixed", errors="coerce")
sales = sales.dropna(subset=["purchased_on"])

if len(sales) == 0:
    st.error("❌ No valid dates found in sales data")
    st.stop()

sales["no_of_orders"] = pd.to_numeric(sales["no_of_orders"], errors="coerce").fillna(0)

# Revenue
sales["discounted_price"] = (
    sales["discounted_price"].astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
sales["discounted_price"] = pd.to_numeric(sales["discounted_price"], errors="coerce").fillna(0)
sales["revenue"] = sales["discounted_price"]

# Commission
if "selling_commission" not in sales.columns:
    sales["selling_commission"] = 0
else:
    sales["selling_commission"] = (
        sales["selling_commission"].astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    sales["selling_commission"] = pd.to_numeric(sales["selling_commission"], errors="coerce").fillna(0)

# Type
if "type" not in sales.columns:
    sales["type"] = "Unknown"

# ---------------- SPEND DATA TYPE CASTING ----------------
if len(channel_spend) > 0:
    st.write("### 🔍 DEBUG: Processing Spend Data")
    
    # Handle date column
    date_col = None
    for col in ["date", "purchased_on"]:
        if col in channel_spend.columns:
            date_col = col
            st.info(f"Using date column: '{col}'")
            break
    
    if date_col:
        channel_spend["date"] = pd.to_datetime(channel_spend[date_col], format="mixed", errors="coerce")
        invalid_dates = channel_spend["date"].isna().sum()
        channel_spend = channel_spend.dropna(subset=["date"])
        if invalid_dates > 0:
            st.warning(f"Removed {invalid_dates} rows with invalid dates")
    
    # Handle ad_spend - ROBUST conversion for both $ and plain numbers
    if "ad_spend" in channel_spend.columns:
        st.write("Before conversion - sample ad_spend values:")
        st.write(channel_spend["ad_spend"].head(10).tolist())
        
        channel_spend["ad_spend"] = channel_spend["ad_spend"].astype(str)
        # Remove ALL currency symbols and formatting
        channel_spend["ad_spend"] = (
            channel_spend["ad_spend"]
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.replace('£', '', regex=False)
            .str.replace('€', '', regex=False)
            .str.strip()
        )
        channel_spend["ad_spend"] = pd.to_numeric(channel_spend["ad_spend"], errors="coerce").fillna(0)
        
        st.write("After conversion - sample ad_spend values:")
        st.write(channel_spend["ad_spend"].head(10).tolist())
        
        total_spend_check = channel_spend["ad_spend"].sum()
        non_zero = (channel_spend["ad_spend"] > 0).sum()
        st.success(f"✓ Ad Spend: {non_zero:,} non-zero rows, Total: ${total_spend_check:,.2f}")

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("📅 Date Range")
min_date = sales["purchased_on"].min().date()
max_date = sales["purchased_on"].max().date()

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")
    st.stop()

st.sidebar.header("🎯 Filters")
available_channels = sales["channel"].unique()
selected_channels = multiselect_with_all("Channels", available_channels)

available_types = sales["type"].unique()
selected_types = multiselect_with_all("Types", available_types)

# Show data info in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Data Info")
    if spend_sheet_info:
        st.markdown("**Spend Sheets:**")
        for info in spend_sheet_info:
            st.caption(f"✓ {info['name']}: {info['rows']:,} rows")
    
    if len(channel_spend) > 0 and "ad_spend" in channel_spend.columns:
        total_spend_all = channel_spend["ad_spend"].sum()
        st.success(f"💰 Total Ad Spend: ${total_spend_all:,.2f}")

# ---------------- FILTER DATA ----------------
start_dt = pd.Timestamp(start_date)
end_dt = pd.Timestamp(end_date)
end_dt_next = end_dt + pd.Timedelta(days=1)

sales_f = sales[
    (sales["purchased_on"] >= start_dt) &
    (sales["purchased_on"] < end_dt_next) &
    (sales["channel"].isin(selected_channels)) &
    (sales["type"].isin(selected_types))
]

if len(sales_f) == 0:
    st.warning("⚠️ No data available for selected filters")
    st.stop()

# ---------------- KPI CALCULATIONS ----------------
total_rev = sales_f["revenue"].sum()
total_orders = sales_f["no_of_orders"].sum()
aov = total_rev / total_orders if total_orders > 0 else 0
total_commission = sales_f["selling_commission"].sum()

# Calculate ad spend
total_spend = 0
if len(channel_spend) > 0 and "date" in channel_spend.columns:
    spend_f = channel_spend[
        (channel_spend["date"] >= start_dt) & 
        (channel_spend["date"] < end_dt_next)
    ]
    if "channel" in channel_spend.columns:
        spend_f = spend_f[spend_f["channel"].isin(selected_channels)]
    total_spend = spend_f["ad_spend"].sum()

net_earning = (total_rev * SAFE_MARGIN) - total_spend - total_commission
roas = total_rev / total_spend if total_spend > 0 else 0
acos = (total_spend / total_rev * 100) if total_rev > 0 else 0

# YoY calculations
year_ago_start = start_dt - pd.DateOffset(years=1)
year_ago_end = end_dt - pd.DateOffset(years=1)

sales_ly = sales[
    (sales["purchased_on"] >= year_ago_start) &
    (sales["purchased_on"] < year_ago_end) &
    (sales["channel"].isin(selected_channels)) &
    (sales["type"].isin(selected_types))
]

ly_rev = sales_ly["revenue"].sum()
ly_orders = sales_ly["no_of_orders"].sum()
ly_commission = sales_ly["selling_commission"].sum()

ly_spend = 0
if len(channel_spend) > 0 and "date" in channel_spend.columns:
    spend_ly = channel_spend[
        (channel_spend["date"] >= year_ago_start) & 
        (channel_spend["date"] < year_ago_end)
    ]
    if "channel" in channel_spend.columns:
        spend_ly = spend_ly[spend_ly["channel"].isin(selected_channels)]
    ly_spend = spend_ly["ad_spend"].sum()

ly_net_earning = (ly_rev * SAFE_MARGIN) - ly_spend - ly_commission

# Calculate percentage changes
rev_change = ((total_rev - ly_rev) / ly_rev * 100) if ly_rev > 0 else 0
orders_change = ((total_orders - ly_orders) / ly_orders * 100) if ly_orders > 0 else 0
spend_change = ((total_spend - ly_spend) / ly_spend * 100) if ly_spend > 0 else 0
net_change = ((net_earning - ly_net_earning) / ly_net_earning * 100) if ly_net_earning != 0 else 0

# ---------------- KPI CARDS ----------------
st.markdown("### 📈 Key Metrics")
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    kpi("Revenue", f"${total_rev:,.0f}", rev_change)
with col2:
    kpi("Orders", f"{total_orders:,.0f}", orders_change)
with col3:
    kpi("AOV", f"${aov:.2f}")
with col4:
    kpi("Ad Spend", f"${total_spend:,.0f}", spend_change)
with col5:
    kpi("Commission", f"${total_commission:,.0f}")
with col6:
    kpi("Net Earning", f"${net_earning:,.0f}", net_change)
with col7:
    kpi("ACOS", f"{acos:.1f}%")

st.markdown("---")

# ---------------- CHARTS ----------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📊 Revenue & Orders Trend")
    trend = (
        sales_f.groupby(pd.Grouper(key="purchased_on", freq="D"))
        .agg({"revenue": "sum", "no_of_orders": "sum"})
        .reset_index()
    )
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=trend["purchased_on"], y=trend["revenue"], name="Revenue",
        line=dict(color="#667eea", width=3), fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    fig1.add_trace(go.Scatter(
        x=trend["purchased_on"], y=trend["no_of_orders"], name="Orders",
        line=dict(color="#10b981", width=3), yaxis="y2"
    ))
    fig1.update_layout(
        yaxis=dict(title="Revenue ($)"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        hovermode="x unified", 
        template="plotly_white", 
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

with col_right:
    st.markdown("### 🛒 Revenue by Channel")
    channel_rev = (
        sales_f.groupby("channel").agg({"revenue": "sum"})
        .reset_index().sort_values("revenue", ascending=False)
    )
    fig2 = px.bar(channel_rev, x="channel", y="revenue", color="revenue",
                  color_continuous_scale=["#667eea", "#764ba2"])
    fig2.update_layout(
        showlegend=False, 
        template="plotly_white", 
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("### 💰 Ad Spend vs Revenue")
    channel_metrics = sales_f.groupby("channel").agg({"revenue": "sum"}).reset_index()
    
    if len(channel_spend) > 0 and "channel" in channel_spend.columns and "date" in channel_spend.columns:
        spend_by_channel = (
            channel_spend[(channel_spend["date"] >= start_dt) & (channel_spend["date"] < end_dt_next)]
            .groupby("channel").agg({"ad_spend": "sum"}).reset_index()
        )
        channel_comparison = channel_metrics.merge(spend_by_channel, on="channel", how="left")
        channel_comparison["ad_spend"] = channel_comparison["ad_spend"].fillna(0)
    else:
        channel_comparison = channel_metrics
        channel_comparison["ad_spend"] = 0
    
    channel_comparison = channel_comparison.sort_values("revenue", ascending=False)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=channel_comparison["channel"], y=channel_comparison["revenue"],
                          name="Revenue", marker_color="#667eea"))
    fig3.add_trace(go.Bar(x=channel_comparison["channel"], y=channel_comparison["ad_spend"],
                          name="Ad Spend", marker_color="#ef4444"))
    fig3.update_layout(
        barmode="group", 
        template="plotly_white", 
        height=400, 
        yaxis_title="Amount ($)",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

with col_right2:
    st.markdown("### 📦 Revenue by Type")
    type_rev = sales_f.groupby("type").agg({"revenue": "sum"}).reset_index()
    fig4 = px.pie(type_rev, values="revenue", names="type", hole=0.4,
                  color_discrete_sequence=px.colors.sequential.Purples_r)
    fig4.update_layout(
        template="plotly_white", 
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})

# ---------------- DATA TABLE ----------------
st.markdown("---")
st.markdown("### 📋 Detailed Data")

summary = (
    sales_f.groupby("channel")
    .agg({"no_of_orders": "sum", "revenue": "sum", "selling_commission": "sum"})
    .reset_index()
)

if len(channel_spend) > 0 and "channel" in channel_spend.columns and "date" in channel_spend.columns:
    spend_summary = (
        channel_spend[(channel_spend["date"] >= start_dt) & (channel_spend["date"] < end_dt_next)]
        .groupby("channel").agg({"ad_spend": "sum"}).reset_index()
    )
    summary = summary.merge(spend_summary, on="channel", how="left")
    summary["ad_spend"] = summary["ad_spend"].fillna(0)
else:
    summary["ad_spend"] = 0

summary["aov"] = summary["revenue"] / summary["no_of_orders"]
summary["net_earning"] = (summary["revenue"] * SAFE_MARGIN) - summary["ad_spend"] - summary["selling_commission"]
summary["roas"] = summary.apply(lambda row: row["revenue"] / row["ad_spend"] if row["ad_spend"] > 0 else 0, axis=1)
summary["acos"] = summary.apply(lambda row: (row["ad_spend"] / row["revenue"] * 100) if row["revenue"] > 0 else 0, axis=1)
summary = summary.sort_values("revenue", ascending=False)

summary.columns = ["Channel", "Orders", "Revenue", "Commission", "Ad Spend", "AOV", "Net Earning", "ROAS", "ACOS"]

st.dataframe(
    summary.style.format({
        "Orders": "{:.0f}", "Revenue": "${:,.0f}", "Commission": "${:,.0f}",
        "Ad Spend": "${:,.0f}", "AOV": "${:,.2f}", "Net Earning": "${:,.0f}",
        "ROAS": "{:.2f}x", "ACOS": "{:.1f}%"
    }),
    use_container_width=True, height=400
)

# ---------------- EXPORT ----------------
st.markdown("---")
csv = summary.to_csv(index=False)
st.download_button(
    label="📥 Export to CSV",
    data=csv,
    file_name=f"dashboard_{start_date}_{end_date}.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | {start_date} to {end_date} | Margin: {SAFE_MARGIN*100:.0f}%")