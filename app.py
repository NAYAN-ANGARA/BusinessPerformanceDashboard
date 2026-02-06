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
    background: linear-gradient(135deg, #1f2937, #111827);
    border-radius: 16px;
    padding: 20px;
    color: white;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
.kpi-title { 
    font-size: 14px; 
    color: #9ca3af; 
    margin-bottom: 8px;
}
.kpi-value { 
    font-size: 34px; 
    font-weight: 700; 
    margin-bottom: 4px;
}
.kpi-change {
    font-size: 12px;
    margin-top: 4px;
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
    /* Make streamlit columns stack on mobile */
    .stColumn {
        width: 100% !important;
        flex: 0 0 100% !important;
    }
    /* Adjust chart heights for mobile */
    .js-plotly-plot {
        height: 300px !important;
    }
    /* Make table scrollable on mobile */
    .stDataFrame {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }
    /* Improve sidebar on mobile */
    [data-testid="stSidebar"] {
        width: 100% !important;
    }
}

@media only screen and (max-width: 480px) {
    .kpi-value {
        font-size: 20px !important;
    }
    .kpi-title {
        font-size: 11px !important;
    }
    h1 {
        font-size: 1.5rem !important;
    }
    h3 {
        font-size: 1.1rem !important;
    }
    /* Stack refresh button on small screens */
    .stButton button {
        width: 100% !important;
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
# Check if we should use mobile layout (using session state as workaround)
if 'mobile_mode' not in st.session_state:
    st.session_state.mobile_mode = False

# On mobile, stack header elements vertically
is_mobile = st.session_state.mobile_mode

if is_mobile:
    st.title("📊 Business Performance Dashboard")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
else:
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
    # Check if this is a sales sheet
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

# Load all spend sheets (including IB marketplace)
spend_sheets = []
spend_sheet_names = []
spend_sheet_info = []

for sheet_name in data.keys():
    # Check if sheet name contains 'spend' or 'marketplace' or 'IB'
    sheet_lower = sheet_name.lower()
    if any(keyword in sheet_lower for keyword in ['spend', 'marketplace', 'ib marketplace']):
        try:
            df = data[sheet_name].copy()
            if not df.empty and len(df) > 0:
                spend_sheets.append(df)
                spend_sheet_names.append(sheet_name)
                # Store info about this sheet
                spend_sheet_info.append({
                    'name': sheet_name,
                    'rows': len(df),
                    'columns': list(df.columns)
                })
        except Exception as e:
            st.warning(f"⚠️ Could not load spend data from sheet '{sheet_name}': {str(e)}")

# Show debug info in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔍 Data Sources")
    if spend_sheet_names:
        st.markdown("**Spend Sheets Loaded:**")
        for info in spend_sheet_info:
            st.text(f"• {info['name']}: {info['rows']:,} rows")
    
if spend_sheets:
    try:
        # Simply concatenate - DO NOT remove any data
        channel_spend = pd.concat(spend_sheets, ignore_index=True)
    except Exception as e:
        st.warning(f"⚠️ Error combining spend sheets: {str(e)}")
        channel_spend = pd.DataFrame(columns=["date", "channel", "spend"])
else:
    channel_spend = pd.DataFrame(columns=["date", "channel", "spend"])

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

# Clean channel names
if "channel" in sales.columns:
    sales["channel"] = (
        sales["channel"]
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )
    sales["channel"] = sales["channel"].str.replace(r'_ebay$', '_eBay', case=False, regex=True)
    sales["channel"] = sales["channel"].str.replace(r'_Ebay$', '_eBay', case=False, regex=True)

if len(channel_spend) > 0:
    channel_spend = normalize_columns(channel_spend)
    
    # Ensure required columns exist
    if "spend" in channel_spend.columns and "ad_spend" not in channel_spend.columns:
        channel_spend["ad_spend"] = channel_spend["spend"]
    elif "ad_spend" not in channel_spend.columns:
        st.warning("⚠️ No 'spend' or 'ad_spend' column found in spend data. Using zero values.")
        channel_spend["ad_spend"] = 0
    
    if "channel" in channel_spend.columns:
        channel_spend["channel"] = (
            channel_spend["channel"]
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
        channel_spend["channel"] = channel_spend["channel"].str.replace(r'_ebay$', '_eBay', case=False, regex=True)
        channel_spend["channel"] = channel_spend["channel"].str.replace(r'_Ebay$', '_eBay', case=False, regex=True)

# ---------------- TYPE CAST SALES ----------------
# Ensure required columns exist
required_sales_cols = ["purchased_on", "no_of_orders", "discounted_price", "channel"]
missing_cols = [col for col in required_sales_cols if col not in sales.columns]

if missing_cols:
    st.error(f"❌ Missing required columns in sales data: {', '.join(missing_cols)}")
    st.info(f"Available columns: {', '.join(sales.columns)}")
    st.stop()

sales["purchased_on"] = pd.to_datetime(sales["purchased_on"], format="mixed", errors="coerce")
sales = sales.dropna(subset=["purchased_on"])

if len(sales) == 0:
    st.error("❌ No valid dates found in sales data")
    st.stop()

sales["no_of_orders"] = pd.to_numeric(sales["no_of_orders"], errors="coerce").fillna(0)

# Revenue - handle dollar signs
sales["discounted_price"] = (
    sales["discounted_price"].astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)
sales["discounted_price"] = pd.to_numeric(sales["discounted_price"], errors="coerce").fillna(0)

# Revenue = discounted_price × no_of_orders
sales["revenue"] = sales["discounted_price"]  # already total revenue per row

# Commission
if "selling_commission" not in sales.columns:
    st.warning("⚠️ 'selling_commission' column not found. Using zero values.")
    sales["selling_commission"] = 0
else:
    sales["selling_commission"] = (
        sales["selling_commission"].astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    sales["selling_commission"] = pd.to_numeric(sales["selling_commission"], errors="coerce").fillna(0)

# Type (optional column)
if "type" not in sales.columns:
    st.info("ℹ️ 'type' column not found. Using 'Unknown' as default.")
    sales["type"] = "Unknown"

# ---------------- SPEND DATA TYPE CASTING ----------------
if len(channel_spend) > 0:
    # Handle date column
    date_cols = ["date", "Date", "purchased_on", "Purchased_On"]
    date_col = None
    for col in date_cols:
        if col in channel_spend.columns:
            date_col = col
            break
    
    if date_col:
        channel_spend["date"] = pd.to_datetime(channel_spend[date_col], format="mixed", errors="coerce")
        # Remove rows with invalid dates
        channel_spend = channel_spend.dropna(subset=["date"])
    else:
        st.error("⚠️ No date column found in spend data. Ad spend will not be time-filtered.")
        channel_spend = pd.DataFrame(columns=["date", "channel", "ad_spend"])
    
    # Handle ad_spend column
    if "ad_spend" in channel_spend.columns:
        channel_spend["ad_spend"] = (
            channel_spend["ad_spend"].astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        channel_spend["ad_spend"] = pd.to_numeric(channel_spend["ad_spend"], errors="coerce").fillna(0)
    
    # Handle channel column
    if "channel" not in channel_spend.columns:
        st.error("⚠️ No 'channel' column found in spend data. Cannot match spend to channels.")
        channel_spend = pd.DataFrame(columns=["date", "channel", "ad_spend"])
    
    # Show total spend data info in sidebar
    st.sidebar.info(f"📊 Total Spend Records: {len(channel_spend):,} rows")

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

# ---------------- FILTER DATA ----------------
start_dt = pd.Timestamp(start_date)
end_dt = pd.Timestamp(end_date)
end_dt_next = end_dt + pd.Timedelta(days=1)  # include end date

sales_f = sales[
    (sales["purchased_on"] >= start_dt) &
    (sales["purchased_on"] < end_dt_next) &
    (sales["channel"].isin(selected_channels)) &
    (sales["type"].isin(selected_types))
]

if len(sales_f) == 0:
    st.warning("⚠️ No data available for selected filters")
    st.stop()

# ---------------- KPI CARDS ----------------
total_rev = sales_f["revenue"].sum()
total_orders = sales_f["no_of_orders"].sum()
aov = total_rev / total_orders if total_orders > 0 else 0
total_commission = sales_f["selling_commission"].sum()

# Calculate ad spend from spend data
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

# Calculate YoY changes
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

# Calculate last year ad spend
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

# ---------------- DEBUG: AD SPEND BREAKDOWN ----------------
with st.expander("🔍 Ad Spend Breakdown (Debug)", expanded=False):
    st.markdown("#### Ad Spend Details by Source")
    
    if len(channel_spend) > 0 and "date" in channel_spend.columns:
        # Filter spend data for current date range
        spend_debug = channel_spend[
            (channel_spend["date"] >= start_dt) & 
            (channel_spend["date"] < end_dt_next)
        ]
        
        if "channel" in channel_spend.columns:
            spend_debug_filtered = spend_debug[spend_debug["channel"].isin(selected_channels)]
        else:
            spend_debug_filtered = spend_debug
        
        st.markdown(f"**Total Rows in All Spend Sheets:** {len(channel_spend):,}")
        st.markdown(f"**Rows in Date Range ({start_date} to {end_date}):** {len(spend_debug):,}")
        st.markdown(f"**Rows After Channel Filter:** {len(spend_debug_filtered):,}")
        st.markdown(f"**Total Ad Spend (Dashboard):** ${total_spend:,.2f}")
        
        # Show spend by channel for selected period
        st.markdown("#### Spend by Channel (Selected Period)")
        if len(spend_debug_filtered) > 0:
            channel_spend_summary = spend_debug_filtered.groupby("channel")["ad_spend"].sum().reset_index()
            channel_spend_summary.columns = ["Channel", "Ad Spend"]
            channel_spend_summary = channel_spend_summary.sort_values("Ad Spend", ascending=False)
            
            # Add a total row
            total_row = pd.DataFrame([["TOTAL", channel_spend_summary["Ad Spend"].sum()]], columns=["Channel", "Ad Spend"])
            display_df = pd.concat([channel_spend_summary, total_row], ignore_index=True)
            
            st.dataframe(
                display_df.style.format({"Ad Spend": "${:,.2f}"}),
                use_container_width=True
            )
        else:
            st.warning("No spend data in selected date range and channels")
        
        # Show ALL channels in the data (not just selected)
        st.markdown("#### All Channels in Spend Data (Selected Period)")
        all_channels_spend = spend_debug.groupby("channel")["ad_spend"].sum().reset_index()
        all_channels_spend.columns = ["Channel", "Ad Spend"]
        all_channels_spend = all_channels_spend.sort_values("Ad Spend", ascending=False)
        st.dataframe(
            all_channels_spend.style.format({"Ad Spend": "${:,.2f}"}),
            use_container_width=True
        )
        
        # Show raw spend data sample
        st.markdown("#### Raw Spend Data Sample (First 20 Rows in Date Range)")
        st.dataframe(spend_debug.head(20), use_container_width=True)
        
    else:
        st.info("No spend data available")

# ---------------- YoY COMPARISON CHART ----------------
st.markdown("---")
st.markdown("### 📊 Year-over-Year Performance")

selected_metric = st.selectbox(
    "Select Metric",
    ["Revenue", "Orders", "Ad Spend", "Commission", "Net Earning", "ACOS"],
    index=0
)

# Check if we have data for last year
has_ly_data = len(sales_ly) > 0

if not has_ly_data:
    st.info(f"ℹ️ No data available for the same period last year ({year_ago_start.date()} to {year_ago_end.date()}). Showing current year data only.")

# Prepare trend data
current_trend = (
    sales_f.groupby(pd.Grouper(key="purchased_on", freq="D"))
    .agg({"revenue": "sum", "no_of_orders": "sum", "selling_commission": "sum"})
    .reset_index()
)

# Only prepare last year trend if we have data
if has_ly_data:
    ly_trend = (
        sales_ly.groupby(pd.Grouper(key="purchased_on", freq="D"))
        .agg({"revenue": "sum", "no_of_orders": "sum", "selling_commission": "sum"})
        .reset_index()
    )
    ly_trend["display_date"] = ly_trend["purchased_on"] + pd.DateOffset(years=1)
else:
    ly_trend = pd.DataFrame(columns=["purchased_on", "revenue", "no_of_orders", "selling_commission", "display_date"])

# Add ad spend to trends
if len(channel_spend) > 0 and "date" in channel_spend.columns:
    # Calculate ad spend for current period
    current_spend_trend = (
        channel_spend[channel_spend["date"].between(start_dt, end_dt)]
        .groupby(pd.Grouper(key="date", freq="D"))
        .agg({"ad_spend": "sum"})
        .reset_index()
    )
    current_trend = current_trend.merge(
        current_spend_trend,
        left_on="purchased_on",
        right_on="date",
        how="left"
    )
    current_trend["ad_spend"] = current_trend["ad_spend"].fillna(0)
    
    # Calculate ad spend for last year (only if we have data)
    if has_ly_data:
        ly_spend_trend = (
            channel_spend[channel_spend["date"].between(year_ago_start, year_ago_end)]
            .groupby(pd.Grouper(key="date", freq="D"))
            .agg({"ad_spend": "sum"})
            .reset_index()
        )
        ly_spend_trend["display_date"] = ly_spend_trend["date"] + pd.DateOffset(years=1)
        ly_trend = ly_trend.merge(
            ly_spend_trend,
            left_on="display_date",
            right_on="display_date",
            how="left"
        )
        ly_trend["ad_spend"] = ly_trend["ad_spend"].fillna(0)
    else:
        ly_trend["ad_spend"] = 0
else:
    current_trend["ad_spend"] = 0
    ly_trend["ad_spend"] = 0

# Calculate net earning for both periods
current_trend["net_earning"] = (current_trend["revenue"] * SAFE_MARGIN) - current_trend["ad_spend"] - current_trend["selling_commission"]
if has_ly_data:
    ly_trend["net_earning"] = (ly_trend["revenue"] * SAFE_MARGIN) - ly_trend["ad_spend"] - ly_trend["selling_commission"]
else:
    ly_trend["net_earning"] = 0

# Calculate ACOS
current_trend["acos"] = (current_trend["ad_spend"] / current_trend["revenue"] * 100).replace([float('inf'), -float('inf')], 0).fillna(0)
if has_ly_data:
    ly_trend["acos"] = (ly_trend["ad_spend"] / ly_trend["revenue"] * 100).replace([float('inf'), -float('inf')], 0).fillna(0)
else:
    ly_trend["acos"] = 0

# Select the metric to display
metric_map = {
    "Revenue": "revenue",
    "Orders": "no_of_orders",
    "Ad Spend": "ad_spend",
    "Commission": "selling_commission",
    "Net Earning": "net_earning",
    "ACOS": "acos"
}

metric_col = metric_map[selected_metric]

# Create YoY comparison chart
fig_yoy = go.Figure()

fig_yoy.add_trace(go.Scatter(
    x=current_trend["purchased_on"],
    y=current_trend[metric_col],
    name=f"{start_date.year} (Current)",
    line=dict(color="#3b82f6", width=3),
    fill='tozeroy',
    fillcolor='rgba(59, 130, 246, 0.1)'
))

# Only add last year trace if we have data
if has_ly_data and len(ly_trend) > 0:
    fig_yoy.add_trace(go.Scatter(
        x=ly_trend["display_date"],
        y=ly_trend[metric_col],
        name=f"{year_ago_start.year} (Last Year)",
        line=dict(color="#10b981", width=3, dash='dash')
    ))

# Format y-axis based on metric
if selected_metric in ["Revenue", "Ad Spend", "Commission", "Net Earning"]:
    yaxis_format = "$,.0f"
    yaxis_title = f"{selected_metric} ($)"
elif selected_metric == "ACOS":
    yaxis_format = ".1f"
    yaxis_title = f"{selected_metric} (%)"
else:
    yaxis_format = ",.0f"
    yaxis_title = selected_metric

fig_yoy.update_layout(
    yaxis=dict(title=yaxis_title, tickformat=yaxis_format),
    xaxis=dict(
        title="Date",
        type='date'
    ),
    hovermode="x unified",
    template="plotly_white",
    height=450,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig_yoy, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

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
        line=dict(color="#3b82f6", width=3), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'
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
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

with col_right:
    st.markdown("### 🛒 Revenue by Channel")
    channel_rev = (
        sales_f.groupby("channel").agg({"revenue": "sum"})
        .reset_index().sort_values("revenue", ascending=False)
    )
    fig2 = px.bar(channel_rev, x="channel", y="revenue", color="revenue",
                  color_continuous_scale="Blues", labels={"revenue": "Revenue ($)", "channel": "Channel"})
    fig2.update_layout(
        showlegend=False, 
        template="plotly_white", 
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("### 💰 Ad Spend vs Revenue by Channel")
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
                          name="Revenue", marker_color="#3b82f6"))
    fig3.add_trace(go.Bar(x=channel_comparison["channel"], y=channel_comparison["ad_spend"],
                          name="Ad Spend", marker_color="#ef4444"))
    fig3.update_layout(
        barmode="group", 
        template="plotly_white", 
        height=400, 
        yaxis_title="Amount ($)",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

with col_right2:
    st.markdown("### 📦 Revenue by Type")
    type_rev = sales_f.groupby("type").agg({"revenue": "sum"}).reset_index()
    fig4 = px.pie(type_rev, values="revenue", names="type", hole=0.4,
                  color_discrete_sequence=px.colors.sequential.Blues_r)
    fig4.update_layout(
        template="plotly_white", 
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

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
    use_container_width=True, height=300
)

channels_no_spend = summary[summary["Ad Spend"] == 0]["Channel"].tolist()
if channels_no_spend:
    st.info(f"ℹ️ Channels with no ad spend data: {', '.join(channels_no_spend)}")

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