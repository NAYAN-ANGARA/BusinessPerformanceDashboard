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
    try:
        return load_all_sheets(
            "secret-envoy-486405-j3-03851d061385.json",
            "USA - DB for Marketplace Dashboard"
        )
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

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
try:
    sales = data["Sales_data"].copy()
except KeyError:
    st.error("❌ Sheet 'Sales_data' not found")
    st.info("Available sheets: " + ", ".join(data.keys()))
    st.stop()

# Load all spend sheets
spend_sheets = []
for sheet_name in data.keys():
    if 'spend' in sheet_name.lower():
        spend_sheets.append(data[sheet_name].copy())

if spend_sheets:
    channel_spend = pd.concat(spend_sheets, ignore_index=True)
else:
    channel_spend = pd.DataFrame(columns=["date", "channel", "spend"])

# ---------------- NORMALIZE COLUMNS ----------------
def normalize_columns(df):
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
sales["purchased_on"] = pd.to_datetime(sales["purchased_on"], format="mixed", errors="coerce")
sales = sales.dropna(subset=["purchased_on"])

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

# Commission - handle dollar signs
commission_col = None
for col in ["selling_commission", "commission", "seller_commission"]:
    if col in sales.columns:
        commission_col = col
        break

if commission_col:
    sales["selling_commission"] = (
        sales[commission_col].astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    sales["selling_commission"] = pd.to_numeric(sales["selling_commission"], errors="coerce").fillna(0)
else:
    sales["selling_commission"] = 0

# ---------------- TYPE CAST CHANNEL SPEND ----------------
if len(channel_spend) > 0:
    channel_spend["date"] = pd.to_datetime(channel_spend["date"], format="mixed", errors="coerce").dt.normalize()
    channel_spend = channel_spend.dropna(subset=["date"])
    
    ad_spend_column = None
    for col_name in ["spend", "ad_spend", "adspend", "cost"]:
        if col_name in channel_spend.columns:
            ad_spend_column = col_name
            break
    
    if ad_spend_column:
        channel_spend["ad_spend"] = pd.to_numeric(channel_spend[ad_spend_column], errors="coerce").fillna(0)
    else:
        channel_spend["ad_spend"] = 0
else:
    channel_spend = pd.DataFrame(columns=["date", "channel", "ad_spend"])

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🎯 Filters")

min_date = sales["purchased_on"].min().date()
max_date = sales["purchased_on"].max().date()

date_range = st.sidebar.date_input(
    "📅 Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = map(pd.to_datetime, date_range)
else:
    start_date = end_date = pd.to_datetime(date_range[0])

# Normalize date boundaries (inclusive end date handling)
start_dt = pd.to_datetime(start_date).normalize()
end_dt = pd.to_datetime(end_date).normalize()
# Use half-open interval [start_dt, end_dt_next) to include the entire end date
end_dt_next = end_dt + pd.Timedelta(days=1)


channels = multiselect_with_all("🛒 Channel", sales["channel"].dropna().unique())
types = multiselect_with_all("📦 Type", sales["type"].dropna().unique())

# Metric selector for YoY comparison
metric_options = ["Revenue", "Orders", "Ad Spend", "Commission", "Net Earning", "ACOS"]
selected_metric = st.sidebar.selectbox("📊 YoY Comparison Metric", metric_options, index=0)

# ---------------- FILTER SALES ----------------
sales_f = sales[
    ((sales["purchased_on"] >= start_dt) & (sales["purchased_on"] < end_dt_next)) &
    (sales["channel"].isin(channels)) &
    (sales["type"].isin(types))
].copy()

# ---------------- CALCULATE AD SPEND ----------------
sales_daily = (
    sales_f.groupby([pd.Grouper(key="purchased_on", freq="D"), "channel"], as_index=False)
    .agg({"revenue": "sum"})
)

if len(channel_spend) > 0:
    spend_daily = (
        channel_spend[(channel_spend["date"] >= start_dt) & (channel_spend["date"] < end_dt_next)]
        .groupby(["date", "channel"], as_index=False)
        .agg({"ad_spend": "sum"})
    )
    
    merged = sales_daily.merge(
        spend_daily,
        left_on=["purchased_on", "channel"],
        right_on=["date", "channel"],
        how="left"
    )
    ad_spend = merged["ad_spend"].sum(skipna=True)
else:
    ad_spend = 0

# ---------------- KPI CALCULATIONS ----------------
orders = sales_f["no_of_orders"].sum()
revenue = sales_f["revenue"].sum()
commission = sales_f["selling_commission"].sum()
aov = revenue / orders if orders > 0 else 0

# Net Earning = (Revenue × Safe Margin) - Ad Spend - Commission
net_earning = (revenue * SAFE_MARGIN) - ad_spend - commission

roas = revenue / ad_spend if ad_spend > 0 else 0
acos = (ad_spend / revenue * 100) if revenue > 0 else 0

# Previous period comparison (for % change indicators)
period_days = (end_date - start_date).days
previous_start = start_date - pd.Timedelta(days=period_days + 1)
previous_end = start_date - pd.Timedelta(days=1)

sales_prev = sales[
    ((sales["purchased_on"] >= previous_start) & (sales["purchased_on"] < (previous_end + pd.Timedelta(days=1)))) &
    (sales["channel"].isin(channels)) &
    (sales["type"].isin(types))
]

prev_revenue = sales_prev["revenue"].sum()
revenue_change = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0

prev_orders = sales_prev["no_of_orders"].sum()
orders_change = ((orders - prev_orders) / prev_orders * 100) if prev_orders > 0 else 0

# ---------------- DASHBOARD KPIs ----------------
st.markdown("### 📈 Key Performance Indicators")

# Detect if we're on mobile (simplified approach using columns)
# First row - 6 KPIs (will stack on mobile due to CSS)
c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
with c1: kpi("Orders", f"{orders:,.0f}", orders_change)
with c2: kpi("Revenue", f"${revenue:,.0f}", revenue_change)
with c3: kpi("AOV", f"${aov:,.2f}")
with c4: kpi("Ad Spend", f"${ad_spend:,.0f}")
with c5: kpi("Commission", f"${commission:,.0f}")
with c6: kpi("Net Earning", f"${net_earning:,.0f}")

st.markdown("---")

# Second row - 3 KPIs (will stack on mobile)
c7, c8, c9 = st.columns([1,1,1])
with c7: 
    kpi("ROAS", f"{roas:.2f}x")
with c8:
    kpi("ACOS", f"{acos:.1f}%")
with c9: 
    avg_comm = commission / orders if orders > 0 else 0
    kpi("Avg Commission", f"${avg_comm:.2f}")

st.markdown("---")

# ---------------- YOY COMPARISON CHART ----------------
st.markdown(f"### 📈 Year-over-Year Comparison: {selected_metric}")

# Calculate same period last year
year_ago_start = start_date - pd.DateOffset(years=1)
year_ago_end = end_date - pd.DateOffset(years=1)

# Filter for last year's data
sales_ly = sales[
    (sales["purchased_on"].between(year_ago_start, year_ago_end)) &
    (sales["channel"].isin(channels)) &
    (sales["type"].isin(types))
].copy()

# Prepare current year data
current_trend = (
    sales_f.groupby(pd.Grouper(key="purchased_on", freq="D"))
    .agg({
        "revenue": "sum",
        "no_of_orders": "sum",
        "selling_commission": "sum"
    })
    .reset_index()
)

# Prepare last year data
ly_trend = (
    sales_ly.groupby(pd.Grouper(key="purchased_on", freq="D"))
    .agg({
        "revenue": "sum",
        "no_of_orders": "sum",
        "selling_commission": "sum"
    })
    .reset_index()
)

# Adjust last year dates to align with current year (for visualization)
ly_trend["display_date"] = ly_trend["purchased_on"] + pd.DateOffset(years=1)

# Calculate ad spend for current period
if len(channel_spend) > 0:
    current_spend_trend = (
        channel_spend[(channel_spend["date"] >= start_dt) & (channel_spend["date"] < end_dt_next)]
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
    
    # Calculate ad spend for last year
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
    current_trend["ad_spend"] = 0
    ly_trend["ad_spend"] = 0

# Calculate net earning for both periods
current_trend["net_earning"] = (current_trend["revenue"] * SAFE_MARGIN) - current_trend["ad_spend"] - current_trend["selling_commission"]
ly_trend["net_earning"] = (ly_trend["revenue"] * SAFE_MARGIN) - ly_trend["ad_spend"] - ly_trend["selling_commission"]

# Calculate ACOS
current_trend["acos"] = (current_trend["ad_spend"] / current_trend["revenue"] * 100).fillna(0)
ly_trend["acos"] = (ly_trend["ad_spend"] / ly_trend["revenue"] * 100).fillna(0)

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
    xaxis=dict(title="Date"),
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
    
    if len(channel_spend) > 0:
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

if len(channel_spend) > 0:
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
    file_name=f"dashboard_{start_date.date()}_{end_date.date()}.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | {start_date.date()} to {end_date.date()} | Margin: {SAFE_MARGIN*100:.0f}%")