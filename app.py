import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets

# Configure Plotly
import plotly.io as pio
pio.templates.default = "plotly_dark"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Business Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CONSTANTS ----------------
SAFE_MARGIN = 0.62

# ---------------- CSS (Subtle Dark Theme) ----------------
st.markdown("""
<style>
    /* Global clean up */
    .stApp {
        background-color: #0e1117;
    }
    
    /* KPI Card Styling */
    .kpi-card {
        background-color: #1a1c24;
        border: 1px solid #2d303e;
        border-radius: 10px;
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 140px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #4b5563;
    }
    .kpi-title {
        font-size: 13px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #f3f4f6;
        margin: 8px 0;
    }
    .kpi-change {
        font-size: 13px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    /* Indicators */
    .trend-up { color: #4ade80; }
    .trend-down { color: #f87171; }
    .trend-neutral { color: #9ca3af; }
    
    /* Left border accents for cards */
    .accent-blue { border-left: 3px solid #3b82f6; }
    .accent-green { border-left: 3px solid #10b981; }
    .accent-orange { border-left: 3px solid #f59e0b; }
    .accent-purple { border-left: 3px solid #8b5cf6; }

    /* Section Headers */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #e5e7eb;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d303e;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------
def kpi(title, value, change=None, accent="blue"):
    change_html = ""
    if change is not None:
        icon = "↑" if change >= 0 else "↓"
        color = "trend-up" if change >= 0 else "trend-down"
        # Invert color logic for cost metrics if needed, but keeping standard for now
        change_html = f'<div class="kpi-change {color}">{icon} {abs(change):.1f}% vs LY</div>'
    else:
        change_html = '<div class="kpi-change trend-neutral">-</div>'
        
    st.markdown(f"""
    <div class="kpi-card accent-{accent}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

def multiselect_with_all(label, options):
    ALL = "All"
    opts = [ALL] + sorted(list(options))
    selected = st.sidebar.multiselect(label, opts, default=[ALL])
    return list(options) if ALL in selected or not selected else selected

# ---------------- DATA LOADING ----------------
@st.cache_data(show_spinner=True, ttl=300)
def load_data():
    all_data = {}
    creds = "secret-envoy-486405-j3-03851d061385.json"
    
    try:
        # Load USA Data
        data1 = load_all_sheets(creds, "USA - DB for Marketplace Dashboard")
        if data1:
            for k, v in data1.items(): all_data[f"USA_{k}"] = v
            
        # Load IB Data
        data2 = load_all_sheets(creds, "IB - Database for Marketplace Dashboard")
        if data2:
            for k, v in data2.items(): all_data[f"IB_{k}"] = v
            
    except Exception as e:
        st.warning(f"⚠️ Connection Warning: {str(e)}")
    
    return all_data if all_data else None

# ---------------- DATA PROCESSING ----------------
data = load_data()
if not data:
    st.error("❌ Failed to load data. Please check credentials.")
    st.stop()

# 1. Process Sales
sales_sheets = [df.copy() for name, df in data.items() if 'sales' in name.lower() and not df.empty]
if not sales_sheets:
    st.error("❌ No sales data found")
    st.stop()

sales = pd.concat(sales_sheets, ignore_index=True)

# Normalize Sales Columns
sales.columns = sales.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# 2. Process Spend (Fix for 'Spend' vs 'Ad Spend')
spend_dfs = []
for name, df in data.items():
    if 'channel' in name.lower() and 'spend' in name.lower():
        df = df.copy()
        
        # Normalize existing columns first
        # We need to find the specific Spend column regardless of case or whitespace
        spend_col = None
        date_col = None
        chan_col = None
        
        for c in df.columns:
            clean_c = c.strip().lower()
            if clean_c in ['spend', 'ad spend', 'ad_spend']:
                spend_col = c
            if clean_c in ['date', 'purchased_on', 'purchased on']:
                date_col = c
            if clean_c in ['channel']:
                chan_col = c
        
        if spend_col and date_col:
            # Standardize this sheet
            temp = pd.DataFrame()
            temp['date'] = pd.to_datetime(df[date_col], errors='coerce')
            temp['channel'] = df[chan_col] if chan_col else "Unknown"
            
            # Clean spend data (remove $, commas)
            temp['ad_spend'] = (
                df[spend_col].astype(str)
                .str.replace(r'[$,]', '', regex=True)
                .str.strip()
            )
            temp['ad_spend'] = pd.to_numeric(temp['ad_spend'], errors='coerce').fillna(0)
            
            spend_dfs.append(temp)

if spend_dfs:
    channel_spend = pd.concat(spend_dfs, ignore_index=True)
else:
    channel_spend = pd.DataFrame(columns=['date', 'channel', 'ad_spend'])

# 3. Final Sales Clean Up
sales["purchased_on"] = pd.to_datetime(sales["purchased_on"], errors="coerce")
sales = sales.dropna(subset=["purchased_on"])

# Clean numericals
def clean_money(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r'[$,]', '', regex=True), 
        errors='coerce'
    ).fillna(0)

sales["revenue"] = clean_money(sales.get("discounted_price", 0))
sales["no_of_orders"] = pd.to_numeric(sales.get("no_of_orders", 0), errors='coerce').fillna(0)
sales["selling_commission"] = clean_money(sales.get("selling_commission", 0))
sales["channel"] = sales.get("channel", "Unknown").astype(str).str.strip()
sales["type"] = sales.get("type", "Unknown")

# ---------------- DASHBOARD LOGIC ----------------

# Sidebar
st.sidebar.title("Filters")
min_d, max_d = sales["purchased_on"].min().date(), sales["purchased_on"].max().date()
start_date = st.sidebar.date_input("Start", min_d, min_value=min_d, max_value=max_d)
end_date = st.sidebar.date_input("End", max_d, min_value=min_d, max_value=max_d)

sel_channels = multiselect_with_all("Channel", sales["channel"].unique())
sel_types = multiselect_with_all("Type", sales["type"].unique())

# Filtering
mask_sales = (
    (sales["purchased_on"].dt.date >= start_date) &
    (sales["purchased_on"].dt.date <= end_date) &
    (sales["channel"].isin(sel_channels)) &
    (sales["type"].isin(sel_types))
)
df_sales = sales[mask_sales]

mask_spend = (
    (channel_spend["date"].dt.date >= start_date) &
    (channel_spend["date"].dt.date <= end_date) &
    (channel_spend["channel"].isin(sel_channels))
)
df_spend = channel_spend[mask_spend]

# YoY Logic
start_ly = start_date - pd.DateOffset(years=1)
end_ly = end_date - pd.DateOffset(years=1)

mask_sales_ly = (
    (sales["purchased_on"].dt.date >= start_ly.date()) &
    (sales["purchased_on"].dt.date <= end_ly.date()) &
    (sales["channel"].isin(sel_channels)) &
    (sales["type"].isin(sel_types))
)
df_sales_ly = sales[mask_sales_ly]

mask_spend_ly = (
    (channel_spend["date"].dt.date >= start_ly.date()) &
    (channel_spend["date"].dt.date <= end_ly.date()) &
    (channel_spend["channel"].isin(sel_channels))
)
df_spend_ly = channel_spend[mask_spend_ly]

# Metrics
curr_rev = df_sales["revenue"].sum()
curr_ord = df_sales["no_of_orders"].sum()
curr_spend = df_spend["ad_spend"].sum()
curr_comm = df_sales["selling_commission"].sum()
curr_net = (curr_rev * SAFE_MARGIN) - curr_spend - curr_comm
curr_roas = (curr_rev / curr_spend) if curr_spend > 0 else 0
curr_acos = (curr_spend / curr_rev * 100) if curr_rev > 0 else 0

# YoY Metrics
ly_rev = df_sales_ly["revenue"].sum()
ly_ord = df_sales_ly["no_of_orders"].sum()
ly_spend = df_spend_ly["ad_spend"].sum()
ly_comm = df_sales_ly["selling_commission"].sum()
ly_net = (ly_rev * SAFE_MARGIN) - ly_spend - ly_comm
ly_roas = (ly_rev / ly_spend) if ly_spend > 0 else 0

def calc_delta(curr, prev):
    if prev == 0: return 0
    return ((curr - prev) / prev) * 100

# ---------------- UI LAYOUT ----------------
st.title("📊 Business Performance")
st.markdown(f"**Period:** {start_date} to {end_date}")

st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
with k1: kpi("Revenue", f"${curr_rev:,.0f}", calc_delta(curr_rev, ly_rev), "blue")
with k2: kpi("Orders", f"{curr_ord:,.0f}", calc_delta(curr_ord, ly_ord), "purple")
with k3: kpi("AOV", f"${(curr_rev/curr_ord if curr_ord else 0):.0f}", None, "blue")
with k4: kpi("Ad Spend", f"${curr_spend:,.0f}", calc_delta(curr_spend, ly_spend), "orange")
with k5: kpi("ROAS", f"{curr_roas:.2f}x", calc_delta(curr_roas, ly_roas), "green")
with k6: kpi("ACOS", f"{curr_acos:.1f}%", None, "orange")
with k7: kpi("Net Profit", f"${curr_net:,.0f}", calc_delta(curr_net, ly_net), "green")

st.markdown("---")

# Charts
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Revenue vs Orders Trend")
    daily = df_sales.groupby(pd.Grouper(key="purchased_on", freq="D")).agg({"revenue": "sum", "no_of_orders": "sum"}).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["purchased_on"], y=daily["revenue"], name="Revenue", 
                             line=dict(color="#3b82f6", width=3), fill='tozeroy'))
    fig.add_trace(go.Scatter(x=daily["purchased_on"], y=daily["no_of_orders"], name="Orders", 
                             yaxis="y2", line=dict(color="#10b981", width=3)))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af"),
        yaxis=dict(title="Revenue", showgrid=True, gridcolor="#2d303e"),
        yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Channel Mix")
    chan_grp = df_sales.groupby("channel")["revenue"].sum().reset_index()
    fig2 = px.pie(chan_grp, values="revenue", names="channel", hole=0.6, 
                  color_discrete_sequence=px.colors.sequential.Bluyl)
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af"),
        showlegend=False
    )
    fig2.update_traces(textposition='outside', textinfo='label+percent')
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TABLE (Using st.column_config - No Matplotlib) ----------------
st.markdown('<div class="section-title">Channel Performance</div>', unsafe_allow_html=True)

# Prepare Summary Data
summ_sales = df_sales.groupby("channel").agg(
    Revenue=("revenue", "sum"),
    Orders=("no_of_orders", "sum"),
    Commission=("selling_commission", "sum")
).reset_index()

summ_spend = df_spend.groupby("channel")["ad_spend"].sum().reset_index()
summ_spend.columns = ["channel", "Ad Spend"]

summ = pd.merge(summ_sales, summ_spend, on="channel", how="left").fillna(0)

# Calculate KPIs
summ["Net"] = (summ["Revenue"] * SAFE_MARGIN) - summ["Ad Spend"] - summ["Commission"]
summ["ROAS"] = summ.apply(lambda x: x["Revenue"] / x["Ad Spend"] if x["Ad Spend"] > 0 else 0, axis=1)
summ["ACOS"] = summ.apply(lambda x: (x["Ad Spend"] / x["Revenue"]) if x["Revenue"] > 0 else 0, axis=1)

# Reorder
summ = summ[["channel", "Orders", "Revenue", "Ad Spend", "ROAS", "ACOS", "Net"]].sort_values("Revenue", ascending=False)

# Display with Native Streamlit Column Config (Fixes Matplotlib Error)
st.dataframe(
    summ,
    column_config={
        "channel": "Channel",
        "Orders": st.column_config.NumberColumn("Orders", format="%d"),
        "Revenue": st.column_config.ProgressColumn(
            "Revenue",
            format="$%d",
            min_value=0,
            max_value=int(summ["Revenue"].max()) if not summ.empty else 1000,
        ),
        "Ad Spend": st.column_config.NumberColumn("Ad Spend", format="$%d"),
        "ROAS": st.column_config.NumberColumn("ROAS", format="%.2fx"),
        "ACOS": st.column_config.NumberColumn("ACOS", format="%.1f%%"),
        "Net": st.column_config.NumberColumn("Net Profit", format="$%d"),
    },
    use_container_width=True,
    hide_index=True,
    height=400
)