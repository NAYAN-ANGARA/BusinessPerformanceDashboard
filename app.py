import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets

# Configure Plotly to respect theme
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

# ---------------- CSS (SUBTLE DARK THEME) ----------------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Card Styling */
    .kpi-card {
        background-color: #1a1c24;
        border: 1px solid #2d303e;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .kpi-title {
        font-size: 14px;
        color: #9ca3af;
        margin-bottom: 8px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    
    .kpi-change {
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    /* Status Colors */
    .text-green { color: #4ade80; }
    .text-red { color: #f87171; }
    .text-blue { color: #60a5fa; }
    
    /* Subtle Borders for Context */
    .border-left-brand { border-left: 4px solid #6366f1; }
    .border-left-success { border-left: 4px solid #4ade80; }
    .border-left-warning { border-left: 4px solid #fbbf24; }
    .border-left-danger { border-left: 4px solid #f87171; }

    /* Section Headers */
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #e5e7eb;
        margin: 32px 0 16px 0;
        border-bottom: 1px solid #2d303e;
        padding-bottom: 8px;
    }

    /* Insight Boxes */
    .insight-box {
        background-color: #1a1c24;
        border: 1px solid #2d303e;
        border-radius: 8px;
        padding: 16px;
        height: 100%;
    }
    .insight-label { color: #9ca3af; font-size: 13px; margin-bottom: 4px; }
    .insight-val { color: #e5e7eb; font-weight: 600; font-size: 15px; }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2d303e; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #4b5563; }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------
def kpi(title, value, change=None, status="neutral"):
    change_html = ""
    if change is not None:
        color_class = "text-green" if change >= 0 else "text-red"
        arrow = "↑" if change >= 0 else "↓"
        change_html = f'<div class="kpi-change {color_class}">{arrow} {abs(change):.1f}% vs LY</div>'
    
    border_class = "border-left-brand"
    if status == "success": border_class = "border-left-success"
    elif status == "warning": border_class = "border-left-warning"
    elif status == "danger": border_class = "border-left-danger"
    
    st.markdown(f"""
    <div class="kpi-card {border_class}">
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

# ---------------- LOAD DATA ----------------
@st.cache_data(show_spinner=True, ttl=300)
def load_data():
    all_data = {}
    creds = "secret-envoy-486405-j3-03851d061385.json"
    
    # Load USA Data
    try:
        data1 = load_all_sheets(creds, "USA - DB for Marketplace Dashboard")
        if data1:
            for sheet_name, df in data1.items():
                all_data[f"USA_{sheet_name}"] = df
    except Exception as e:
        st.warning(f"⚠️ USA Data Error: {str(e)}")
    
    # Load IB Data
    try:
        data2 = load_all_sheets(creds, "IB - Database for Marketplace Dashboard")
        if data2:
            for sheet_name, df in data2.items():
                all_data[f"IB_{sheet_name}"] = df
    except Exception as e:
        st.warning(f"⚠️ IB Data Error: {str(e)}")
    
    return all_data if all_data else None

# ---------------- HEADER ----------------
col1, col2 = st.columns([5, 2])
with col1:
    st.title("📊 Business Performance")
    st.caption("Real-time financial tracking and channel analytics")

with col2:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

data = load_data()
if not data:
    st.error("❌ No data loaded. Please check your Google Sheets connection.")
    st.stop()

# ---------------- PROCESS DATA ----------------

# 1. Collect Sales Data
sales_sheets = [df.copy() for name, df in data.items() if 'sales' in name.lower() and not df.empty]
if not sales_sheets:
    st.error("❌ No sales data found in the loaded sheets.")
    st.stop()

sales = pd.concat(sales_sheets, ignore_index=True)

# 2. Collect & Normalize Spend Data (Specific Fix)
spend_dfs = []

for key, df in data.items():
    key_lower = key.lower()
    df_copy = df.copy()
    
    # Only process sheets that look like spend data
    if "channel_spend_data" in key_lower:
        
        # Check for USA style "Spend"
        if "Spend" in df_copy.columns:
            df_copy["ad_spend_clean"] = df_copy["Spend"]
            
        # Check for IB style "Ad Spend"
        elif "Ad Spend" in df_copy.columns:
            df_copy["ad_spend_clean"] = df_copy["Ad Spend"]
        
        # Fallback: look for normalized version if raw col missing
        else:
             cols = [c for c in df_copy.columns if "spend" in c.lower()]
             if cols:
                 df_copy["ad_spend_clean"] = df_copy[cols[0]]
             else:
                 continue # Skip if no spend column found
        
        # Normalize Date column
        date_col = next((c for c in df_copy.columns if c.lower() in ["date", "purchased_on", "purchased on"]), None)
        if date_col:
            df_copy["date_clean"] = pd.to_datetime(df_copy[date_col], errors="coerce")
            
        # Append if valid
        if "ad_spend_clean" in df_copy.columns and "date_clean" in df_copy.columns:
            # Clean up Channel Name
            chan_col = next((c for c in df_copy.columns if "channel" in c.lower()), None)
            if chan_col:
                df_copy["channel_clean"] = df_copy[chan_col].astype(str).str.strip()
            else:
                df_copy["channel_clean"] = "Unknown"

            spend_dfs.append(df_copy[["date_clean", "channel_clean", "ad_spend_clean"]])

if spend_dfs:
    channel_spend = pd.concat(spend_dfs, ignore_index=True)
    # Clean currency chars
    channel_spend["ad_spend"] = (
        channel_spend["ad_spend_clean"]
        .astype(str)
        .str.replace(r'[$,₹£€]', '', regex=True)
        .str.replace(',', '', regex=False)
    )
    channel_spend["ad_spend"] = pd.to_numeric(channel_spend["ad_spend"], errors="coerce").fillna(0)
    channel_spend.rename(columns={"date_clean": "date", "channel_clean": "channel"}, inplace=True)
else:
    channel_spend = pd.DataFrame(columns=["date", "channel", "ad_spend"])

# 3. Finalize Sales Columns
def clean_currency(col):
    if col in sales.columns:
        return pd.to_numeric(
            sales[col].astype(str).str.replace(r'[$,]', '', regex=True), 
            errors="coerce"
        ).fillna(0)
    return 0

# Normalize column names for sales
sales.columns = sales.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# Ensure core columns exist
sales["purchased_on"] = pd.to_datetime(sales["purchased_on"], errors="coerce")
sales["channel"] = sales.get("channel", "Unknown").astype(str).str.strip()
sales["type"] = sales.get("type", "Unknown")
sales["no_of_orders"] = pd.to_numeric(sales["no_of_orders"], errors="coerce").fillna(0)
sales["revenue"] = clean_currency("discounted_price")
sales["selling_commission"] = clean_currency("selling_commission")

sales = sales.dropna(subset=["purchased_on"])

# ---------------- FILTERS ----------------
st.sidebar.header("📅 Settings")
min_date = sales["purchased_on"].min().date()
max_date = sales["purchased_on"].max().date()

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

st.sidebar.divider()
st.sidebar.subheader("Filters")
selected_channels = multiselect_with_all("Sales Channels", sales["channel"].unique())
selected_types = multiselect_with_all("Order Types", sales["type"].unique())

# ---------------- CALCULATION ENGINE ----------------
# Date Masks
mask_curr = (sales["purchased_on"].dt.date >= start_date) & (sales["purchased_on"].dt.date <= end_date)
mask_curr_spend = (channel_spend["date"].dt.date >= start_date) & (channel_spend["date"].dt.date <= end_date)

# Previous Period (YoY)
start_ly = start_date - pd.DateOffset(years=1)
end_ly = end_date - pd.DateOffset(years=1)
mask_ly = (sales["purchased_on"].dt.date >= start_ly.date()) & (sales["purchased_on"].dt.date <= end_ly.date())
mask_ly_spend = (channel_spend["date"].dt.date >= start_ly.date()) & (channel_spend["date"].dt.date <= end_ly.date())

# Filter Dataframes
sales_f = sales[mask_curr & sales["channel"].isin(selected_channels) & sales["type"].isin(selected_types)]
sales_ly = sales[mask_ly & sales["channel"].isin(selected_channels) & sales["type"].isin(selected_types)]

spend_f = channel_spend[mask_curr_spend & channel_spend["channel"].isin(selected_channels)]
spend_ly_df = channel_spend[mask_ly_spend & channel_spend["channel"].isin(selected_channels)]

# Aggregate Metrics
curr_rev = sales_f["revenue"].sum()
curr_orders = sales_f["no_of_orders"].sum()
curr_comm = sales_f["selling_commission"].sum()
curr_spend = spend_f["ad_spend"].sum()

ly_rev = sales_ly["revenue"].sum()
ly_orders = sales_ly["no_of_orders"].sum()
ly_comm = sales_ly["selling_commission"].sum()
ly_spend = spend_ly_df["ad_spend"].sum()

# Derived Metrics
curr_net = (curr_rev * SAFE_MARGIN) - curr_spend - curr_comm
ly_net = (ly_rev * SAFE_MARGIN) - ly_spend - ly_comm

curr_roas = (curr_rev / curr_spend) if curr_spend > 0 else 0
ly_roas = (ly_rev / ly_spend) if ly_spend > 0 else 0

curr_acos = (curr_spend / curr_rev * 100) if curr_rev > 0 else 0
curr_margin = ((curr_rev * SAFE_MARGIN - curr_spend) / curr_rev * 100) if curr_rev > 0 else 0

# Changes
def calc_change(curr, prev):
    if prev == 0: return 0 if curr == 0 else 100
    return ((curr - prev) / prev) * 100

rev_chg = calc_change(curr_rev, ly_rev)
ord_chg = calc_change(curr_orders, ly_orders)
spend_chg = calc_change(curr_spend, ly_spend)
roas_chg = calc_change(curr_roas, ly_roas)
net_chg = calc_change(curr_net, ly_net)

# ---------------- DASHBOARD LAYOUT ----------------

# KPI ROW
st.markdown('<div class="section-header">📈 Performance Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

with k1: kpi("Revenue", f"${curr_rev:,.0f}", rev_chg, "neutral" if rev_chg > 0 else "danger")
with k2: kpi("Orders", f"{curr_orders:,.0f}", ord_chg, "success")
with k3: kpi("AOV", f"${(curr_rev/curr_orders if curr_orders else 0):.0f}", None, "neutral")
with k4: kpi("Ad Spend", f"${curr_spend:,.0f}", spend_chg, "warning")
with k5: kpi("ROAS", f"{curr_roas:.2f}x", roas_chg, "success" if roas_chg > 0 else "danger")
with k6: kpi("ACOS", f"{curr_acos:.1f}%", None, "success" if curr_acos < 30 else "warning")
with k7: kpi("Net Earning", f"${curr_net:,.0f}", net_chg, "success" if curr_net > 0 else "danger")

# INSIGHTS ROW
st.markdown("---")
i1, i2, i3 = st.columns(3)
with i1:
    top_chan = sales_f.groupby("channel")["revenue"].sum().idxmax() if not sales_f.empty else "N/A"
    top_val = sales_f.groupby("channel")["revenue"].sum().max() if not sales_f.empty else 0
    st.markdown(f"""<div class="insight-box">
        <div class="insight-label">🏆 Top Revenue Channel</div>
        <div class="insight-val">{top_chan} (${top_val:,.0f})</div>
    </div>""", unsafe_allow_html=True)
with i2:
    status_icon = "🔥" if roas_chg > 0 else "📉"
    status_text = "Improving" if roas_chg > 0 else "Declining"
    st.markdown(f"""<div class="insight-box">
        <div class="insight-label">📊 Efficiency Trend</div>
        <div class="insight-val">{status_icon} ROAS is {status_text} ({roas_chg:+.1f}%)</div>
    </div>""", unsafe_allow_html=True)
with i3:
    budget_status = "Excellent" if curr_acos < 20 else "Fair" if curr_acos < 40 else "High"
    st.markdown(f"""<div class="insight-box">
        <div class="insight-label">💰 Budget Health</div>
        <div class="insight-val">{budget_status} (ACOS: {curr_acos:.1f}%)</div>
    </div>""", unsafe_allow_html=True)

# CHARTS ROW
st.markdown('<div class="section-header">📊 Detailed Analytics</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📉 Trends & Channels", "📋 Raw Data"])

with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("**Revenue vs Orders Trend**")
        daily = sales_f.groupby(pd.Grouper(key="purchased_on", freq="D")).agg({"revenue": "sum", "no_of_orders": "sum"}).reset_index()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily["purchased_on"], y=daily["revenue"], name="Revenue",
            line=dict(color="#6366f1", width=3),
            fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.1)'
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily["purchased_on"], y=daily["no_of_orders"], name="Orders",
            line=dict(color="#4ade80", width=3),
            yaxis="y2"
        ))
        
        # FIXED: Updated layout syntax to avoid errors
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9ca3af"),
            yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
            yaxis2=dict(
                title="Orders", 
                overlaying="y", 
                side="right", 
                showgrid=False
            ),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=10, b=0),
            height=350,
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        st.markdown("**Revenue by Channel**")
        chan_rev = sales_f.groupby("channel")["revenue"].sum().reset_index().sort_values("revenue", ascending=True)
        fig_bar = px.bar(
            chan_rev, y="channel", x="revenue", 
            orientation='h',
            text_auto='.2s'
        )
        fig_bar.update_traces(marker_color="#6366f1", textfont_color="white")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9ca3af"),
            xaxis=dict(showgrid=True, gridcolor="#2d303e", title=""),
            yaxis=dict(title=""),
            margin=dict(l=0, r=0, t=10, b=0),
            height=350
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Spend vs Revenue
    st.markdown("**💰 Ad Spend vs Revenue vs ROAS**")
    
    # Merge Sales and Spend by Channel
    ch_sales = sales_f.groupby("channel").agg({"revenue": "sum"}).reset_index()
    ch_spend = spend_f.groupby("channel").agg({"ad_spend": "sum"}).reset_index()
    
    merged = pd.merge(ch_sales, ch_spend, on="channel", how="outer").fillna(0)
    merged["roas"] = merged.apply(lambda x: x["revenue"] / x["ad_spend"] if x["ad_spend"] > 0 else 0, axis=1)
    merged = merged[merged["revenue"] > 0].sort_values("revenue", ascending=False)
    
    fig_mix = go.Figure()
    fig_mix.add_trace(go.Bar(name="Revenue", x=merged["channel"], y=merged["revenue"], marker_color="#6366f1"))
    fig_mix.add_trace(go.Bar(name="Ad Spend", x=merged["channel"], y=merged["ad_spend"], marker_color="#f87171"))
    fig_mix.add_trace(go.Scatter(
        name="ROAS", x=merged["channel"], y=merged["roas"], 
        yaxis="y2", mode='markers+lines', 
        line=dict(color="#fbbf24", width=2)
    ))
    
    fig_mix.update_layout(
        barmode='group',
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af"),
        yaxis=dict(title="Amount ($)", gridcolor="#2d303e"),
        yaxis2=dict(title="ROAS (x)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.1),
        height=400,
        margin=dict(t=30)
    )
    st.plotly_chart(fig_mix, use_container_width=True)

with tab2:
    st.markdown("### 📋 Channel Performance Summary")
    
    # Create Summary Table
    summ = merged.copy()
    summ["acos"] = summ.apply(lambda x: (x["ad_spend"]/x["revenue"]*100) if x["revenue"] > 0 else 0, axis=1)
    
    # Add Orders and Commission
    ch_xtra = sales_f.groupby("channel").agg({"no_of_orders": "sum", "selling_commission": "sum"}).reset_index()
    summ = summ.merge(ch_xtra, on="channel", how="left").fillna(0)
    
    summ["net"] = (summ["revenue"] * SAFE_MARGIN) - summ["ad_spend"] - summ["selling_commission"]
    
    # Reorder
    cols = ["channel", "no_of_orders", "revenue", "ad_spend", "selling_commission", "net", "roas", "acos"]
    summ = summ[cols]
    summ.columns = ["Channel", "Orders", "Revenue", "Ad Spend", "Comm.", "Net", "ROAS", "ACOS"]
    
    st.dataframe(
        summ.style.format({
            "Revenue": "${:,.0f}", "Ad Spend": "${:,.0f}", "Comm.": "${:,.0f}", "Net": "${:,.0f}",
            "Orders": "{:,.0f}", "ROAS": "{:.2f}x", "ACOS": "{:.1f}%"
        }).background_gradient(subset=["Net"], cmap="Greens", vmin=0),
        use_container_width=True,
        height=400
    )
    
    # Download
    csv = summ.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Summary CSV",
        csv,
        "channel_summary.csv",
        "text/csv",
        key='download-csv'
    )

st.markdown("---")
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")