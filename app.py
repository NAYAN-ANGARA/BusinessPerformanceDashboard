import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets
from datetime import date, timedelta
import numpy as np

# Configure Plotly
import plotly.io as pio
pio.templates.default = "plotly_dark"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Business Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CONSTANTS ----------------
SAFE_MARGIN = 0.62  # Profit margin after COGS but before Ads/Commission

# ---------------- ENHANCED CSS ----------------
st.markdown("""
<style>
    /* Main Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f1116 0%, #1a1d29 100%);
    }
    
    /* KPI Cards with Enhanced Glassmorphism */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 32, 40, 0.8) 0%, rgba(42, 45, 58, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(15px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-color), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-6px);
        border-color: rgba(255, 255, 255, 0.25);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #9ca3af;
        margin-bottom: 10px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.2;
        background: linear-gradient(135deg, #fff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Dynamic Metric Accents */
    .accent-blue { --accent-color: #3b82f6; }
    .accent-green { --accent-color: #10b981; }
    .accent-orange { --accent-color: #f97316; }
    .accent-purple { --accent-color: #8b5cf6; }
    .accent-pink { --accent-color: #ec4899; }
    .accent-cyan { --accent-color: #06b6d4; }
    .accent-yellow { --accent-color: #eab308; }
    .accent-red { --accent-color: #ef4444; }
    
    /* Delta Badge with Glow */
    .delta-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 800;
        gap: 4px;
    }
    .delta-pos { 
        background: rgba(16, 185, 129, 0.25); 
        color: #34d399;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }
    .delta-neg { 
        background: rgba(239, 68, 68, 0.25); 
        color: #f87171;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    
    /* Section Headers with Icons */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #f3f4f6;
        margin: 40px 0 20px 0;
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
        color: #e0e7ff;
        font-size: 14px;
    }
    
    /* Tabs Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 32, 40, 0.5);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    
    /* Custom Button Styles */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    /* Hide Plotly Modebar */
    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(30, 32, 40, 0.4);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------
def metric_card(label, value, delta=None, prefix="", suffix="", color="blue", inverse=False, icon=""):
    delta_html = ""
    if delta is not None:
        is_pos = delta >= 0
        is_good = not is_pos if inverse else is_pos
        
        delta_class = "delta-pos" if is_good else "delta-neg"
        arrow = "↑" if is_pos else "↓"
        delta_html = f'<span class="delta-badge {delta_class}">{arrow} {abs(delta):.1f}%</span>'
    else:
        delta_html = '<span style="color:#6b7280; font-size:11px">No prev data</span>'
    
    icon_html = f'<span style="font-size: 16px;">{icon}</span>' if icon else ''
        
    st.markdown(f"""
    <div class="metric-card accent-{color}">
        <div class="metric-label">{icon_html}{label}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def multiselect_with_all(label, options):
    ALL = "All"
    opts = [ALL] + sorted(list(options))
    selected = st.sidebar.multiselect(label, opts, default=[ALL])
    return list(options) if ALL in selected or not selected else selected

# ---------------- DATA LOADER ----------------
@st.cache_data(show_spinner=True, ttl=600)
def load_and_process_data():
    creds = "secret-envoy-486405-j3-03851d061385.json"
    
    try:
        data1 = load_all_sheets(creds, "USA - DB for Marketplace Dashboard")
        data2 = load_all_sheets(creds, "IB - Database for Marketplace Dashboard")
        
        all_dfs = {}
        if data1: 
            for k, v in data1.items(): all_dfs[f"USA_{k}"] = v
        if data2: 
            for k, v in data2.items(): all_dfs[f"IB_{k}"] = v
            
    except Exception as e:
        return None, None, str(e)

    if not all_dfs: return None, None, "No data found."

    # Process Sales
    sales_list = []
    for name, df in all_dfs.items():
        if 'sales' in name.lower() and not df.empty:
            sales_list.append(df)
            
    if not sales_list: return None, None, "No Sales sheets found."
    
    sales = pd.concat(sales_list, ignore_index=True)
    sales.columns = sales.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Convert Money
    for col in ["discounted_price", "selling_commission"]:
        if col in sales.columns:
            sales[col] = pd.to_numeric(
                sales[col].astype(str).str.replace(r'[$,]', '', regex=True), 
                errors='coerce'
            ).fillna(0)
    
    sales["revenue"] = sales.get("discounted_price", 0)
    sales["orders"] = pd.to_numeric(sales.get("no_of_orders", 0), errors='coerce').fillna(0)
    sales["date"] = pd.to_datetime(sales["purchased_on"], errors="coerce")
    sales["channel"] = sales.get("channel", "Unknown").astype(str).str.strip()
    sales["type"] = sales.get("type", "Unknown").astype(str).str.strip()
    
    # Add product info if available
    if "product_name" in sales.columns:
        sales["product"] = sales["product_name"].astype(str).str.strip()
    else:
        sales["product"] = "Unknown"
    
    sales = sales.dropna(subset=["date"])

    # Process Spend
    spend_list = []
    for name, df in all_dfs.items():
        if 'channel' in name.lower() and 'spend' in name.lower():
            df = df.copy()
            col_map = {}
            for c in df.columns:
                clean = c.strip().lower()
                if clean in ['spend', 'ad spend', 'ad_spend']: col_map[c] = 'spend'
                if clean in ['date', 'purchased_on']: col_map[c] = 'date'
                if clean in ['channel']: col_map[c] = 'channel'
            
            df = df.rename(columns=col_map)
            
            if 'spend' in df.columns and 'date' in df.columns:
                df['spend'] = pd.to_numeric(
                    df['spend'].astype(str).str.replace(r'[$,]', '', regex=True), 
                    errors='coerce'
                ).fillna(0)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['channel'] = df['channel'].astype(str).str.strip() if 'channel' in df.columns else "Unknown"
                spend_list.append(df[['date', 'channel', 'spend']])

    spend = pd.concat(spend_list, ignore_index=True) if spend_list else pd.DataFrame(columns=['date', 'channel', 'spend'])
    
    return sales, spend, None

# ---------------- LOAD STATE ----------------
with st.spinner("⚡ Loading business intelligence..."):
    result = load_and_process_data()
    
    if result[2]:  # Error
        st.error(f"❌ **Data Load Failed:** {result[2]}")
        st.stop()
    
    sales_df, spend_df = result[0], result[1]
    
    if sales_df is None or sales_df.empty:
        st.warning("⚠️ No sales data available.")
        st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🎛️ Control Panel")

# Date Range
min_date = sales_df["date"].min().date()
max_date = sales_df["date"].max().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", max_date - timedelta(days=30), min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Multi-Selects
selected_channels = multiselect_with_all("📺 Channels", sales_df["channel"].unique())

if "type" in sales_df.columns:
    selected_types = multiselect_with_all("🏷️ Product Types", sales_df["type"].unique())
else:
    selected_types = []

# Comparison Period
st.sidebar.markdown("---")
comparison_period = st.sidebar.selectbox(
    "📊 Compare Against",
    ["Year over Year", "Month over Month", "Week over Week", "Previous Period"]
)

# ---------------- APPLY FILTERS ----------------
mask_sales = (
    (sales_df["date"].dt.date >= start_date) & 
    (sales_df["date"].dt.date <= end_date) &
    (sales_df["channel"].isin(selected_channels)) &
    (sales_df["type"].isin(selected_types) if "type" in sales_df.columns and selected_types else True)
)
df_s = sales_df[mask_sales]

mask_spend = (
    (spend_df["date"].dt.date >= start_date) & 
    (spend_df["date"].dt.date <= end_date) &
    (spend_df["channel"].isin(selected_channels))
)
df_sp = spend_df[mask_spend]

# Previous Period Calculation
days_diff = (end_date - start_date).days + 1

if comparison_period == "Year over Year":
    start_ly = start_date - pd.DateOffset(years=1)
    end_ly = end_date - pd.DateOffset(years=1)
elif comparison_period == "Month over Month":
    start_ly = start_date - pd.DateOffset(months=1)
    end_ly = end_date - pd.DateOffset(months=1)
elif comparison_period == "Week over Week":
    start_ly = start_date - timedelta(days=7)
    end_ly = end_date - timedelta(days=7)
else:  # Previous Period
    start_ly = start_date - timedelta(days=days_diff)
    end_ly = start_date - timedelta(days=1)

mask_sales_ly = (
    (sales_df["date"].dt.date >= start_ly.date()) & 
    (sales_df["date"].dt.date <= end_ly.date()) &
    (sales_df["channel"].isin(selected_channels))
)
df_s_ly = sales_df[mask_sales_ly]

mask_spend_ly = (
    (spend_df["date"].dt.date >= start_ly.date()) & 
    (spend_df["date"].dt.date <= end_ly.date()) &
    (spend_df["channel"].isin(selected_channels))
)
df_sp_ly = spend_df[mask_spend_ly]

# ---------------- METRIC CALCULATIONS ----------------
def calc_metrics(sales, spend):
    rev = sales["revenue"].sum()
    comm = sales["selling_commission"].sum() if "selling_commission" in sales.columns else 0
    ads = spend["spend"].sum()
    orders = sales["orders"].sum()
    
    net = (rev * SAFE_MARGIN) - ads - comm
    roas = (rev / ads) if ads > 0 else 0
    acos = (ads / rev * 100) if rev > 0 else 0
    aov = (rev / orders) if orders > 0 else 0
    
    return {
        "Revenue": rev, "Orders": orders, "Spend": ads, "Commission": comm,
        "Net": net, "ROAS": roas, "ACOS": acos, "AOV": aov
    }

curr = calc_metrics(df_s, df_sp)
prev = calc_metrics(df_s_ly, df_sp_ly)

def delta(k): 
    if prev[k] == 0: return 0
    return ((curr[k] - prev[k]) / prev[k]) * 100

# ---------------- UI: HEADER ----------------
c1, c2 = st.columns([3, 1])
with c1:
    st.title("⚡ Executive Command Center")
    st.caption(f"Analyzing performance from **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}** • {comparison_period}")
with c2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ---------------- UI: ENHANCED KPI GRID ----------------
st.markdown('<div class="section-header">💎 Key Performance Indicators</div>', unsafe_allow_html=True)

# First Row - Primary Metrics
k1, k2, k3, k4 = st.columns(4)
with k1: 
    metric_card("Total Revenue", f"{curr['Revenue']:,.0f}", delta("Revenue"), prefix="$", color="blue", icon="💰")
with k2: 
    metric_card("Total Orders", f"{curr['Orders']:,.0f}", delta("Orders"), color="cyan", icon="🛒")
with k3: 
    metric_card("Average Order Value", f"{curr['AOV']:,.2f}", delta("AOV"), prefix="$", color="purple", icon="📊")
with k4: 
    metric_card("Net Profit", f"{curr['Net']:,.0f}", delta("Net"), prefix="$", color="green", icon="💹")

st.markdown("")

# Second Row - Performance Metrics
k5, k6, k7, k8 = st.columns(4)
with k5: 
    metric_card("Ad Spend", f"{curr['Spend']:,.0f}", delta("Spend"), prefix="$", color="orange", inverse=True, icon="📢")
with k6: 
    metric_card("Selling Commission", f"{curr['Commission']:,.0f}", delta("Commission"), prefix="$", color="pink", inverse=True, icon="💳")
with k7: 
    metric_card("ROAS", f"{curr['ROAS']:.2f}", delta("ROAS"), suffix="x", color="yellow", icon="🎯")
with k8: 
    metric_card("ACOS", f"{curr['ACOS']:.1f}", delta("ACOS"), suffix="%", color="red", inverse=True, icon="📈")

# ---------------- UI: ENHANCED ANALYSIS TABS ----------------
st.markdown("")
tabs = st.tabs([
    "📈 Performance Trends", 
    "🛒 Channel Analysis", 
    "🎁 Product Insights",
    "📊 Profitability Deep Dive",
    "📋 Data Explorer"
])

# TAB 1: Performance Trends
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Revenue, Orders & Efficiency Timeline**")
        
        # Daily aggregation
        daily_rev = df_s.groupby(pd.Grouper(key="date", freq="D")).agg({
            "revenue": "sum",
            "orders": "sum"
        }).reset_index()
        daily_spend = df_sp.groupby(pd.Grouper(key="date", freq="D"))["spend"].sum().reset_index()
        daily_trend = pd.merge(daily_rev, daily_spend, on="date", how="outer").fillna(0)
        daily_trend["roas"] = daily_trend.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)
        daily_trend["aov"] = daily_trend.apply(lambda x: x["revenue"]/x["orders"] if x["orders"]>0 else 0, axis=1)
        
        fig_multi = go.Figure()
        
        # Revenue Bar
        fig_multi.add_trace(go.Bar(
            x=daily_trend["date"], y=daily_trend["revenue"],
            name="Revenue", marker_color="#3b82f6", opacity=0.7, yaxis="y"
        ))
        
        # Orders Line
        fig_multi.add_trace(go.Scatter(
            x=daily_trend["date"], y=daily_trend["orders"],
            name="Orders", line=dict(color="#ec4899", width=2), yaxis="y2"
        ))
        
        # ROAS Line
        fig_multi.add_trace(go.Scatter(
            x=daily_trend["date"], y=daily_trend["roas"],
            name="ROAS", line=dict(color="#10b981", width=3, dash='dot'), yaxis="y3"
        ))
        
        fig_multi.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
            yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False),
            yaxis3=dict(title="ROAS", overlaying="y", side="right", position=0.95, showgrid=False),
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=0, r=80, t=60, b=0),
            height=420
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    
    with col2:
        st.markdown("**AOV Trend Analysis**")
        
        # Weekly AOV
        weekly_aov = df_s.groupby(pd.Grouper(key="date", freq="W")).agg({
            "revenue": "sum",
            "orders": "sum"
        }).reset_index()
        weekly_aov["aov"] = weekly_aov.apply(lambda x: x["revenue"]/x["orders"] if x["orders"]>0 else 0, axis=1)
        
        fig_aov = go.Figure()
        fig_aov.add_trace(go.Scatter(
            x=weekly_aov["date"], y=weekly_aov["aov"],
            fill='tozeroy',
            line=dict(color="#8b5cf6", width=3),
            fillcolor="rgba(139, 92, 246, 0.2)"
        ))
        
        fig_aov.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis=dict(title="AOV ($)", showgrid=True, gridcolor="#2d303e"),
            xaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=40, b=0),
            height=420
        )
        st.plotly_chart(fig_aov, use_container_width=True)
    
    # Commission Over Time
    st.markdown("**Commission & Spend Comparison**")
    daily_comm = df_s.groupby(pd.Grouper(key="date", freq="D"))["selling_commission"].sum().reset_index() if "selling_commission" in df_s.columns else pd.DataFrame()
    
    if not daily_comm.empty:
        daily_costs = pd.merge(daily_spend, daily_comm, on="date", how="outer").fillna(0)
        
        fig_costs = go.Figure()
        fig_costs.add_trace(go.Bar(
            x=daily_costs["date"], y=daily_costs["spend"],
            name="Ad Spend", marker_color="#f97316"
        ))
        fig_costs.add_trace(go.Bar(
            x=daily_costs["date"], y=daily_costs["selling_commission"],
            name="Commission", marker_color="#ec4899"
        ))
        
        fig_costs.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode='stack',
            yaxis=dict(title="Cost ($)", showgrid=True, gridcolor="#2d303e"),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=0, r=0, t=40, b=0),
            height=350
        )
        st.plotly_chart(fig_costs, use_container_width=True)

# TAB 2: Channel Analysis
with tabs[1]:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**Channel Performance Matrix**")
        
        ch_rev = df_s.groupby("channel").agg({
            "revenue": "sum",
            "orders": "sum",
            "selling_commission": "sum" if "selling_commission" in df_s.columns else "count"
        }).reset_index()
        ch_sp = df_sp.groupby("channel")["spend"].sum().reset_index()
        ch_matrix = pd.merge(ch_rev, ch_sp, on="channel", how="outer").fillna(0)
        ch_matrix["roas"] = ch_matrix.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)
        ch_matrix["aov"] = ch_matrix.apply(lambda x: x["revenue"]/x["orders"] if x["orders"]>0 else 0, axis=1)
        ch_matrix = ch_matrix[ch_matrix["revenue"] > 0]
        
        fig_bubble = px.scatter(
            ch_matrix, x="spend", y="revenue", 
            size="roas", color="aov",
            hover_name="channel",
            hover_data={"orders": ":,", "roas": ":.2f", "aov": ":$.2f"},
            labels={"spend": "Ad Spend ($)", "revenue": "Revenue ($)", "aov": "AOV ($)"},
            size_max=80,
            text="channel",
            color_continuous_scale="viridis"
        )
        
        fig_bubble.update_traces(textposition='top center', textfont_size=10)
        fig_bubble.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.05)",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    
    with col2:
        st.markdown("**Channel Revenue Share**")
        
        fig_pie = px.pie(
            ch_matrix, values="revenue", names="channel",
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=250,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("**Channel Efficiency Ranking**")
        
        ch_rank = ch_matrix.sort_values("roas", ascending=False)[["channel", "roas", "acos"]].head(10)
        ch_rank["acos"] = ch_rank.apply(lambda x: (ch_matrix[ch_matrix["channel"]==x["channel"]]["spend"].sum() / ch_matrix[ch_matrix["channel"]==x["channel"]]["revenue"].sum() * 100) if ch_matrix[ch_matrix["channel"]==x["channel"]]["revenue"].sum() > 0 else 0, axis=1)
        
        fig_rank = go.Figure()
        fig_rank.add_trace(go.Bar(
            y=ch_rank["channel"], x=ch_rank["roas"],
            orientation='h',
            marker=dict(
                color=ch_rank["roas"],
                colorscale='Viridis',
                showscale=False
            ),
            text=ch_rank["roas"].apply(lambda x: f"{x:.2f}x"),
            textposition='outside'
        ))
        
        fig_rank.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="ROAS", showgrid=True, gridcolor="#2d303e"),
            yaxis=dict(title=""),
            margin=dict(l=0, r=0, t=20, b=0),
            height=250
        )
        st.plotly_chart(fig_rank, use_container_width=True)

# TAB 3: Product Insights
with tabs[2]:
    if "product" in df_s.columns and df_s["product"].nunique() > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Products by Revenue**")
            
            prod_perf = df_s.groupby("product").agg({
                "revenue": "sum",
                "orders": "sum"
            }).reset_index()
            prod_perf["aov"] = prod_perf["revenue"] / prod_perf["orders"]
            prod_perf = prod_perf.sort_values("revenue", ascending=False).head(10)
            
            fig_prod = px.bar(
                prod_perf, x="revenue", y="product",
                orientation='h',
                color="aov",
                color_continuous_scale="Blues",
                labels={"revenue": "Revenue ($)", "product": "Product", "aov": "AOV ($)"}
            )
            
            fig_prod.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_prod, use_container_width=True)
        
        with col2:
            st.markdown("**Product Performance Metrics**")
            
            top_products = prod_perf.head(5)
            
            for idx, row in top_products.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(30, 32, 40, 0.5); padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 3px solid #3b82f6;">
                        <div style="font-weight: 600; color: #e5e7eb; margin-bottom: 8px;">{row['product'][:40]}...</div>
                        <div style="display: flex; justify-content: space-between; font-size: 13px;">
                            <span style="color: #9ca3af;">Revenue: <strong style="color: #3b82f6;">${row['revenue']:,.0f}</strong></span>
                            <span style="color: #9ca3af;">Orders: <strong style="color: #10b981;">{row['orders']:,.0f}</strong></span>
                            <span style="color: #9ca3af;">AOV: <strong style="color: #8b5cf6;">${row['aov']:,.2f}</strong></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("**Revenue Distribution**")
            
            fig_tree = px.treemap(
                prod_perf.head(15), 
                path=['product'], 
                values='revenue',
                color='aov',
                color_continuous_scale='Viridis'
            )
            
            fig_tree.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
                height=300
            )
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.info("📦 Product-level data not available in the current dataset.")

# TAB 4: Profitability Deep Dive
with tabs[3]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Profit Waterfall**")
        cost_goods = curr['Revenue'] * (1 - SAFE_MARGIN)
        
        fig_water = go.Figure(go.Waterfall(
            name = "Profitability", orientation = "v",
            measure = ["relative", "relative", "relative", "relative", "total"],
            x = ["Gross Revenue", "COGS", "Commission", "Ad Spend", "Net Profit"],
            textposition = "outside",
            text = [f"${curr['Revenue']/1000:.1f}k", f"-${cost_goods/1000:.1f}k", 
                    f"-${curr['Commission']/1000:.1f}k", f"-${curr['Spend']/1000:.1f}k", f"${curr['Net']/1000:.1f}k"],
            y = [curr['Revenue'], -cost_goods, -curr['Commission'], -curr['Spend'], curr['Net']],
            connector = {"line":{"color":"#6366f1"}},
            decreasing = {"marker":{"color":"#f87171"}},
            increasing = {"marker":{"color":"#10b981"}},
            totals = {"marker":{"color":"#3b82f6"}}
        ))
        
        fig_water.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=40, b=0),
            height=450
        )
        st.plotly_chart(fig_water, use_container_width=True)
    
    with col2:
        st.markdown("**Cost Breakdown Analysis**")
        
        costs_data = pd.DataFrame({
            "Category": ["COGS", "Ad Spend", "Commission", "Net Profit"],
            "Amount": [cost_goods, curr['Spend'], curr['Commission'], curr['Net']],
            "Percentage": [
                (cost_goods / curr['Revenue'] * 100) if curr['Revenue'] > 0 else 0,
                (curr['Spend'] / curr['Revenue'] * 100) if curr['Revenue'] > 0 else 0,
                (curr['Commission'] / curr['Revenue'] * 100) if curr['Revenue'] > 0 else 0,
                (curr['Net'] / curr['Revenue'] * 100) if curr['Revenue'] > 0 else 0
            ]
        })
        
        fig_costs_pie = px.pie(
            costs_data, values="Amount", names="Category",
            hole=0.6,
            color_discrete_sequence=['#f87171', '#f97316', '#ec4899', '#10b981']
        )
        fig_costs_pie.update_traces(textposition='outside', textinfo='percent+label')
        fig_costs_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            height=450,
            margin=dict(l=0, r=0, t=40, b=0),
            annotations=[dict(text=f'${curr["Revenue"]/1000:.0f}k<br>Total', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig_costs_pie, use_container_width=True)
    
    # Profitability metrics table
    st.markdown("**Profitability Metrics Summary**")
    
    profit_metrics = pd.DataFrame({
        "Metric": ["Gross Revenue", "COGS", "Gross Margin", "Ad Spend", "Commission", "Total Costs", "Net Profit", "Profit Margin"],
        "Amount": [
            f"${curr['Revenue']:,.0f}",
            f"${cost_goods:,.0f}",
            f"${curr['Revenue'] - cost_goods:,.0f}",
            f"${curr['Spend']:,.0f}",
            f"${curr['Commission']:,.0f}",
            f"${cost_goods + curr['Spend'] + curr['Commission']:,.0f}",
            f"${curr['Net']:,.0f}",
            f"{(curr['Net'] / curr['Revenue'] * 100) if curr['Revenue'] > 0 else 0:.2f}%"
        ]
    })
    
    st.dataframe(profit_metrics, use_container_width=True, hide_index=True)

# TAB 5: Data Explorer
with tabs[4]:
    st.markdown('<div class="section-header">📋 Performance Data Explorer</div>', unsafe_allow_html=True)
    
    # Channel Performance Table
    tbl = ch_matrix.copy()
    if "selling_commission" in df_s.columns:
        tbl["commission"] = df_s.groupby("channel")["selling_commission"].sum().values
    else:
        tbl["commission"] = 0
    
    tbl["acos"] = tbl.apply(lambda x: (x["spend"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
    tbl["net"] = (tbl["revenue"] * SAFE_MARGIN) - tbl["spend"] - tbl.get("commission", 0)
    tbl["profit_margin"] = tbl.apply(lambda x: (x["net"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
    
    st.dataframe(
        tbl,
        column_config={
            "channel": st.column_config.TextColumn("Channel", width="medium"),
            "revenue": st.column_config.ProgressColumn("Revenue", format="$%d", min_value=0, max_value=int(tbl["revenue"].max())),
            "orders": st.column_config.NumberColumn("Orders", format="%d"),
            "aov": st.column_config.NumberColumn("AOV", format="$%.2f"),
            "spend": st.column_config.NumberColumn("Ad Spend", format="$%d"),
            "commission": st.column_config.NumberColumn("Commission", format="$%d"),
            "roas": st.column_config.NumberColumn("ROAS", format="%.2fx"),
            "acos": st.column_config.NumberColumn("ACOS", format="%.1f%%"),
            "net": st.column_config.NumberColumn("Net Profit", format="$%d"),
            "profit_margin": st.column_config.NumberColumn("Profit Margin", format="%.1f%%"),
        },
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Export Options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = tbl.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Channel Report (CSV)", csv, "channel_performance.csv", "text/csv", use_container_width=True)
    
    with col2:
        if not df_s.empty:
            sales_csv = df_s.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Sales Data (CSV)", sales_csv, "sales_data.csv", "text/csv", use_container_width=True)
    
    with col3:
        if not df_sp.empty:
            spend_csv = df_sp.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Spend Data (CSV)", spend_csv, "spend_data.csv", "text/csv", use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div style='text-align: left; color: #6b7280; font-size: 12px;'>📅 Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='text-align: center; color: #6b7280; font-size: 12px;'>⚙️ Safe Margin: {SAFE_MARGIN*100:.0f}%</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div style='text-align: right; color: #6b7280; font-size: 12px;'>📊 Data Points: {len(df_s):,}</div>", unsafe_allow_html=True)