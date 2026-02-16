import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets
from datetime import date, timedelta, datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
import hashlib

# Configure Plotly
import plotly.io as pio
pio.templates.default = "plotly_dark"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Marketplace Business Insights",
    page_icon="📊",
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
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(30, 32, 40, 0.6);
        border-radius: 8px;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(42, 45, 58, 0.8);
    }

    /* RECOMMENDATION CARDS */
    .rec-card {
        background: rgba(30, 32, 40, 0.6);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        transition: transform 0.2s;
    }
    .rec-card:hover {
        transform: translateX(5px);
        background: rgba(42, 45, 58, 0.8);
    }
    .rec-title {
        font-weight: 700;
        font-size: 16px;
        color: #fff;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .rec-body {
        color: #9ca3af;
        font-size: 14px;
        margin-top: 4px;
    }
    .rec-high { border-left-color: #10b981; } /* Green - Scale */
    .rec-warn { border-left-color: #f59e0b; } /* Orange - Optimize */
    .rec-crit { border-left-color: #ef4444; } /* Red - Cut */
    .rec-info { border-left-color: #3b82f6; } /* Blue - Info */
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
        # Check for sales sheets (usually named 'Sales_data')
        if 'sales' in name.lower() and not df.empty:
            df = df.copy()
            
            # Normalize columns map
            col_map = {}
            for col in df.columns:
                clean_col = col.strip().lower().replace(' ', '_').replace('-', '_')
                col_map[col] = clean_col
                
                # Explicitly map SKU/Parent columns regardless of case
                if clean_col == 'parent': col_map[col] = 'Parent'
                elif clean_col == 'sku': col_map[col] = 'SKU'
                
            df = df.rename(columns=col_map)
            sales_list.append(df)
            
    if not sales_list: return None, None, "No Sales sheets found."
    
    sales = pd.concat(sales_list, ignore_index=True)
    
    # Ensure all required lowercase columns exist
    sales.columns = [c if c in ['Parent', 'SKU'] else c.lower() for c in sales.columns]
    
    # Convert Money & Numbers
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
    
    # Handle SKU information (Fill missing if not found)
    if "Parent" not in sales.columns:
        sales["Parent"] = "Unknown"
    else:
        sales["Parent"] = sales["Parent"].astype(str).str.strip()
        
    if "SKU" not in sales.columns:
        sales["SKU"] = "Unknown"
    else:
        sales["SKU"] = sales["SKU"].astype(str).str.strip()
    
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
selected_channels = multiselect_with_all("📺 Marketplaces", sales_df["channel"].unique())

if "type" in sales_df.columns:
    selected_types = multiselect_with_all("🏷️ Product Types", sales_df["type"].unique())
else:
    selected_types = []

# Comparison Period
st.sidebar.markdown("---")
comparison_period = st.sidebar.selectbox(
    "📊 Compare Against",
    ["Year over Year", "Month over Month"]
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

def generate_insights(df_channel, current_metrics):
    insights = []
    
    # 1. CHANNEL SCALING OPPORTUNITIES (High ROAS)
    if 'roas' in df_channel.columns:
        scale_ops = df_channel[df_channel['roas'] >= 3.0]
        for _, row in scale_ops.iterrows():
            insights.append({
                "type": "scale",
                "title": f"🚀 Scale Up: {row['channel']}",
                "msg": f"ROAS is strong at {row['roas']:.2f}x. Consider increasing daily budget by 15-20% to maximize volume while maintaining profitability.",
                "metric": f"{row['roas']:.2f}x ROAS"
            })

    # 2. BLEEDING CAMPAIGNS (Low ROAS / High Spend)
    if 'roas' in df_channel.columns and 'spend' in df_channel.columns:
        bleeding = df_channel[(df_channel['roas'] < 1.5) & (df_channel['spend'] > 500)]
        for _, row in bleeding.iterrows():
            insights.append({
                "type": "crit",
                "title": f"🛑 High Spend / Low Return: {row['channel']}",
                "msg": f"This channel has spent ${row['spend']:,.0f} with only {row['roas']:.2f}x ROAS. Review search terms, pause bleeding keywords, or lower bids immediately.",
                "metric": f"${row['spend']:,.0f} Spend"
            })

    # 3. PROFITABILITY WARNING
    if current_metrics['Net'] < 0:
        insights.append({
            "type": "crit",
            "title": "📉 Net Loss Alert",
            "msg": "The business is currently operating at a net loss for the selected period. Prioritize cutting Ad Spend on channels with < 2.0 ROAS immediately.",
            "metric": f"${current_metrics['Net']:,.0f}"
        })
    elif current_metrics['Revenue'] > 0 and (current_metrics['Net'] / current_metrics['Revenue']) < 0.10:
        insights.append({
            "type": "warn",
            "title": "⚠️ Thin Margins",
            "msg": "Net Profit margin is below 10%. Keep a close eye on COGS and Commission rates.",
            "metric": f"{(current_metrics['Net']/current_metrics['Revenue']*100):.1f}% Margin"
        })

    # 4. AOV OPPORTUNITIES
    if current_metrics['AOV'] > 0 and current_metrics['AOV'] < 50: # Example threshold
        insights.append({
            "type": "info",
            "title": "📦 Bundle Opportunity",
            "msg": "AOV is below $50. Consider creating 'Buy 2 Save 10%' bundles or adding post-purchase upsells to increase basket size.",
            "metric": f"${current_metrics['AOV']:.2f} Avg"
        })

    return insights

curr = calc_metrics(df_s, df_sp)
prev = calc_metrics(df_s_ly, df_sp_ly)

def delta(k): 
    if prev[k] == 0: return 0
    return ((curr[k] - prev[k]) / prev[k]) * 100

# ---------------- UI: HEADER ----------------
c1, c2 = st.columns([3, 1])
with c1:
    st.title("📊 Marketplace Business Insights")
    st.caption(f"Analyzing performance from **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}** • {comparison_period}")
with c2:
    if st.button("🔄 Refresh Data", key="refresh_btn"):
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
    "🚀 Strategy & Recommendations",
    "📈 Performance Trends", 
    "🛒 Marketplace Analysis", 
    "🏷️ SKU Analysis",
    "📊 Profitability Deep Dive",
    "🔮 Forecasting & Predictions",
    "🧪 A/B Test Tracker",
    "📅 Weekly Reports",
    "📋 Data Explorer"
])

# TAB 1: Strategy & Recommendations
with tabs[0]:
    st.markdown('<div class="section-header">🧠 AI Strategic Insights</div>', unsafe_allow_html=True)
    
    # Generate insights based on the Channel Matrix
    ch_rev_rec = df_s.groupby("channel")["revenue"].sum().reset_index()
    ch_sp_rec = df_sp.groupby("channel")["spend"].sum().reset_index()
    ch_matrix_rec = pd.merge(ch_rev_rec, ch_sp_rec, on="channel", how="outer").fillna(0)
    ch_matrix_rec["roas"] = ch_matrix_rec.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)
    
    recommendations = generate_insights(ch_matrix_rec, curr)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not recommendations:
            st.info("✅ Business looks stable. No critical alerts found based on current thresholds.")
        else:
            for rec in recommendations:
                # Map type to CSS class and icon
                css_map = {
                    "scale": ("rec-high", "📈"),
                    "warn": ("rec-warn", "⚠️"),
                    "crit": ("rec-crit", "🚨"),
                    "info": ("rec-info", "💡")
                }
                style_class, icon = css_map.get(rec['type'], ("rec-info", "ℹ️"))
                
                st.markdown(f"""
                <div class="rec-card {style_class}">
                    <div class="rec-title">{icon} {rec['title']} <span style="margin-left:auto; font-size:12px; opacity:0.8; background:rgba(255,255,255,0.1); padding:2px 8px; border-radius:10px;">{rec['metric']}</span></div>
                    <div class="rec-body">{rec['msg']}</div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("**🎯 Projected Outcome**")
        st.caption("If you optimize based on these insights:")
        
        # Simple projection logic
        potential_savings = ch_matrix_rec[ch_matrix_rec['roas'] < 1.5]['spend'].sum() * 0.5 # Assume we cut 50% of bad spend
        potential_gain = ch_matrix_rec[ch_matrix_rec['roas'] >= 3.0]['revenue'].sum() * 0.2 # Assume 20% growth on good channels
        
        new_net = curr['Net'] + potential_savings + (potential_gain * 0.2) # Assuming 20% margin on new rev
        
        st.metric("Potential Wasted Ad Spend", f"${potential_savings:,.0f}", help="Spend on channels with < 1.5 ROAS")
        st.metric("Revenue Growth Opportunity", f"${potential_gain:,.0f}", help="Projected lift from scaling high ROAS channels")
        
        st.markdown("---")
        st.markdown(f"**Projected Net Profit:**")
        st.markdown(f"<h2 style='color:#10b981'>${new_net:,.0f}</h2>", unsafe_allow_html=True)
        st.caption(f"Vs Current: ${curr['Net']:,.0f}")

# TAB 2: Performance Trends
with tabs[1]:
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
        st.plotly_chart(fig_multi, config={'displayModeBar': False})
    
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
        st.plotly_chart(fig_aov, config={'displayModeBar': False})
    
    # Commission Over Time
    st.markdown("**Commission & Spend Comparison**")
    if "selling_commission" in df_s.columns:
        daily_comm = df_s.groupby(pd.Grouper(key="date", freq="D"))["selling_commission"].sum().reset_index()
        
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
            st.plotly_chart(fig_costs, config={'displayModeBar': False})

# TAB 3: Marketplace Analysis
with tabs[2]:
    st.markdown('<div class="section-header">🛒 Marketplace Performance Analysis</div>', unsafe_allow_html=True)
    
    # Info box explaining the analysis
    st.markdown("""
    <div class="info-box">
    📊 <strong>Understanding This Analysis:</strong><br>
    • <strong>Bubble Size</strong> = ROAS (bigger bubbles = better efficiency)<br>
    • <strong>Bubble Color</strong> = Average Order Value (darker = higher AOV)<br>
    • <strong>Top Right</strong> = High revenue + High spend (established channels)<br>
    • <strong>Top Left</strong> = High revenue + Low spend (highly efficient, scale these!)<br>
    • <strong>Bottom Right</strong> = Low revenue + High spend (bleeding money, optimize or cut)
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**Marketplace Performance Matrix**")
        
        ch_rev = df_s.groupby("channel").agg({
            "revenue": "sum",
            "orders": "sum"
        }).reset_index()
        
        if "selling_commission" in df_s.columns:
            ch_comm = df_s.groupby("channel")["selling_commission"].sum().reset_index()
            ch_rev = pd.merge(ch_rev, ch_comm, on="channel", how="left").fillna(0)
        
        ch_sp = df_sp.groupby("channel")["spend"].sum().reset_index()
        ch_matrix = pd.merge(ch_rev, ch_sp, on="channel", how="outer").fillna(0)
        ch_matrix["roas"] = ch_matrix.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)
        ch_matrix["aov"] = ch_matrix.apply(lambda x: x["revenue"]/x["orders"] if x["orders"]>0 else 0, axis=1)
        ch_matrix["acos"] = ch_matrix.apply(lambda x: (x["spend"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
        ch_matrix = ch_matrix[ch_matrix["revenue"] > 0]
        
        fig_bubble = px.scatter(
            ch_matrix, x="spend", y="revenue", 
            size="roas", color="aov",
            hover_name="channel",
            hover_data={"orders": ":,", "roas": ":.2f", "aov": ":$.2f", "acos": ":.1f%"},
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
        st.plotly_chart(fig_bubble, config={'displayModeBar': False})
    
    with col2:
        st.markdown("**Marketplace Revenue Share**")
        st.caption("Distribution of revenue across marketplaces")
        
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
        st.plotly_chart(fig_pie, config={'displayModeBar': False})
        
        st.markdown("**Marketplace Efficiency Ranking**")
        st.caption("Ranked by ROAS (Return on Ad Spend)")
        
        ch_rank = ch_matrix.sort_values("roas", ascending=False)[["channel", "roas", "revenue", "spend"]].head(10).copy()
        ch_rank["acos"] = ch_rank.apply(lambda x: (x["spend"] / x["revenue"] * 100) if x["revenue"] > 0 else 0, axis=1)
        
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
        st.plotly_chart(fig_rank, config={'displayModeBar': False})
        
        # Quick Actions
        st.markdown("**⚡ Quick Actions**")
        st.caption("Recommended actions based on ROAS performance")
        for _, row in ch_matrix.sort_values('roas', ascending=False).head(3).iterrows():
            if row['roas'] >= 3.0:
                st.success(f"**{row['channel']}**: Scale budget +20%")
            elif row['roas'] < 1.5:
                st.error(f"**{row['channel']}**: Reduce spend -30%")
            else:
                st.info(f"**{row['channel']}**: Optimize campaigns")
    
    # Detailed Marketplace Breakdown Table (full width below charts)
    st.markdown("---")
    st.markdown("**📋 Detailed Marketplace Metrics**")
    st.caption("Complete performance breakdown for all marketplaces")
    
    display_ch = ch_matrix.copy()
    display_ch = display_ch.sort_values('revenue', ascending=False)
    display_ch['profit_margin'] = display_ch.apply(
        lambda x: ((x['revenue'] * SAFE_MARGIN - x['spend'] - x.get('selling_commission', 0)) / x['revenue'] * 100) if x['revenue'] > 0 else 0, 
        axis=1
    )
    
    st.dataframe(
        display_ch[['channel', 'revenue', 'orders', 'aov', 'spend', 'roas', 'acos', 'profit_margin']],
        column_config={
            "channel": "Marketplace",
            "revenue": st.column_config.NumberColumn("Revenue", format="$%d"),
            "orders": st.column_config.NumberColumn("Orders", format="%d"),
            "aov": st.column_config.NumberColumn("AOV", format="$%.2f"),
            "spend": st.column_config.NumberColumn("Ad Spend", format="$%d"),
            "roas": st.column_config.NumberColumn("ROAS", format="%.2fx"),
            "acos": st.column_config.NumberColumn("ACOS", format="%.1f%%"),
            "profit_margin": st.column_config.NumberColumn("Profit %", format="%.1f%%"),
        },
        hide_index=True,
        height=350
    )

# TAB 4: SKU Analysis
with tabs[3]:
    st.markdown('<div class="section-header">🏷️ SKU Performance Analysis</div>', unsafe_allow_html=True)
    
    if "Parent" in df_s.columns and df_s["Parent"].nunique() > 1:
        # Calculate Parent SKU Performance
        Parent_perf = df_s.groupby("Parent").agg({
            "revenue": "sum",
            "orders": "sum"
        }).reset_index()
        Parent_perf["aov"] = Parent_perf["revenue"] / Parent_perf["orders"]
        Parent_perf = Parent_perf.sort_values("revenue", ascending=False).head(10)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Top 10 Parent SKUs by Revenue**")
            
            fig_sku_bar = px.bar(
                Parent_perf, 
                x="revenue", 
                y="Parent",
                orientation='h',
                color="orders",
                color_continuous_scale="Blues",
                labels={"revenue": "Revenue ($)", "Parent": "Parent SKU", "orders": "Orders"}
            )
            
            fig_sku_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(tickmode='linear')
            )
            st.plotly_chart(fig_sku_bar, config={'displayModeBar': False})
        
        with col2:
            st.markdown("**SKU Revenue Distribution**")
            
            fig_sku_tree = px.treemap(
                Parent_perf, 
                path=['Parent'], 
                values='revenue',
                color='aov',
                color_continuous_scale='Viridis',
                labels={"revenue": "Revenue", "aov": "AOV"}
            )
            
            fig_sku_tree.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
                height=450
            )
            st.plotly_chart(fig_sku_tree, config={'displayModeBar': False})
        
        # Detailed SKU Cards with Child SKUs
        st.markdown("---")
        st.markdown("**📦 Detailed SKU Breakdown (Click to expand for Child SKUs)**")
        
        for idx, parent_row in Parent_perf.iterrows():
            parent = parent_row['Parent']
            
            # Get child SKUs for this parent
            if "SKU" in df_s.columns:
                child_data = df_s[df_s["Parent"] == parent].groupby("SKU").agg({
                    "revenue": "sum",
                    "orders": "sum"
                }).reset_index()
                child_data["aov"] = child_data["revenue"] / child_data["orders"]
                child_data = child_data.sort_values("revenue", ascending=False)
                
                has_children = len(child_data) > 0 and child_data["SKU"].iloc[0] != "Unknown"
            else:
                has_children = False
                child_data = pd.DataFrame()
            
            # Create expander for each parent SKU
            with st.expander(f"🏷️ {parent} - ${parent_row['revenue']:,.0f} Revenue", expanded=False):
                # Parent SKU metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Revenue", f"${parent_row['revenue']:,.0f}")
                with col2:
                    st.metric("Orders", f"{parent_row['orders']:,.0f}")
                with col3:
                    st.metric("AOV", f"${parent_row['aov']:,.2f}")
                with col4:
                    if has_children:
                        st.metric("Child SKUs", f"{len(child_data)}")
                    else:
                        st.metric("Child SKUs", "N/A")
                
                # Show child SKUs if available
                if has_children:
                    st.markdown("---")
                    st.markdown("**Child SKUs Performance:**")
                    
                    # Create a nice table for child SKUs
                    child_display = child_data.copy()
                    child_display["revenue"] = child_display["revenue"].apply(lambda x: f"${x:,.0f}")
                    child_display["orders"] = child_display["orders"].apply(lambda x: f"{x:,.0f}")
                    child_display["aov"] = child_display["aov"].apply(lambda x: f"${x:,.2f}")
                    
                    st.dataframe(
                        child_display,
                        column_config={
                            "SKU": st.column_config.TextColumn("SKU", width="medium"),
                            "revenue": st.column_config.TextColumn("Revenue", width="small"),
                            "orders": st.column_config.TextColumn("Orders", width="small"),
                            "aov": st.column_config.TextColumn("AOV", width="small"),
                        },
                        hide_index=True,
                        height=min(300, 50 + len(child_display) * 35)
                    )
                    
                    # Visual breakdown
                    if len(child_data) > 1:
                        fig_child = px.pie(
                            child_data, 
                            values="revenue", 
                            names="SKU",
                            title="Revenue Distribution by Child SKU",
                            color_discrete_sequence=px.colors.sequential.Plasma
                        )
                        fig_child.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=300,
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        st.plotly_chart(fig_child, config={'displayModeBar': False})
                else:
                    st.info("ℹ️ No child SKU data available for this parent SKU")
        
    else:
        st.info("📦 SKU data not available in the current dataset. Please ensure 'Parent' column exists in your data.")

# TAB 5: Profitability Deep Dive
with tabs[4]:
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
        st.plotly_chart(fig_water, config={'displayModeBar': False})
    
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
        st.plotly_chart(fig_costs_pie, config={'displayModeBar': False})
    
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
    
    st.dataframe(profit_metrics, hide_index=True)

# TAB 6: Forecasting & Predictions
with tabs[5]:
    st.markdown('<div class="section-header">🔮 Revenue & Performance Forecasting</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    📊 <strong>Forecasting Model:</strong> Uses Linear Regression on historical trends to predict future performance. 
    Predictions become more accurate with more historical data (recommended: 3+ months).
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**📈 Revenue Forecast (Next 30 Days)**")
        
        # Prepare data for forecasting
        daily_revenue = df_s.groupby(pd.Grouper(key="date", freq="D"))["revenue"].sum().reset_index()
        daily_revenue = daily_revenue.sort_values("date")
        
        if len(daily_revenue) >= 14:  # Need at least 2 weeks of data
            # Prepare features
            daily_revenue['days_since_start'] = (daily_revenue['date'] - daily_revenue['date'].min()).dt.days
            X = daily_revenue['days_since_start'].values.reshape(-1, 1)
            y = daily_revenue['revenue'].values
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict next 30 days
            last_day = daily_revenue['days_since_start'].max()
            future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
            future_dates = pd.date_range(daily_revenue['date'].max() + timedelta(days=1), periods=30)
            predictions = model.predict(future_days)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'predicted_revenue': predictions
            })
            
            # Combine historical and forecast
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=daily_revenue['date'], 
                y=daily_revenue['revenue'],
                name='Historical Revenue',
                line=dict(color='#3b82f6', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['predicted_revenue'],
                name='Forecast',
                line=dict(color='#10b981', width=2, dash='dash')
            ))
            
            fig_forecast.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
                xaxis=dict(title="Date", showgrid=False),
                legend=dict(orientation="h", y=1.1, x=0),
                margin=dict(l=0, r=0, t=40, b=0),
                height=400
            )
            st.plotly_chart(fig_forecast, config={'displayModeBar': False})
            
            # Forecast summary
            st.markdown("**📊 Forecast Summary**")
            forecast_total = forecast_df['predicted_revenue'].sum()
            historical_avg = daily_revenue['revenue'].tail(30).mean()
            forecast_avg = forecast_df['predicted_revenue'].mean()
            growth_rate = ((forecast_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Predicted Revenue (30 days)", f"${forecast_total:,.0f}")
            with col_b:
                st.metric("Avg Daily Revenue (Forecast)", f"${forecast_avg:,.0f}")
            with col_c:
                st.metric("Growth Rate", f"{growth_rate:+.1f}%")
        else:
            st.warning("⚠️ Need at least 14 days of historical data for accurate forecasting. Current data: {} days".format(len(daily_revenue)))
    
    with col2:
        st.markdown("**🎯 Key Predictions**")
        
        if len(daily_revenue) >= 14:
            # ROAS prediction
            daily_full = daily_revenue.merge(
                df_sp.groupby(pd.Grouper(key="date", freq="D"))["spend"].sum().reset_index(),
                on="date", how="left"
            ).fillna(0)
            daily_full['roas'] = daily_full.apply(lambda x: x['revenue']/x['spend'] if x['spend'] > 0 else 0, axis=1)
            
            avg_roas = daily_full['roas'].mean()
            recent_roas = daily_full['roas'].tail(7).mean()
            
            st.metric("Predicted ROAS (7 days)", f"{recent_roas:.2f}x", 
                     delta=f"{((recent_roas - avg_roas) / avg_roas * 100) if avg_roas > 0 else 0:.1f}%")
            
            # Orders prediction
            daily_orders = df_s.groupby(pd.Grouper(key="date", freq="D"))["orders"].sum().reset_index()
            avg_orders = daily_orders['orders'].tail(30).mean()
            
            st.metric("Predicted Daily Orders", f"{avg_orders:.0f}")
            
            # Confidence level
            confidence = min(100, (len(daily_revenue) / 90) * 100)  # More data = higher confidence
            st.markdown(f"**Model Confidence:** {confidence:.0f}%")
            st.progress(confidence / 100)
            
            st.markdown("---")
            st.markdown("**💡 Forecast Insights**")
            if growth_rate > 10:
                st.success("📈 Strong growth expected! Consider scaling operations.")
            elif growth_rate < -10:
                st.error("📉 Decline predicted. Review marketing strategy.")
            else:
                st.info("➡️ Stable performance expected.")
        else:
            st.info("Add more historical data to see predictions")

# TAB 7: A/B Test Tracker
with tabs[6]:
    st.markdown('<div class="section-header">🧪 A/B Test Performance Tracker</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    💡 <strong>How it works:</strong> Track performance of different campaigns, products, or strategies. 
    Add test variants below and compare their metrics side-by-side.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for A/B tests
    if 'ab_tests' not in st.session_state:
        st.session_state.ab_tests = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**➕ Create New A/B Test**")
        
        with st.form("ab_test_form"):
            test_name = st.text_input("Test Name", placeholder="e.g., Holiday Campaign 2024")
            
            col_a, col_b = st.columns(2)
            with col_a:
                variant_a_name = st.text_input("Variant A Name", value="Control", placeholder="e.g., Original Ad")
                variant_a_channel = st.selectbox("Variant A Marketplace", df_s["channel"].unique(), key="var_a_ch")
                var_a_date_start = st.date_input("Variant A Start", value=start_date, key="var_a_start")
                var_a_date_end = st.date_input("Variant A End", value=end_date, key="var_a_end")
            
            with col_b:
                variant_b_name = st.text_input("Variant B Name", value="Test", placeholder="e.g., New Ad Creative")
                variant_b_channel = st.selectbox("Variant B Marketplace", df_s["channel"].unique(), key="var_b_ch")
                var_b_date_start = st.date_input("Variant B Start", value=start_date, key="var_b_start")
                var_b_date_end = st.date_input("Variant B End", value=end_date, key="var_b_end")
            
            submitted = st.form_submit_button("🚀 Create Test")
            
            if submitted and test_name:
                # Calculate metrics for both variants
                # Variant A
                mask_a = (
                    (sales_df["date"].dt.date >= var_a_date_start) & 
                    (sales_df["date"].dt.date <= var_a_date_end) &
                    (sales_df["channel"] == variant_a_channel)
                )
                df_a = sales_df[mask_a]
                mask_spend_a = (
                    (spend_df["date"].dt.date >= var_a_date_start) & 
                    (spend_df["date"].dt.date <= var_a_date_end) &
                    (spend_df["channel"] == variant_a_channel)
                )
                spend_a = spend_df[mask_spend_a]["spend"].sum()
                
                # Variant B
                mask_b = (
                    (sales_df["date"].dt.date >= var_b_date_start) & 
                    (sales_df["date"].dt.date <= var_b_date_end) &
                    (sales_df["channel"] == variant_b_channel)
                )
                df_b = sales_df[mask_b]
                mask_spend_b = (
                    (spend_df["date"].dt.date >= var_b_date_start) & 
                    (spend_df["date"].dt.date <= var_b_date_end) &
                    (spend_df["channel"] == variant_b_channel)
                )
                spend_b = spend_df[mask_spend_b]["spend"].sum()
                
                test_data = {
                    'test_name': test_name,
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'variant_a': {
                        'name': variant_a_name,
                        'channel': variant_a_channel,
                        'revenue': df_a['revenue'].sum(),
                        'orders': df_a['orders'].sum(),
                        'spend': spend_a,
                        'roas': (df_a['revenue'].sum() / spend_a) if spend_a > 0 else 0
                    },
                    'variant_b': {
                        'name': variant_b_name,
                        'channel': variant_b_channel,
                        'revenue': df_b['revenue'].sum(),
                        'orders': df_b['orders'].sum(),
                        'spend': spend_b,
                        'roas': (df_b['revenue'].sum() / spend_b) if spend_b > 0 else 0
                    }
                }
                
                st.session_state.ab_tests.append(test_data)
                st.success(f"✅ Test '{test_name}' created successfully!")
                st.rerun()
    
    with col2:
        st.markdown("**📊 Test Guidelines**")
        st.markdown("""
        **Best Practices:**
        - Run tests for at least 7 days
        - Keep one variable different
        - Use same date ranges when possible
        - Track statistical significance
        
        **Metrics to Compare:**
        - Revenue & Orders
        - ROAS & ACOS
        - AOV (Average Order Value)
        - Conversion Rate
        """)
    
    # Display existing tests
    if st.session_state.ab_tests:
        st.markdown("---")
        st.markdown("**🔬 Active & Completed Tests**")
        
        for idx, test in enumerate(st.session_state.ab_tests):
            with st.expander(f"🧪 {test['test_name']} - Created: {test['created_at']}", expanded=True):
                # Comparison metrics
                col1, col2, col3 = st.columns(3)
                
                var_a = test['variant_a']
                var_b = test['variant_b']
                
                # Calculate improvements
                revenue_improvement = ((var_b['revenue'] - var_a['revenue']) / var_a['revenue'] * 100) if var_a['revenue'] > 0 else 0
                orders_improvement = ((var_b['orders'] - var_a['orders']) / var_a['orders'] * 100) if var_a['orders'] > 0 else 0
                roas_improvement = ((var_b['roas'] - var_a['roas']) / var_a['roas'] * 100) if var_a['roas'] > 0 else 0
                
                # Display comparison
                st.markdown(f"**{var_a['name']}** vs **{var_b['name']}**")
                
                comparison_data = pd.DataFrame({
                    'Metric': ['Revenue', 'Orders', 'Ad Spend', 'ROAS'],
                    var_a['name']: [
                        f"${var_a['revenue']:,.0f}",
                        f"{var_a['orders']:,.0f}",
                        f"${var_a['spend']:,.0f}",
                        f"{var_a['roas']:.2f}x"
                    ],
                    var_b['name']: [
                        f"${var_b['revenue']:,.0f}",
                        f"{var_b['orders']:,.0f}",
                        f"${var_b['spend']:,.0f}",
                        f"{var_b['roas']:.2f}x"
                    ],
                    'Change': [
                        f"{revenue_improvement:+.1f}%",
                        f"{orders_improvement:+.1f}%",
                        f"{((var_b['spend'] - var_a['spend']) / var_a['spend'] * 100) if var_a['spend'] > 0 else 0:+.1f}%",
                        f"{roas_improvement:+.1f}%"
                    ]
                })
                
                st.dataframe(comparison_data, hide_index=True, use_container_width=True)
                
                # Winner determination
                st.markdown("**🏆 Test Result:**")
                if revenue_improvement > 10 and roas_improvement > 5:
                    st.success(f"✅ **{var_b['name']} is the clear winner!** (+{revenue_improvement:.1f}% revenue, +{roas_improvement:.1f}% ROAS)")
                elif revenue_improvement < -10 or roas_improvement < -5:
                    st.error(f"❌ **{var_a['name']} performs better.** Stick with control.")
                else:
                    st.info(f"➡️ **Results are inconclusive.** Consider running test longer or with larger sample size.")
                
                # Delete button
                if st.button(f"🗑️ Delete Test", key=f"delete_{idx}"):
                    st.session_state.ab_tests.pop(idx)
                    st.rerun()
    else:
        st.info("📝 No A/B tests created yet. Use the form above to create your first test!")

# TAB 8: Weekly Reports
with tabs[7]:
    st.markdown('<div class="section-header">📅 Weekly Performance Reports</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    📊 <strong>Generate Weekly Reports:</strong> Create comprehensive performance summaries with key metrics, 
    marketplace breakdowns, and actionable recommendations. Export as PDF or share the summary.
    </div>
    """, unsafe_allow_html=True)
    
    # Report Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**📋 Report Generator**")
        
        # Date range selector for report
        st.markdown("**Select Report Period:**")
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            report_start = st.date_input(
                "From Date",
                value=max_date - timedelta(days=7),
                min_value=min_date,
                max_value=max_date,
                key="report_start"
            )
        with col_date2:
            report_end = st.date_input(
                "To Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="report_end"
            )
        
        # Report sections to include
        st.markdown("**📑 Include in Report:**")
        include_kpis = st.checkbox("Key Performance Indicators", value=True)
        include_trends = st.checkbox("Performance Trends", value=True)
        include_marketplaces = st.checkbox("Marketplace Breakdown", value=True)
        include_skus = st.checkbox("Top SKU Performance", value=True)
        include_recommendations = st.checkbox("Strategic Recommendations", value=True)
        
        # Generate report button
        if st.button("📊 Generate Report", type="primary", key="generate_report"):
            # Filter data for report period
            report_mask_sales = (
                (sales_df["date"].dt.date >= report_start) & 
                (sales_df["date"].dt.date <= report_end) &
                (sales_df["channel"].isin(selected_channels))
            )
            report_df_s = sales_df[report_mask_sales]
            
            report_mask_spend = (
                (spend_df["date"].dt.date >= report_start) & 
                (spend_df["date"].dt.date <= report_end) &
                (spend_df["channel"].isin(selected_channels))
            )
            report_df_sp = spend_df[report_mask_spend]
            
            # Calculate metrics for report period
            report_metrics = calc_metrics(report_df_s, report_df_sp)
            
            # Store in session state to display
            st.session_state.current_report = {
                'period': f"{report_start.strftime('%b %d, %Y')} - {report_end.strftime('%b %d, %Y')}",
                'metrics': report_metrics,
                'sales_data': report_df_s,
                'spend_data': report_df_sp,
                'sections': {
                    'kpis': include_kpis,
                    'trends': include_trends,
                    'marketplaces': include_marketplaces,
                    'skus': include_skus,
                    'recommendations': include_recommendations
                }
            }
            st.success("✅ Report generated successfully!")
            st.rerun()
    
    with col2:
        st.markdown("**💡 Report Tips**")
        st.markdown("""
        **Best Practices:**
        - Weekly reports: Use 7-day periods
        - Monthly reports: Full month data
        - Include recommendations for action items
        
        **Common Report Types:**
        - **Weekly Summary** (last 7 days)
        - **Monthly Review** (full month)
        - **Quarter Analysis** (90 days)
        - **Campaign Performance** (campaign dates)
        
        **Export Options:**
        - View in dashboard
        - Copy formatted text
        - Download as markdown
        - Share with team
        """)
    
    # Display generated report
    if 'current_report' in st.session_state:
        report = st.session_state.current_report
        
        st.markdown("---")
        st.markdown(f"## 📊 Performance Report")
        st.markdown(f"**Period:** {report['period']}")
        
        # KPIs Section
        if report['sections']['kpis']:
            st.markdown("### 💎 Key Performance Indicators")
            
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.metric("Total Revenue", f"${report['metrics']['Revenue']:,.0f}")
            with kpi_col2:
                st.metric("Total Orders", f"{report['metrics']['Orders']:,.0f}")
            with kpi_col3:
                st.metric("ROAS", f"{report['metrics']['ROAS']:.2f}x")
            with kpi_col4:
                st.metric("Net Profit", f"${report['metrics']['Net']:,.0f}")
            
            kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
            with kpi_col5:
                st.metric("Ad Spend", f"${report['metrics']['Spend']:,.0f}")
            with kpi_col6:
                st.metric("Commission", f"${report['metrics']['Commission']:,.0f}")
            with kpi_col7:
                st.metric("ACOS", f"{report['metrics']['ACOS']:.1f}%")
            with kpi_col8:
                st.metric("AOV", f"${report['metrics']['AOV']:.2f}")
        
        # Marketplace Breakdown
        ch_report = pd.DataFrame()  # Initialize ch_report to avoid NameError
        
        if report['sections']['marketplaces']:
            st.markdown("### 🛒 Marketplace Performance")
            
            # Calculate marketplace data
            ch_report = report['sales_data'].groupby("channel").agg({
                "revenue": "sum",
                "orders": "sum"
            }).reset_index()
            
            ch_sp_report = report['spend_data'].groupby("channel")["spend"].sum().reset_index()
            ch_report = pd.merge(ch_report, ch_sp_report, on="channel", how="outer").fillna(0)
            ch_report["roas"] = ch_report.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)
            ch_report["acos"] = ch_report.apply(lambda x: (x["spend"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
            ch_report = ch_report.sort_values("revenue", ascending=False)
            
            # Display marketplace table
            st.dataframe(
                ch_report[['channel', 'revenue', 'orders', 'spend', 'roas', 'acos']],
                column_config={
                    "channel": "Marketplace",
                    "revenue": st.column_config.NumberColumn("Revenue", format="$%d"),
                    "orders": st.column_config.NumberColumn("Orders", format="%d"),
                    "spend": st.column_config.NumberColumn("Ad Spend", format="$%d"),
                    "roas": st.column_config.NumberColumn("ROAS", format="%.2fx"),
                    "acos": st.column_config.NumberColumn("ACOS", format="%.1f%%"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Top performer
            if len(ch_report) > 0:
                top_marketplace = ch_report.iloc[0]
                st.success(f"🏆 **Top Performer:** {top_marketplace['channel']} - ${top_marketplace['revenue']:,.0f} revenue with {top_marketplace['roas']:.2f}x ROAS")
        
        # Top SKUs
        if report['sections']['skus'] and "Parent" in report['sales_data'].columns:
            st.markdown("### 🏷️ Top Performing SKUs")
            
            sku_report = report['sales_data'].groupby("Parent").agg({
                "revenue": "sum",
                "orders": "sum"
            }).reset_index()
            sku_report["aov"] = sku_report["revenue"] / sku_report["orders"]
            sku_report = sku_report.sort_values("revenue", ascending=False).head(5)
            
            st.dataframe(
                sku_report,
                column_config={
                    "Parent": "SKU",
                    "revenue": st.column_config.NumberColumn("Revenue", format="$%d"),
                    "orders": st.column_config.NumberColumn("Orders", format="%d"),
                    "aov": st.column_config.NumberColumn("AOV", format="$%.2f"),
                },
                hide_index=True,
                use_container_width=True
            )
        
        # Recommendations
        if report['sections']['recommendations']:
            st.markdown("### 🚀 Strategic Recommendations")
            
            try:
                # Generate recommendations based on report data
                # Ensure ch_report has the right structure
                if len(ch_report) > 0 and 'roas' in ch_report.columns:
                    ch_matrix_rec = ch_report
                else:
                    # Create empty dataframe with expected structure if no marketplace data
                    ch_matrix_rec = pd.DataFrame(columns=['channel', 'revenue', 'spend', 'roas'])
                
                # Call generate_insights with correct number of parameters (only 2)
                recommendations_list = generate_insights(ch_matrix_rec, report['metrics'])
                
                if recommendations_list:
                    for rec in recommendations_list[:5]:  # Top 5 recommendations
                        rec_type_map = {
                            "scale": ("success", "📈"),
                            "warn": ("warning", "⚠️"),
                            "crit": ("error", "🚨"),
                            "info": ("info", "💡")
                        }
                        status, icon = rec_type_map.get(rec['type'], ("info", "💡"))
                        
                        if status == "success":
                            st.success(f"{icon} **{rec['title']}** - {rec['msg']}")
                        elif status == "warning":
                            st.warning(f"{icon} **{rec['title']}** - {rec['msg']}")
                        elif status == "error":
                            st.error(f"{icon} **{rec['title']}** - {rec['msg']}")
                        else:
                            st.info(f"{icon} **{rec['title']}** - {rec['msg']}")
                else:
                    st.info("✅ No critical recommendations at this time. Performance is stable.")
            except Exception as e:
                st.warning(f"⚠️ Could not generate recommendations. Please ensure marketplace data is included in the report.")
                # Optional: Show error details for debugging
                # st.error(f"Debug: {str(e)}")
        
        # Performance Trends
        if report['sections']['trends']:
            st.markdown("### 📈 Performance Trend")
            
            # Daily trend for report period
            daily_report = report['sales_data'].groupby(pd.Grouper(key="date", freq="D")).agg({
                "revenue": "sum",
                "orders": "sum"
            }).reset_index()
            
            fig_report_trend = go.Figure()
            fig_report_trend.add_trace(go.Scatter(
                x=daily_report["date"],
                y=daily_report["revenue"],
                name="Revenue",
                fill='tozeroy',
                line=dict(color='#3b82f6', width=2)
            ))
            
            fig_report_trend.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
                xaxis=dict(title="Date", showgrid=False),
                margin=dict(l=0, r=0, t=20, b=0),
                height=300
            )
            st.plotly_chart(fig_report_trend, config={'displayModeBar': False})
        
        # Export options
        st.markdown("---")
        st.markdown("### 📤 Export Report")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Generate markdown report
            markdown_report = f"""# Performance Report
**Period:** {report['period']}

## Key Metrics
- **Revenue:** ${report['metrics']['Revenue']:,.0f}
- **Orders:** {report['metrics']['Orders']:,.0f}
- **ROAS:** {report['metrics']['ROAS']:.2f}x
- **Net Profit:** ${report['metrics']['Net']:,.0f}
- **ACOS:** {report['metrics']['ACOS']:.1f}%
- **AOV:** ${report['metrics']['AOV']:.2f}
"""
            
            if report['sections']['marketplaces'] and len(ch_report) > 0:
                markdown_report += f"\n## Top Marketplaces\n"
                for _, row in ch_report.head(3).iterrows():
                    markdown_report += f"- **{row['channel']}**: ${row['revenue']:,.0f} revenue, {row['roas']:.2f}x ROAS\n"
            
            if report['sections']['recommendations'] and recommendations_list:
                markdown_report += f"\n## Recommendations\n"
                for rec in recommendations_list[:3]:
                    markdown_report += f"- **{rec['title']}**: {rec['msg']}\n"
            
            st.download_button(
                "📥 Download as Markdown",
                markdown_report,
                f"weekly_report_{report_start}_{report_end}.md",
                "text/markdown",
                key="download_markdown"
            )
        
        with export_col2:
            # Copy to clipboard button (show text area)
            if st.button("📋 Show Copyable Text", key="show_copy"):
                st.text_area("Copy this report:", markdown_report, height=200)

# TAB 9: Data Explorer
with tabs[8]:
    st.markdown('<div class="section-header">📋 Performance Data Explorer</div>', unsafe_allow_html=True)
    
    # Channel Performance Table
    tbl = ch_matrix.copy()
    if "selling_commission" in ch_matrix.columns:
        tbl["commission"] = ch_matrix["selling_commission"]
    else:
        tbl["commission"] = 0
    
    tbl["acos"] = tbl.apply(lambda x: (x["spend"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
    tbl["net"] = (tbl["revenue"] * SAFE_MARGIN) - tbl["spend"] - tbl.get("commission", 0)
    tbl["profit_margin"] = tbl.apply(lambda x: (x["net"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
    
    # Select columns to display
    display_cols = ["channel", "revenue", "orders", "aov", "spend", "commission", "roas", "acos", "net", "profit_margin"]
    display_tbl = tbl[[col for col in display_cols if col in tbl.columns]]
    
    st.dataframe(
        display_tbl,
        column_config={
            "channel": st.column_config.TextColumn("Marketplace", width="medium"),
            "revenue": st.column_config.ProgressColumn("Revenue", format="$%d", min_value=0, max_value=int(display_tbl["revenue"].max()) if "revenue" in display_tbl.columns else 100),
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
        height=400
    )
    
    # Export Options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = display_tbl.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Marketplace Report (CSV)", csv, "marketplace_performance.csv", "text/csv", key="download_channel")
    
    with col2:
        if not df_s.empty:
            sales_csv = df_s.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Sales Data (CSV)", sales_csv, "sales_data.csv", "text/csv", key="download_sales")
    
    with col3:
        if not df_sp.empty:
            spend_csv = df_sp.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Spend Data (CSV)", spend_csv, "spend_data.csv", "text/csv", key="download_spend")



# ---------------- FOOTER ----------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div style='text-align: left; color: #6b7280; font-size: 12px;'>📅 Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='text-align: center; color: #6b7280; font-size: 12px;'>⚙️ Safe Margin: {SAFE_MARGIN*100:.0f}%</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div style='text-align: right; color: #6b7280; font-size: 12px;'>📊 Data Points: {len(df_s):,}</div>", unsafe_allow_html=True)