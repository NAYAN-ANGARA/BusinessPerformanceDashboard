import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets
from datetime import date, timedelta, datetime
import numpy as np
import json
import hashlib

# Configure Plotly
import plotly.io as pio
p

# ---------------- SAFE HELPERS ----------------
def _fillna_numeric_only(df: pd.DataFrame, value=0):
    """Fill NaNs only in numeric columns (prevents categorical fill errors)."""
    if df is None or df.empty:
        return df
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(value)
    return df

@st.cache_data(show_spinner=False, ttl=900)
def compute_channel_matrix(df_s: pd.DataFrame, df_sp: pd.DataFrame) -> pd.DataFrame:
    """Channel-level revenue/orders/spend/commission matrix. Safe to call from any tab."""
    if df_s is None or df_s.empty:
        base = pd.DataFrame({"channel": []})
        base["revenue"] = []
        base["orders"] = []
        base["selling_commission"] = []
        base["spend"] = []
        base["roas"] = []
        base["aov"] = []
        return base

    ch_rev = df_s.groupby("channel", dropna=False).agg(revenue=("revenue", "sum"), orders=("orders", "sum")).reset_index()

    if "selling_commission" in df_s.columns:
        ch_comm = df_s.groupby("channel", dropna=False)["selling_commission"].sum().reset_index()
        ch_rev = pd.merge(ch_rev, ch_comm, on="channel", how="left")
    else:
        ch_rev["selling_commission"] = 0.0

    if df_sp is None or df_sp.empty or "spend" not in df_sp.columns:
        ch_sp = pd.DataFrame({"channel": ch_rev["channel"], "spend": 0.0})
    else:
        ch_sp = df_sp.groupby("channel", dropna=False)["spend"].sum().reset_index()

    ch_matrix = pd.merge(ch_rev, ch_sp, on="channel", how="outer")
    ch_matrix = _fillna_numeric_only(ch_matrix, 0)

    # derived metrics
    ch_matrix["roas"] = np.where(ch_matrix["spend"] > 0, ch_matrix["revenue"] / ch_matrix["spend"], 0)
    ch_matrix["aov"] = np.where(ch_matrix["orders"] > 0, ch_matrix["revenue"] / ch_matrix["orders"], 0)
    return ch_matrix
io.templates.default = "plotly_dark"

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
    import tempfile, json as _json, os as _os
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            _json.dump(creds_dict, tmp)
            creds = tmp.name
    except Exception as e:
        return None, None, f"Credentials error: {e}"
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
    finally:
        try: _os.unlink(creds)
        except Exception: pass

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
    "📋 Data Explorer",
    "💎 Merchandising Intel"
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
        
        # ── SKU SEARCH / LOOKUP ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">🔍 SKU Deep-Dive Search</div>', unsafe_allow_html=True)

        all_parent_skus_list = sorted(df_s["Parent"].dropna().unique().tolist())

        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            sku_search_query = st.text_input(
                "Search SKU",
                placeholder="Type a Parent SKU name...",
                key="sku_search_input",
                label_visibility="collapsed"
            )
        with search_col2:
            sku_search_exact = st.selectbox(
                "Match",
                options=["Contains", "Exact"],
                key="sku_search_mode",
                label_visibility="collapsed"
            )

        # Filter matching SKUs
        if sku_search_query.strip():
            q = sku_search_query.strip()
            if sku_search_exact == "Exact":
                matching_skus = [s for s in all_parent_skus_list if s.lower() == q.lower()]
            else:
                matching_skus = [s for s in all_parent_skus_list if q.lower() in s.lower()]
        else:
            matching_skus = []

        if sku_search_query.strip() and not matching_skus:
            st.warning(f"No SKUs found matching **'{sku_search_query}'**. Try a shorter keyword or switch to Contains mode.")

        elif matching_skus:
            # If multiple matches show a selector, otherwise jump straight in
            if len(matching_skus) > 1:
                st.caption(f"Found **{len(matching_skus)}** matching SKUs — select one to inspect:")
                selected_sku = st.selectbox(
                    "Select SKU",
                    options=matching_skus,
                    key="sku_search_select",
                    label_visibility="collapsed"
                )
            else:
                selected_sku = matching_skus[0]

            # ── Pull data for selected SKU ──────────────────────────────────
            sku_df = df_s[df_s["Parent"] == selected_sku].copy()

            total_rev    = sku_df["revenue"].sum()
            total_orders = sku_df["orders"].sum()
            aov          = (total_rev / total_orders) if total_orders > 0 else 0
            active_days  = sku_df["date"].nunique()
            first_sale   = sku_df["date"].min().strftime("%b %d, %Y")
            last_sale    = sku_df["date"].max().strftime("%b %d, %Y")

            # ── Header ───────────────────────────────────────────────────────
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(59,130,246,0.15),rgba(139,92,246,0.1));
                        border:1px solid rgba(59,130,246,0.35); border-radius:12px;
                        padding:16px 20px; margin:12px 0;'>
                <div style='font-size:20px; font-weight:800; color:#f3f4f6;'>🏷️ {selected_sku}</div>
                <div style='font-size:12px; color:#9ca3af; margin-top:4px;'>
                    First sale: {first_sale} &nbsp;·&nbsp; Last sale: {last_sale} &nbsp;·&nbsp; {active_days} active days
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── KPI row ──────────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("💰 Revenue",    f"${total_rev:,.0f}")
            with m2:
                st.metric("🛒 Orders",     f"{total_orders:,.0f}")
            with m3:
                st.metric("📊 AOV",        f"${aov:,.2f}")
            with m4:
                revenue_share = (total_rev / df_s["revenue"].sum() * 100) if df_s["revenue"].sum() > 0 else 0
                st.metric("📈 Rev Share",  f"{revenue_share:.1f}%")

            st.markdown("")

            chart_col, info_col = st.columns([3, 2])

            # ── Daily revenue trend ──────────────────────────────────────────
            with chart_col:
                st.markdown("**📅 Daily Revenue Trend**")
                daily_sku = (
                    sku_df.groupby(pd.Grouper(key="date", freq="D"))["revenue"]
                    .sum().reset_index().sort_values("date")
                )
                # 7-day rolling average
                daily_sku["rolling7"] = daily_sku["revenue"].rolling(7, min_periods=1).mean()

                fig_sku_trend = go.Figure()
                fig_sku_trend.add_trace(go.Bar(
                    x=daily_sku["date"], y=daily_sku["revenue"],
                    name="Daily Revenue",
                    marker_color="rgba(59,130,246,0.5)"
                ))
                fig_sku_trend.add_trace(go.Scatter(
                    x=daily_sku["date"], y=daily_sku["rolling7"],
                    name="7-day Avg",
                    line=dict(color="#10b981", width=2),
                    mode="lines"
                ))
                fig_sku_trend.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", y=1.15),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#2d303e", title="Revenue ($)")
                )
                st.plotly_chart(fig_sku_trend, config={"displayModeBar": False})

            # ── Marketplace breakdown ────────────────────────────────────────
            with info_col:
                st.markdown("**🛒 Revenue by Marketplace**")
                mp_breakdown = (
                    sku_df.groupby("channel")
                    .agg(revenue=("revenue","sum"), orders=("orders","sum"))
                    .reset_index()
                    .sort_values("revenue", ascending=False)
                )
                mp_breakdown["share"] = (
                    mp_breakdown["revenue"] / mp_breakdown["revenue"].sum() * 100
                ).round(1)

                if len(mp_breakdown) > 0:
                    fig_mp_pie = px.pie(
                        mp_breakdown, values="revenue", names="channel",
                        hole=0.55,
                        color_discrete_sequence=["#3b82f6","#10b981","#f59e0b",
                                                  "#8b5cf6","#ec4899","#06b6d4"]
                    )
                    fig_mp_pie.update_traces(textposition="outside", textinfo="percent+label")
                    fig_mp_pie.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        height=280,
                        margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig_mp_pie, config={"displayModeBar": False})

            # ── Marketplace table ────────────────────────────────────────────
            st.dataframe(
                mp_breakdown,
                column_config={
                    "channel": st.column_config.TextColumn("Marketplace"),
                    "revenue": st.column_config.NumberColumn("Revenue",  format="$%d"),
                    "orders":  st.column_config.NumberColumn("Orders",   format="%d"),
                    "share":   st.column_config.NumberColumn("Share %",  format="%.1f%%"),
                },
                hide_index=True, use_container_width=True
            )

            # ── Child SKUs ───────────────────────────────────────────────────
            if "SKU" in df_s.columns:
                child_skus = (
                    sku_df.groupby("SKU")
                    .agg(revenue=("revenue","sum"), orders=("orders","sum"))
                    .reset_index()
                    .sort_values("revenue", ascending=False)
                )
                child_skus["aov"]   = child_skus["revenue"] / child_skus["orders"].replace(0, np.nan)
                child_skus["share"] = (child_skus["revenue"] / child_skus["revenue"].sum() * 100).round(1)
                valid_children      = child_skus[child_skus["SKU"] != "Unknown"]

                if len(valid_children) > 0:
                    st.markdown(f"**📦 Child SKUs ({len(valid_children)} variants)**")
                    st.dataframe(
                        valid_children,
                        column_config={
                            "SKU":     st.column_config.TextColumn("Child SKU", width="large"),
                            "revenue": st.column_config.ProgressColumn(
                                "Revenue", format="$%d",
                                min_value=0, max_value=int(valid_children["revenue"].max())
                            ),
                            "orders":  st.column_config.NumberColumn("Orders",  format="%d"),
                            "aov":     st.column_config.NumberColumn("AOV",     format="$%.2f"),
                            "share":   st.column_config.NumberColumn("Share %", format="%.1f%%"),
                        },
                        hide_index=True, use_container_width=True,
                        height=min(400, 60 + len(valid_children) * 38)
                    )

        else:
            st.caption("🔎 Type a SKU name above to search. Supports partial matches.")

        # ── Detailed SKU Cards with Child SKUs ───────────────────────────────
        st.markdown("---")
        st.markdown("**📦 Top 10 SKU Breakdown (Click to expand for Child SKUs)**")
        
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
    st.markdown('<div class="section-header">🔮 Advanced Ensemble ML Forecasting</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    🤖 <strong>Ensemble Forecasting Engine:</strong> Combines <strong>5 models</strong> — 
    Gradient Boosting, Random Forest, Ridge Regression, Exponential Smoothing, and YoY Seasonal Adjustment — 
    then blends them by inverse-error weighting so the best-performing model on your data gets the highest vote.
    Confidence is calculated from cross-validation R² scores, not just data volume.
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        forecast_period = st.selectbox(
            "📅 Forecast Period",
            ["Next 7 Days", "Next 30 Days", "Next Quarter (90 Days)"],
            index=1
        )
    with fc2:
        forecast_type = st.selectbox(
            "📊 Forecast Type",
            ["Revenue & Orders", "SKU Performance", "Marketplace Performance"],
            index=0
        )
    with fc3:
        use_yoy = st.checkbox("Year-over-Year Seasonal Boost", value=True,
                              help="Adjust predictions using same-period last year growth rate")

    forecast_days = {"Next 7 Days": 7, "Next 30 Days": 30, "Next Quarter (90 Days)": 90}[forecast_period]


    # ── Lazy import: ML only loads when this tab is first opened ─────────
    try:
        from forecast_engine import ensemble_forecast, forecast_all_skus
    except Exception as _fe_err:
        st.error(f"⚠️ Forecasting engine failed to load: {_fe_err}")
        st.stop()

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # ========== REVENUE & ORDERS FORECASTING ==========
    # ═══════════════════════════════════════════════════════════════════════
    if forecast_type == "Revenue & Orders":
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**📈 Ensemble Revenue Forecast — {forecast_period}**")

            daily_revenue = df_s.groupby(pd.Grouper(key="date", freq="D")).agg(
                {"revenue": "sum", "orders": "sum"}
            ).reset_index().sort_values("date")

            if len(daily_revenue) >= 14:
                future_dates = pd.date_range(
                    daily_revenue["date"].max() + timedelta(days=1), periods=forecast_days
                )

                # YoY data
                yoy_revenue = None
                if use_yoy:
                    yoy_mask = (
                        sales_df["date"].dt.date >= (daily_revenue["date"].max() - pd.DateOffset(years=1) - timedelta(days=forecast_days)).date()
                    ) & (
                        sales_df["date"].dt.date <= (daily_revenue["date"].max() - pd.DateOffset(years=1)).date()
                    )
                    yoy_raw = sales_df[yoy_mask]
                    if len(yoy_raw) > 0:
                        yoy_revenue = yoy_raw.groupby(pd.Grouper(key="date", freq="D"))["revenue"].sum().values

                with st.spinner("🤖 Training ensemble (5 models)…"):
                    rev_pred, rev_std, confidence, weighted_r2, model_info = ensemble_forecast(
                        daily_revenue["date"], daily_revenue["revenue"].values,
                        future_dates, yoy_revenue
                    )
                    ord_pred, _, _, _, _ = ensemble_forecast(
                        daily_revenue["date"], daily_revenue["orders"].values,
                        future_dates
                    )

                forecast_df = pd.DataFrame({
                    "date": future_dates,
                    "predicted_revenue": rev_pred,
                    "predicted_orders":  ord_pred,
                    "upper": rev_pred + rev_std,
                    "lower": np.maximum(rev_pred - rev_std, 0)
                })

                # Chart
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=daily_revenue["date"], y=daily_revenue["revenue"],
                    name="Historical", line=dict(color="#3b82f6", width=2),
                    fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"
                ))
                # Confidence band
                fig_fc.add_trace(go.Scatter(
                    x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
                    y=list(forecast_df["upper"]) + list(forecast_df["lower"])[::-1],
                    fill="toself", fillcolor="rgba(16,185,129,0.15)",
                    line=dict(width=0), name="Confidence Band", showlegend=True
                ))
                fig_fc.add_trace(go.Scatter(
                    x=forecast_df["date"], y=forecast_df["predicted_revenue"],
                    name="Ensemble Forecast", line=dict(color="#10b981", width=3)
                ))
                if yoy_revenue is not None:
                    yoy_dates = pd.date_range(
                        daily_revenue["date"].max() + timedelta(days=1) - pd.DateOffset(years=1),
                        periods=min(len(yoy_revenue), forecast_days)
                    )
                    fig_fc.add_trace(go.Scatter(
                        x=yoy_dates, y=yoy_revenue[:len(yoy_dates)],
                        name="Last Year Same Period",
                        line=dict(color="#f59e0b", width=1.5, dash="dot"), opacity=0.7
                    ))
                fig_fc.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                    yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
                    xaxis=dict(showgrid=False),
                    legend=dict(orientation="h", y=1.18),
                    margin=dict(l=0, r=0, t=60, b=0), height=440
                )
                st.plotly_chart(fig_fc, config={"displayModeBar": False})

                # Summary metrics
                total_rev = forecast_df["predicted_revenue"].sum()
                total_ord = forecast_df["predicted_orders"].sum()
                hist_avg  = daily_revenue["revenue"].tail(forecast_days).mean()
                growth    = ((forecast_df["predicted_revenue"].mean() - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
                yoy_vs    = None
                if yoy_revenue is not None:
                    yoy_tot = yoy_revenue.sum()
                    yoy_vs  = ((total_rev - yoy_tot) / yoy_tot * 100) if yoy_tot > 0 else None

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric(f"Predicted Revenue ({forecast_days}d)", f"${total_rev:,.0f}")
                mc2.metric(f"Predicted Orders ({forecast_days}d)",  f"{total_ord:,.0f}")
                mc3.metric("Growth vs Historical",                  f"{growth:+.1f}%")
                if yoy_vs is not None:
                    mc4.metric("vs Same Period Last Year",          f"{yoy_vs:+.1f}%")
                else:
                    mc4.metric("YoY Data",                          "Not available")

            else:
                st.warning(f"⚠️ Need at least 14 days of data. Currently have {len(daily_revenue)} days.")

        with col2:
            st.markdown("**🎯 Model Performance**")
            if len(daily_revenue) >= 14:
                # Confidence gauge
                conf_color = "#10b981" if confidence >= 75 else ("#f59e0b" if confidence >= 55 else "#ef4444")
                st.markdown(
                    f"<div style='text-align:center; padding:16px; background:rgba(0,0,0,0.3); border-radius:10px; border:2px solid {conf_color}'>"
                    f"<p style='margin:0; color:#9ca3af; font-size:12px;'>ENSEMBLE CONFIDENCE</p>"
                    f"<p style='margin:4px 0; font-size:42px; font-weight:900; color:{conf_color}'>{confidence:.0f}%</p>"
                    f"<p style='margin:0; color:#9ca3af; font-size:11px;'>Weighted CV R² = {weighted_r2*100:.1f}%</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.progress(confidence / 100)

                st.markdown("---")
                st.markdown("**📊 Model Breakdown**")
                for mname, minfo in model_info.items():
                    if mname == "_weights":
                        continue
                    weight = model_info["_weights"].get(mname, 0)
                    r2_val = minfo["r2"]
                    bar_w  = int(r2_val)
                    color  = "#10b981" if r2_val >= 70 else ("#f59e0b" if r2_val >= 40 else "#ef4444")
                    st.markdown(
                        f"<div style='margin:4px 0'>"
                        f"<span style='font-size:11px; color:#9ca3af'>{mname}</span>"
                        f"<div style='background:#1e2030; border-radius:4px; height:6px; margin:2px 0'>"
                        f"<div style='background:{color}; width:{bar_w}%; height:6px; border-radius:4px'></div></div>"
                        f"<span style='font-size:10px; color:{color}'>R²={r2_val}% · weight={weight}%</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("---")
                st.markdown("**💡 Insight**")
                if growth > 10:
                    st.success("📈 Strong growth expected.")
                elif growth < -10:
                    st.error("📉 Decline predicted. Review strategy.")
                else:
                    st.info("➡️ Stable performance expected.")
                if yoy_vs is not None:
                    if yoy_vs > 20:   st.success(f"🎉 {yoy_vs:.0f}% above last year!")
                    elif yoy_vs < -10: st.warning(f"⚠️ {abs(yoy_vs):.0f}% below last year")


    
    # ═══════════════════════════════════════════════════════════════════════
    # ========== SKU PERFORMANCE FORECASTING ==========
    # ═══════════════════════════════════════════════════════════════════════
    elif forecast_type == "SKU Performance":
        st.markdown(f"**🏷️ Top SKU Performance Forecast - {forecast_period}**")
        st.markdown("""
        <div class="info-box">
        🤖 <strong>Auto-Ranked SKU Predictions:</strong> The model runs on all SKUs with sufficient data, 
        scores each by predicted growth momentum, and ranks the <strong>Top 20 Best Opportunity SKUs</strong> for you — no manual selection needed.
        </div>
        """, unsafe_allow_html=True)

        if "Parent" in df_s.columns:
            all_parent_skus = df_s["Parent"].dropna().unique().tolist()
            all_sku_forecasts = []

            with st.spinner(f"⚡ Forecasting {len(all_parent_skus)} SKUs in parallel..."):
                all_sku_forecasts = forecast_all_skus(
                    df_s["revenue"].values,
                    df_s["date"].values,
                    df_s["Parent"].values,
                    tuple(sorted(all_parent_skus)),
                    forecast_days,
                    use_yoy,
                )

            if not all_sku_forecasts:
                st.warning("⚠️ No SKUs had enough data (7+ days) to forecast.")
            else:
                df_sku_rank = pd.DataFrame(all_sku_forecasts)

                # Rank by composite score: growth + momentum + confidence
                df_sku_rank["_score"] = (
                    df_sku_rank["Growth %"].clip(-200, 200) * 0.4 +
                    df_sku_rank["Momentum %"].clip(-200, 200) * 0.4 +
                    df_sku_rank["Confidence %"] * 0.2
                )
                df_sku_rank = df_sku_rank.sort_values("_score", ascending=False).head(20).reset_index(drop=True)
                df_sku_rank["Rank"] = df_sku_rank.index + 1

                # ── TOP 20 SUMMARY TABLE ──────────────────────────────────────
                display_cols = [
                    "Rank", "SKU", "Historical Avg", "Recent 2wk Avg",
                    "Forecast Avg", f"Total Forecast ({forecast_days}d)",
                    "Growth %", "Momentum %", "YoY Change %", "Confidence %"
                ]
                df_display = df_sku_rank[display_cols].copy()

                st.markdown(f"### 🏆 Top 20 SKUs by ML Forecast Score")
                st.caption("Ranked by composite score: Growth (40%) + Short-term Momentum (40%) + Model Confidence (20%)")

                st.dataframe(
                    df_display,
                    column_config={
                        "Rank":         st.column_config.NumberColumn("Rank", format="%d"),
                        "SKU":          st.column_config.TextColumn("SKU"),
                        "Historical Avg":      st.column_config.NumberColumn("Hist. Avg/Day", format="$%.0f"),
                        "Recent 2wk Avg":      st.column_config.NumberColumn("Recent 2wk Avg", format="$%.0f"),
                        "Forecast Avg":        st.column_config.NumberColumn("Forecast Avg/Day", format="$%.0f"),
                        f"Total Forecast ({forecast_days}d)": st.column_config.NumberColumn(f"Total Forecast", format="$%.0f"),
                        "Growth %":            st.column_config.NumberColumn("Growth %", format="%.1f%%"),
                        "Momentum %":          st.column_config.NumberColumn("Momentum %", format="%.1f%%"),
                        "YoY Change %":        st.column_config.NumberColumn("YoY Change %", format="%.1f%%"),
                        "Confidence %":        st.column_config.NumberColumn("Confidence %", format="%.0f%%"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=550
                )

                # ── TOP 5 VISUAL CHARTS ───────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📈 Forecast Charts — Top 5 SKUs")

                top5 = df_sku_rank.head(5)
                chart_cols = st.columns(min(len(top5), 5))

                for i, (_, row) in enumerate(top5.iterrows()):
                    with chart_cols[i]:
                        fig_mini = go.Figure()
                        hist = row["_hist"]
                        fig_mini.add_trace(go.Bar(
                            x=hist["date"], y=hist["revenue"],
                            name="Historical", marker_color="#3b82f6", opacity=0.7,
                            showlegend=False
                        ))
                        fig_mini.add_trace(go.Scatter(
                            x=row["_dates"], y=row["_pred"],
                            name="Forecast", line=dict(color="#10b981", width=2),
                            mode="lines", showlegend=False
                        ))
                        fig_mini.update_layout(
                            title=dict(text=f"#{row['Rank']} {row['SKU']}", font_size=12),
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=0, r=0, t=30, b=0),
                            height=200,
                            yaxis=dict(showgrid=False, showticklabels=False),
                            xaxis=dict(showgrid=False, showticklabels=False)
                        )
                        st.plotly_chart(fig_mini, config={"displayModeBar": False}, use_container_width=True)
                        growth_val = row["Growth %"]
                        color = "green" if growth_val >= 0 else "red"
                        st.markdown(
                            f"<p style='text-align:center; color:{color}; font-size:13px; font-weight:700;'>"
                            f"${row[f'Total Forecast ({forecast_days}d)']:,.0f} | {growth_val:+.1f}%</p>",
                            unsafe_allow_html=True
                        )

                # ── GROWTH vs REVENUE SCATTER ─────────────────────────────────
                st.markdown("---")
                st.markdown("### 🔵 Opportunity Matrix — Growth vs Forecasted Revenue")
                st.caption("Top-right = High growth + High revenue = Best opportunities to scale")

                fig_scatter = px.scatter(
                    df_sku_rank,
                    x=f"Total Forecast ({forecast_days}d)",
                    y="Growth %",
                    text="SKU",
                    size="Confidence %",
                    color="Momentum %",
                    color_continuous_scale="RdYlGn",
                    labels={
                        f"Total Forecast ({forecast_days}d)": "Forecasted Revenue ($)",
                        "Growth %": "Growth vs Historical (%)"
                    },
                    size_max=30
                )
                fig_scatter.update_traces(textposition="top center", textfont_size=9)
                fig_scatter.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.03)",
                    height=450,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_scatter, config={"displayModeBar": False})

        else:
            st.info("📦 SKU data not available. Ensure 'Parent' column exists in your data.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ========== MARKETPLACE PERFORMANCE FORECASTING ==========
    # ═══════════════════════════════════════════════════════════════════════
    elif forecast_type == "Marketplace Performance":
        st.markdown(f"**🛒 Ensemble Marketplace Forecast — {forecast_period}**")

        marketplaces       = df_s["channel"].unique().tolist()
        marketplace_forecasts = []

        with st.spinner(f"🤖 Running ensemble on {len(marketplaces)} marketplaces…"):
            for marketplace in marketplaces:
                mp_data = (
                    df_s[df_s["channel"] == marketplace]
                    .groupby(pd.Grouper(key="date", freq="D"))["revenue"]
                    .sum().reset_index().sort_values("date")
                )
                if len(mp_data) < 7:
                    continue

                mp_future_dates = pd.date_range(
                    mp_data["date"].max() + timedelta(days=1), periods=forecast_days
                )

                # YoY data for this marketplace
                yoy_mp_vals = None
                if use_yoy:
                    yoy_mp_mask = (
                        df_s["channel"].eq(marketplace) &
                        df_s["date"].dt.date.between(
                            (mp_data["date"].max() - pd.DateOffset(years=1) - timedelta(days=forecast_days)).date(),
                            (mp_data["date"].max() - pd.DateOffset(years=1)).date()
                        )
                    )
                    yoy_mp_chunk = df_s[yoy_mp_mask]["revenue"].values
                    if len(yoy_mp_chunk) > 0:
                        yoy_mp_vals = yoy_mp_chunk

                mp_pred, mp_std, mp_conf, mp_r2, _ = ensemble_forecast(
                    mp_data["date"], mp_data["revenue"].values,
                    mp_future_dates, yoy_mp_vals
                )

                hist_avg = mp_data["revenue"].mean()
                fore_avg = mp_pred.mean()
                growth   = ((fore_avg - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0

                marketplace_forecasts.append({
                    "Marketplace":       marketplace,
                    "Historical Revenue":mp_data["revenue"].sum(),
                    "Forecast Revenue":  mp_pred.sum(),
                    "Hist. Daily Avg":   hist_avg,
                    "Forecast Daily Avg":fore_avg,
                    "Growth %":          growth,
                    "Confidence %":      mp_conf,
                    "_pred":             mp_pred,
                    "_std":              mp_std,
                    "_dates":            mp_future_dates,
                    "_hist":             mp_data,
                })

        if not marketplace_forecasts:
            st.warning("⚠️ Insufficient data for marketplace forecasting.")
        else:
            df_mp = pd.DataFrame(marketplace_forecasts).sort_values("Forecast Revenue", ascending=False)

            # ── Bar chart: historical vs forecast ──────────────────────────
            fig_mp = go.Figure()
            fig_mp.add_trace(go.Bar(
                x=df_mp["Marketplace"], y=df_mp["Historical Revenue"],
                name="Historical", marker_color="#3b82f6", opacity=0.85
            ))
            fig_mp.add_trace(go.Bar(
                x=df_mp["Marketplace"], y=df_mp["Forecast Revenue"],
                name=f"Forecast ({forecast_days}d)",
                marker_color="#10b981", opacity=0.85,
                text=df_mp["Forecast Revenue"].apply(lambda v: f"${v:,.0f}"),
                textposition="outside"
            ))
            fig_mp.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", barmode="group",
                yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
                xaxis=dict(title="Marketplace"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=0, r=0, t=40, b=0), height=380
            )
            st.plotly_chart(fig_mp, config={"displayModeBar": False})

            # ── Trend lines per marketplace ─────────────────────────────────
            st.markdown("**📈 Forecast Trend Lines by Marketplace**")
            fig_lines = go.Figure()
            colors = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#14b8a6","#f97316"]
            for i, row in enumerate(marketplace_forecasts):
                col = colors[i % len(colors)]
                hist = row["_hist"]
                fig_lines.add_trace(go.Scatter(
                    x=hist["date"], y=hist["revenue"],
                    name=f"{row['Marketplace']} (hist)",
                    line=dict(color=col, width=1.5), opacity=0.5
                ))
                fig_lines.add_trace(go.Scatter(
                    x=row["_dates"], y=row["_pred"],
                    name=f"{row['Marketplace']} (fcst)",
                    line=dict(color=col, width=2.5, dash="dash")
                ))
            fig_lines.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
                xaxis=dict(showgrid=False),
                legend=dict(orientation="h", y=1.15),
                margin=dict(l=0, r=0, t=50, b=0), height=380
            )
            st.plotly_chart(fig_lines, config={"displayModeBar": False})

            # ── Detail table with confidence ────────────────────────────────
            st.markdown("**📊 Detailed Marketplace Forecast**")
            st.dataframe(
                df_mp[["Marketplace","Historical Revenue","Forecast Revenue",
                        "Hist. Daily Avg","Forecast Daily Avg","Growth %","Confidence %"]],
                column_config={
                    "Marketplace":        "Channel",
                    "Historical Revenue": st.column_config.NumberColumn("Historical",          format="$%.0f"),
                    "Forecast Revenue":   st.column_config.NumberColumn(f"Forecast ({forecast_days}d)", format="$%.0f"),
                    "Hist. Daily Avg":    st.column_config.NumberColumn("Hist. Daily Avg",     format="$%.0f"),
                    "Forecast Daily Avg": st.column_config.NumberColumn("Forecast Daily Avg",  format="$%.0f"),
                    "Growth %":           st.column_config.NumberColumn("Growth %",            format="%.1f%%"),
                    "Confidence %":       st.column_config.NumberColumn("Confidence %",        format="%.0f%%"),
                },
                hide_index=True, use_container_width=True
            )

            # ── Callouts ────────────────────────────────────────────────────
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            top_rev  = df_mp.iloc[0]
            top_grow = df_mp.nlargest(1, "Growth %").iloc[0]
            top_conf = df_mp.nlargest(1, "Confidence %").iloc[0]
            c1.success(f"💰 **Top Revenue:** {top_rev['Marketplace']} — ${top_rev['Forecast Revenue']:,.0f}")
            c2.info(   f"🚀 **Highest Growth:** {top_grow['Marketplace']} ({top_grow['Growth %']:+.1f}%)")
            c3.info(   f"🎯 **Most Confident:** {top_conf['Marketplace']} ({top_conf['Confidence %']:.0f}%)")

# TAB 7: A/B Test Tracker
with tabs[6]:
    st.markdown('<div class="section-header">🧪 Advanced A/B Test Performance Tracker</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    💡 <strong>Enhanced Testing:</strong> Compare campaigns, products, or strategies. Now with support for 
    <strong>multiple marketplace comparison</strong>, time period analysis, and statistical significance testing.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for A/B tests
    if 'ab_tests' not in st.session_state:
        st.session_state.ab_tests = []
    
    # Test mode selector
    test_mode = st.radio(
        "**Test Mode:**",
        ["Single Marketplace Comparison", "Multi-Marketplace Comparison", "Time Period Comparison"],
        horizontal=True
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**➕ Create New A/B Test**")
        
        # ========== SINGLE MARKETPLACE COMPARISON ==========
        if test_mode == "Single Marketplace Comparison":
            with st.form("ab_test_single_form"):
                test_name = st.text_input("Test Name", placeholder="e.g., Amazon - Old vs New Campaign")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Variant A (Control)**")
                    variant_a_name = st.text_input("Name", value="Control", placeholder="e.g., Original Campaign", key="single_a_name")
                    variant_a_channel = st.selectbox("Marketplace", df_s["channel"].unique(), key="single_a_ch")
                    var_a_date_start = st.date_input("Start Date", value=start_date, key="single_a_start")
                    var_a_date_end = st.date_input("End Date", value=end_date, key="single_a_end")
                
                with col_b:
                    st.markdown("**Variant B (Test)**")
                    variant_b_name = st.text_input("Name", value="Test", placeholder="e.g., New Campaign", key="single_b_name")
                    variant_b_channel = st.selectbox("Marketplace", df_s["channel"].unique(), key="single_b_ch")
                    var_b_date_start = st.date_input("Start Date", value=start_date, key="single_b_start")
                    var_b_date_end = st.date_input("End Date", value=end_date, key="single_b_end")
                
                submitted = st.form_submit_button("🚀 Create Single Marketplace Test", type="primary")
                
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
                        'test_type': 'Single Marketplace',
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'variant_a': {
                            'name': variant_a_name,
                            'channel': variant_a_channel,
                            'revenue': df_a['revenue'].sum(),
                            'orders': df_a['orders'].sum(),
                            'spend': spend_a,
                            'roas': (df_a['revenue'].sum() / spend_a) if spend_a > 0 else 0,
                            'aov': (df_a['revenue'].sum() / df_a['orders'].sum()) if df_a['orders'].sum() > 0 else 0
                        },
                        'variant_b': {
                            'name': variant_b_name,
                            'channel': variant_b_channel,
                            'revenue': df_b['revenue'].sum(),
                            'orders': df_b['orders'].sum(),
                            'spend': spend_b,
                            'roas': (df_b['revenue'].sum() / spend_b) if spend_b > 0 else 0,
                            'aov': (df_b['revenue'].sum() / df_b['orders'].sum()) if df_b['orders'].sum() > 0 else 0
                        }
                    }
                    
                    st.session_state.ab_tests.append(test_data)
                    st.success(f"✅ Test '{test_name}' created successfully!")
                    st.rerun()
        
        # ========== MULTI-MARKETPLACE COMPARISON ==========
        elif test_mode == "Multi-Marketplace Comparison":
            with st.form("ab_test_multi_form"):
                test_name = st.text_input("Test Name", placeholder="e.g., Amazon vs Walmart vs eBay Performance")
                
                st.markdown("**Select Marketplaces to Compare:**")
                marketplaces_to_compare = st.multiselect(
                    "Choose 2-5 marketplaces",
                    df_s["channel"].unique().tolist(),
                    max_selections=5
                )
                
                st.markdown("**Time Period:**")
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    multi_date_start = st.date_input("Start Date", value=start_date, key="multi_start")
                with col_date2:
                    multi_date_end = st.date_input("End Date", value=end_date, key="multi_end")
                
                submitted_multi = st.form_submit_button("🚀 Create Multi-Marketplace Test", type="primary")
                
                if submitted_multi and test_name and len(marketplaces_to_compare) >= 2:
                    # Calculate metrics for each marketplace
                    marketplace_results = []
                    
                    for mp in marketplaces_to_compare:
                        mask_mp = (
                            (sales_df["date"].dt.date >= multi_date_start) & 
                            (sales_df["date"].dt.date <= multi_date_end) &
                            (sales_df["channel"] == mp)
                        )
                        df_mp = sales_df[mask_mp]
                        
                        mask_spend_mp = (
                            (spend_df["date"].dt.date >= multi_date_start) & 
                            (spend_df["date"].dt.date <= multi_date_end) &
                            (spend_df["channel"] == mp)
                        )
                        spend_mp = spend_df[mask_spend_mp]["spend"].sum()
                        
                        marketplace_results.append({
                            'marketplace': mp,
                            'revenue': df_mp['revenue'].sum(),
                            'orders': df_mp['orders'].sum(),
                            'spend': spend_mp,
                            'roas': (df_mp['revenue'].sum() / spend_mp) if spend_mp > 0 else 0,
                            'aov': (df_mp['revenue'].sum() / df_mp['orders'].sum()) if df_mp['orders'].sum() > 0 else 0,
                            'acos': (spend_mp / df_mp['revenue'].sum() * 100) if df_mp['revenue'].sum() > 0 else 0
                        })
                    
                    test_data = {
                        'test_name': test_name,
                        'test_type': 'Multi-Marketplace',
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'marketplaces': marketplace_results,
                        'period': f"{multi_date_start} to {multi_date_end}"
                    }
                    
                    st.session_state.ab_tests.append(test_data)
                    st.success(f"✅ Multi-marketplace test '{test_name}' created with {len(marketplaces_to_compare)} marketplaces!")
                    st.rerun()
                elif submitted_multi and len(marketplaces_to_compare) < 2:
                    st.error("❌ Please select at least 2 marketplaces to compare.")
        
        # ========== TIME PERIOD COMPARISON ==========
        elif test_mode == "Time Period Comparison":
            with st.form("ab_test_time_form"):
                test_name = st.text_input("Test Name", placeholder="e.g., Q4 2024 vs Q4 2023")
                
                marketplace_time = st.selectbox("Select Marketplace", ["All Marketplaces"] + df_s["channel"].unique().tolist())
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown("**Period A**")
                    period_a_start = st.date_input("Start", value=start_date - timedelta(days=90), key="time_a_start")
                    period_a_end = st.date_input("End", value=start_date - timedelta(days=1), key="time_a_end")
                
                with col_p2:
                    st.markdown("**Period B**")
                    period_b_start = st.date_input("Start", value=start_date, key="time_b_start")
                    period_b_end = st.date_input("End", value=end_date, key="time_b_end")
                
                submitted_time = st.form_submit_button("🚀 Create Time Period Test", type="primary")
                
                if submitted_time and test_name:
                    # Period A
                    if marketplace_time == "All Marketplaces":
                        mask_period_a = (sales_df["date"].dt.date >= period_a_start) & (sales_df["date"].dt.date <= period_a_end)
                        mask_spend_a = (spend_df["date"].dt.date >= period_a_start) & (spend_df["date"].dt.date <= period_a_end)
                    else:
                        mask_period_a = (
                            (sales_df["date"].dt.date >= period_a_start) & 
                            (sales_df["date"].dt.date <= period_a_end) &
                            (sales_df["channel"] == marketplace_time)
                        )
                        mask_spend_a = (
                            (spend_df["date"].dt.date >= period_a_start) & 
                            (spend_df["date"].dt.date <= period_a_end) &
                            (spend_df["channel"] == marketplace_time)
                        )
                    
                    df_period_a = sales_df[mask_period_a]
                    spend_period_a = spend_df[mask_spend_a]["spend"].sum()
                    
                    # Period B
                    if marketplace_time == "All Marketplaces":
                        mask_period_b = (sales_df["date"].dt.date >= period_b_start) & (sales_df["date"].dt.date <= period_b_end)
                        mask_spend_b = (spend_df["date"].dt.date >= period_b_start) & (spend_df["date"].dt.date <= period_b_end)
                    else:
                        mask_period_b = (
                            (sales_df["date"].dt.date >= period_b_start) & 
                            (sales_df["date"].dt.date <= period_b_end) &
                            (sales_df["channel"] == marketplace_time)
                        )
                        mask_spend_b = (
                            (spend_df["date"].dt.date >= period_b_start) & 
                            (spend_df["date"].dt.date <= period_b_end) &
                            (spend_df["channel"] == marketplace_time)
                        )
                    
                    df_period_b = sales_df[mask_period_b]
                    spend_period_b = spend_df[mask_spend_b]["spend"].sum()
                    
                    test_data = {
                        'test_name': test_name,
                        'test_type': 'Time Period',
                        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'marketplace': marketplace_time,
                        'period_a': {
                            'name': f"{period_a_start} to {period_a_end}",
                            'revenue': df_period_a['revenue'].sum(),
                            'orders': df_period_a['orders'].sum(),
                            'spend': spend_period_a,
                            'roas': (df_period_a['revenue'].sum() / spend_period_a) if spend_period_a > 0 else 0,
                            'aov': (df_period_a['revenue'].sum() / df_period_a['orders'].sum()) if df_period_a['orders'].sum() > 0 else 0
                        },
                        'period_b': {
                            'name': f"{period_b_start} to {period_b_end}",
                            'revenue': df_period_b['revenue'].sum(),
                            'orders': df_period_b['orders'].sum(),
                            'spend': spend_period_b,
                            'roas': (df_period_b['revenue'].sum() / spend_period_b) if spend_period_b > 0 else 0,
                            'aov': (df_period_b['revenue'].sum() / df_period_b['orders'].sum()) if df_period_b['orders'].sum() > 0 else 0
                        }
                    }
                    
                    st.session_state.ab_tests.append(test_data)
                    st.success(f"✅ Time period test '{test_name}' created successfully!")
                    st.rerun()
    
    with col2:
        st.markdown("**💡 Testing Guide**")
        
        if test_mode == "Single Marketplace Comparison":
            st.markdown("""
            **Best for:**
            - Comparing campaigns on same marketplace
            - Testing different strategies
            - Before/after analysis
            
            **Tips:**
            - Use same time periods for accuracy
            - Run tests for 7+ days minimum
            - Isolate one variable at a time
            """)
        
        elif test_mode == "Multi-Marketplace Comparison":
            st.markdown("""
            **Best for:**
            - Comparing marketplace performance
            - Channel allocation decisions
            - Finding best-performing platforms
            
            **Tips:**
            - Compare 2-5 marketplaces
            - Use same time period
            - Look at ROAS and AOV together
            """)
        
        else:  # Time Period
            st.markdown("""
            **Best for:**
            - YoY comparisons
            - Seasonal analysis
            - Campaign performance over time
            
            **Tips:**
            - Compare similar periods (e.g., Q4 vs Q4)
            - Account for seasonality
            - Look at growth percentages
            """)
    
    # Display existing tests
    if st.session_state.ab_tests:
        st.markdown("---")
        st.markdown("**🔬 Test Results**")
        
        for idx, test in enumerate(st.session_state.ab_tests):
            # ========== SINGLE MARKETPLACE TEST DISPLAY ==========
            if test['test_type'] == 'Single Marketplace':
                with st.expander(f"🧪 {test['test_name']} ({test['test_type']}) - {test['created_at']}", expanded=True):
                    var_a = test['variant_a']
                    var_b = test['variant_b']
                    
                    # Calculate improvements
                    revenue_improvement = ((var_b['revenue'] - var_a['revenue']) / var_a['revenue'] * 100) if var_a['revenue'] > 0 else 0
                    orders_improvement = ((var_b['orders'] - var_a['orders']) / var_a['orders'] * 100) if var_a['orders'] > 0 else 0
                    roas_improvement = ((var_b['roas'] - var_a['roas']) / var_a['roas'] * 100) if var_a['roas'] > 0 else 0
                    
                    # Display comparison
                    st.markdown(f"**{var_a['name']} ({var_a['channel']})** vs **{var_b['name']} ({var_b['channel']})**")
                    
                    comparison_data = pd.DataFrame({
                        'Metric': ['Revenue', 'Orders', 'Ad Spend', 'ROAS', 'AOV'],
                        var_a['name']: [
                            f"${var_a['revenue']:,.0f}",
                            f"{var_a['orders']:,.0f}",
                            f"${var_a['spend']:,.0f}",
                            f"{var_a['roas']:.2f}x",
                            f"${var_a['aov']:.2f}"
                        ],
                        var_b['name']: [
                            f"${var_b['revenue']:,.0f}",
                            f"{var_b['orders']:,.0f}",
                            f"${var_b['spend']:,.0f}",
                            f"{var_b['roas']:.2f}x",
                            f"${var_b['aov']:.2f}"
                        ],
                        'Change': [
                            f"{revenue_improvement:+.1f}%",
                            f"{orders_improvement:+.1f}%",
                            f"{((var_b['spend'] - var_a['spend']) / var_a['spend'] * 100) if var_a['spend'] > 0 else 0:+.1f}%",
                            f"{roas_improvement:+.1f}%",
                            f"{((var_b['aov'] - var_a['aov']) / var_a['aov'] * 100) if var_a['aov'] > 0 else 0:+.1f}%"
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
                    if st.button(f"🗑️ Delete Test", key=f"delete_single_{idx}"):
                        st.session_state.ab_tests.pop(idx)
                        st.rerun()
            
            # ========== MULTI-MARKETPLACE TEST DISPLAY ==========
            elif test['test_type'] == 'Multi-Marketplace':
                with st.expander(f"🛒 {test['test_name']} (Multi-Marketplace) - {test['created_at']}", expanded=True):
                    st.markdown(f"**Period:** {test['period']}")
                    st.markdown(f"**Comparing {len(test['marketplaces'])} Marketplaces**")
                    
                    # Create comparison dataframe
                    mp_comparison = pd.DataFrame(test['marketplaces'])
                    mp_comparison = mp_comparison.sort_values('revenue', ascending=False)
                    
                    # Visual comparison
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Bar chart comparison
                        fig_mp_compare = go.Figure()
                        
                        fig_mp_compare.add_trace(go.Bar(
                            x=mp_comparison['marketplace'],
                            y=mp_comparison['revenue'],
                            name='Revenue',
                            marker_color='#3b82f6',
                            text=mp_comparison['revenue'].apply(lambda x: f'${x:,.0f}'),
                            textposition='outside'
                        ))
                        
                        fig_mp_compare.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
                            xaxis=dict(title="Marketplace"),
                            margin=dict(l=0, r=0, t=20, b=0),
                            height=300
                        )
                        st.plotly_chart(fig_mp_compare, config={'displayModeBar': False})
                    
                    with col2:
                        # Top performers
                        best_revenue = mp_comparison.iloc[0]
                        best_roas = mp_comparison.nlargest(1, 'roas').iloc[0]
                        
                        st.success(f"💰 **Best Revenue:**\n{best_revenue['marketplace']}\n${best_revenue['revenue']:,.0f}")
                        st.info(f"🎯 **Best ROAS:**\n{best_roas['marketplace']}\n{best_roas['roas']:.2f}x")
                    
                    # Detailed table
                    st.markdown("**📊 Detailed Comparison:**")
                    st.dataframe(
                        mp_comparison,
                        column_config={
                            "marketplace": "Marketplace",
                            "revenue": st.column_config.NumberColumn("Revenue", format="$%d"),
                            "orders": st.column_config.NumberColumn("Orders", format="%d"),
                            "spend": st.column_config.NumberColumn("Ad Spend", format="$%d"),
                            "roas": st.column_config.NumberColumn("ROAS", format="%.2fx"),
                            "aov": st.column_config.NumberColumn("AOV", format="$%.2f"),
                            "acos": st.column_config.NumberColumn("ACOS", format="%.1f%%"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Recommendations
                    st.markdown("**💡 Recommendations:**")
                    top_performer = mp_comparison.iloc[0]['marketplace']
                    worst_performer = mp_comparison.iloc[-1]['marketplace']
                    
                    st.success(f"✅ **Scale:** {top_performer} is your top performer. Consider increasing ad budget here.")
                    if mp_comparison.iloc[-1]['roas'] < 2.0:
                        st.warning(f"⚠️ **Review:** {worst_performer} has low ROAS. Consider optimizing or reducing spend.")
                    
                    # Delete button
                    if st.button(f"🗑️ Delete Test", key=f"delete_multi_{idx}"):
                        st.session_state.ab_tests.pop(idx)
                        st.rerun()
            
            # ========== TIME PERIOD TEST DISPLAY ==========
            elif test['test_type'] == 'Time Period':
                with st.expander(f"📅 {test['test_name']} (Time Period) - {test['created_at']}", expanded=True):
                    st.markdown(f"**Marketplace:** {test['marketplace']}")
                    
                    period_a = test['period_a']
                    period_b = test['period_b']
                    
                    # Calculate changes
                    revenue_change = ((period_b['revenue'] - period_a['revenue']) / period_a['revenue'] * 100) if period_a['revenue'] > 0 else 0
                    orders_change = ((period_b['orders'] - period_a['orders']) / period_a['orders'] * 100) if period_a['orders'] > 0 else 0
                    roas_change = ((period_b['roas'] - period_a['roas']) / period_a['roas'] * 100) if period_a['roas'] > 0 else 0
                    
                    # Display comparison
                    time_comparison = pd.DataFrame({
                        'Metric': ['Revenue', 'Orders', 'Ad Spend', 'ROAS', 'AOV'],
                        period_a['name']: [
                            f"${period_a['revenue']:,.0f}",
                            f"{period_a['orders']:,.0f}",
                            f"${period_a['spend']:,.0f}",
                            f"{period_a['roas']:.2f}x",
                            f"${period_a['aov']:.2f}"
                        ],
                        period_b['name']: [
                            f"${period_b['revenue']:,.0f}",
                            f"{period_b['orders']:,.0f}",
                            f"${period_b['spend']:,.0f}",
                            f"{period_b['roas']:.2f}x",
                            f"${period_b['aov']:.2f}"
                        ],
                        'Change': [
                            f"{revenue_change:+.1f}%",
                            f"{orders_change:+.1f}%",
                            f"{((period_b['spend'] - period_a['spend']) / period_a['spend'] * 100) if period_a['spend'] > 0 else 0:+.1f}%",
                            f"{roas_change:+.1f}%",
                            f"{((period_b['aov'] - period_a['aov']) / period_a['aov'] * 100) if period_a['aov'] > 0 else 0:+.1f}%"
                        ]
                    })
                    
                    st.dataframe(time_comparison, hide_index=True, use_container_width=True)
                    
                    # Analysis
                    st.markdown("**📈 Period Analysis:**")
                    if revenue_change > 20:
                        st.success(f"🎉 Excellent growth of {revenue_change:.1f}%! Business is scaling well.")
                    elif revenue_change > 0:
                        st.info(f"📈 Positive growth of {revenue_change:.1f}%. Continue current strategies.")
                    elif revenue_change > -10:
                        st.warning(f"⚠️ Slight decline of {abs(revenue_change):.1f}%. Monitor trends closely.")
                    else:
                        st.error(f"🚨 Significant decline of {abs(revenue_change):.1f}%. Review strategy immediately.")
                    
                    # Delete button
                    if st.button(f"🗑️ Delete Test", key=f"delete_time_{idx}"):
                        st.session_state.ab_tests.pop(idx)
                        st.rerun()
    else:
        st.info("📝 No A/B tests created yet. Use the forms above to create your first test!")
# TAB 8: Weekly Reports
with tabs[7]:
    st.markdown('<div class="section-header">📅 Weekly Performance Reports</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    📊 <strong>Generate Weekly Reports with Year-over-Year Comparison:</strong> Every metric is automatically 
    compared against the <strong>exact same period last year</strong> so you instantly see whether you're growing or declining.
    </div>
    """, unsafe_allow_html=True)

    # ── CONTROLS ─────────────────────────────────────────────────────────────
    cfg_col1, cfg_col2 = st.columns([2, 1])

    with cfg_col1:
        st.markdown("**📋 Report Generator**")
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            report_start = st.date_input(
                "From Date",
                value=max_date - timedelta(days=7),
                min_value=min_date, max_value=max_date,
                key="report_start"
            )
        with col_date2:
            report_end = st.date_input(
                "To Date",
                value=max_date,
                min_value=min_date, max_value=max_date,
                key="report_end"
            )

        # ── Marketplace filter ────────────────────────────────────────────
        all_marketplaces = sorted(sales_df["channel"].dropna().unique().tolist())
        report_marketplace_options = ["All Marketplaces"] + all_marketplaces
        report_selected_mp = st.multiselect(
            "🛒 Filter by Marketplace",
            options=report_marketplace_options,
            default=["All Marketplaces"],
            key="report_marketplace_filter",
            help="Select one or more marketplaces to scope this report. Defaults to all."
        )
        # Resolve to actual channel list
        if not report_selected_mp or "All Marketplaces" in report_selected_mp:
            report_channels = all_marketplaces
        else:
            report_channels = report_selected_mp

        # Show active filter badge
        if report_channels != all_marketplaces:
            st.info(f"📌 Report scoped to: **{', '.join(report_channels)}**")

        st.markdown("**📑 Include Sections:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            include_kpis            = st.checkbox("KPI Summary",          value=True)
            include_trends          = st.checkbox("Trend Chart",           value=True)
        with c2:
            include_marketplaces    = st.checkbox("Marketplace Breakdown", value=True)
            include_skus            = st.checkbox("Top SKUs",              value=True)
        with c3:
            include_recommendations = st.checkbox("Recommendations",       value=True)
            include_yoy             = st.checkbox("YoY Comparison",        value=True)

        if st.button("📊 Generate Report", type="primary", key="generate_report"):
            period_len = (report_end - report_start).days + 1

            # ── Current period ──────────────────────────────────────────────
            mask_s = (
                (sales_df["date"].dt.date >= report_start) &
                (sales_df["date"].dt.date <= report_end) &
                (sales_df["channel"].isin(report_channels))
            )
            mask_sp = (
                (spend_df["date"].dt.date >= report_start) &
                (spend_df["date"].dt.date <= report_end) &
                (spend_df["channel"].isin(report_channels))
            )
            report_df_s  = sales_df[mask_s]
            report_df_sp = spend_df[mask_sp]
            report_metrics = calc_metrics(report_df_s, report_df_sp)

            # ── Same period LAST YEAR ───────────────────────────────────────
            yoy_start = report_start - timedelta(days=365)
            yoy_end   = report_end   - timedelta(days=365)
            mask_yoy_s = (
                (sales_df["date"].dt.date >= yoy_start) &
                (sales_df["date"].dt.date <= yoy_end) &
                (sales_df["channel"].isin(report_channels))
            )
            mask_yoy_sp = (
                (spend_df["date"].dt.date >= yoy_start) &
                (spend_df["date"].dt.date <= yoy_end) &
                (spend_df["channel"].isin(report_channels))
            )
            yoy_df_s  = sales_df[mask_yoy_s]
            yoy_df_sp = spend_df[mask_yoy_sp]
            yoy_metrics = calc_metrics(yoy_df_s, yoy_df_sp) if len(yoy_df_s) > 0 else None

            mp_label = ", ".join(report_channels) if report_channels != all_marketplaces else "All Marketplaces"
            st.session_state.current_report = {
                'period':        f"{report_start.strftime('%b %d, %Y')} – {report_end.strftime('%b %d, %Y')}",
                'yoy_period':    f"{yoy_start.strftime('%b %d, %Y')} – {yoy_end.strftime('%b %d, %Y')}",
                'metrics':       report_metrics,
                'yoy_metrics':   yoy_metrics,
                'sales_data':    report_df_s,
                'spend_data':    report_df_sp,
                'yoy_sales':     yoy_df_s,
                'yoy_spend':     yoy_df_sp,
                'report_start':  report_start,
                'report_end':    report_end,
                'marketplace_label': mp_label,
                'sections': {
                    'kpis':            include_kpis,
                    'trends':          include_trends,
                    'marketplaces':    include_marketplaces,
                    'skus':            include_skus,
                    'recommendations': include_recommendations,
                    'yoy':             include_yoy,
                }
            }
            st.success("✅ Report generated!")
            st.rerun()

    with cfg_col2:
        st.markdown("**💡 Tips**")
        st.markdown("""
        **Periods:**
        - Last 7 days → Weekly
        - Last 30 days → Monthly
        - Last 90 days → Quarter

        **YoY Comparison:**
        Automatically uses exact same
        calendar period one year ago.

        **No data last year?**
        YoY section is hidden gracefully.

        **Export:**
        Download full markdown report
        or copy text for email/Slack.
        """)

    # ── RENDER REPORT ─────────────────────────────────────────────────────────
    if 'current_report' in st.session_state:
        report = st.session_state.current_report
        m      = report['metrics']
        ym     = report['yoy_metrics']   # None if no last-year data
        has_yoy = ym is not None and report['sections']['yoy']

        def _delta(curr, prev, fmt="$"):
            if prev is None or prev == 0:
                return None
            pct = (curr - prev) / abs(prev) * 100
            if fmt == "$":
                return f"{pct:+.1f}% vs last year (was ${prev:,.0f})"
            elif fmt == "x":
                return f"{pct:+.1f}% vs last year (was {prev:.2f}x)"
            else:
                return f"{pct:+.1f}% vs last year (was {prev:.1f}{fmt})"

        st.markdown("---")

        # ── HEADER BANNER ────────────────────────────────────────────────────
        mp_scope = report.get('marketplace_label', 'All Marketplaces')
        mp_badge = (
            f"&nbsp;|&nbsp; 🛒 <strong>Marketplace:</strong> {mp_scope}"
            if mp_scope != "All Marketplaces"
            else ""
        )
        st.markdown(
            f"<h2 style='margin:0'>📊 Performance Report</h2>"
            f"<p style='color:#9ca3af; margin:4px 0 16px 0;'>"
            f"📅 <strong>This period:</strong> {report['period']} &nbsp;|&nbsp; "
            f"📅 <strong>Last year same period:</strong> {report['yoy_period']}"
            f"{mp_badge}"
            f"</p>",
            unsafe_allow_html=True
        )

        # ── KPI SECTION ───────────────────────────────────────────────────────
        if report['sections']['kpis']:
            st.markdown("### 💎 Key Performance Indicators")
            st.caption("Delta shown vs same period last year. Green = improvement, Red = decline.")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("💰 Revenue",   f"${m['Revenue']:,.0f}",
                          delta=_delta(m['Revenue'], ym['Revenue'] if has_yoy else None) if has_yoy else None)
            with k2:
                st.metric("🛒 Orders",    f"{m['Orders']:,.0f}",
                          delta=_delta(m['Orders'], ym['Orders'] if has_yoy else None, fmt="") if has_yoy else None)
            with k3:
                st.metric("🎯 ROAS",      f"{m['ROAS']:.2f}x",
                          delta=_delta(m['ROAS'], ym['ROAS'] if has_yoy else None, fmt="x") if has_yoy else None)
            with k4:
                st.metric("💹 Net Profit", f"${m['Net']:,.0f}",
                          delta=_delta(m['Net'], ym['Net'] if has_yoy else None) if has_yoy else None)

            k5, k6, k7, k8 = st.columns(4)
            with k5:
                st.metric("📢 Ad Spend",   f"${m['Spend']:,.0f}",
                          delta=_delta(m['Spend'], ym['Spend'] if has_yoy else None) if has_yoy else None,
                          delta_color="inverse")
            with k6:
                st.metric("🏪 Commission", f"${m['Commission']:,.0f}",
                          delta=_delta(m['Commission'], ym['Commission'] if has_yoy else None) if has_yoy else None,
                          delta_color="inverse")
            with k7:
                st.metric("📊 ACOS",       f"{m['ACOS']:.1f}%",
                          delta=_delta(m['ACOS'], ym['ACOS'] if has_yoy else None, fmt="%") if has_yoy else None,
                          delta_color="inverse")
            with k8:
                st.metric("🧾 AOV",        f"${m['AOV']:.2f}",
                          delta=_delta(m['AOV'], ym['AOV'] if has_yoy else None) if has_yoy else None)

        # ── YoY VISUAL COMPARISON ─────────────────────────────────────────────
        if has_yoy and report['sections']['yoy']:
            st.markdown("### 📅 This Week vs Same Week Last Year")

            metrics_to_compare = {
                "Revenue": ("$", m['Revenue'], ym['Revenue']),
                "Orders":  ("",  m['Orders'],  ym['Orders']),
                "Ad Spend":("$", m['Spend'],   ym['Spend']),
                "Net Profit":("$",m['Net'],    ym['Net']),
                "ROAS":    ("x", m['ROAS'],    ym['ROAS']),
                "ACOS %":  ("%", m['ACOS'],    ym['ACOS']),
            }

            yoy_rows = []
            for metric_name, (unit, curr_val, prev_val) in metrics_to_compare.items():
                pct = ((curr_val - prev_val) / abs(prev_val) * 100) if prev_val != 0 else 0
                if unit == "$":
                    curr_str = f"${curr_val:,.0f}"
                    prev_str = f"${prev_val:,.0f}"
                elif unit == "x":
                    curr_str = f"{curr_val:.2f}x"
                    prev_str = f"{prev_val:.2f}x"
                elif unit == "%":
                    curr_str = f"{curr_val:.1f}%"
                    prev_str = f"{prev_val:.1f}%"
                else:
                    curr_str = f"{curr_val:,.0f}"
                    prev_str = f"{prev_val:,.0f}"
                yoy_rows.append({
                    "Metric":         metric_name,
                    "This Period":    curr_str,
                    "Last Year":      prev_str,
                    "Change %":       pct,
                    "Trend":          "📈" if pct > 0 else ("📉" if pct < 0 else "➡️")
                })

            df_yoy = pd.DataFrame(yoy_rows)
            st.dataframe(
                df_yoy,
                column_config={
                    "Metric":      st.column_config.TextColumn("Metric"),
                    "This Period": st.column_config.TextColumn("This Period"),
                    "Last Year":   st.column_config.TextColumn("Same Period Last Year"),
                    "Change %":    st.column_config.NumberColumn("Change %", format="%.1f%%"),
                    "Trend":       st.column_config.TextColumn(""),
                },
                hide_index=True,
                use_container_width=True
            )

            # YoY revenue trend chart side-by-side
            daily_curr = report['sales_data'].groupby(pd.Grouper(key="date", freq="D"))["revenue"].sum().reset_index()
            daily_yoy  = report['yoy_sales'].groupby(pd.Grouper(key="date", freq="D"))["revenue"].sum().reset_index()

            # Align on day-offset so both plot on the same x-axis
            daily_curr["day"] = range(len(daily_curr))
            daily_yoy["day"]  = range(len(daily_yoy))
            max_days = max(len(daily_curr), len(daily_yoy))

            fig_yoy = go.Figure()
            fig_yoy.add_trace(go.Scatter(
                x=daily_curr["day"], y=daily_curr["revenue"],
                name=f"This Period ({report['period'][:6]})",
                line=dict(color="#3b82f6", width=3),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.15)"
            ))
            fig_yoy.add_trace(go.Scatter(
                x=daily_yoy["day"], y=daily_yoy["revenue"],
                name=f"Last Year ({report['yoy_period'][:6]})",
                line=dict(color="#f59e0b", width=2, dash="dot")
            ))
            fig_yoy.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Day of Period", showgrid=False),
                yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
                legend=dict(orientation="h", y=1.12),
                margin=dict(l=0, r=0, t=30, b=0),
                height=320
            )
            st.plotly_chart(fig_yoy, config={"displayModeBar": False})

            # Overall YoY summary callout
            rev_growth = ((m['Revenue'] - ym['Revenue']) / ym['Revenue'] * 100) if ym['Revenue'] > 0 else 0
            if rev_growth > 10:
                st.success(f"🎉 **Strong YoY Growth!** Revenue is up **{rev_growth:.1f}%** vs same period last year.")
            elif rev_growth > 0:
                st.info(f"📈 **Positive YoY Growth.** Revenue up {rev_growth:.1f}% vs last year.")
            elif rev_growth > -10:
                st.warning(f"⚠️ **Slight YoY Decline.** Revenue down {abs(rev_growth):.1f}% vs last year.")
            else:
                st.error(f"🚨 **Significant YoY Decline.** Revenue down {abs(rev_growth):.1f}% — review strategy.")

        # ── TREND CHART ───────────────────────────────────────────────────────
        if report['sections']['trends']:
            st.markdown("### 📈 Revenue Trend — This Period")
            daily_r = report['sales_data'].groupby(pd.Grouper(key="date", freq="D"))["revenue"].sum().reset_index()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=daily_r["date"], y=daily_r["revenue"],
                fill="tozeroy", line=dict(color="#3b82f6", width=2)
            ))
            fig_trend.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
                xaxis=dict(showgrid=False),
                margin=dict(l=0, r=0, t=10, b=0), height=260
            )
            st.plotly_chart(fig_trend, config={"displayModeBar": False})

        # ── MARKETPLACE BREAKDOWN ─────────────────────────────────────────────
        ch_report = pd.DataFrame()
        if report['sections']['marketplaces']:
            st.markdown("### 🛒 Marketplace Performance")

            ch_now = report['sales_data'].groupby("channel").agg({"revenue":"sum","orders":"sum"}).reset_index()
            ch_sp_now = report['spend_data'].groupby("channel")["spend"].sum().reset_index()
            ch_report = pd.merge(ch_now, ch_sp_now, on="channel", how="outer").fillna(0)
            ch_report["roas"] = ch_report.apply(lambda r: r["revenue"]/r["spend"] if r["spend"]>0 else 0, axis=1)
            ch_report["acos"] = ch_report.apply(lambda r: r["spend"]/r["revenue"]*100 if r["revenue"]>0 else 0, axis=1)
            ch_report = ch_report.sort_values("revenue", ascending=False)

            if has_yoy:
                # Merge in YoY revenue per channel
                ch_yoy = report['yoy_sales'].groupby("channel")["revenue"].sum().reset_index().rename(columns={"revenue":"yoy_revenue"})
                ch_report = ch_report.merge(ch_yoy, on="channel", how="left").fillna(0)
                ch_report["yoy_growth"] = ch_report.apply(
                    lambda r: (r["revenue"]-r["yoy_revenue"])/r["yoy_revenue"]*100 if r["yoy_revenue"]>0 else float("nan"), axis=1
                )

                display_mp_cols = ["channel","revenue","orders","spend","roas","acos","yoy_revenue","yoy_growth"]
                col_cfg_mp = {
                    "channel":     "Marketplace",
                    "revenue":     st.column_config.NumberColumn("Revenue",        format="$%d"),
                    "orders":      st.column_config.NumberColumn("Orders",         format="%d"),
                    "spend":       st.column_config.NumberColumn("Ad Spend",       format="$%d"),
                    "roas":        st.column_config.NumberColumn("ROAS",           format="%.2fx"),
                    "acos":        st.column_config.NumberColumn("ACOS",           format="%.1f%%"),
                    "yoy_revenue": st.column_config.NumberColumn("Last Year Rev",  format="$%d"),
                    "yoy_growth":  st.column_config.NumberColumn("YoY Growth",     format="%.1f%%"),
                }
            else:
                display_mp_cols = ["channel","revenue","orders","spend","roas","acos"]
                col_cfg_mp = {
                    "channel": "Marketplace",
                    "revenue": st.column_config.NumberColumn("Revenue",  format="$%d"),
                    "orders":  st.column_config.NumberColumn("Orders",   format="%d"),
                    "spend":   st.column_config.NumberColumn("Ad Spend", format="$%d"),
                    "roas":    st.column_config.NumberColumn("ROAS",     format="%.2fx"),
                    "acos":    st.column_config.NumberColumn("ACOS",     format="%.1f%%"),
                }

            st.dataframe(ch_report[display_mp_cols], column_config=col_cfg_mp,
                         hide_index=True, use_container_width=True)

            if len(ch_report) > 0:
                top_mp = ch_report.iloc[0]
                st.success(f"🏆 **Top Marketplace:** {top_mp['channel']} — ${top_mp['revenue']:,.0f} revenue | {top_mp['roas']:.2f}x ROAS")

        # ── TOP SKUs ──────────────────────────────────────────────────────────
        if report['sections']['skus'] and "Parent" in report['sales_data'].columns:
            st.markdown("### 🏷️ Top SKU Performance")

            sku_now = report['sales_data'].groupby("Parent").agg({"revenue":"sum","orders":"sum"}).reset_index()
            sku_now["aov"] = sku_now["revenue"] / sku_now["orders"].replace(0, np.nan)
            sku_now = sku_now.sort_values("revenue", ascending=False).head(10)

            if has_yoy and "Parent" in report['yoy_sales'].columns:
                sku_yoy = report['yoy_sales'].groupby("Parent")["revenue"].sum().reset_index().rename(columns={"revenue":"yoy_revenue"})
                sku_now = sku_now.merge(sku_yoy, on="Parent", how="left").fillna(0)
                sku_now["yoy_growth"] = sku_now.apply(
                    lambda r: (r["revenue"]-r["yoy_revenue"])/r["yoy_revenue"]*100 if r["yoy_revenue"]>0 else float("nan"), axis=1
                )
                st.dataframe(sku_now, column_config={
                    "Parent":      "SKU",
                    "revenue":     st.column_config.NumberColumn("Revenue",       format="$%d"),
                    "orders":      st.column_config.NumberColumn("Orders",        format="%d"),
                    "aov":         st.column_config.NumberColumn("AOV",           format="$%.2f"),
                    "yoy_revenue": st.column_config.NumberColumn("Last Year Rev", format="$%d"),
                    "yoy_growth":  st.column_config.NumberColumn("YoY Growth",    format="%.1f%%"),
                }, hide_index=True, use_container_width=True)
            else:
                st.dataframe(sku_now[["Parent","revenue","orders","aov"]], column_config={
                    "Parent":  "SKU",
                    "revenue": st.column_config.NumberColumn("Revenue", format="$%d"),
                    "orders":  st.column_config.NumberColumn("Orders",  format="%d"),
                    "aov":     st.column_config.NumberColumn("AOV",     format="$%.2f"),
                }, hide_index=True, use_container_width=True)

        # ── RECOMMENDATIONS ───────────────────────────────────────────────────
        recommendations_list = []
        if report['sections']['recommendations']:
            st.markdown("### 🚀 Strategic Recommendations")
            try:
                ch_matrix_rec = ch_report if (len(ch_report) > 0 and 'roas' in ch_report.columns) \
                    else pd.DataFrame(columns=['channel','revenue','spend','roas'])
                recommendations_list = generate_insights(ch_matrix_rec, m)
                for rec in recommendations_list[:5]:
                    icon_map = {"scale":"📈","warn":"⚠️","crit":"🚨","info":"💡"}
                    icon = icon_map.get(rec['type'], "💡")
                    if rec['type'] == "scale":   st.success(f"{icon} **{rec['title']}** — {rec['msg']}")
                    elif rec['type'] == "warn":  st.warning(f"{icon} **{rec['title']}** — {rec['msg']}")
                    elif rec['type'] == "crit":  st.error(f"{icon} **{rec['title']}** — {rec['msg']}")
                    else:                        st.info(f"{icon} **{rec['title']}** — {rec['msg']}")
                if not recommendations_list:
                    st.info("✅ No critical issues. Performance is stable.")
            except Exception:
                st.warning("⚠️ Could not generate recommendations. Include Marketplace Breakdown for best results.")

        # ── EXPORT ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📤 Export Report")

        # Build markdown text
        yoy_md = ""
        if has_yoy:
            yoy_md = f"""
## 📅 Year-over-Year Comparison
Same period last year: **{report['yoy_period']}**

| Metric | This Period | Last Year | Change |
|--------|------------|-----------|--------|
| Revenue | ${m['Revenue']:,.0f} | ${ym['Revenue']:,.0f} | {((m['Revenue']-ym['Revenue'])/ym['Revenue']*100) if ym['Revenue'] else 0:+.1f}% |
| Orders | {m['Orders']:,.0f} | {ym['Orders']:,.0f} | {((m['Orders']-ym['Orders'])/ym['Orders']*100) if ym['Orders'] else 0:+.1f}% |
| ROAS | {m['ROAS']:.2f}x | {ym['ROAS']:.2f}x | {((m['ROAS']-ym['ROAS'])/ym['ROAS']*100) if ym['ROAS'] else 0:+.1f}% |
| Net Profit | ${m['Net']:,.0f} | ${ym['Net']:,.0f} | {((m['Net']-ym['Net'])/abs(ym['Net'])*100) if ym['Net'] else 0:+.1f}% |
| ACOS | {m['ACOS']:.1f}% | {ym['ACOS']:.1f}% | {((m['ACOS']-ym['ACOS'])/ym['ACOS']*100) if ym['ACOS'] else 0:+.1f}% |
"""

        mp_md = ""
        if report['sections']['marketplaces'] and len(ch_report) > 0:
            mp_md = "\n## 🛒 Marketplace Breakdown\n"
            for _, row in ch_report.head(5).iterrows():
                yoy_str = f" | Last Year: ${row.get('yoy_revenue', 0):,.0f} ({row.get('yoy_growth', float('nan')):+.1f}%)" \
                          if 'yoy_revenue' in ch_report.columns else ""
                mp_md += f"- **{row['channel']}**: ${row['revenue']:,.0f} rev | {row['roas']:.2f}x ROAS{yoy_str}\n"

        rec_md = ""
        if recommendations_list:
            rec_md = "\n## 🚀 Recommendations\n"
            for rec in recommendations_list[:5]:
                rec_md += f"- **{rec['title']}**: {rec['msg']}\n"

        markdown_report = f"""# 📊 Performance Report
**Period:** {report['period']}

## 💎 Key Metrics
| Metric | This Period |
|--------|------------|
| Revenue | ${m['Revenue']:,.0f} |
| Orders | {m['Orders']:,.0f} |
| ROAS | {m['ROAS']:.2f}x |
| Net Profit | ${m['Net']:,.0f} |
| Ad Spend | ${m['Spend']:,.0f} |
| ACOS | {m['ACOS']:.1f}% |
| AOV | ${m['AOV']:.2f} |
{yoy_md}{mp_md}{rec_md}
---
*Generated by Marketplace Business Insights Dashboard*
"""

        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "📥 Download Markdown",
                markdown_report,
                f"report_{report['report_start']}_{report['report_end']}.md",
                "text/markdown",
                key="download_md"
            )
        with ex2:
            if st.button("📋 Show Copyable Text", key="show_copy"):
                st.text_area("Select all & copy:", markdown_report, height=200)

# TAB 9: Data Explorer
with tabs[8]:
    st.markdown('<div class="section-header">📋 Performance Data Explorer</div>', unsafe_allow_html=True)
    
    # Channel Performance Table
    ch_matrix = compute_channel_matrix(df_s, df_sp)
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




# TAB 10: Merchandising Intel
with tabs[9]:
    st.markdown('<div class="section-header">💎 Merchandising Intelligence</div>', unsafe_allow_html=True)

    # ── Load & deduplicate merchandising reference data ───────────────────────
    @st.cache_data(show_spinner=False, ttl=3600)
    def _load_merch():
        import os
        # Streamlit Cloud mounts the repo at /mount/src/<repo-name>/
        # We also try the cwd and the directory of app.py
        candidates = [
            "/mount/src/businessperformancedashboard/Merchandising_data.xlsx",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "Merchandising_data.xlsx"),
            "Merchandising_data.xlsx",
        ]
        found_path = None
        for p in candidates:
            if os.path.exists(p):
                found_path = p
                break

        if found_path is None:
            return None, "FILE_NOT_FOUND"

        # Check openpyxl is available before reading
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            return None, "OPENPYXL_MISSING"

        try:
            raw = pd.read_excel(
                found_path, engine="openpyxl",
                usecols=["Parent", "Design Code", "jewelry_type", "stone"]
            )
        except Exception as e:
            return None, f"READ_ERROR: {e}"

        raw = raw.rename(columns={"Design Code": "design_code"})
        for col in ["Parent", "design_code", "jewelry_type", "stone"]:
            raw[col] = raw[col].astype(str).str.strip()
        lookup = raw.drop_duplicates(subset="Parent").reset_index(drop=True)
        return lookup, "OK"

    merch_lookup, merch_status = _load_merch()

    if merch_status == "OPENPYXL_MISSING":
        st.error(
            "⚠️ **openpyxl is not installed.** Add `openpyxl>=3.1.0` to your "
            "`requirements.txt`, commit it, and redeploy."
        )
        st.stop()
    elif merch_status == "FILE_NOT_FOUND":
        st.error(
            "⚠️ **Merchandising_data.xlsx not found.** "
            "Make sure the file is committed to your GitHub repo in the same folder as app.py."
        )
        st.stop()
    elif merch_status != "OK" or merch_lookup is None:
        st.error(f"⚠️ Failed to load merchandising data: {merch_status}")
        st.stop()

    # ── Mapping stats ─────────────────────────────────────────────────────────
    sales_parents  = set(df_s["Parent"].dropna().unique())
    merch_parents  = set(merch_lookup["Parent"].unique())
    matched_parents = sales_parents & merch_parents

    match_pct = len(matched_parents) / len(sales_parents) * 100 if sales_parents else 0

    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("💎 Merch Catalogue",  f"{len(merch_parents):,} SKUs")
    bm2.metric("📦 Sales SKUs",       f"{len(sales_parents):,} SKUs")
    bm3.metric("✅ Matched",          f"{len(matched_parents):,} SKUs")
    bm4.metric("🔗 Match Rate",       f"{match_pct:.1f}%")

    if match_pct == 0:
        st.warning("No Parent SKUs could be matched between sales data and the merchandising sheet. "
                   "Check that Parent SKU names are formatted the same way in both sources.")

    st.markdown("---")

    # ── Enrich sales data with merch attributes ───────────────────────────────
    df_enriched = df_s.merge(
        merch_lookup[["Parent","design_code","jewelry_type","stone"]],
        on="Parent", how="left"
    )

    # ── Tab-level filters ─────────────────────────────────────────────────────
    st.markdown("### 🔧 Filters")
    fcol1, fcol2, fcol3 = st.columns([2, 2, 1])

    all_jtypes = sorted(merch_lookup["jewelry_type"].dropna().unique().tolist())
    all_stones = sorted(merch_lookup["stone"].dropna().unique().tolist())

    with fcol1:
        sel_jtype = st.multiselect(
            "💍 Jewelry Type",
            options=all_jtypes,
            default=[],
            key="merch_jtype_filter",
            placeholder="All jewelry types…"
        )
    with fcol2:
        sel_stone = st.multiselect(
            "💠 Stone",
            options=all_stones,
            default=[],
            key="merch_stone_filter",
            placeholder="All stones…"
        )
    with fcol3:
        matched_only = st.toggle(
            "Matched SKUs only",
            value=True,
            key="merch_matched_only",
            help="When ON, only Parent SKUs found in both sales and merchandising data are shown."
        )

    # Apply filters
    df_m = df_enriched.copy()
    if matched_only:
        df_m = df_m[df_m["design_code"].notna()]
    if sel_jtype:
        df_m = df_m[df_m["jewelry_type"].isin(sel_jtype)]
    if sel_stone:
        df_m = df_m[df_m["stone"].isin(sel_stone)]

    if df_m.empty:
        st.warning("No records match the current filters. Try removing some filter selections.")
        st.stop()

    # Active filter badges
    active = []
    if sel_jtype: active.append(f"💍 {', '.join(sel_jtype)}")
    if sel_stone: active.append(f"💠 {', '.join(sel_stone[:3])}{'…+more' if len(sel_stone) > 3 else ''}")
    if active:
        st.info("📌 Active: " + "  |  ".join(active) +
                f"  ·  **{df_m['Parent'].nunique():,} SKUs** · "
                f"**${df_m['revenue'].sum():,.0f}** revenue")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 1 ── Revenue by Jewelry Type & Stone
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown("### 📊 Category Revenue Overview")

    ov_left, ov_right = st.columns(2)

    # ── Jewelry Type bar ─────────────────────────────────────────────────────
    with ov_left:
        st.markdown("**💍 Revenue by Jewelry Type**")
        jtype_agg = (
            df_m.groupby("jewelry_type", dropna=False)
            .agg(revenue=("revenue","sum"), orders=("orders","sum"))
            .reset_index()
            .sort_values("revenue", ascending=True)
        )
        jtype_agg["aov"] = (jtype_agg["revenue"] / jtype_agg["orders"].replace(0, np.nan)).fillna(0)

        fig_jtype = px.bar(
            jtype_agg, x="revenue", y="jewelry_type", orientation="h",
            color="aov", color_continuous_scale="Blues",
            custom_data=["orders","aov"],
            labels={"revenue":"Revenue ($)","jewelry_type":"","aov":"AOV ($)"},
            text=jtype_agg["revenue"].apply(lambda v: f"${v/1000:.0f}k")
        )
        fig_jtype.update_traces(
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.0f}<br>Orders: %{customdata[0]:,.0f}<br>AOV: $%{customdata[1]:.2f}<extra></extra>"
        )
        fig_jtype.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=max(350, len(jtype_agg)*35+60),
            margin=dict(l=0, r=70, t=10, b=0),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="#2d303e"),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_jtype, config={"displayModeBar":False}, use_container_width=True)

    # ── Stone pie ────────────────────────────────────────────────────────────
    with ov_right:
        st.markdown("**💠 Top 15 Stones by Revenue**")
        stone_agg = (
            df_m.groupby("stone", dropna=False)["revenue"]
            .sum().reset_index()
            .sort_values("revenue", ascending=False)
            .head(15)
        )
        fig_stone = px.pie(
            stone_agg, values="revenue", names="stone", hole=0.48,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_stone.update_traces(
            textposition="outside", textinfo="percent+label",
            textfont_size=10,
            hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>"
        )
        fig_stone.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            height=max(350, len(jtype_agg)*35+60),
            margin=dict(l=0, r=0, t=10, b=40),
            showlegend=False
        )
        st.plotly_chart(fig_stone, config={"displayModeBar":False}, use_container_width=True)

    # ── KPI summary row ───────────────────────────────────────────────────────
    kp1, kp2, kp3, kp4, kp5 = st.columns(5)
    kp1.metric("💰 Total Revenue",    f"${df_m['revenue'].sum():,.0f}")
    kp2.metric("🛒 Total Orders",     f"{df_m['orders'].sum():,.0f}")
    aov_all = df_m['revenue'].sum() / df_m['orders'].sum() if df_m['orders'].sum() > 0 else 0
    kp3.metric("📊 Blended AOV",      f"${aov_all:,.2f}")
    kp4.metric("🏷️ Active Parent SKUs", f"{df_m['Parent'].nunique():,}")
    kp5.metric("🎨 Design Codes",     f"{df_m['design_code'].nunique():,}")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 2 ── Parent SKU Performance Table
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown("### 🏷️ Parent SKU Performance")
    st.caption("Each Parent SKU mapped to its Design Code, Jewelry Type and Stone from the merchandising catalogue.")

    parent_agg = (
        df_m.groupby("Parent")
        .agg(
            revenue     =("revenue","sum"),
            orders      =("orders","sum"),
            design_code =("design_code",  lambda x: x.dropna().iloc[0] if not x.dropna().empty else "—"),
            jewelry_type=("jewelry_type", lambda x: x.dropna().iloc[0] if not x.dropna().empty else "—"),
            stone       =("stone",        lambda x: x.dropna().iloc[0] if not x.dropna().empty else "—"),
        )
        .reset_index()
    )
    parent_agg["aov"]           = (parent_agg["revenue"] / parent_agg["orders"].replace(0, np.nan)).fillna(0)
    parent_agg["revenue_share"] = (parent_agg["revenue"] / parent_agg["revenue"].sum() * 100).round(2)
    parent_agg = parent_agg.sort_values("revenue", ascending=False).reset_index(drop=True)

    # Inline search
    ps1, ps2 = st.columns([3,1])
    with ps1:
        p_search = st.text_input("🔎 Search Parent SKU or Design Code",
                                  placeholder="e.g. EJ_SE or FC_SB…",
                                  key="merch_parent_search",
                                  label_visibility="collapsed")
    with ps2:
        top_n = st.selectbox("Show top", [25, 50, 100, "All"],
                              key="merch_parent_topn", label_visibility="collapsed")

    p_disp = parent_agg.copy()
    if p_search.strip():
        q = p_search.strip()
        p_disp = p_disp[
            p_disp["Parent"].str.contains(q, case=False, na=False) |
            p_disp["design_code"].str.contains(q, case=False, na=False)
        ]
    if top_n != "All":
        p_disp = p_disp.head(int(top_n))

    st.dataframe(
        p_disp[["Parent","design_code","jewelry_type","stone",
                "revenue","orders","aov","revenue_share"]],
        column_config={
            "Parent":        st.column_config.TextColumn("Parent SKU",   width="medium"),
            "design_code":   st.column_config.TextColumn("Design Code",  width="medium"),
            "jewelry_type":  st.column_config.TextColumn("Jewelry Type", width="small"),
            "stone":         st.column_config.TextColumn("Stone",        width="medium"),
            "revenue":       st.column_config.ProgressColumn(
                                "Revenue ($)", format="$%d",
                                min_value=0, max_value=int(parent_agg["revenue"].max())),
            "orders":        st.column_config.NumberColumn("Orders",      format="%d"),
            "aov":           st.column_config.NumberColumn("AOV ($)",     format="$%.2f"),
            "revenue_share": st.column_config.NumberColumn("Rev Share %", format="%.2f%%"),
        },
        hide_index=True, use_container_width=True, height=430
    )
    st.caption(f"Showing {len(p_disp):,} of {len(parent_agg):,} Parent SKUs")
    st.download_button(
        "📥 Download Parent SKU Report (CSV)",
        p_disp.to_csv(index=False).encode("utf-8"),
        "parent_sku_performance.csv", "text/csv", key="dl_merch_parent"
    )

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 3 ── Design Code Performance
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown("### 🎨 Design Code Performance")
    st.caption("A Design Code groups multiple Parent SKUs (different stones/variants of the same design). Revenue is aggregated across all its variants.")

    design_agg = (
        df_m[df_m["design_code"].notna() & (df_m["design_code"] != "nan")]
        .groupby("design_code")
        .agg(
            revenue     =("revenue","sum"),
            orders      =("orders","sum"),
            variants    =("Parent","nunique"),
            jewelry_type=("jewelry_type", lambda x: x.dropna().iloc[0] if not x.dropna().empty else "—"),
            stones      =("stone",        lambda x: ", ".join(sorted(set(x.dropna().astype(str).tolist())))),
        )
        .reset_index()
    )
    design_agg["aov"]           = (design_agg["revenue"] / design_agg["orders"].replace(0, np.nan)).fillna(0)
    design_agg["revenue_share"] = (design_agg["revenue"] / design_agg["revenue"].sum() * 100).round(2)
    design_agg = design_agg.sort_values("revenue", ascending=False).reset_index(drop=True)

    # Top 20 chart
    top20 = design_agg.head(20).copy()
    fig_dc = px.bar(
        top20, x="design_code", y="revenue",
        color="jewelry_type",
        custom_data=["orders","aov","variants","stones"],
        labels={"revenue":"Revenue ($)","design_code":"Design Code","jewelry_type":"Type"},
        text=top20["revenue"].apply(lambda v: f"${v/1000:.1f}k")
    )
    fig_dc.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Revenue: $%{y:,.0f}<br>"
            "Orders: %{customdata[0]:,.0f}<br>"
            "AOV: $%{customdata[1]:.2f}<br>"
            "Variants: %{customdata[2]}<br>"
            "Stones: %{customdata[3]}<extra></extra>"
        )
    )
    fig_dc.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(tickangle=-40, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#2d303e"),
        legend=dict(title="Jewelry Type", orientation="h", y=1.14),
        bargap=0.25
    )
    st.plotly_chart(fig_dc, config={"displayModeBar":False}, use_container_width=True)

    # Design code searchable table
    ds1, ds2 = st.columns([3,1])
    with ds1:
        dc_search = st.text_input("🔎 Search Design Code or Stone",
                                   placeholder="e.g. FC_SB or Diamond…",
                                   key="merch_dc_search",
                                   label_visibility="collapsed")
    with ds2:
        dc_top_n = st.selectbox("Show top", [25, 50, 100, "All"],
                                 key="merch_dc_topn", label_visibility="collapsed")

    dc_disp = design_agg.copy()
    if dc_search.strip():
        q2 = dc_search.strip()
        dc_disp = dc_disp[
            dc_disp["design_code"].str.contains(q2, case=False, na=False) |
            dc_disp["stones"].str.contains(q2, case=False, na=False)
        ]
    if dc_top_n != "All":
        dc_disp = dc_disp.head(int(dc_top_n))

    st.dataframe(
        dc_disp[["design_code","jewelry_type","stones","variants",
                 "revenue","orders","aov","revenue_share"]],
        column_config={
            "design_code":   st.column_config.TextColumn("Design Code",   width="medium"),
            "jewelry_type":  st.column_config.TextColumn("Jewelry Type",  width="small"),
            "stones":        st.column_config.TextColumn("Stones",        width="large"),
            "variants":      st.column_config.NumberColumn("# Variants",  format="%d"),
            "revenue":       st.column_config.ProgressColumn(
                                "Revenue ($)", format="$%d",
                                min_value=0, max_value=int(design_agg["revenue"].max())),
            "orders":        st.column_config.NumberColumn("Orders",      format="%d"),
            "aov":           st.column_config.NumberColumn("AOV ($)",     format="$%.2f"),
            "revenue_share": st.column_config.NumberColumn("Rev Share %", format="%.2f%%"),
        },
        hide_index=True, use_container_width=True, height=430
    )
    st.caption(f"Showing {len(dc_disp):,} of {len(design_agg):,} Design Codes")
    st.download_button(
        "📥 Download Design Code Report (CSV)",
        dc_disp.to_csv(index=False).encode("utf-8"),
        "design_code_performance.csv", "text/csv", key="dl_merch_design"
    )

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 4 ── Jewelry Type × Stone Heatmap
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown("### 🔥 Revenue Heatmap — Jewelry Type × Stone")
    st.caption("Top 15 stones shown. Colour intensity and label = revenue. Hover for exact figure.")

    heat_raw = (
        df_m.groupby(["jewelry_type","stone"])["revenue"]
        .sum().reset_index()
    )
    # Keep top 15 stones by total revenue so the chart stays readable
    top15_stones = (
        heat_raw.groupby("stone")["revenue"].sum()
        .nlargest(15).index.tolist()
    )
    heat_raw = heat_raw[heat_raw["stone"].isin(top15_stones)]
    pivot = heat_raw.pivot(index="jewelry_type", columns="stone", values="revenue").fillna(0)

    text_matrix = [
        [f"${v/1000:.0f}k" if v > 0 else "" for v in row]
        for row in pivot.values
    ]
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Blues",
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Revenue: $%{z:,.0f}<extra></extra>",
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 9}
    ))
    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(320, len(pivot) * 42 + 80),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickangle=-42, side="bottom"),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_heat, config={"displayModeBar":False}, use_container_width=True)

    # Unmatched callout (collapsible)
    if len(sales_parents - merch_parents) > 0:
        with st.expander(f"ℹ️ {len(sales_parents - merch_parents):,} sales SKUs with no merchandising match"):
            st.caption("These Parent SKUs have sales data but were not found in Merchandising_data.xlsx.")
            unmatched_df = pd.DataFrame(sorted(sales_parents - merch_parents), columns=["Parent SKU"])
            st.dataframe(unmatched_df, hide_index=True, use_container_width=True, height=250)
            st.download_button("📥 Download Unmatched SKU List",
                               unmatched_df.to_csv(index=False).encode("utf-8"),
                               "unmatched_skus.csv", "text/csv", key="dl_unmatched")

# ---------------- FOOTER ----------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div style='text-align: left; color: #6b7280; font-size: 12px;'>📅 Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='text-align: center; color: #6b7280; font-size: 12px;'>⚙️ Safe Margin: {SAFE_MARGIN*100:.0f}%</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div style='text-align: right; color: #6b7280; font-size: 12px;'>📊 Data Points: {len(df_s):,}</div>", unsafe_allow_html=True)
