import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets
from datetime import date, timedelta, datetime
import numpy as np
import json
import hashlib
import re
import os

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
    .stApp {
        background: linear-gradient(135deg, #0f1116 0%, #1a1d29 100%);
    }
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
        top: 0; left: 0; right: 0;
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
    .metric-card:hover::before { opacity: 1; }
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
    .accent-blue   { --accent-color: #3b82f6; }
    .accent-green  { --accent-color: #10b981; }
    .accent-orange { --accent-color: #f97316; }
    .accent-purple { --accent-color: #8b5cf6; }
    .accent-pink   { --accent-color: #ec4899; }
    .accent-cyan   { --accent-color: #06b6d4; }
    .accent-yellow { --accent-color: #eab308; }
    .accent-red    { --accent-color: #ef4444; }
    .delta-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 800;
        gap: 4px;
    }
    .delta-pos { background: rgba(16,185,129,0.25); color:#34d399; box-shadow:0 0 20px rgba(16,185,129,0.3); }
    .delta-neg { background: rgba(239,68,68,0.25);  color:#f87171; box-shadow:0 0 20px rgba(239,68,68,0.3); }
    .section-header {
        font-size: 20px; font-weight: 700; color: #f3f4f6;
        margin: 40px 0 20px 0;
        display: flex; align-items: center; gap: 12px;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: rgba(30,32,40,0.5); padding: 8px; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 12px 24px; font-weight: 600; transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    .stButton > button {
        border-radius: 10px; font-weight: 600; transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
    .js-plotly-plot .plotly .modebar { display: none !important; }
    .streamlit-expanderHeader {
        background: rgba(30,32,40,0.6); border-radius: 8px; font-weight: 600;
    }
    .streamlit-expanderHeader:hover { background: rgba(42,45,58,0.8); }
    .rec-card {
        background: rgba(30,32,40,0.6);
        border-left: 4px solid #3b82f6;
        border-radius: 8px; padding: 16px; margin-bottom: 12px; transition: transform 0.2s;
    }
    .rec-card:hover { transform: translateX(5px); background: rgba(42,45,58,0.8); }
    .rec-title { font-weight:700; font-size:16px; color:#fff; display:flex; align-items:center; gap:8px; }
    .rec-body  { color:#9ca3af; font-size:14px; margin-top:4px; }
    .rec-high { border-left-color: #10b981; }
    .rec-warn { border-left-color: #f59e0b; }
    .rec-crit { border-left-color: #ef4444; }
    .rec-info { border-left-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------
def metric_card(label, value, delta=None, prefix="", suffix="", color="blue", inverse=False, icon=""):
    delta_html = ""
    if delta is not None:
        is_pos  = delta >= 0
        is_good = not is_pos if inverse else is_pos
        delta_class = "delta-pos" if is_good else "delta-neg"
        arrow = "↑" if is_pos else "↓"
        delta_html = f'<span class="delta-badge {delta_class}">{arrow} {abs(delta):.1f}%</span>'
    else:
        delta_html = '<span style="color:#6b7280; font-size:11px">No prev data</span>'
    icon_html = f'<span style="font-size:16px;">{icon}</span>' if icon else ''
    st.markdown(f"""
    <div class="metric-card accent-{color}">
        <div class="metric-label">{icon_html}{label}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def multiselect_with_all(label, options):
    ALL  = "All"
    opts = [ALL] + sorted(list(options))
    selected = st.sidebar.multiselect(label, opts, default=[ALL])
    return list(options) if ALL in selected or not selected else selected

# ============================================================
# FIX 1: Robust date parser (handles multiple formats)
# ============================================================
def robust_parse_dates(series):
    """
    Try multiple date parsing strategies.
    Returns a Series of datetime (NaT for failures).
    """
    # Ensure we have a pandas Series
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception:
            return pd.Series([pd.NaT] * len(series) if hasattr(series, '__len__') else pd.NaT)

    # First attempt: let pandas guess
    try:
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        if parsed.notna().sum() > 0:
            return parsed
    except Exception:
        pass

    # Try with dayfirst=True (common for DD/MM/YYYY)
    try:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if parsed.notna().sum() > 0:
            return parsed
    except Exception:
        pass

    # Try with explicit formats (most common)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y", "%Y%m%d"):
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed
        except Exception:
            continue

    # Last resort: return all NaT
    return pd.Series([pd.NaT] * len(series))

# ============================================================
# FIX 2: Improved data loader with fuzzy column matching
# ============================================================
@st.cache_data(show_spinner=True, ttl=600)
def load_and_process_data():
    import tempfile
    import json as _json
    import os as _os

    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            _json.dump(creds_dict, tmp)
            creds = tmp.name
    except Exception as e:
        return None, None, f"Credentials error: {e}"

    try:
        workbook = load_all_sheets(creds, "New BI Dashboard")
    except Exception as e:
        return None, None, str(e)
    finally:
        try:
            _os.unlink(creds)
        except Exception:
            pass

    if not workbook:
        return None, None, "No sheets found."

    sales_df = None
    spend_df = None

    # Helper to normalise column names
    def _norm_cols(df):
        return [str(c).strip().lower() for c in df.columns]

    # Helper to find the best matching column (exact or contains)
    def _find_column(df, candidates, default=None):
        cols_norm = _norm_cols(df)
        # Exact match
        for cand in candidates:
            if cand in cols_norm:
                return df.columns[cols_norm.index(cand)]
        # Fuzzy match (contains)
        for cand in candidates:
            for i, col in enumerate(cols_norm):
                if cand in col:
                    return df.columns[i]
        return default

    for sheet_name, df in workbook.items():
        cols_norm = _norm_cols(df)

        # ── SALES TAB detection ────────────────────────────────────────
        if ("purchased on" in cols_norm and
            "discounted price" in cols_norm and
            "no of orders" in cols_norm):
            sales_df = df.copy()

        # ── SPEND TAB detection ────────────────────────────────────────
        # Flexible: must have a date-like column AND a spend-like column,
        # and should NOT be the sales sheet.
        has_date = any("date" in c or "day" in c for c in cols_norm)
        has_spend = any("spend" in c or "cost" in c or "ads" in c for c in cols_norm)
        if has_date and has_spend and "purchased on" not in cols_norm:
            spend_df = df.copy()

    if sales_df is None:
        return None, None, "Sales sheet not found (missing columns: 'purchased on', 'discounted price', 'no of orders')."

    if spend_df is None:
        # Create empty DataFrame if no spend sheet found
        spend_df = pd.DataFrame(columns=["date", "channel", "spend"])
        st.warning("⚠️ Spend sheet not found. Using empty spend data.")
    else:
        # Normalise spend column names
        spend_df.columns = [str(c).strip().lower().replace(" ", "_") for c in spend_df.columns]

        # Find actual date column
        date_col = _find_column(spend_df, ["date", "day", "purchase_date", "transaction_date"])
        if date_col is None:
            spend_df["date"] = pd.NaT
        else:
            # Ensure we extract a Series (not a DataFrame)
            date_series = spend_df[date_col]
            if isinstance(date_series, pd.DataFrame):
                date_series = date_series.iloc[:, 0]
            spend_df["date"] = robust_parse_dates(date_series)

        # Find actual spend column
        spend_col = _find_column(spend_df, ["spend", "cost", "ad_spend", "spend_usd", "ads_spend", "amount"])
        if spend_col is None:
            spend_df["spend"] = 0.0
        else:
            spend_df["spend"] = pd.to_numeric(spend_df[spend_col], errors="coerce").fillna(0)

        # Find channel column (optional)
        channel_col = _find_column(spend_df, ["channel", "marketplace", "platform", "market", "source"])
        if channel_col is None:
            spend_df["channel"] = "All"
        else:
            spend_df["channel"] = spend_df[channel_col].astype(str).str.strip()
            spend_df["channel"] = spend_df["channel"].replace(
                {"": "All", "nan": "All", "None": "All", "NaN": "All"}
            )

        # Keep only necessary columns
        keep_cols = ["date", "channel", "spend"]
        for col in keep_cols:
            if col not in spend_df.columns:
                spend_df[col] = None
        spend_df = spend_df[keep_cols].dropna(subset=["date"])

    # =====================================================================
    # SALES PROCESSING (with robust column detection)
    # =====================================================================
    sales_df.columns = [str(c).strip().lower().replace(" ", "_") for c in sales_df.columns]

    # Find and parse date column
    date_col = _find_column(sales_df, ["purchased_on", "date", "order_date", "transaction_date"])
    if date_col is None:
        return None, None, "Sales sheet missing a date column."
    sales_df["date"] = robust_parse_dates(sales_df[date_col])

    # Revenue (discounted price)
    rev_col = _find_column(sales_df, ["discounted_price", "revenue", "sale_amount", "total", "price"])
    if rev_col is None:
        return None, None, "Sales sheet missing a revenue column (expected 'discounted price')."
    sales_df["revenue"] = pd.to_numeric(sales_df[rev_col], errors="coerce").fillna(0)

    # Orders
    ord_col = _find_column(sales_df, ["no_of_orders", "orders", "quantity", "units", "order_count"])
    if ord_col is None:
        sales_df["orders"] = 1  # assume one per row
    else:
        sales_df["orders"] = pd.to_numeric(sales_df[ord_col], errors="coerce").fillna(1)

    # Selling commission
    comm_col = _find_column(sales_df, ["selling_commission", "commission", "marketplace_fee", "fee"])
    if comm_col is None:
        sales_df["selling_commission"] = 0.0
    else:
        sales_df["selling_commission"] = pd.to_numeric(sales_df[comm_col], errors="coerce").fillna(0)

    # Channel / marketplace
    channel_col = _find_column(sales_df, ["channel", "marketplace", "platform", "source"])
    if channel_col is None:
        sales_df["channel"] = "Unknown"
    else:
        sales_df["channel"] = sales_df[channel_col].astype(str).str.strip()

    # Product type
    type_col = _find_column(sales_df, ["type", "product_type", "category"])
    if type_col is None:
        sales_df["type"] = "Unknown"
    else:
        sales_df["type"] = sales_df[type_col].astype(str).str.strip()

    # Parent SKU
    parent_col = _find_column(sales_df, ["parent", "parent_sku", "sku_parent", "product_id"])
    if parent_col is None:
        sales_df["Parent"] = "Unknown"
    else:
        sales_df["Parent"] = sales_df[parent_col].astype(str).str.strip()

    # Child SKU
    sku_col = _find_column(sales_df, ["sku", "child_sku", "variant_sku", "item_sku"])
    if sku_col is None:
        sales_df["SKU"] = "Unknown"
    else:
        sales_df["SKU"] = sales_df[sku_col].astype(str).str.strip()

    # Drop rows with invalid dates
    sales_df = sales_df.dropna(subset=["date"])

    # Optional debug info (enable via secrets)
    if st.secrets.get("debug_mode", False):
        st.write("Sales date range after parsing:", sales_df["date"].min(), "→", sales_df["date"].max())
        st.write("Spend date range after parsing:", spend_df["date"].min() if not spend_df.empty else "No spend data")

    return sales_df, spend_df, None

# ---------------- LOAD STATE ----------------
with st.spinner("⚡ Loading business intelligence..."):
    result = load_and_process_data()

    if result[2]:
        st.error(f"❌ **Data Load Failed:** {result[2]}")
        st.stop()

    sales_df, spend_df = result[0], result[1]

    # Ensure datetime conversion
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
    if not spend_df.empty:
        spend_df["date"] = pd.to_datetime(spend_df["date"], errors="coerce")

    if sales_df is None or sales_df.empty:
        st.warning("⚠️ No sales data available.")
        st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🎛️ Control Panel")

min_date = sales_df["date"].min().date()
max_date = sales_df["date"].max().date()

st.sidebar.info(
    f"Available Data\n\n{min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}"
)

default_end = max_date
default_start = max(min_date, max_date - timedelta(days=30))

col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=min_date,
        max_value=max_date
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=default_end,
        min_value=min_date,
        max_value=max_date
    )

if start_date > end_date:
    st.error("Start date cannot be after End date")
    st.stop()

selected_channels = multiselect_with_all("📺 Marketplaces", sales_df["channel"].unique())

if "type" in sales_df.columns:
    def _sidebar_remap(v):
        v = str(v).strip()
        if v in ("", "nan", "None", "NaN", "<NA>", "Unknown"):
            return "Rings"
        _m = {
            "ring":"Rings","rings":"Rings",
            "pendant":"Pendants","pendants":"Pendants",
            "necklace":"Pendants","necklaces":"Pendants",
            "earring":"Earrings","earrings":"Earrings",
            "bracelet":"Bracelets","bracelets":"Bracelets",
            "band":"Band","bands":"Band",
            "bangle":"Bangles","bangles":"Bangles",
            "lapel pin":"Lapel Pin","misc":"MISC",
            "men's band":"Men's Band","mens band":"Men's Band",
        }
        return _m.get(v.lower(), v)

    _type_display_series = sales_df["type"].apply(_sidebar_remap)
    _type_options        = sorted(_type_display_series.unique().tolist())
    selected_types_display = multiselect_with_all("🏷️ Product Types", _type_options)
    selected_types = sales_df.loc[
        _type_display_series.isin(selected_types_display), "type"
    ].unique().tolist()
else:
    selected_types         = []
    selected_types_display = []

st.sidebar.markdown("---")
comparison_period = st.sidebar.selectbox(
    "📊 Compare Against",
    ["Year over Year", "Month over Month"]
)

# ---------------- APPLY FILTERS ----------------
start_ts = pd.to_datetime(start_date)
end_ts   = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask_sales = (
    (sales_df["date"] >= start_ts) &
    (sales_df["date"] <= end_ts) &
    (sales_df["channel"].isin(selected_channels)) &
    (sales_df["type"].isin(selected_types) if "type" in sales_df.columns and selected_types else True)
)
df_s = sales_df[mask_sales]

if spend_df.empty:
    df_sp = pd.DataFrame(columns=["date", "channel", "spend"])
else:
    mask_spend = (
        (spend_df["date"] >= start_ts) &
        (spend_df["date"] <= end_ts) &
        (spend_df["channel"].isin(selected_channels))
    )
    df_sp = spend_df.loc[mask_spend]

# Previous Period
days_diff = (end_date - start_date).days + 1

if comparison_period == "Year over Year":
    start_ly = start_date - pd.DateOffset(years=1)
    end_ly   = end_date   - pd.DateOffset(years=1)
elif comparison_period == "Month over Month":
    start_ly = start_date - pd.DateOffset(months=1)
    end_ly   = end_date   - pd.DateOffset(months=1)
elif comparison_period == "Week over Week":
    start_ly = start_date - timedelta(days=7)
    end_ly   = end_date   - timedelta(days=7)
else:
    start_ly = start_date - timedelta(days=days_diff)
    end_ly   = start_date - timedelta(days=1)

start_ly_ts = pd.to_datetime(start_ly)
end_ly_ts   = pd.to_datetime(end_ly) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask_sales_ly = (
    (sales_df["date"] >= start_ly_ts) &
    (sales_df["date"] <= end_ly_ts) &
    (sales_df["channel"].isin(selected_channels))
)
df_s_ly = sales_df[mask_sales_ly]

if spend_df.empty:
    df_sp_ly = pd.DataFrame(columns=["date", "channel", "spend"])
else:
    mask_spend_ly = (
        (spend_df["date"] >= start_ly_ts) &
        (spend_df["date"] <= end_ly_ts) &
        (spend_df["channel"].isin(selected_channels))
    )
    df_sp_ly = spend_df.loc[mask_spend_ly]

# ---------------- METRIC CALCULATIONS ----------------
def calc_metrics(sales, spend):
    rev    = sales["revenue"].sum()
    comm   = sales["selling_commission"].sum() if "selling_commission" in sales.columns else 0
    ads    = spend["spend"].sum() if not spend.empty else 0
    orders = sales["orders"].sum()
    net    = (rev * SAFE_MARGIN) - ads - comm
    roas   = (rev / ads)       if ads > 0  else 0
    acos   = (ads / rev * 100) if rev > 0  else 0
    aov    = (rev / orders)    if orders > 0 else 0
    return {
        "Revenue": rev, "Orders": orders, "Spend": ads, "Commission": comm,
        "Net": net, "ROAS": roas, "ACOS": acos, "AOV": aov
    }

def generate_insights(df_channel, current_metrics):
    insights = []
    if 'roas' in df_channel.columns:
        for _, row in df_channel[df_channel['roas'] >= 3.0].iterrows():
            insights.append({
                "type": "scale",
                "title": f"🚀 Scale Up: {row['channel']}",
                "msg": f"ROAS is strong at {row['roas']:.2f}x. Consider increasing daily budget by 15-20%.",
                "metric": f"{row['roas']:.2f}x ROAS"
            })
    if 'roas' in df_channel.columns and 'spend' in df_channel.columns:
        for _, row in df_channel[(df_channel['roas'] < 1.5) & (df_channel['spend'] > 500)].iterrows():
            insights.append({
                "type": "crit",
                "title": f"🛑 High Spend / Low Return: {row['channel']}",
                "msg": f"Spent ${row['spend']:,.0f} with only {row['roas']:.2f}x ROAS. Review search terms or lower bids.",
                "metric": f"${row['spend']:,.0f} Spend"
            })
    if current_metrics['Net'] < 0:
        insights.append({
            "type": "crit",
            "title": "📉 Net Loss Alert",
            "msg": "Operating at a net loss for this period. Cut spend on channels with < 2.0 ROAS immediately.",
            "metric": f"${current_metrics['Net']:,.0f}"
        })
    elif current_metrics['Revenue'] > 0 and (current_metrics['Net'] / current_metrics['Revenue']) < 0.10:
        insights.append({
            "type": "warn",
            "title": "⚠️ Thin Margins",
            "msg": "Net margin is below 10%. Watch COGS and commission rates closely.",
            "metric": f"{(current_metrics['Net']/current_metrics['Revenue']*100):.1f}% Margin"
        })
    if current_metrics['AOV'] > 0 and current_metrics['AOV'] < 50:
        insights.append({
            "type": "info",
            "title": "📦 Bundle Opportunity",
            "msg": "AOV is below $50. Try 'Buy 2 Save 10%' bundles or post-purchase upsells.",
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

# ---------------- KPI GRID ----------------
st.markdown('<div class="section-header">💎 Key Performance Indicators</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1: metric_card("Total Revenue",        f"{curr['Revenue']:,.0f}",  delta("Revenue"),    prefix="$", color="blue",   icon="💰")
with k2: metric_card("Total Orders",         f"{curr['Orders']:,.0f}",   delta("Orders"),                color="cyan",   icon="🛒")
with k3: metric_card("Average Order Value",  f"{curr['AOV']:,.2f}",      delta("AOV"),        prefix="$", color="purple", icon="📊")
with k4: metric_card("Net Profit",           f"{curr['Net']:,.0f}",      delta("Net"),        prefix="$", color="green",  icon="💹")

st.markdown("")

k5, k6, k7, k8 = st.columns(4)
with k5: metric_card("Ad Spend",         f"{curr['Spend']:,.0f}",      delta("Spend"),      prefix="$", color="orange", inverse=True, icon="📢")
with k6: metric_card("Selling Commission",f"{curr['Commission']:,.0f}", delta("Commission"), prefix="$", color="pink",   inverse=True, icon="💳")
with k7: metric_card("ROAS",             f"{curr['ROAS']:.2f}",         delta("ROAS"),                  suffix="x", color="yellow", icon="🎯")
with k8: metric_card("ACOS",             f"{curr['ACOS']:.1f}",         delta("ACOS"),                  suffix="%", color="red",    inverse=True, icon="📈")

# ---------------- TABS ----------------
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

# ==================== ALL FOLLOWING TABS REMAIN IDENTICAL TO YOUR ORIGINAL CODE ====================
# (They are unchanged – we only fixed the data loading and date parsing)

# Since the rest of the tabs (TAB 1 through TAB 10) are unchanged from your original working code,
# we skip repeating them here for brevity. In your actual deployment, copy the exact same tab content
# from your original file after the line "# ---------------- TABS ----------------".

# ==================== FOOTER ====================
st.markdown("---")
f1, f2, f3 = st.columns(3)
with f1: st.markdown(f"<div style='text-align:left;color:#6b7280;font-size:12px;'>📅 Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
with f2: st.markdown(f"<div style='text-align:center;color:#6b7280;font-size:12px;'>⚙️ Safe Margin: {SAFE_MARGIN*100:.0f}%</div>", unsafe_allow_html=True)
with f3: st.markdown(f"<div style='text-align:right;color:#6b7280;font-size:12px;'>📊 Data Points: {len(df_s):,}</div>", unsafe_allow_html=True)
