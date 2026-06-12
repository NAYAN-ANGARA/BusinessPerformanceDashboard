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
# FIX: robust column normaliser used everywhere in loader
# ============================================================
def _norm_cols(df):
    """Return list of stripped-lowercase column names."""
    return [str(c).strip().lower() for c in df.columns]

# ---------------- DATA LOADER ----------------
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

    for sheet_name, df in workbook.items():
        # ── Normalise column names for detection only ──────────────────
        cols_norm = _norm_cols(df)

        # ── SALES TAB detection ────────────────────────────────────────
        if (
            "purchased on"    in cols_norm
            and "discounted price" in cols_norm
            and "no of orders"     in cols_norm
        ):
            sales_df = df.copy()

        # ── SPEND TAB detection ────────────────────────────────────────
        # FIX 1: Use substring matching (any("spend" in c ...)) instead of
        # exact list-element check ("spend" in cols_norm).
        # This ensures sheets with columns like "Ad Spend" or "Total Spend"
        # are correctly identified — previously they were silently skipped.
        elif (
            any("date"  in c for c in cols_norm)
            and any("spend" in c for c in cols_norm)
            and "purchased on" not in cols_norm   # make sure it's not the sales sheet
        ):
            spend_df = df.copy()

    if sales_df is None:
        return None, None, "Sales sheet not found."

    if spend_df is None:
        # Last-resort fallback: match sheet by name
        for sheet_name, df in workbook.items():
            sn_key = sheet_name.lower().replace(" ", "").replace("_", "")
            if any(kw in sn_key for kw in ["spend", "adspend", "advertising", "adsdata"]):
                cols_chk = _norm_cols(df)
                if any("date" in c for c in cols_chk):
                    spend_df = df.copy()
                    break

    if spend_df is None:
        spend_df = pd.DataFrame(columns=["date", "channel", "spend"])

    # =====================================================================
    # SALES PROCESSING
    # =====================================================================
    # Normalise column names to snake_case
    sales_df.columns = [str(c).strip().lower().replace(" ", "_") for c in sales_df.columns]

    # FIX 2: Use format="mixed" so pandas parses each row independently.
    # Without this, pandas infers a single format from the first N rows;
    # if the sheet's date format changed mid-history (e.g. after May 14),
    # all later rows are silently coerced to NaT and then dropped.
    _raw_dates = sales_df["purchased_on"].astype(str).str.strip()
    sales_df["date"] = pd.to_datetime(
        _raw_dates, errors="coerce", format="mixed", dayfirst=False
    )
    # If dayfirst=False left too many NaTs, retry with dayfirst=True
    if sales_df["date"].isna().mean() > 0.2:
        sales_df["date"] = pd.to_datetime(
            _raw_dates, errors="coerce", format="mixed", dayfirst=True
        )

    sales_df["revenue"] = pd.to_numeric(
        sales_df.get("discounted_price", 0), errors="coerce"
    ).fillna(0)

    sales_df["orders"] = pd.to_numeric(
        sales_df.get("no_of_orders", 0), errors="coerce"
    ).fillna(0)

    # ── selling_commission — tolerate missing column ────────────────────
    if "selling_commission" in sales_df.columns:
        sales_df["selling_commission"] = pd.to_numeric(
            sales_df["selling_commission"], errors="coerce"
        ).fillna(0)
    else:
        _comm_candidates = [
            c for c in sales_df.columns
            if "commission" in c or "comm" in c
        ]
        if _comm_candidates:
            sales_df["selling_commission"] = pd.to_numeric(
                sales_df[_comm_candidates[0]], errors="coerce"
            ).fillna(0)
        else:
            sales_df["selling_commission"] = 0.0

    sales_df["channel"] = sales_df["channel"].astype(str).str.strip()
    sales_df["type"]    = sales_df["type"].astype(str).str.strip() if "type" in sales_df.columns else "Unknown"

    if "parent" in sales_df.columns:
        sales_df["Parent"] = sales_df["parent"].astype(str).str.strip()
    else:
        sales_df["Parent"] = "Unknown"

    if "sku" in sales_df.columns:
        sales_df["SKU"] = sales_df["sku"].astype(str).str.strip()
    else:
        sales_df["SKU"] = "Unknown"

    sales_df = sales_df.dropna(subset=["date"])

    # =====================================================================
    # SPEND PROCESSING
    # =====================================================================
    spend_df.columns = [str(c).strip().lower().replace(" ", "_") for c in spend_df.columns]

    if len(spend_df) > 0:
        # FIX 3a: Standardise the date column name.
        # After normalisation "Date" stays "date" but exotic headers like
        # "Week Date" become "week_date" — rename whichever one has "date".
        if "date" not in spend_df.columns:
            _date_col = next((c for c in spend_df.columns if "date" in c), None)
            if _date_col:
                spend_df = spend_df.rename(columns={_date_col: "date"})

        # FIX 3b: Standardise the spend column name.
        # "Ad Spend" → "ad_spend" after normalisation; "spend" won't exist.
        # Mirror the same pattern already used for the channel column below.
        if "spend" not in spend_df.columns:
            _spend_col = next((c for c in spend_df.columns if "spend" in c), None)
            if _spend_col:
                spend_df = spend_df.rename(columns={_spend_col: "spend"})
            else:
                spend_df["spend"] = 0.0

        # FIX 2 (spend): Use format="mixed" for the same reason as sales above.
        spend_df["date"]  = pd.to_datetime(spend_df["date"],  errors="coerce", format="mixed")
        spend_df["spend"] = pd.to_numeric(spend_df["spend"], errors="coerce").fillna(0)

        # Channel column may be named differently or absent
        if "channel" not in spend_df.columns:
            _ch_candidates = [c for c in spend_df.columns if "channel" in c or "marketplace" in c or "platform" in c]
            if _ch_candidates:
                spend_df["channel"] = spend_df[_ch_candidates[0]].astype(str).str.strip()
            else:
                spend_df["channel"] = "All"
        else:
            spend_df["channel"] = spend_df["channel"].astype(str).str.strip()

        # Replace blank/nan channel values
        spend_df["channel"] = spend_df["channel"].replace(
            {"": "All", "nan": "All", "None": "All", "NaN": "All"}
        )

        spend_df = spend_df.dropna(subset=["date"])
    else:
        spend_df = pd.DataFrame({
            "date":    pd.to_datetime([]),
            "channel": [],
            "spend":   [],
        })

    return sales_df, spend_df, None


# ---------------- LOAD STATE ----------------
with st.spinner("⚡ Loading business intelligence..."):
    result = load_and_process_data()

    if result[2]:
        st.error(f"❌ **Data Load Failed:** {result[2]}")
        st.stop()

    sales_df, spend_df = result[0], result[1]

    # FIX 4: Use format="mixed" here too — dates are already datetime objects
    # from the cache, but if Streamlit ever serialises/deserialises them as
    # strings this guarantees they are re-parsed correctly.
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce", format="mixed")
    spend_df["date"] = pd.to_datetime(spend_df["date"], errors="coerce", format="mixed")

    if sales_df is None or sales_df.empty:
        st.warning("⚠️ No sales data available.")
        st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🎛️ Control Panel")

min_date = sales_df["date"].min().date()
max_date = sales_df["date"].max().date()

# Show available data range
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

# Multi-Selects
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Strategy & Recommendations
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">🧠 AI Strategic Insights</div>', unsafe_allow_html=True)

    ch_rev_rec = df_s.groupby("channel")["revenue"].sum().reset_index()
    ch_sp_rec  = df_sp.groupby("channel")["spend"].sum().reset_index() if not df_sp.empty else pd.DataFrame(columns=["channel","spend"])
    ch_matrix_rec = pd.merge(ch_rev_rec, ch_sp_rec, on="channel", how="outer").fillna(0)
    ch_matrix_rec["roas"] = ch_matrix_rec.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)

    recommendations = generate_insights(ch_matrix_rec, curr)

    col1, col2 = st.columns([2, 1])
    with col1:
        if not recommendations:
            st.info("✅ Business looks stable. No critical alerts found.")
        else:
            css_map = {"scale":("rec-high","📈"), "warn":("rec-warn","⚠️"), "crit":("rec-crit","🚨"), "info":("rec-info","💡")}
            for rec in recommendations:
                style_class, icon = css_map.get(rec['type'], ("rec-info","ℹ️"))
                st.markdown(f"""
                <div class="rec-card {style_class}">
                    <div class="rec-title">{icon} {rec['title']}
                        <span style="margin-left:auto;font-size:12px;opacity:0.8;background:rgba(255,255,255,0.1);padding:2px 8px;border-radius:10px;">{rec['metric']}</span>
                    </div>
                    <div class="rec-body">{rec['msg']}</div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("**🎯 Projected Outcome**")
        st.caption("If you optimise based on these insights:")
        potential_savings = ch_matrix_rec[ch_matrix_rec['roas'] < 1.5]['spend'].sum() * 0.5
        potential_gain    = ch_matrix_rec[ch_matrix_rec['roas'] >= 3.0]['revenue'].sum() * 0.2
        new_net = curr['Net'] + potential_savings + (potential_gain * 0.2)
        st.metric("Potential Wasted Ad Spend",    f"${potential_savings:,.0f}")
        st.metric("Revenue Growth Opportunity",   f"${potential_gain:,.0f}")
        st.markdown("---")
        st.markdown("**Projected Net Profit:**")
        st.markdown(f"<h2 style='color:#10b981'>${new_net:,.0f}</h2>", unsafe_allow_html=True)
        st.caption(f"Vs Current: ${curr['Net']:,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Performance Trends
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Revenue, Orders & Efficiency Timeline**")
        daily_rev   = df_s.groupby(pd.Grouper(key="date", freq="D")).agg({"revenue":"sum","orders":"sum"}).reset_index()
        daily_spend = df_sp.groupby(pd.Grouper(key="date", freq="D"))["spend"].sum().reset_index() if not df_sp.empty else pd.DataFrame(columns=["date","spend"])
        daily_trend = pd.merge(daily_rev, daily_spend, on="date", how="outer").fillna(0)
        daily_trend["roas"] = daily_trend.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)

        fig_multi = go.Figure()
        fig_multi.add_trace(go.Bar(x=daily_trend["date"], y=daily_trend["revenue"], name="Revenue", marker_color="#3b82f6", opacity=0.7, yaxis="y"))
        fig_multi.add_trace(go.Scatter(x=daily_trend["date"], y=daily_trend["orders"], name="Orders", line=dict(color="#ec4899", width=2), yaxis="y2"))
        fig_multi.add_trace(go.Scatter(x=daily_trend["date"], y=daily_trend["roas"],   name="ROAS",   line=dict(color="#10b981", width=3, dash='dot'), yaxis="y3"))
        fig_multi.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
            yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False),
            yaxis3=dict(title="ROAS", overlaying="y", side="right", position=0.95, showgrid=False),
            legend=dict(orientation="h", y=1.15, x=0),
            margin=dict(l=0, r=80, t=60, b=0), height=420
        )
        st.plotly_chart(fig_multi, config={'displayModeBar': False})

    with col2:
        st.markdown("**AOV Trend Analysis**")
        weekly_aov = df_s.groupby(pd.Grouper(key="date", freq="W")).agg({"revenue":"sum","orders":"sum"}).reset_index()
        weekly_aov["aov"] = (weekly_aov["revenue"] / weekly_aov["orders"].replace(0, np.nan)).fillna(0)
        fig_aov = go.Figure()
        fig_aov.add_trace(go.Scatter(x=weekly_aov["date"], y=weekly_aov["aov"], fill='tozeroy', line=dict(color="#8b5cf6", width=3), fillcolor="rgba(139,92,246,0.2)"))
        fig_aov.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis=dict(title="AOV ($)", showgrid=True, gridcolor="#2d303e"),
            xaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=40, b=0), height=420
        )
        st.plotly_chart(fig_aov, config={'displayModeBar': False})

    st.markdown("**Commission & Spend Comparison**")
    if "selling_commission" in df_s.columns:
        daily_comm = df_s.groupby(pd.Grouper(key="date", freq="D"))["selling_commission"].sum().reset_index()
        if not daily_comm.empty and not daily_spend.empty:
            daily_costs = pd.merge(daily_spend, daily_comm, on="date", how="outer").fillna(0)
            fig_costs = go.Figure()
            fig_costs.add_trace(go.Bar(x=daily_costs["date"], y=daily_costs["spend"],               name="Ad Spend",   marker_color="#f97316"))
            fig_costs.add_trace(go.Bar(x=daily_costs["date"], y=daily_costs["selling_commission"],   name="Commission", marker_color="#ec4899"))
            fig_costs.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                barmode='stack',
                yaxis=dict(title="Cost ($)", showgrid=True, gridcolor="#2d303e"),
                legend=dict(orientation="h", y=1.1, x=0),
                margin=dict(l=0, r=0, t=40, b=0), height=350
            )
            st.plotly_chart(fig_costs, config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Marketplace Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">🛒 Marketplace Performance Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**Marketplace Performance Matrix**")
        ch_rev = df_s.groupby("channel").agg({"revenue":"sum","orders":"sum"}).reset_index()
        if "selling_commission" in df_s.columns:
            ch_comm = df_s.groupby("channel")["selling_commission"].sum().reset_index()
            ch_rev  = pd.merge(ch_rev, ch_comm, on="channel", how="left").fillna(0)
        ch_sp = df_sp.groupby("channel")["spend"].sum().reset_index() if not df_sp.empty else pd.DataFrame(columns=["channel","spend"])
        ch_matrix = pd.merge(ch_rev, ch_sp, on="channel", how="outer").fillna(0)
        ch_matrix["roas"] = ch_matrix.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)
        ch_matrix["aov"]  = (ch_matrix["revenue"] / ch_matrix["orders"].replace(0, np.nan)).fillna(0)
        ch_matrix["acos"] = ch_matrix.apply(lambda x: (x["spend"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
        ch_matrix = ch_matrix[ch_matrix["revenue"] > 0]

        fig_bubble = px.scatter(
            ch_matrix, x="spend", y="revenue", size="roas", color="aov",
            hover_name="channel",
            hover_data={"orders":":.0f","roas":":.2f","aov":":$.2f","acos":":.1f"},
            labels={"spend":"Ad Spend ($)","revenue":"Revenue ($)","aov":"AOV ($)"},
            size_max=80, text="channel", color_continuous_scale="viridis"
        )
        fig_bubble.update_traces(textposition='top center', textfont_size=10)
        fig_bubble.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.05)",
            height=500, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_bubble, config={'displayModeBar': False})

    with col2:
        st.markdown("**Marketplace Revenue Share**")
        fig_pie = px.pie(ch_matrix, values="revenue", names="channel", hole=0.5,
                         color_discrete_sequence=px.colors.sequential.Viridis)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_pie, config={'displayModeBar': False})

        st.markdown("**Marketplace Efficiency Ranking**")
        ch_rank = ch_matrix.sort_values("roas", ascending=False)[["channel","roas","revenue","spend"]].head(10).copy()
        ch_rank["acos"] = ch_rank.apply(lambda x: (x["spend"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
        fig_rank = go.Figure()
        fig_rank.add_trace(go.Bar(y=ch_rank["channel"], x=ch_rank["roas"], orientation='h',
                                  marker=dict(color=ch_rank["roas"], colorscale='Viridis', showscale=False),
                                  text=ch_rank["roas"].apply(lambda x: f"{x:.2f}x"), textposition='outside'))
        fig_rank.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="ROAS", showgrid=True, gridcolor="#2d303e"),
            yaxis=dict(title=""),
            margin=dict(l=0, r=0, t=20, b=0), height=250
        )
        st.plotly_chart(fig_rank, config={'displayModeBar': False})

        st.markdown("**⚡ Quick Actions**")
        for _, row in ch_matrix.sort_values('roas', ascending=False).head(3).iterrows():
            if row['roas'] >= 3.0:
                st.success(f"**{row['channel']}**: Scale budget +20%")
            elif row['roas'] < 1.5:
                st.error(f"**{row['channel']}**: Reduce spend -30%")
            else:
                st.info(f"**{row['channel']}**: Optimise campaigns")

    st.markdown("---")
    st.markdown("**📋 Detailed Marketplace Metrics**")
    display_ch = ch_matrix.copy().sort_values('revenue', ascending=False)
    display_ch['profit_margin'] = display_ch.apply(
        lambda x: ((x['revenue']*SAFE_MARGIN - x['spend'] - x.get('selling_commission',0))/x['revenue']*100) if x['revenue']>0 else 0, axis=1
    )
    st.dataframe(
        display_ch[['channel','revenue','orders','aov','spend','roas','acos','profit_margin']],
        column_config={
            "channel":       "Marketplace",
            "revenue":       st.column_config.NumberColumn("Revenue",      format="$%d"),
            "orders":        st.column_config.NumberColumn("Orders",       format="%d"),
            "aov":           st.column_config.NumberColumn("AOV",          format="$%.2f"),
            "spend":         st.column_config.NumberColumn("Ad Spend",     format="$%d"),
            "roas":          st.column_config.NumberColumn("ROAS",         format="%.2fx"),
            "acos":          st.column_config.NumberColumn("ACOS",         format="%.1f%%"),
            "profit_margin": st.column_config.NumberColumn("Profit %",     format="%.1f%%"),
        },
        hide_index=True, height=350
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SKU Analysis  (Ads data from Supabase)
# ══════════════════════════════════════════════════════════════════════════════
import requests as _requests

def _get_supabase_creds():
    def _get(key):
        val = os.environ.get(key, "").strip()
        if val: return val
        try:
            import streamlit as _st
            if hasattr(_st, "secrets"):
                try: return "".join(str(_st.secrets[key]).split())
                except KeyError: pass
        except Exception: pass
        return ""
    url = _get("SUPABASE_URL")
    key = _get("SUPABASE_SERVICE_KEY")
    if not key:
        key = _get("SUPABASE_KEY_1") + _get("SUPABASE_KEY_2")
    return url, key

def _load_sku_ads_raw(start: str, end: str, _url: str = "", _key: str = "") -> pd.DataFrame:
    if not _url or not _key:
        return pd.DataFrame({"_error": ["Supabase credentials not set."]})
    try:
        headers = {"apikey": _key, "Authorization": f"Bearer {_key}", "Accept": "application/json", "Prefer": "count=none"}
        all_data, page_size, offset = [], 1000, 0
        while True:
            params = f"select=*&date=gte.{start}&date=lte.{end}&limit={page_size}&offset={offset}"
            r = _requests.get(f"{_url}/rest/v1/sku_ads_cache?{params}", headers=headers, timeout=30)
            if r.status_code != 200:
                return pd.DataFrame({"_error": [f"Supabase error {r.status_code}: {r.text[:300]}"]})
            batch = r.json()
            if not batch: break
            all_data.extend(batch)
            if len(batch) < page_size: break
            offset += page_size
        if not all_data: return pd.DataFrame()
        df = pd.DataFrame(all_data)
        df = df.rename(columns={"date":"Date","market":"Market","parent_sku":"Parent_SKU","sku":"SKU","asin":"ASIN",
                                  "impressions":"Impressions","clicks":"Clicks","spend":"Spend","ad_sales":"Ad_Sales","ad_orders":"Ad_Orders"})
        df["Date"] = pd.to_datetime(df["Date"])
        if "SKU" in df.columns:
            df = df[df["SKU"].notna() & (df["SKU"].astype(str).str.strip() != "")]
        needed = ["Date","Market","Parent_SKU","SKU","ASIN","Impressions","Clicks","Spend","Ad_Sales","Ad_Orders"]
        return df[[c for c in needed if c in df.columns]]
    except Exception as exc:
        return pd.DataFrame({"_error": [str(exc)]})

def _get_supabase_date_range(_url: str = "", _key: str = "") -> tuple:
    if not _url or not _key: return ("","")
    try:
        headers = {"apikey": _key, "Authorization": f"Bearer {_key}"}
        r_min = _requests.get(f"{_url}/rest/v1/sku_ads_cache?select=date&order=date.asc&limit=1",  headers=headers, timeout=10)
        r_max = _requests.get(f"{_url}/rest/v1/sku_ads_cache?select=date&order=date.desc&limit=1", headers=headers, timeout=10)
        min_d = r_min.json()[0].get("date","") if r_min.status_code==200 and r_min.json() else ""
        max_d = r_max.json()[0].get("date","") if r_max.status_code==200 and r_max.json() else ""
        return (min_d, max_d)
    except Exception: return ("","")

def _aggregate_ads(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["Market","Parent_SKU"], as_index=False).agg(
        Impressions=("Impressions","sum"), Clicks=("Clicks","sum"),
        Spend=("Spend","sum"), Ad_Sales=("Ad_Sales","sum"), Ad_Orders=("Ad_Orders","sum")
    )
    agg["CTR"]  = (agg["Clicks"]  / agg["Impressions"].replace(0, float("nan"))) * 100
    agg["CPC"]  = (agg["Spend"]   / agg["Clicks"].replace(0, float("nan")))
    agg["ACOS"] = (agg["Spend"]   / agg["Ad_Sales"].replace(0, float("nan"))) * 100
    agg[["CTR","CPC","ACOS"]] = agg[["CTR","CPC","ACOS"]].fillna(0)
    return agg.sort_values("Spend", ascending=False).reset_index(drop=True)

with tabs[3]:
    st.markdown('<div class="section-header">🏷️ SKU Performance Analysis</div>', unsafe_allow_html=True)

    _SB_URL, _SB_KEY = _get_supabase_creds()

    with st.expander("📡 Amazon Ads Data  —  Spend · Impressions · Clicks per SKU", expanded=True):
        _ads_raw = _load_sku_ads_raw(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), _url=_SB_URL, _key=_SB_KEY)

        if not _ads_raw.empty and "_error" in _ads_raw.columns:
            st.error(f"❌ {_ads_raw['_error'].iloc[0]}")
        elif _ads_raw.empty:
            sb_min, sb_max = _get_supabase_date_range(_url=_SB_URL, _key=_SB_KEY)
            if sb_min and sb_max:
                st.warning(f"No ads data for **{start_date}** → **{end_date}**. Supabase has data from **{sb_min}** → **{sb_max}**.")
            else:
                st.info("ℹ️ No ads data in Supabase yet.")
        else:
            ads_filtered = _ads_raw.copy()
            sb_min, sb_max = _get_supabase_date_range(_url=_SB_URL, _key=_SB_KEY)
            st.caption(f"📅 Supabase covers **{sb_min}** → **{sb_max}** · Showing: **{start_date.strftime('%d %b %Y')}** → **{end_date.strftime('%d %b %Y')}** · refreshed daily at 9 AM IST")

            if ads_filtered.empty:
                st.warning(f"No ads data for selected range.")
            else:
                mkt_filter = st.radio("Market", ["All","US","CA"], horizontal=True, key="ads_mkt_filter")
                if mkt_filter != "All":
                    ads_filtered = ads_filtered[ads_filtered["Market"] == mkt_filter]

                if "type" in df_s.columns and "Parent" in df_s.columns and selected_types:
                    type_skus = set(df_s[df_s["type"].isin(selected_types)]["Parent"].dropna().unique())
                    ads_filtered = ads_filtered[ads_filtered["Parent_SKU"].isin(type_skus)]

                ads_df_raw = _aggregate_ads(ads_filtered)
                total_imp      = ads_df_raw["Impressions"].sum()
                total_clk      = ads_df_raw["Clicks"].sum()
                total_spend    = ads_df_raw["Spend"].sum()
                total_ad_sales = ads_df_raw["Ad_Sales"].sum()
                blended_acos   = (total_spend / total_ad_sales * 100) if total_ad_sales > 0 else 0

                ak1,ak2,ak3,ak4,ak5 = st.columns(5)
                ak1.metric("👁️ Impressions",  f"{total_imp:,.0f}")
                ak2.metric("🖱️ Clicks",       f"{total_clk:,.0f}")
                ak3.metric("💸 Ad Spend",     f"${total_spend:,.2f}")
                ak4.metric("📈 Ad Sales",     f"${total_ad_sales:,.2f}")
                ak5.metric("🎯 Blended ACOS", f"{blended_acos:.1f}%")
                st.markdown("---")

                top_spend = ads_df_raw.nlargest(15, "Spend")
                fig_ads_bar = px.bar(
                    top_spend, x="Spend", y="Parent_SKU", orientation="h",
                    color="ACOS", color_continuous_scale="RdYlGn_r", range_color=[0,60],
                    custom_data=["Impressions","Clicks","ACOS","CTR","CPC","Ad_Sales"],
                    labels={"Spend":"Ad Spend ($)","Parent_SKU":"Parent SKU","ACOS":"ACOS %"},
                    title="Top 15 SKUs by Ad Spend (colour = ACOS%)",
                )
                fig_ads_bar.update_traces(hovertemplate=(
                    "<b>%{y}</b><br>Spend: $%{x:,.2f}<br>Impressions: %{customdata[0]:,.0f}<br>"
                    "Clicks: %{customdata[1]:,.0f}<br>CTR: %{customdata[3]:.2f}%<br>"
                    "CPC: $%{customdata[4]:.2f}<br>Ad Sales: $%{customdata[5]:,.2f}<br>ACOS: %{customdata[2]:.1f}%<extra></extra>"
                ))
                fig_ads_bar.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=430, margin=dict(l=0,r=0,t=40,b=0),
                    yaxis=dict(autorange="reversed"), coloraxis_colorbar=dict(title="ACOS %")
                )
                st.plotly_chart(fig_ads_bar, config={"displayModeBar":False}, use_container_width=True)

                fig_scatter = px.scatter(
                    ads_df_raw[ads_df_raw["Impressions"]>0], x="Impressions", y="Clicks",
                    size="Spend", color="ACOS", color_continuous_scale="RdYlGn_r", range_color=[0,60],
                    hover_name="Parent_SKU", custom_data=["Spend","ACOS","CTR"],
                    title="Impressions vs Clicks (bubble = Spend, colour = ACOS%)",
                )
                fig_scatter.update_traces(hovertemplate=(
                    "<b>%{hovertext}</b><br>Impressions: %{x:,.0f}<br>Clicks: %{y:,.0f}<br>"
                    "Spend: $%{customdata[0]:,.2f}<br>ACOS: %{customdata[1]:.1f}%<br>CTR: %{customdata[2]:.2f}%<extra></extra>"
                ))
                fig_scatter.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                    height=400, margin=dict(l=0,r=0,t=40,b=0)
                )
                st.plotly_chart(fig_scatter, config={"displayModeBar":False}, use_container_width=True)

                sheet_orders = pd.DataFrame()
                if "Parent" in df_s.columns and "orders" in df_s.columns:
                    sheet_orders = (df_s.groupby("Parent", as_index=False)["orders"]
                                    .sum().rename(columns={"Parent":"Parent_SKU","orders":"Total_Orders"}))

                disp_ads = ads_df_raw.sort_values("Spend", ascending=False).copy()
                if not sheet_orders.empty:
                    disp_ads = disp_ads.merge(sheet_orders, on="Parent_SKU", how="left")
                    disp_ads["Total_Orders"] = disp_ads["Total_Orders"].fillna(0).astype(int)
                else:
                    disp_ads["Total_Orders"] = 0

                st.markdown("**📋 Full SKU Ads Data**")
                st.dataframe(
                    disp_ads,
                    column_config={
                        "Market":       st.column_config.TextColumn("Market",        width="small"),
                        "Parent_SKU":   st.column_config.TextColumn("Parent SKU",    width="medium"),
                        "Impressions":  st.column_config.NumberColumn("Impressions",  format="%d"),
                        "Clicks":       st.column_config.NumberColumn("Clicks",       format="%d"),
                        "Spend":        st.column_config.NumberColumn("Spend ($)",    format="$%.2f"),
                        "Ad_Sales":     st.column_config.NumberColumn("Ad Sales ($)", format="$%.2f"),
                        "Ad_Orders":    st.column_config.NumberColumn("Ad Orders",    format="%d"),
                        "Total_Orders": st.column_config.NumberColumn("Total Orders", format="%d"),
                        "CTR":          st.column_config.NumberColumn("CTR %",        format="%.2f%%"),
                        "CPC":          st.column_config.NumberColumn("CPC ($)",      format="$%.2f"),
                        "ACOS":         st.column_config.NumberColumn("ACOS %",       format="%.1f%%"),
                    },
                    hide_index=True, use_container_width=True, height=430
                )

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button("📥 Download Parent-SKU Summary (CSV)", disp_ads.to_csv(index=False).encode("utf-8"),
                                       f"sku_ads_summary_{start_date}_{end_date}.csv","text/csv",key="dl_sku_ads_summary",use_container_width=True)
                with dl_col2:
                    sku_level_cols = [c for c in ["Market","Parent_SKU","SKU","ASIN","Impressions","Clicks","Spend","Ad_Sales","Ad_Orders"] if c in ads_filtered.columns]
                    sku_export = (ads_filtered[sku_level_cols]
                                  .groupby([c for c in sku_level_cols if c not in ("Impressions","Clicks","Spend","Ad_Sales","Ad_Orders")], as_index=False)
                                  .agg(Impressions=("Impressions","sum"),Clicks=("Clicks","sum"),Spend=("Spend","sum"),Ad_Sales=("Ad_Sales","sum"),Ad_Orders=("Ad_Orders","sum")))
                    if not sheet_orders.empty:
                        sku_export = sku_export.merge(sheet_orders, on="Parent_SKU", how="left")
                        sku_export["Total_Orders"] = sku_export["Total_Orders"].fillna(0).astype(int)
                    else:
                        sku_export["Total_Orders"] = 0
                    sku_export["CTR"]  = (sku_export["Clicks"]  / sku_export["Impressions"].replace(0, float("nan"))) * 100
                    sku_export["CPC"]  = (sku_export["Spend"]   / sku_export["Clicks"].replace(0, float("nan")))
                    sku_export["ACOS"] = (sku_export["Spend"]   / sku_export["Ad_Sales"].replace(0, float("nan"))) * 100
                    sku_export[["CTR","CPC","ACOS"]] = sku_export[["CTR","CPC","ACOS"]].fillna(0)
                    sku_export = sku_export.sort_values(["Market","Spend"], ascending=[True,False]).reset_index(drop=True)
                    st.download_button("📥 Download Child-SKU Level Data (CSV)", sku_export.to_csv(index=False).encode("utf-8"),
                                       f"sku_ads_child_level_{start_date}_{end_date}.csv","text/csv",key="dl_sku_ads_child",use_container_width=True)

    st.markdown("---")

    if not _ads_raw.empty and "_error" not in _ads_raw.columns:
        _ads_summary_tab = _aggregate_ads(_ads_raw)
    else:
        _ads_summary_tab = pd.DataFrame()

    def _get_ads_for_sku(parent_sku: str):
        if _ads_summary_tab.empty: return None
        row = _ads_summary_tab[_ads_summary_tab["Parent_SKU"] == parent_sku]
        if row.empty: return None
        r = row.iloc[0]
        return {"Impressions":int(r["Impressions"]),"Clicks":int(r["Clicks"]),"Spend":float(r["Spend"]),
                "Ad_Sales":float(r["Ad_Sales"]),"CTR":float(r["CTR"]),"CPC":float(r["CPC"]),"ACOS":float(r["ACOS"])}

    if "Parent" in df_s.columns and df_s["Parent"].nunique() > 1:
        tn_col1, tn_col2 = st.columns([3,1])
        with tn_col1: st.markdown("**🏷️ Parent SKU Performance**")
        with tn_col2:
            top_n_sku = st.selectbox("Show top", options=[10,20,50,100], index=0, key="sku_top_n",
                                     label_visibility="collapsed", format_func=lambda x: f"Top {x} SKUs")

        Parent_perf_all = df_s.groupby("Parent").agg({"revenue":"sum","orders":"sum"}).reset_index()
        Parent_perf_all["aov"] = (Parent_perf_all["revenue"] / Parent_perf_all["orders"].replace(0, np.nan)).fillna(0)
        Parent_perf_all = Parent_perf_all.sort_values("revenue", ascending=False)
        Parent_perf = Parent_perf_all.head(top_n_sku)

        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"**Top {top_n_sku} Parent SKUs by Revenue**")
            fig_sku_bar = px.bar(Parent_perf, x="revenue", y="Parent", orientation='h',
                                 color="orders", color_continuous_scale="Blues",
                                 labels={"revenue":"Revenue ($)","Parent":"Parent SKU","orders":"Orders"})
            fig_sku_bar.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=max(450, top_n_sku*28), margin=dict(l=0,r=0,t=20,b=0),
                yaxis=dict(tickmode='linear')
            )
            st.plotly_chart(fig_sku_bar, config={'displayModeBar': False})

        with col2:
            st.markdown("**SKU Revenue Distribution**")
            fig_sku_tree = px.treemap(Parent_perf, path=['Parent'], values='revenue', color='aov',
                                      color_continuous_scale='Viridis', labels={"revenue":"Revenue","aov":"AOV"})
            fig_sku_tree.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                       margin=dict(l=0,r=0,t=20,b=0), height=450)
            st.plotly_chart(fig_sku_tree, config={'displayModeBar': False})

        st.markdown("---")
        st.markdown('<div class="section-header">🔍 SKU Deep-Dive Search</div>', unsafe_allow_html=True)
        all_parent_skus_list = sorted(df_s["Parent"].dropna().unique().tolist())

        search_col1, search_col2 = st.columns([3,1])
        with search_col1:
            sku_search_query = st.text_input("Search SKU", placeholder="Type a Parent SKU name...", key="sku_search_input", label_visibility="collapsed")
        with search_col2:
            sku_search_exact = st.selectbox("Match", options=["Contains","Exact"], key="sku_search_mode", label_visibility="collapsed")

        if sku_search_query.strip():
            q = sku_search_query.strip()
            matching_skus = ([s for s in all_parent_skus_list if s.lower()==q.lower()]
                             if sku_search_exact=="Exact"
                             else [s for s in all_parent_skus_list if q.lower() in s.lower()])
        else:
            matching_skus = []

        if sku_search_query.strip() and not matching_skus:
            st.warning(f"No SKUs found matching **'{sku_search_query}'**.")
        elif matching_skus:
            selected_sku = (st.selectbox("Select SKU", options=matching_skus, key="sku_search_select", label_visibility="collapsed")
                            if len(matching_skus)>1 else matching_skus[0])
            if len(matching_skus)>1:
                st.caption(f"Found **{len(matching_skus)}** matching SKUs:")

            sku_df       = df_s[df_s["Parent"]==selected_sku].copy()
            total_rev    = sku_df["revenue"].sum()
            total_orders = sku_df["orders"].sum()
            aov          = (total_rev / total_orders) if total_orders>0 else 0
            active_days  = sku_df["date"].nunique()
            first_sale   = sku_df["date"].min().strftime("%b %d, %Y")
            last_sale    = sku_df["date"].max().strftime("%b %d, %Y")

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(59,130,246,0.15),rgba(139,92,246,0.1));
                        border:1px solid rgba(59,130,246,0.35);border-radius:12px;padding:16px 20px;margin:12px 0;'>
                <div style='font-size:20px;font-weight:800;color:#f3f4f6;'>🏷️ {selected_sku}</div>
                <div style='font-size:12px;color:#9ca3af;margin-top:4px;'>
                    First sale: {first_sale} &nbsp;·&nbsp; Last sale: {last_sale} &nbsp;·&nbsp; {active_days} active days
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("💰 Revenue", f"${total_rev:,.0f}")
            m2.metric("🛒 Orders",  f"{total_orders:,.0f}")
            m3.metric("📊 AOV",     f"${aov:,.2f}")
            revenue_share = (total_rev / df_s["revenue"].sum() * 100) if df_s["revenue"].sum()>0 else 0
            m4.metric("📈 Rev Share", f"{revenue_share:.1f}%")

            ads_info = _get_ads_for_sku(selected_sku)
            if ads_info:
                st.markdown("")
                st.markdown("**📡 Amazon Ads Performance (fetched period)**")
                a1,a2,a3,a4,a5 = st.columns(5)
                a1.metric("👁️ Impressions", f"{ads_info['Impressions']:,}")
                a2.metric("🖱️ Clicks",      f"{ads_info['Clicks']:,}")
                a3.metric("💸 Spend",       f"${ads_info['Spend']:,.2f}")
                a4.metric("🎯 ACOS",        f"{ads_info['ACOS']:.1f}%")
                a5.metric("📊 CTR",         f"{ads_info['CTR']:.2f}%")

            chart_col, info_col = st.columns([3,2])
            with chart_col:
                st.markdown("**📅 Daily Revenue Trend**")
                daily_sku = sku_df.groupby(pd.Grouper(key="date", freq="D"))["revenue"].sum().reset_index().sort_values("date")
                daily_sku["rolling7"] = daily_sku["revenue"].rolling(7, min_periods=1).mean()
                fig_sku_trend = go.Figure()
                fig_sku_trend.add_trace(go.Bar(x=daily_sku["date"], y=daily_sku["revenue"], name="Daily Revenue", marker_color="rgba(59,130,246,0.5)"))
                fig_sku_trend.add_trace(go.Scatter(x=daily_sku["date"], y=daily_sku["rolling7"], name="7-day Avg", line=dict(color="#10b981",width=2), mode="lines"))
                fig_sku_trend.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=280, margin=dict(l=0,r=0,t=10,b=0),
                    legend=dict(orientation="h",y=1.15),
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True,gridcolor="#2d303e",title="Revenue ($)")
                )
                st.plotly_chart(fig_sku_trend, config={"displayModeBar":False})

            with info_col:
                st.markdown("**🛒 Revenue by Marketplace**")
                mp_breakdown = (sku_df.groupby("channel").agg(revenue=("revenue","sum"),orders=("orders","sum"))
                                .reset_index().sort_values("revenue",ascending=False))
                mp_breakdown["share"] = (mp_breakdown["revenue"]/mp_breakdown["revenue"].sum()*100).round(1)
                if len(mp_breakdown)>0:
                    fig_mp_pie = px.pie(mp_breakdown, values="revenue", names="channel", hole=0.55,
                                        color_discrete_sequence=["#3b82f6","#10b981","#f59e0b","#8b5cf6","#ec4899","#06b6d4"])
                    fig_mp_pie.update_traces(textposition="outside", textinfo="percent+label")
                    fig_mp_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=280, margin=dict(l=0,r=0,t=10,b=0))
                    st.plotly_chart(fig_mp_pie, config={"displayModeBar":False})

            st.dataframe(mp_breakdown, column_config={
                "channel":st.column_config.TextColumn("Marketplace"),
                "revenue":st.column_config.NumberColumn("Revenue",format="$%d"),
                "orders": st.column_config.NumberColumn("Orders", format="%d"),
                "share":  st.column_config.NumberColumn("Share %",format="%.1f%%"),
            }, hide_index=True, use_container_width=True)

            if "SKU" in df_s.columns:
                child_skus = (sku_df.groupby("SKU").agg(revenue=("revenue","sum"),orders=("orders","sum"))
                              .reset_index().sort_values("revenue",ascending=False))
                child_skus["aov"]   = (child_skus["revenue"]/child_skus["orders"].replace(0,np.nan)).fillna(0)
                child_skus["share"] = (child_skus["revenue"]/child_skus["revenue"].sum()*100).round(1)
                valid_children = child_skus[child_skus["SKU"]!="Unknown"]
                if len(valid_children)>0:
                    st.markdown(f"**📦 Child SKUs ({len(valid_children)} variants)**")
                    st.dataframe(valid_children, column_config={
                        "SKU":    st.column_config.TextColumn("Child SKU",width="large"),
                        "revenue":st.column_config.NumberColumn("Revenue",format="$%d"),
                        "orders": st.column_config.NumberColumn("Orders", format="%d"),
                        "aov":    st.column_config.NumberColumn("AOV",    format="$%.2f"),
                        "share":  st.column_config.NumberColumn("Share %",format="%.1f%%"),
                    }, hide_index=True, use_container_width=True, height=min(400,60+len(valid_children)*38))
        else:
            st.caption("🔎 Type a SKU name above to search.")

        st.markdown("---")
        st.markdown(f"**📦 Top {top_n_sku} SKU Breakdown (Click to expand for Child SKUs)**")
        for idx, parent_row in Parent_perf.iterrows():
            parent = parent_row['Parent']
            if "SKU" in df_s.columns:
                child_data = df_s[df_s["Parent"]==parent].groupby("SKU").agg({"revenue":"sum","orders":"sum"}).reset_index()
                child_data["aov"] = (child_data["revenue"]/child_data["orders"].replace(0,np.nan)).fillna(0)
                child_data = child_data.sort_values("revenue",ascending=False)
                has_children = len(child_data)>0 and child_data["SKU"].iloc[0]!="Unknown"
            else:
                has_children, child_data = False, pd.DataFrame()

            with st.expander(f"🏷️ {parent} - ${parent_row['revenue']:,.0f} Revenue", expanded=False):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Revenue", f"${parent_row['revenue']:,.0f}")
                c2.metric("Orders",  f"{parent_row['orders']:,.0f}")
                c3.metric("AOV",     f"${parent_row['aov']:,.2f}")
                c4.metric("Child SKUs", f"{len(child_data)}" if has_children else "N/A")

                ads_info_card = _get_ads_for_sku(parent)
                if ads_info_card:
                    st.markdown("**📡 Amazon Ads** (fetched period)")
                    ac1,ac2,ac3,ac4,ac5 = st.columns(5)
                    ac1.metric("👁️ Impressions", f"{ads_info_card['Impressions']:,}")
                    ac2.metric("🖱️ Clicks",      f"{ads_info_card['Clicks']:,}")
                    ac3.metric("💸 Spend",       f"${ads_info_card['Spend']:,.2f}")
                    ac4.metric("🎯 ACOS",        f"{ads_info_card['ACOS']:.1f}%")
                    ac5.metric("📊 CTR",         f"{ads_info_card['CTR']:.2f}%")

                if has_children:
                    st.markdown("---")
                    st.markdown("**Child SKUs Performance:**")
                    child_display = child_data.copy()
                    child_display["revenue"] = child_display["revenue"].apply(lambda x: f"${x:,.0f}")
                    child_display["orders"]  = child_display["orders"].apply(lambda x: f"{x:,.0f}")
                    child_display["aov"]     = child_display["aov"].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(child_display, column_config={
                        "SKU":    st.column_config.TextColumn("SKU",    width="medium"),
                        "revenue":st.column_config.TextColumn("Revenue",width="small"),
                        "orders": st.column_config.TextColumn("Orders", width="small"),
                        "aov":    st.column_config.TextColumn("AOV",    width="small"),
                    }, hide_index=True, height=min(300,50+len(child_display)*35))
                    if len(child_data)>1:
                        fig_child = px.pie(child_data, values="revenue", names="SKU",
                                           title="Revenue Distribution by Child SKU",
                                           color_discrete_sequence=px.colors.sequential.Plasma)
                        fig_child.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=0,r=0,t=40,b=0))
                        st.plotly_chart(fig_child, config={'displayModeBar': False})
                else:
                    st.info("ℹ️ No child SKU data available.")
    else:
        st.info("📦 SKU data not available. Ensure 'Parent' column exists in your data.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Profitability Deep Dive
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    col1, col2 = st.columns([1,2])
    cost_goods = curr['Revenue'] * (1 - SAFE_MARGIN)

    with col1:
        st.markdown("**Profit Waterfall**")
        fig_water = go.Figure(go.Waterfall(
            name="Profitability", orientation="v",
            measure=["relative","relative","relative","relative","total"],
            x=["Gross Revenue","COGS","Commission","Ad Spend","Net Profit"],
            textposition="outside",
            text=[f"${curr['Revenue']/1000:.1f}k", f"-${cost_goods/1000:.1f}k",
                  f"-${curr['Commission']/1000:.1f}k", f"-${curr['Spend']/1000:.1f}k", f"${curr['Net']/1000:.1f}k"],
            y=[curr['Revenue'], -cost_goods, -curr['Commission'], -curr['Spend'], curr['Net']],
            connector={"line":{"color":"#6366f1"}},
            decreasing={"marker":{"color":"#f87171"}},
            increasing={"marker":{"color":"#10b981"}},
            totals={"marker":{"color":"#3b82f6"}}
        ))
        fig_water.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=False), margin=dict(l=0,r=0,t=40,b=0), height=450
        )
        st.plotly_chart(fig_water, config={'displayModeBar': False})

    with col2:
        st.markdown("**Cost Breakdown Analysis**")
        costs_data = pd.DataFrame({
            "Category": ["COGS","Ad Spend","Commission","Net Profit"],
            "Amount":   [cost_goods, curr['Spend'], curr['Commission'], curr['Net']],
            "Percentage": [
                (cost_goods/curr['Revenue']*100) if curr['Revenue']>0 else 0,
                (curr['Spend']/curr['Revenue']*100) if curr['Revenue']>0 else 0,
                (curr['Commission']/curr['Revenue']*100) if curr['Revenue']>0 else 0,
                (curr['Net']/curr['Revenue']*100) if curr['Revenue']>0 else 0,
            ]
        })
        fig_costs_pie = px.pie(costs_data, values="Amount", names="Category", hole=0.6,
                               color_discrete_sequence=['#f87171','#f97316','#ec4899','#10b981'])
        fig_costs_pie.update_traces(textposition='outside', textinfo='percent+label')
        fig_costs_pie.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
            height=450, margin=dict(l=0,r=0,t=40,b=0),
            annotations=[dict(text=f'${curr["Revenue"]/1000:.0f}k<br>Total', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig_costs_pie, config={'displayModeBar': False})

    st.markdown("**Profitability Metrics Summary**")
    profit_metrics = pd.DataFrame({
        "Metric": ["Gross Revenue","COGS","Gross Margin","Ad Spend","Commission","Total Costs","Net Profit","Profit Margin"],
        "Amount": [
            f"${curr['Revenue']:,.0f}", f"${cost_goods:,.0f}",
            f"${curr['Revenue']-cost_goods:,.0f}", f"${curr['Spend']:,.0f}",
            f"${curr['Commission']:,.0f}", f"${cost_goods+curr['Spend']+curr['Commission']:,.0f}",
            f"${curr['Net']:,.0f}",
            f"{(curr['Net']/curr['Revenue']*100) if curr['Revenue']>0 else 0:.2f}%"
        ]
    })
    st.dataframe(profit_metrics, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: Forecasting & Predictions
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">🔮 Advanced Ensemble ML Forecasting</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        forecast_period = st.selectbox("📅 Forecast Period",
                                       ["Next 7 Days","Next 30 Days","Next Quarter (90 Days)"], index=1)
    with fc2:
        forecast_type = st.selectbox("📊 Forecast Type",
                                     ["Revenue & Orders","SKU Performance","Marketplace Performance"], index=0)
    with fc3:
        use_yoy = st.checkbox("Year-over-Year Seasonal Boost", value=True)

    forecast_days = {"Next 7 Days":7,"Next 30 Days":30,"Next Quarter (90 Days)":90}[forecast_period]

    try:
        from forecast_engine import ensemble_forecast, forecast_all_skus
    except Exception as _fe_err:
        st.error(f"⚠️ Forecasting engine failed to load: {_fe_err}")
        st.stop()

    st.markdown("---")

    if forecast_type == "Revenue & Orders":
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f"**📈 Ensemble Revenue Forecast — {forecast_period}**")
            daily_revenue = df_s.groupby(pd.Grouper(key="date", freq="D")).agg({"revenue":"sum","orders":"sum"}).reset_index().sort_values("date")
            if len(daily_revenue) >= 14:
                future_dates = pd.date_range(daily_revenue["date"].max() + timedelta(days=1), periods=forecast_days)
                yoy_revenue = None
                if use_yoy:
                    yoy_start_ts = pd.to_datetime(daily_revenue["date"].max() - pd.DateOffset(years=1) - timedelta(days=forecast_days))
                    yoy_end_ts   = pd.to_datetime(daily_revenue["date"].max() - pd.DateOffset(years=1)) + pd.Timedelta(days=1, microseconds=-1)
                    yoy_raw = sales_df[(sales_df["date"]>=yoy_start_ts) & (sales_df["date"]<=yoy_end_ts)]
                    if len(yoy_raw)>0:
                        yoy_revenue = yoy_raw.groupby(pd.Grouper(key="date", freq="D"))["revenue"].sum().values

                with st.spinner("🤖 Training ensemble (5 models)…"):
                    rev_pred, rev_std, confidence, weighted_r2, model_info = ensemble_forecast(
                        daily_revenue["date"], daily_revenue["revenue"].values, future_dates, yoy_revenue)
                    ord_pred, _, _, _, _ = ensemble_forecast(
                        daily_revenue["date"], daily_revenue["orders"].values, future_dates)

                forecast_df = pd.DataFrame({
                    "date": future_dates, "predicted_revenue": rev_pred, "predicted_orders": ord_pred,
                    "upper": rev_pred+rev_std, "lower": np.maximum(rev_pred-rev_std, 0)
                })

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=daily_revenue["date"], y=daily_revenue["revenue"],
                                            name="Historical", line=dict(color="#3b82f6",width=2), fill="tozeroy", fillcolor="rgba(59,130,246,0.1)"))
                fig_fc.add_trace(go.Scatter(
                    x=list(forecast_df["date"])+list(forecast_df["date"])[::-1],
                    y=list(forecast_df["upper"])+list(forecast_df["lower"])[::-1],
                    fill="toself", fillcolor="rgba(16,185,129,0.15)", line=dict(width=0), name="Confidence Band"))
                fig_fc.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["predicted_revenue"],
                                            name="Ensemble Forecast", line=dict(color="#10b981",width=3)))
                fig_fc.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                    yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
                    xaxis=dict(showgrid=False),
                    legend=dict(orientation="h", y=1.18),
                    margin=dict(l=0,r=0,t=60,b=0), height=440
                )
                st.plotly_chart(fig_fc, config={"displayModeBar":False})

                total_rev = forecast_df["predicted_revenue"].sum()
                total_ord = forecast_df["predicted_orders"].sum()
                hist_avg  = daily_revenue["revenue"].tail(forecast_days).mean()
                growth    = ((forecast_df["predicted_revenue"].mean()-hist_avg)/hist_avg*100) if hist_avg>0 else 0

                mc1,mc2,mc3,mc4 = st.columns(4)
                mc1.metric(f"Predicted Revenue ({forecast_days}d)", f"${total_rev:,.0f}")
                mc2.metric(f"Predicted Orders ({forecast_days}d)",  f"{total_ord:,.0f}")
                mc3.metric("Growth vs Historical",                  f"{growth:+.1f}%")
                mc4.metric("YoY Data", "Enabled" if yoy_revenue is not None else "Not available")
            else:
                st.warning(f"⚠️ Need at least 14 days of data. Currently have {len(daily_revenue)} days.")

        with col2:
            st.markdown("**🎯 Model Performance**")
            if len(daily_revenue) >= 14:
                conf_color = "#10b981" if confidence>=75 else ("#f59e0b" if confidence>=55 else "#ef4444")
                st.markdown(
                    f"<div style='text-align:center;padding:16px;background:rgba(0,0,0,0.3);border-radius:10px;border:2px solid {conf_color}'>"
                    f"<p style='margin:0;color:#9ca3af;font-size:12px;'>ENSEMBLE CONFIDENCE</p>"
                    f"<p style='margin:4px 0;font-size:42px;font-weight:900;color:{conf_color}'>{confidence:.0f}%</p>"
                    f"<p style='margin:0;color:#9ca3af;font-size:11px;'>Weighted CV R² = {weighted_r2*100:.1f}%</p>"
                    f"</div>", unsafe_allow_html=True)
                st.progress(confidence/100)
                st.markdown("---")
                st.markdown("**📊 Model Breakdown**")
                for mname, minfo in model_info.items():
                    if mname=="_weights": continue
                    weight = model_info["_weights"].get(mname,0)
                    r2_val = minfo["r2"]
                    color  = "#10b981" if r2_val>=70 else ("#f59e0b" if r2_val>=40 else "#ef4444")
                    st.markdown(
                        f"<div style='margin:4px 0'><span style='font-size:11px;color:#9ca3af'>{mname}</span>"
                        f"<div style='background:#1e2030;border-radius:4px;height:6px;margin:2px 0'>"
                        f"<div style='background:{color};width:{int(r2_val)}%;height:6px;border-radius:4px'></div></div>"
                        f"<span style='font-size:10px;color:{color}'>R²={r2_val}% · weight={weight}%</span></div>",
                        unsafe_allow_html=True)

    elif forecast_type == "SKU Performance":
        st.markdown(f"**🏷️ Top SKU Performance Forecast - {forecast_period}**")
        if "Parent" in df_s.columns:
            all_parent_skus = df_s["Parent"].dropna().unique().tolist()
            with st.spinner(f"⚡ Forecasting {len(all_parent_skus)} SKUs…"):
                all_sku_forecasts = forecast_all_skus(
                    df_s["revenue"].values, df_s["date"].values, df_s["Parent"].values,
                    tuple(sorted(all_parent_skus)), forecast_days, use_yoy)
            if not all_sku_forecasts:
                st.warning("⚠️ No SKUs had enough data (7+ days) to forecast.")
            else:
                df_sku_rank = pd.DataFrame(all_sku_forecasts)
                df_sku_rank["_score"] = (
                    df_sku_rank["Growth %"].clip(-200,200)*0.4 +
                    df_sku_rank["Momentum %"].clip(-200,200)*0.4 +
                    df_sku_rank["Confidence %"]*0.2
                )
                df_sku_rank = df_sku_rank.sort_values("_score", ascending=False).head(20).reset_index(drop=True)
                df_sku_rank["Rank"] = df_sku_rank.index + 1
                display_cols = ["Rank","SKU","Historical Avg","Recent 2wk Avg","Forecast Avg",
                                f"Total Forecast ({forecast_days}d)","Growth %","Momentum %","YoY Change %","Confidence %"]
                st.dataframe(df_sku_rank[[c for c in display_cols if c in df_sku_rank.columns]],
                             hide_index=True, use_container_width=True, height=550)
        else:
            st.info("📦 SKU data not available.")

    elif forecast_type == "Marketplace Performance":
        st.markdown(f"**🛒 Ensemble Marketplace Forecast — {forecast_period}**")
        marketplaces = df_s["channel"].unique().tolist()
        marketplace_forecasts = []
        with st.spinner(f"🤖 Running ensemble on {len(marketplaces)} marketplaces…"):
            for marketplace in marketplaces:
                mp_data = (df_s[df_s["channel"]==marketplace]
                           .groupby(pd.Grouper(key="date",freq="D"))["revenue"].sum().reset_index().sort_values("date"))
                if len(mp_data)<7: continue
                mp_future_dates = pd.date_range(mp_data["date"].max()+timedelta(days=1), periods=forecast_days)
                mp_pred, mp_std, mp_conf, mp_r2, _ = ensemble_forecast(
                    mp_data["date"], mp_data["revenue"].values, mp_future_dates)
                hist_avg = mp_data["revenue"].mean()
                fore_avg = mp_pred.mean()
                growth   = ((fore_avg-hist_avg)/hist_avg*100) if hist_avg>0 else 0
                marketplace_forecasts.append({
                    "Marketplace": marketplace, "Historical Revenue": mp_data["revenue"].sum(),
                    "Forecast Revenue": mp_pred.sum(), "Growth %": growth, "Confidence %": mp_conf,
                    "_pred": mp_pred, "_dates": mp_future_dates, "_hist": mp_data
                })
        if marketplace_forecasts:
            df_mp = pd.DataFrame(marketplace_forecasts).sort_values("Forecast Revenue", ascending=False)
            st.dataframe(df_mp[["Marketplace","Historical Revenue","Forecast Revenue","Growth %","Confidence %"]],
                         column_config={
                             "Historical Revenue": st.column_config.NumberColumn("Historical",format="$%.0f"),
                             "Forecast Revenue":   st.column_config.NumberColumn(f"Forecast ({forecast_days}d)",format="$%.0f"),
                             "Growth %":           st.column_config.NumberColumn("Growth %",format="%.1f%%"),
                             "Confidence %":       st.column_config.NumberColumn("Confidence %",format="%.0f%%"),
                         }, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: A/B Test Tracker
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">🧪 Advanced A/B Test Performance Tracker</div>', unsafe_allow_html=True)
    if 'ab_tests' not in st.session_state:
        st.session_state.ab_tests = []

    test_mode = st.radio("**Test Mode:**",
                         ["Single Marketplace Comparison","Multi-Marketplace Comparison","Time Period Comparison"],
                         horizontal=True)
    st.markdown("---")
    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("**➕ Create New A/B Test**")

        if test_mode == "Single Marketplace Comparison":
            with st.form("ab_test_single_form"):
                test_name = st.text_input("Test Name", placeholder="e.g., Amazon - Old vs New Campaign")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Variant A (Control)**")
                    variant_a_name    = st.text_input("Name", value="Control", key="single_a_name")
                    variant_a_channel = st.selectbox("Marketplace", df_s["channel"].unique(), key="single_a_ch")
                    var_a_date_start  = st.date_input("Start Date", value=start_date, key="single_a_start")
                    var_a_date_end    = st.date_input("End Date",   value=end_date,   key="single_a_end")
                with col_b:
                    st.markdown("**Variant B (Test)**")
                    variant_b_name    = st.text_input("Name", value="Test", key="single_b_name")
                    variant_b_channel = st.selectbox("Marketplace", df_s["channel"].unique(), key="single_b_ch")
                    var_b_date_start  = st.date_input("Start Date", value=start_date, key="single_b_start")
                    var_b_date_end    = st.date_input("End Date",   value=end_date,   key="single_b_end")

                submitted = st.form_submit_button("🚀 Create Single Marketplace Test", type="primary")
                if submitted and test_name:
                    def _ab_metrics(channel, ds, de):
                        s_ts = pd.to_datetime(ds); e_ts = pd.to_datetime(de) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        d = sales_df[(sales_df["date"]>=s_ts)&(sales_df["date"]<=e_ts)&(sales_df["channel"]==channel)]
                        sp = spend_df[(spend_df["date"]>=s_ts)&(spend_df["date"]<=e_ts)&(spend_df["channel"]==channel)]["spend"].sum() if not spend_df.empty else 0
                        rev = d["revenue"].sum(); ord_ = d["orders"].sum()
                        return {"name":channel,"revenue":rev,"orders":ord_,"spend":sp,
                                "roas":(rev/sp) if sp>0 else 0,"aov":(rev/ord_) if ord_>0 else 0}
                    test_data = {
                        "test_name":test_name,"test_type":"Single Marketplace",
                        "created_at":datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "variant_a":{**_ab_metrics(variant_a_channel,var_a_date_start,var_a_date_end),"name":variant_a_name},
                        "variant_b":{**_ab_metrics(variant_b_channel,var_b_date_start,var_b_date_end),"name":variant_b_name},
                    }
                    st.session_state.ab_tests.append(test_data)
                    st.success(f"✅ Test '{test_name}' created!")
                    st.rerun()

        elif test_mode == "Multi-Marketplace Comparison":
            with st.form("ab_test_multi_form"):
                test_name = st.text_input("Test Name", placeholder="e.g., Amazon vs Walmart vs eBay")
                marketplaces_to_compare = st.multiselect("Choose 2-5 marketplaces", df_s["channel"].unique().tolist(), max_selections=5)
                col_date1, col_date2 = st.columns(2)
                with col_date1: multi_date_start = st.date_input("Start Date", value=start_date, key="multi_start")
                with col_date2: multi_date_end   = st.date_input("End Date",   value=end_date,   key="multi_end")
                submitted_multi = st.form_submit_button("🚀 Create Multi-Marketplace Test", type="primary")
                if submitted_multi and test_name and len(marketplaces_to_compare)>=2:
                    s_ts = pd.to_datetime(multi_date_start); e_ts = pd.to_datetime(multi_date_end)+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)
                    results = []
                    for mp in marketplaces_to_compare:
                        d  = sales_df[(sales_df["date"]>=s_ts)&(sales_df["date"]<=e_ts)&(sales_df["channel"]==mp)]
                        sp = spend_df[(spend_df["date"]>=s_ts)&(spend_df["date"]<=e_ts)&(spend_df["channel"]==mp)]["spend"].sum() if not spend_df.empty else 0
                        rev=d["revenue"].sum(); ord_=d["orders"].sum()
                        results.append({"marketplace":mp,"revenue":rev,"orders":ord_,"spend":sp,
                                        "roas":(rev/sp) if sp>0 else 0,"aov":(rev/ord_) if ord_>0 else 0,
                                        "acos":(sp/rev*100) if rev>0 else 0})
                    st.session_state.ab_tests.append({
                        "test_name":test_name,"test_type":"Multi-Marketplace",
                        "created_at":datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "marketplaces":results,"period":f"{multi_date_start} to {multi_date_end}"
                    })
                    st.success(f"✅ Multi-marketplace test created!")
                    st.rerun()

        elif test_mode == "Time Period Comparison":
            with st.form("ab_test_time_form"):
                test_name = st.text_input("Test Name", placeholder="e.g., Q4 2024 vs Q4 2023")
                marketplace_time = st.selectbox("Select Marketplace", ["All Marketplaces"]+df_s["channel"].unique().tolist())
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown("**Period A**")
                    period_a_start = st.date_input("Start", value=start_date-timedelta(days=90), key="time_a_start")
                    period_a_end   = st.date_input("End",   value=start_date-timedelta(days=1),  key="time_a_end")
                with col_p2:
                    st.markdown("**Period B**")
                    period_b_start = st.date_input("Start", value=start_date, key="time_b_start")
                    period_b_end   = st.date_input("End",   value=end_date,   key="time_b_end")
                submitted_time = st.form_submit_button("🚀 Create Time Period Test", type="primary")
                if submitted_time and test_name:
                    def _period_metrics(ps, pe, mp):
                        s_ts=pd.to_datetime(ps); e_ts=pd.to_datetime(pe)+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)
                        mask_s = (sales_df["date"]>=s_ts)&(sales_df["date"]<=e_ts)
                        mask_sp = (spend_df["date"]>=s_ts)&(spend_df["date"]<=e_ts) if not spend_df.empty else pd.Series(dtype=bool)
                        if mp != "All Marketplaces":
                            mask_s  &= (sales_df["channel"]==mp)
                            if not spend_df.empty: mask_sp &= (spend_df["channel"]==mp)
                        d=sales_df[mask_s]; sp=spend_df[mask_sp]["spend"].sum() if not spend_df.empty else 0
                        rev=d["revenue"].sum(); ord_=d["orders"].sum()
                        return {"name":f"{ps} to {pe}","revenue":rev,"orders":ord_,"spend":sp,
                                "roas":(rev/sp) if sp>0 else 0,"aov":(rev/ord_) if ord_>0 else 0}
                    st.session_state.ab_tests.append({
                        "test_name":test_name,"test_type":"Time Period",
                        "created_at":datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "marketplace":marketplace_time,
                        "period_a":_period_metrics(period_a_start,period_a_end,marketplace_time),
                        "period_b":_period_metrics(period_b_start,period_b_end,marketplace_time),
                    })
                    st.success(f"✅ Time period test created!")
                    st.rerun()

    with col2:
        st.markdown("**💡 Testing Guide**")
        guides = {
            "Single Marketplace Comparison": "Compare campaigns on same marketplace. Use same time periods for accuracy. Run 7+ days minimum.",
            "Multi-Marketplace Comparison":  "Compare 2-5 marketplaces. Use same time period. Look at ROAS and AOV together.",
            "Time Period Comparison":        "Compare YoY or seasonal periods. Account for seasonality. Look at growth %.",
        }
        st.info(guides.get(test_mode,""))

    if st.session_state.ab_tests:
        st.markdown("---")
        st.markdown("**🔬 Test Results**")
        for idx, test in enumerate(st.session_state.ab_tests):
            if test['test_type'] == 'Single Marketplace':
                with st.expander(f"🧪 {test['test_name']} ({test['test_type']}) - {test['created_at']}", expanded=True):
                    va, vb = test['variant_a'], test['variant_b']
                    rev_imp  = ((vb['revenue']-va['revenue'])/va['revenue']*100) if va['revenue']>0 else 0
                    roas_imp = ((vb['roas']-va['roas'])/va['roas']*100) if va['roas']>0 else 0
                    st.markdown(f"**{va['name']} ({va.get('channel',va['name'])})** vs **{vb['name']} ({vb.get('channel',vb['name'])})**")
                    comp = pd.DataFrame({
                        "Metric":["Revenue","Orders","Ad Spend","ROAS","AOV"],
                        va['name']:[f"${va['revenue']:,.0f}",f"{va['orders']:,.0f}",f"${va['spend']:,.0f}",f"{va['roas']:.2f}x",f"${va['aov']:.2f}"],
                        vb['name']:[f"${vb['revenue']:,.0f}",f"{vb['orders']:,.0f}",f"${vb['spend']:,.0f}",f"{vb['roas']:.2f}x",f"${vb['aov']:.2f}"],
                        "Change":[f"{rev_imp:+.1f}%",f"{((vb['orders']-va['orders'])/va['orders']*100) if va['orders']>0 else 0:+.1f}%",
                                  f"{((vb['spend']-va['spend'])/va['spend']*100) if va['spend']>0 else 0:+.1f}%",
                                  f"{roas_imp:+.1f}%",f"{((vb['aov']-va['aov'])/va['aov']*100) if va['aov']>0 else 0:+.1f}%"]
                    })
                    st.dataframe(comp, hide_index=True, use_container_width=True)
                    if rev_imp>10 and roas_imp>5: st.success(f"✅ **{vb['name']} wins!** +{rev_imp:.1f}% revenue, +{roas_imp:.1f}% ROAS")
                    elif rev_imp<-10 or roas_imp<-5: st.error(f"❌ **{va['name']} performs better.** Stick with control.")
                    else: st.info("➡️ **Results inconclusive.** Run longer or with larger sample.")
                    if st.button("🗑️ Delete", key=f"del_single_{idx}"):
                        st.session_state.ab_tests.pop(idx); st.rerun()

            elif test['test_type'] == 'Multi-Marketplace':
                with st.expander(f"🛒 {test['test_name']} (Multi-Marketplace) - {test['created_at']}", expanded=True):
                    mp_df = pd.DataFrame(test['marketplaces']).sort_values('revenue', ascending=False)
                    st.dataframe(mp_df, column_config={
                        "marketplace":st.column_config.TextColumn("Marketplace"),
                        "revenue":st.column_config.NumberColumn("Revenue",format="$%d"),
                        "orders": st.column_config.NumberColumn("Orders", format="%d"),
                        "spend":  st.column_config.NumberColumn("Ad Spend",format="$%d"),
                        "roas":   st.column_config.NumberColumn("ROAS",   format="%.2fx"),
                        "aov":    st.column_config.NumberColumn("AOV",    format="$%.2f"),
                        "acos":   st.column_config.NumberColumn("ACOS",   format="%.1f%%"),
                    }, hide_index=True, use_container_width=True)
                    st.success(f"🏆 **Top:** {mp_df.iloc[0]['marketplace']} — ${mp_df.iloc[0]['revenue']:,.0f}")
                    if st.button("🗑️ Delete", key=f"del_multi_{idx}"):
                        st.session_state.ab_tests.pop(idx); st.rerun()

            elif test['test_type'] == 'Time Period':
                with st.expander(f"📅 {test['test_name']} (Time Period) - {test['created_at']}", expanded=True):
                    pa, pb = test['period_a'], test['period_b']
                    rev_ch = ((pb['revenue']-pa['revenue'])/pa['revenue']*100) if pa['revenue']>0 else 0
                    comp = pd.DataFrame({
                        "Metric":["Revenue","Orders","Ad Spend","ROAS","AOV"],
                        pa['name']:[f"${pa['revenue']:,.0f}",f"{pa['orders']:,.0f}",f"${pa['spend']:,.0f}",f"{pa['roas']:.2f}x",f"${pa['aov']:.2f}"],
                        pb['name']:[f"${pb['revenue']:,.0f}",f"{pb['orders']:,.0f}",f"${pb['spend']:,.0f}",f"{pb['roas']:.2f}x",f"${pb['aov']:.2f}"],
                        "Change":[f"{rev_ch:+.1f}%",f"{((pb['orders']-pa['orders'])/pa['orders']*100) if pa['orders']>0 else 0:+.1f}%",
                                  f"{((pb['spend']-pa['spend'])/pa['spend']*100) if pa['spend']>0 else 0:+.1f}%",
                                  f"{((pb['roas']-pa['roas'])/pa['roas']*100) if pa['roas']>0 else 0:+.1f}%",
                                  f"{((pb['aov']-pa['aov'])/pa['aov']*100) if pa['aov']>0 else 0:+.1f}%"]
                    })
                    st.dataframe(comp, hide_index=True, use_container_width=True)
                    if rev_ch>20: st.success(f"🎉 Excellent growth of {rev_ch:.1f}%!")
                    elif rev_ch>0: st.info(f"📈 Positive growth of {rev_ch:.1f}%.")
                    elif rev_ch>-10: st.warning(f"⚠️ Slight decline of {abs(rev_ch):.1f}%.")
                    else: st.error(f"🚨 Significant decline of {abs(rev_ch):.1f}%.")
                    if st.button("🗑️ Delete", key=f"del_time_{idx}"):
                        st.session_state.ab_tests.pop(idx); st.rerun()
    else:
        st.info("📝 No A/B tests yet. Use the forms above to create your first test!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8: Weekly Reports
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">📅 Weekly Performance Reports</div>', unsafe_allow_html=True)

    cfg_col1, cfg_col2 = st.columns([2,1])
    with cfg_col1:
        st.markdown("**📋 Report Generator**")
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            report_start = st.date_input("From Date", value=max_date-timedelta(days=7),
                                         min_value=min_date, max_value=date.today(), key="report_start")
        with col_date2:
            report_end   = st.date_input("To Date", value=max_date,
                                         min_value=min_date, max_value=date.today(), key="report_end")

        all_marketplaces = sorted(sales_df["channel"].dropna().unique().tolist())
        report_selected_mp = st.multiselect("🛒 Filter by Marketplace",
                                             options=["All Marketplaces"]+all_marketplaces,
                                             default=["All Marketplaces"], key="report_marketplace_filter")
        report_channels = all_marketplaces if (not report_selected_mp or "All Marketplaces" in report_selected_mp) else report_selected_mp
        if report_channels != all_marketplaces:
            st.info(f"📌 Report scoped to: **{', '.join(report_channels)}**")

        st.markdown("**📑 Include Sections:**")
        c1,c2,c3 = st.columns(3)
        with c1: include_kpis=st.checkbox("KPI Summary",value=True); include_trends=st.checkbox("Trend Chart",value=True)
        with c2: include_marketplaces=st.checkbox("Marketplace Breakdown",value=True); include_skus=st.checkbox("Top SKUs",value=True)
        with c3: include_recommendations=st.checkbox("Recommendations",value=True); include_yoy=st.checkbox("YoY Comparison",value=True)

        if st.button("📊 Generate Report", type="primary", key="generate_report"):
            rep_start_ts = pd.to_datetime(report_start)
            rep_end_ts   = pd.to_datetime(report_end)+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)
            mask_s  = (sales_df["date"]>=rep_start_ts)&(sales_df["date"]<=rep_end_ts)&(sales_df["channel"].isin(report_channels))
            mask_sp = (spend_df["date"]>=rep_start_ts)&(spend_df["date"]<=rep_end_ts)&(spend_df["channel"].isin(report_channels)) if not spend_df.empty else pd.Series(dtype=bool)
            report_df_s  = sales_df[mask_s]
            report_df_sp = spend_df[mask_sp] if not spend_df.empty else pd.DataFrame(columns=["date","channel","spend"])
            report_metrics = calc_metrics(report_df_s, report_df_sp)

            yoy_start = report_start-timedelta(days=365); yoy_end = report_end-timedelta(days=365)
            yoy_s_ts  = pd.to_datetime(yoy_start); yoy_e_ts = pd.to_datetime(yoy_end)+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)
            mask_yoy_s  = (sales_df["date"]>=yoy_s_ts)&(sales_df["date"]<=yoy_e_ts)&(sales_df["channel"].isin(report_channels))
            mask_yoy_sp = (spend_df["date"]>=yoy_s_ts)&(spend_df["date"]<=yoy_e_ts)&(spend_df["channel"].isin(report_channels)) if not spend_df.empty else pd.Series(dtype=bool)
            yoy_df_s  = sales_df[mask_yoy_s]
            yoy_df_sp = spend_df[mask_yoy_sp] if not spend_df.empty else pd.DataFrame(columns=["date","channel","spend"])
            yoy_metrics = calc_metrics(yoy_df_s, yoy_df_sp) if len(yoy_df_s)>0 else None

            mp_label = ", ".join(report_channels) if report_channels!=all_marketplaces else "All Marketplaces"
            st.session_state.current_report = {
                'period':    f"{report_start.strftime('%b %d, %Y')} – {report_end.strftime('%b %d, %Y')}",
                'yoy_period':f"{yoy_start.strftime('%b %d, %Y')} – {yoy_end.strftime('%b %d, %Y')}",
                'metrics':   report_metrics, 'yoy_metrics': yoy_metrics,
                'sales_data':report_df_s,'spend_data':report_df_sp,
                'yoy_sales': yoy_df_s,'yoy_spend':yoy_df_sp,
                'report_start':report_start,'report_end':report_end,
                'marketplace_label':mp_label,
                'sections':{'kpis':include_kpis,'trends':include_trends,'marketplaces':include_marketplaces,
                            'skus':include_skus,'recommendations':include_recommendations,'yoy':include_yoy}
            }
            st.success("✅ Report generated!")
            st.rerun()

    with cfg_col2:
        st.markdown("**💡 Tips**")
        st.markdown("""
        **Periods:** Last 7 days → Weekly · Last 30 → Monthly · Last 90 → Quarter

        **YoY:** Uses exact same calendar period one year ago.

        **Export:** Download as Markdown or PDF.
        """)

    if 'current_report' in st.session_state:
        report = st.session_state.current_report
        m = report['metrics']; ym = report['yoy_metrics']
        has_yoy = ym is not None and report['sections']['yoy']

        def _delta(curr_v, prev_v, fmt="$"):
            if prev_v is None or prev_v==0: return None
            pct = (curr_v-prev_v)/abs(prev_v)*100
            if fmt=="$": return f"{pct:+.1f}% vs last year (was ${prev_v:,.0f})"
            elif fmt=="x": return f"{pct:+.1f}% vs last year (was {prev_v:.2f}x)"
            else: return f"{pct:+.1f}% vs last year (was {prev_v:.1f}{fmt})"

        st.markdown("---")
        mp_scope = report.get('marketplace_label','All Marketplaces')
        st.markdown(
            f"<h2 style='margin:0'>📊 Performance Report</h2>"
            f"<p style='color:#9ca3af;margin:4px 0 16px 0;'>📅 <strong>This period:</strong> {report['period']} &nbsp;|&nbsp; "
            f"📅 <strong>Last year:</strong> {report['yoy_period']} &nbsp;|&nbsp; 🛒 {mp_scope}</p>",
            unsafe_allow_html=True)

        if report['sections']['kpis']:
            st.markdown("### 💎 Key Performance Indicators")
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("💰 Revenue",    f"${m['Revenue']:,.0f}",    delta=_delta(m['Revenue'],   ym['Revenue']    if has_yoy else None))
            k2.metric("🛒 Orders",     f"{m['Orders']:,.0f}",      delta=_delta(m['Orders'],    ym['Orders']     if has_yoy else None,""))
            k3.metric("🎯 ROAS",       f"{m['ROAS']:.2f}x",        delta=_delta(m['ROAS'],      ym['ROAS']       if has_yoy else None,"x"))
            k4.metric("💹 Net Profit", f"${m['Net']:,.0f}",        delta=_delta(m['Net'],       ym['Net']        if has_yoy else None))
            k5,k6,k7,k8 = st.columns(4)
            k5.metric("📢 Ad Spend",   f"${m['Spend']:,.0f}",      delta=_delta(m['Spend'],     ym['Spend']      if has_yoy else None), delta_color="inverse")
            k6.metric("🏪 Commission", f"${m['Commission']:,.0f}", delta=_delta(m['Commission'],ym['Commission'] if has_yoy else None), delta_color="inverse")
            k7.metric("📊 ACOS",       f"{m['ACOS']:.1f}%",        delta=_delta(m['ACOS'],      ym['ACOS']       if has_yoy else None,"%"), delta_color="inverse")
            k8.metric("🧾 AOV",        f"${m['AOV']:.2f}",         delta=_delta(m['AOV'],       ym['AOV']        if has_yoy else None))

        ch_report = pd.DataFrame()
        if report['sections']['marketplaces']:
            st.markdown("### 🛒 Marketplace Performance")
            ch_now  = report['sales_data'].groupby("channel").agg({"revenue":"sum","orders":"sum"}).reset_index()
            ch_sp_n = report['spend_data'].groupby("channel")["spend"].sum().reset_index() if not report['spend_data'].empty else pd.DataFrame(columns=["channel","spend"])
            ch_report = pd.merge(ch_now, ch_sp_n, on="channel", how="outer").fillna(0)
            ch_report["roas"] = ch_report.apply(lambda r: r["revenue"]/r["spend"] if r["spend"]>0 else 0, axis=1)
            ch_report["acos"] = ch_report.apply(lambda r: r["spend"]/r["revenue"]*100 if r["revenue"]>0 else 0, axis=1)
            ch_report = ch_report.sort_values("revenue", ascending=False)
            st.dataframe(ch_report[["channel","revenue","orders","spend","roas","acos"]], column_config={
                "channel":st.column_config.TextColumn("Marketplace"),
                "revenue":st.column_config.NumberColumn("Revenue",format="$%d"),
                "orders": st.column_config.NumberColumn("Orders", format="%d"),
                "spend":  st.column_config.NumberColumn("Ad Spend",format="$%d"),
                "roas":   st.column_config.NumberColumn("ROAS",   format="%.2fx"),
                "acos":   st.column_config.NumberColumn("ACOS",   format="%.1f%%"),
            }, hide_index=True, use_container_width=True)

        if report['sections']['skus'] and "Parent" in report['sales_data'].columns:
            st.markdown("### 🏷️ Top SKU Performance")
            sku_now = report['sales_data'].groupby("Parent").agg({"revenue":"sum","orders":"sum"}).reset_index()
            sku_now["aov"] = (sku_now["revenue"]/sku_now["orders"].replace(0,pd.NA)).fillna(0)
            sku_now = sku_now.sort_values("revenue",ascending=False).head(10)
            st.dataframe(sku_now[["Parent","revenue","orders","aov"]], column_config={
                "Parent": "SKU",
                "revenue":st.column_config.NumberColumn("Revenue",format="$%d"),
                "orders": st.column_config.NumberColumn("Orders", format="%d"),
                "aov":    st.column_config.NumberColumn("AOV",    format="$%.2f"),
            }, hide_index=True, use_container_width=True)

        if report['sections']['recommendations']:
            st.markdown("### 🚀 Strategic Recommendations")
            try:
                recs = generate_insights(ch_report if len(ch_report)>0 else pd.DataFrame(columns=["channel","revenue","spend","roas"]), m)
                for rec in recs[:5]:
                    icon_map={"scale":"📈","warn":"⚠️","crit":"🚨","info":"💡"}
                    icon=icon_map.get(rec['type'],"💡")
                    if rec['type']=="scale":   st.success(f"{icon} **{rec['title']}** — {rec['msg']}")
                    elif rec['type']=="warn":  st.warning(f"{icon} **{rec['title']}** — {rec['msg']}")
                    elif rec['type']=="crit":  st.error(f"{icon} **{rec['title']}** — {rec['msg']}")
                    else:                      st.info(f"{icon} **{rec['title']}** — {rec['msg']}")
                if not recs: st.info("✅ No critical issues. Performance is stable.")
            except Exception: st.warning("⚠️ Could not generate recommendations.")

        st.markdown("---")
        st.markdown("### 📤 Export Report")
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
---
*Generated by Marketplace Business Insights Dashboard*
"""
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button("📥 Download Markdown", markdown_report,
                               f"report_{report['report_start']}_{report['report_end']}.md","text/markdown",key="download_md")
        with ex2:
            if st.button("📋 Show Copyable Text", key="show_copy"):
                st.text_area("Select all & copy:", markdown_report, height=200)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9: Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="section-header">📋 Performance Data Explorer</div>', unsafe_allow_html=True)

    tbl = ch_matrix.copy()
    tbl["commission"]     = ch_matrix.get("selling_commission", 0)
    tbl["acos"]           = tbl.apply(lambda x: (x["spend"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
    tbl["net"]            = (tbl["revenue"]*SAFE_MARGIN) - tbl["spend"] - tbl.get("commission",0)
    tbl["profit_margin"]  = tbl.apply(lambda x: (x["net"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)

    display_cols = ["channel","revenue","orders","aov","spend","commission","roas","acos","net","profit_margin"]
    st.dataframe(
        tbl[[c for c in display_cols if c in tbl.columns]],
        column_config={
            "channel":       st.column_config.TextColumn("Marketplace",   width="medium"),
            "revenue":       st.column_config.NumberColumn("Revenue",      format="$%,.0f"),
            "orders":        st.column_config.NumberColumn("Orders",       format="%d"),
            "aov":           st.column_config.NumberColumn("AOV",          format="$%.2f"),
            "spend":         st.column_config.NumberColumn("Ad Spend",     format="$%d"),
            "commission":    st.column_config.NumberColumn("Commission",   format="$%d"),
            "roas":          st.column_config.NumberColumn("ROAS",         format="%.2fx"),
            "acos":          st.column_config.NumberColumn("ACOS",         format="%.1f%%"),
            "net":           st.column_config.NumberColumn("Net Profit",   format="$%d"),
            "profit_margin": st.column_config.NumberColumn("Profit Margin",format="%.1f%%"),
        },
        hide_index=True, height=400
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        csv = tbl.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Marketplace Report (CSV)", csv, "marketplace_performance.csv", "text/csv", key="download_channel")
    with col2:
        if not df_s.empty:
            st.download_button("📥 Download Sales Data (CSV)", df_s.to_csv(index=False).encode('utf-8'), "sales_data.csv","text/csv",key="download_sales")
    with col3:
        if not df_sp.empty:
            st.download_button("📥 Download Spend Data (CSV)", df_sp.to_csv(index=False).encode('utf-8'),"spend_data.csv","text/csv",key="download_spend")

    # ── Jewelry Type Performance ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💍 Jewelry Type Performance")
    _jt_src = df_s.copy()

    def _remap_jt(v):
        v = str(v).strip()
        if v in ("","nan","None","NaN","<NA>","Unknown"): return None
        _m = {"ring":"Rings","rings":"Rings","pendant":"Pendants","pendants":"Pendants",
              "necklace":"Pendants","necklaces":"Pendants","earring":"Earrings","earrings":"Earrings",
              "bracelet":"Bracelets","bracelets":"Bracelets","band":"Band","bands":"Band",
              "bangle":"Bangles","bangles":"Bangles","lapel pin":"Lapel Pin","misc":"MISC",
              "men's band":"Men's Band","mens band":"Men's Band"}
        return _m.get(v.lower(), v)

    if "type" in _jt_src.columns:
        _jt_src["jewelry_type"] = _jt_src["type"].apply(_remap_jt)
        _jt_src = _jt_src[_jt_src["jewelry_type"].notna()]
        jt_tbl = (_jt_src.groupby("jewelry_type", as_index=False)
                  .agg(Revenue=("revenue","sum"), Orders=("orders","sum"))
                  .sort_values("Revenue", ascending=False))
        jt_tbl["AOV"] = (jt_tbl["Revenue"]/jt_tbl["Orders"].replace(0,np.nan)).fillna(0)
        _total_rev = jt_tbl["Revenue"].sum(); _total_ord = jt_tbl["Orders"].sum()
        jt_tbl["Rev_Share"] = jt_tbl["Revenue"]/_total_rev*100 if _total_rev>0 else 0
        jt_tbl["Ord_Share"] = jt_tbl["Orders"]/_total_ord*100  if _total_ord>0 else 0
        _totals = {"jewelry_type":"🔢 TOTAL","Revenue":_total_rev,"Orders":_total_ord,
                   "AOV":_total_rev/_total_ord if _total_ord>0 else 0,"Rev_Share":100.0,"Ord_Share":100.0}
        jt_tbl = pd.concat([jt_tbl, pd.DataFrame([_totals])], ignore_index=True)
        st.dataframe(jt_tbl[["jewelry_type","Revenue","Orders","AOV","Rev_Share","Ord_Share"]], column_config={
            "jewelry_type":st.column_config.TextColumn("Jewelry Type",width="medium"),
            "Revenue":     st.column_config.NumberColumn("Revenue ($)", format="$%,.0f"),
            "Orders":      st.column_config.NumberColumn("Orders",      format="%,.0f"),
            "AOV":         st.column_config.NumberColumn("AOV ($)",     format="$%.2f"),
            "Rev_Share":   st.column_config.NumberColumn("Rev Share %", format="%.1f%%"),
            "Ord_Share":   st.column_config.NumberColumn("Ord Share %", format="%.1f%%"),
        }, hide_index=True, use_container_width=True)
        st.download_button("📥 Download Jewelry Type Report (CSV)",
                           jt_tbl.to_csv(index=False).encode("utf-8"),
                           "jewelry_type_performance.csv","text/csv",key="dl_jtype_explorer")
    else:
        st.info("No 'type' column found in sales data.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 10: Merchandising Intel
# ══════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="section-header">💎 Merchandising Intelligence</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False, ttl=3600)
    def _load_merch(_version="v4"):
        candidates = [
            "/mount/src/businessperformancedashboard/Merchandising_data.xlsx",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "Merchandising_data.xlsx"),
            "Merchandising_data.xlsx",
        ]
        found_path = next((p for p in candidates if os.path.exists(p)), None)
        if found_path is None: return None, "FILE_NOT_FOUND"
        try:
            import openpyxl  # noqa
        except ImportError:
            return None, "OPENPYXL_MISSING"
        try:
            raw = pd.read_excel(found_path, engine="openpyxl",
                                usecols=["Parent","Design Code","jewelry_type","stone"])
        except Exception as e:
            return None, f"READ_ERROR: {e}"
        raw = raw.rename(columns={"Design Code":"design_code"})
        for col in ["Parent","design_code","jewelry_type","stone"]:
            raw[col] = raw[col].astype(str).str.strip()
        _JTYPE_MAP = {"ring":"Rings","rings":"Rings","pendant":"Pendants","pendants":"Pendants",
                      "necklace":"Pendants","necklaces":"Pendants","earring":"Earrings","earrings":"Earrings",
                      "bracelet":"Bracelets","bracelets":"Bracelets","band":"Band","bands":"Band","bangle":"Bangles","bangles":"Bangles"}
        def _rj(v):
            v=str(v).strip()
            if v in ("","nan","None","NaN","<NA>"): return "Rings"
            return _JTYPE_MAP.get(v.lower(), v)
        raw["jewelry_type"] = raw["jewelry_type"].apply(_rj)
        return raw.drop_duplicates(subset="Parent").reset_index(drop=True), "OK"

    merch_lookup, merch_status = _load_merch("v4")

    if merch_status == "OPENPYXL_MISSING":
        st.error("⚠️ **openpyxl is not installed.** Add `openpyxl>=3.1.0` to requirements.txt.")
        st.stop()
    elif merch_status == "FILE_NOT_FOUND":
        st.error("⚠️ **Merchandising_data.xlsx not found.** Ensure it is committed to the repo root.")
        st.stop()
    elif merch_status != "OK" or merch_lookup is None:
        st.error(f"⚠️ Failed to load merchandising data: {merch_status}")
        st.stop()

    mask_merch_sales = (sales_df["date"]>=start_ts)&(sales_df["date"]<=end_ts)&(sales_df["channel"].isin(selected_channels))
    df_s_merch = sales_df[mask_merch_sales]

    sales_parents   = set(df_s_merch["Parent"].dropna().unique())
    merch_parents   = set(merch_lookup["Parent"].unique())
    matched_parents = sales_parents & merch_parents
    match_pct = len(matched_parents)/len(sales_parents)*100 if sales_parents else 0

    bm1,bm2,bm3,bm4 = st.columns(4)
    bm1.metric("💎 Merch Catalogue", f"{len(merch_parents):,} SKUs")
    bm2.metric("📦 Sales SKUs",      f"{len(sales_parents):,} SKUs")
    bm3.metric("✅ Matched",         f"{len(matched_parents):,} SKUs")
    bm4.metric("🔗 Match Rate",      f"{match_pct:.1f}%")

    if match_pct==0:
        st.warning("No Parent SKUs matched. Check that Parent SKU names are consistent between sheets.")

    st.markdown("---")

    merch_lookup_dedup = merch_lookup[["Parent","design_code","jewelry_type","stone"]].drop_duplicates(subset="Parent",keep="first").copy()
    df_enriched = df_s_merch.merge(merch_lookup_dedup, on="Parent", how="left", suffixes=("_sales","_merch"))

    def _coalesce(df, base):
        mc=f"{base}_merch"; sc=f"{base}_sales"
        if mc in df.columns:   df[base]=df[mc]
        elif sc in df.columns: df[base]=df[sc]
        elif base not in df.columns: df[base]=np.nan
        return base
    for _c in ["design_code","jewelry_type","stone"]: _coalesce(df_enriched, _c)
    df_enriched.drop(columns=[c for c in df_enriched.columns if c.endswith("_sales") or c.endswith("_merch")], inplace=True)

    st.markdown("### 🔧 Filters")
    fcol1, fcol2, fcol3 = st.columns([2,2,1])
    all_jtypes = sorted([v for v in merch_lookup_dedup["jewelry_type"].unique() if str(v).strip() not in ("","nan","None","NaN","<NA>")])
    all_stones = sorted(merch_lookup["stone"].dropna().unique().tolist())
    with fcol1: sel_jtype = st.multiselect("💍 Jewelry Type", options=all_jtypes, default=[], key="merch_jtype_filter", placeholder="All jewelry types…")
    with fcol2: sel_stone = st.multiselect("💠 Stone", options=all_stones, default=[], key="merch_stone_filter", placeholder="All stones…")
    with fcol3: matched_only = st.toggle("Matched SKUs only", value=True, key="merch_matched_only")

    FILTER_COLS = ["Parent","design_code","jewelry_type","stone","revenue","orders"]
    _df_merch_base = df_enriched[[c for c in FILTER_COLS if c in df_enriched.columns]].copy()

    # Apply filters
    df_m = _df_merch_base.copy()
    for _c in ["design_code","jewelry_type","stone"]:
        if _c in df_m.columns: df_m[_c] = df_m[_c].astype("string").fillna("").str.strip()

    if matched_only:
        dc = df_m.get("design_code","")
        df_m = df_m[dc.notna() & (dc.astype("string").str.strip()!="") & (dc.astype("string").str.lower()!="nan")]
    if sel_jtype:
        jt_norm = [s.lower() for s in sel_jtype]
        df_m = df_m[df_m["jewelry_type"].str.lower().isin(jt_norm)]
    if sel_stone:
        st_norm = [s.lower() for s in sel_stone]
        pat = r"(?:^|,\s*)({})(?:\s*,|$)".format("|".join(re.escape(s) for s in st_norm))
        df_m = df_m[df_m["stone"].str.lower().str.contains(pat, na=False, regex=True)]

    active = []
    if sel_jtype: active.append(f"💍 {', '.join(sel_jtype)}")
    if sel_stone: active.append(f"💠 {', '.join(sel_stone[:3])}")
    if active: st.info("📌 Active: " + "  |  ".join(active) + f"  ·  **{df_m['Parent'].nunique():,} SKUs** · **${df_m['revenue'].sum():,.0f}** revenue")

    st.markdown("---")
    st.markdown("### 📊 Category Revenue Overview")
    ov_left, ov_right = st.columns(2)

    with ov_left:
        st.markdown("**💍 Revenue by Jewelry Type**")
        jtype_agg = (df_m.groupby("jewelry_type", dropna=False)
                     .agg(revenue=("revenue","sum"),orders=("orders","sum")).reset_index()
                     .sort_values("revenue",ascending=True))
        jtype_agg["aov"] = (jtype_agg["revenue"]/jtype_agg["orders"].replace(0,np.nan)).fillna(0)
        if jtype_agg.empty: st.info("No jewelry type sales for current selection.")
        else:
            fig_jtype = px.bar(jtype_agg, x="revenue", y="jewelry_type", orientation="h",
                               color="aov", color_continuous_scale="Blues",
                               custom_data=["orders","aov"],
                               labels={"revenue":"Revenue ($)","jewelry_type":"","aov":"AOV ($)"},
                               text=jtype_agg["revenue"].apply(lambda v: f"${v/1000:.0f}k"))
            fig_jtype.update_traces(textposition="outside",
                                    hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.0f}<br>Orders: %{customdata[0]:,.0f}<br>AOV: $%{customdata[1]:.2f}<extra></extra>")
            fig_jtype.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    height=max(350,len(jtype_agg)*35+60), margin=dict(l=0,r=70,t=10,b=0),
                                    coloraxis_showscale=False)
            st.plotly_chart(fig_jtype, config={"displayModeBar":False}, use_container_width=True)

    with ov_right:
        st.markdown("**💠 Top 15 Stones by Revenue**")
        stone_agg = (df_m.groupby("stone",dropna=False)["revenue"].sum().reset_index()
                     .sort_values("revenue",ascending=False).head(15))
        if stone_agg.empty: st.info("No stone sales for current selection.")
        else:
            fig_stone = px.pie(stone_agg, values="revenue", names="stone", hole=0.48,
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_stone.update_traces(textposition="outside", textinfo="percent+label", textfont_size=10)
            fig_stone.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                    height=max(350,len(jtype_agg)*35+60), margin=dict(l=0,r=0,t=10,b=40), showlegend=False)
            st.plotly_chart(fig_stone, config={"displayModeBar":False}, use_container_width=True)

    kp1,kp2,kp3,kp4,kp5 = st.columns(5)
    kp1.metric("💰 Total Revenue",      f"${df_m['revenue'].sum():,.0f}")
    kp2.metric("🛒 Total Orders",       f"{df_m['orders'].sum():,.0f}")
    aov_all = df_m['revenue'].sum()/df_m['orders'].sum() if df_m['orders'].sum()>0 else 0
    kp3.metric("📊 Blended AOV",        f"${aov_all:,.2f}")
    kp4.metric("🏷️ Active Parent SKUs", f"{df_m['Parent'].nunique():,}")
    kp5.metric("🎨 Design Codes",       f"{df_m['design_code'].nunique():,}")

    st.markdown("---")
    st.markdown("### 🏷️ Parent SKU Performance")

    parent_sum  = df_m.groupby("Parent").agg(revenue=("revenue","sum"),orders=("orders","sum")).reset_index()
    parent_attr = df_m[["Parent","design_code","jewelry_type","stone"]].drop_duplicates(subset="Parent",keep="first")
    parent_agg  = parent_sum.merge(parent_attr, on="Parent", how="left")
    for _c in ["design_code","jewelry_type","stone"]:
        if _c in parent_agg.columns: parent_agg[_c] = parent_agg[_c].astype("string").fillna("—")
    parent_agg["aov"]           = (parent_agg["revenue"]/parent_agg["orders"].replace(0,np.nan)).fillna(0)
    parent_agg["revenue_share"] = (parent_agg["revenue"]/parent_agg["revenue"].sum()*100).round(2)
    parent_agg = parent_agg.sort_values("revenue", ascending=False).reset_index(drop=True)

    ps1, ps2 = st.columns([3,1])
    with ps1: p_search = st.text_input("🔎 Search Parent SKU or Design Code", placeholder="e.g. EJ_SE…", key="merch_parent_search", label_visibility="collapsed")
    with ps2: top_n    = st.selectbox("Show top", [25,50,100,"All"], key="merch_parent_topn", label_visibility="collapsed")

    p_disp = parent_agg.copy()
    if p_search.strip():
        p_disp = p_disp[p_disp["Parent"].str.contains(p_search.strip(),case=False,na=False)|p_disp["design_code"].str.contains(p_search.strip(),case=False,na=False)]
    if top_n != "All": p_disp = p_disp.head(int(top_n))

    st.dataframe(p_disp[["Parent","design_code","jewelry_type","stone","revenue","orders","aov","revenue_share"]], column_config={
        "Parent":        st.column_config.TextColumn("Parent SKU",  width="medium"),
        "design_code":   st.column_config.TextColumn("Design Code", width="medium"),
        "jewelry_type":  st.column_config.TextColumn("Jewelry Type",width="small"),
        "stone":         st.column_config.TextColumn("Stone",       width="medium"),
        "revenue":       st.column_config.NumberColumn("Revenue ($)",format="$%,.0f"),
        "orders":        st.column_config.NumberColumn("Orders",     format="%d"),
        "aov":           st.column_config.NumberColumn("AOV ($)",    format="$%.2f"),
        "revenue_share": st.column_config.NumberColumn("Rev Share %",format="%.2f%%"),
    }, hide_index=True, use_container_width=True, height=430)
    st.caption(f"Showing {len(p_disp):,} of {len(parent_agg):,} Parent SKUs")
    st.download_button("📥 Download Parent SKU Report (CSV)", p_disp.to_csv(index=False).encode("utf-8"),
                       "parent_sku_performance.csv","text/csv",key="dl_merch_parent")

    st.markdown("---")
    st.markdown("### 🎨 Design Code Performance")

    ddf = df_m[df_m["design_code"].notna()&(df_m["design_code"].astype(str).str.lower()!="nan")&(df_m["design_code"].astype(str).str.strip()!="")].copy()
    if not ddf.empty:
        design_sum   = ddf.groupby("design_code").agg(revenue=("revenue","sum"),orders=("orders","sum"),variants=("Parent","nunique")).reset_index()
        design_jtype = ddf[["design_code","jewelry_type"]].drop_duplicates(subset="design_code",keep="first")
        stones_s     = (ddf[["design_code","stone"]].dropna(subset=["stone"]).astype({"stone":str})
                        .groupby("design_code")["stone"]
                        .apply(lambda s:", ".join(sorted(set([x.strip() for x in s.tolist() if str(x).strip()]))))
                        .reset_index(name="stones"))
        design_agg = design_sum.merge(design_jtype,on="design_code",how="left").merge(stones_s,on="design_code",how="left")
        design_agg["jewelry_type"]  = design_agg["jewelry_type"].astype("string").fillna("—")
        design_agg["stones"]        = design_agg["stones"].astype("string").fillna("—")
        design_agg["aov"]           = (design_agg["revenue"]/design_agg["orders"].replace(0,np.nan)).fillna(0)
        design_agg["revenue_share"] = (design_agg["revenue"]/design_agg["revenue"].sum()*100).round(2)
        design_agg = design_agg.sort_values("revenue",ascending=False).reset_index(drop=True)

        top20 = design_agg.head(20).copy()
        fig_dc = px.bar(top20, x="design_code", y="revenue", color="jewelry_type",
                        custom_data=["orders","aov","variants","stones"],
                        labels={"revenue":"Revenue ($)","design_code":"Design Code","jewelry_type":"Type"},
                        text=top20["revenue"].apply(lambda v: f"${v/1000:.1f}k"))
        fig_dc.update_traces(textposition="outside",
                             hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<br>Orders: %{customdata[0]:,.0f}<br>AOV: $%{customdata[1]:.2f}<br>Variants: %{customdata[2]}<br>Stones: %{customdata[3]}<extra></extra>")
        fig_dc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             height=380, margin=dict(l=0,r=0,t=20,b=0),
                             xaxis=dict(tickangle=-40,showgrid=False), yaxis=dict(showgrid=True,gridcolor="#2d303e"),
                             legend=dict(title="Jewelry Type",orientation="h",y=1.14), bargap=0.25)
        st.plotly_chart(fig_dc, config={"displayModeBar":False}, use_container_width=True)

        ds1, ds2 = st.columns([3,1])
        with ds1: dc_search = st.text_input("🔎 Search Design Code or Stone", placeholder="e.g. FC_SB or Diamond…", key="merch_dc_search", label_visibility="collapsed")
        with ds2: dc_top_n  = st.selectbox("Show top", [25,50,100,"All"], key="merch_dc_topn", label_visibility="collapsed")
        dc_disp = design_agg.copy()
        if dc_search.strip():
            dc_disp = dc_disp[dc_disp["design_code"].str.contains(dc_search.strip(),case=False,na=False)|dc_disp["stones"].str.contains(dc_search.strip(),case=False,na=False)]
        if dc_top_n != "All": dc_disp = dc_disp.head(int(dc_top_n))
        st.dataframe(dc_disp[["design_code","jewelry_type","stones","variants","revenue","orders","aov","revenue_share"]], column_config={
            "design_code":   st.column_config.TextColumn("Design Code",  width="medium"),
            "jewelry_type":  st.column_config.TextColumn("Jewelry Type", width="small"),
            "stones":        st.column_config.TextColumn("Stones",       width="large"),
            "variants":      st.column_config.NumberColumn("# Variants", format="%d"),
            "revenue":       st.column_config.NumberColumn("Revenue ($)",format="$%,.0f"),
            "orders":        st.column_config.NumberColumn("Orders",     format="%d"),
            "aov":           st.column_config.NumberColumn("AOV ($)",    format="$%.2f"),
            "revenue_share": st.column_config.NumberColumn("Rev Share %",format="%.2f%%"),
        }, hide_index=True, use_container_width=True, height=430)
        st.download_button("📥 Download Design Code Report (CSV)", dc_disp.to_csv(index=False).encode("utf-8"),
                           "design_code_performance.csv","text/csv",key="dl_merch_design")

    st.markdown("---")
    st.markdown("### 🔥 Revenue Heatmap — Jewelry Type × Stone")
    heat_raw    = df_m.groupby(["jewelry_type","stone"],dropna=False)["revenue"].sum().reset_index()
    top15_stones = heat_raw.groupby("stone")["revenue"].sum().nlargest(15).index.tolist()
    heat_filt   = heat_raw[heat_raw["stone"].isin(top15_stones)]
    if not heat_filt.empty:
        pivot = heat_filt.pivot(index="jewelry_type", columns="stone", values="revenue").fillna(0)
        text_matrix = [[f"${v/1000:.0f}k" if v>0 else "" for v in row] for row in pivot.values]
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale="Blues", hoverongaps=False,
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Revenue: $%{z:,.0f}<extra></extra>",
            text=text_matrix, texttemplate="%{text}", textfont={"size":9}
        ))
        fig_heat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               height=max(320,len(pivot)*42+80), margin=dict(l=0,r=0,t=10,b=0),
                               xaxis=dict(tickangle=-42,side="bottom"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_heat, config={"displayModeBar":False}, use_container_width=True)

    if len(sales_parents - merch_parents) > 0:
        with st.expander(f"ℹ️ {len(sales_parents-merch_parents):,} sales SKUs with no merchandising match"):
            unmatched_df = pd.DataFrame(sorted(sales_parents-merch_parents), columns=["Parent SKU"])
            st.dataframe(unmatched_df, hide_index=True, use_container_width=True, height=250)
            st.download_button("📥 Download Unmatched SKU List", unmatched_df.to_csv(index=False).encode("utf-8"),
                               "unmatched_skus.csv","text/csv",key="dl_unmatched")

# ---------------- FOOTER ----------------
st.markdown("---")
f1, f2, f3 = st.columns(3)
with f1: st.markdown(f"<div style='text-align:left;color:#6b7280;font-size:12px;'>📅 Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
with f2: st.markdown(f"<div style='text-align:center;color:#6b7280;font-size:12px;'>⚙️ Safe Margin: {SAFE_MARGIN*100:.0f}%</div>", unsafe_allow_html=True)
with f3: st.markdown(f"<div style='text-align:right;color:#6b7280;font-size:12px;'>📊 Data Points: {len(df_s):,}</div>", unsafe_allow_html=True)
