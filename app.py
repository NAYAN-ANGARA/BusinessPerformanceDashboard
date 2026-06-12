import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets
from datetime import date, timedelta, datetime
import numpy as np
import json
import re
import os

import plotly.io as pio
pio.templates.default = "plotly_dark"

st.set_page_config(page_title="Marketplace Business Insights", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

SAFE_MARGIN = 0.62

# ---------- CSS (same as before) ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f1116 0%, #1a1d29 100%); }
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
    .metric-card:hover { transform: translateY(-6px); border-color: rgba(255,255,255,0.25); box-shadow: 0 16px 48px rgba(0,0,0,0.5); }
    .metric-card:hover::before { opacity: 1; }
    .metric-label { font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: #9ca3af; margin-bottom: 10px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
    .metric-value { font-size: 32px; font-weight: 900; color: #ffffff; margin-bottom: 8px; line-height: 1.2; background: linear-gradient(135deg, #fff 0%, #e0e0e0 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .accent-blue   { --accent-color: #3b82f6; }
    .accent-green  { --accent-color: #10b981; }
    .accent-orange { --accent-color: #f97316; }
    .accent-purple { --accent-color: #8b5cf6; }
    .accent-pink   { --accent-color: #ec4899; }
    .accent-cyan   { --accent-color: #06b6d4; }
    .accent-yellow { --accent-color: #eab308; }
    .accent-red    { --accent-color: #ef4444; }
    .delta-badge { display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 800; gap: 4px; }
    .delta-pos { background: rgba(16,185,129,0.25); color:#34d399; box-shadow:0 0 20px rgba(16,185,129,0.3); }
    .delta-neg { background: rgba(239,68,68,0.25);  color:#f87171; box-shadow:0 0 20px rgba(239,68,68,0.3); }
    .section-header { font-size: 20px; font-weight: 700; color: #f3f4f6; margin: 40px 0 20px 0; display: flex; align-items: center; gap: 12px; padding-bottom: 12px; border-bottom: 2px solid rgba(255,255,255,0.1); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: rgba(30,32,40,0.5); padding: 8px; border-radius: 12px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 12px 24px; font-weight: 600; transition: all 0.3s ease; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }
    .stButton > button { border-radius: 10px; font-weight: 600; transition: all 0.3s ease; border: 1px solid rgba(255,255,255,0.1); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

# ---------- Helper functions ----------
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
    icon_html = f'<span style="font-size:16px;">{icon}</span>' if icon else ''
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

def _norm_cols(df):
    return [str(c).strip().lower() for c in df.columns]

def robust_parse_dates(series):
    # Try default, then dayfirst=True for ambiguous dates
    parsed = pd.to_datetime(series, errors='coerce', format='mixed' if hasattr(pd, 'to_datetime') and 'mixed' in pd.to_datetime.__code__.co_names else None)
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(series, errors='coerce', dayfirst=True)

# ---------- Data loader with debug UI ----------
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
        return None, None, f"Credentials error: {e}", {}

    try:
        workbook = load_all_sheets(creds, "New BI Dashboard")
    except Exception as e:
        return None, None, str(e), {}
    finally:
        try:
            _os.unlink(creds)
        except Exception:
            pass

    if not workbook:
        return None, None, "No sheets found.", {}

    # Debug info to show in UI
    sheet_debug = {}
    sales_df = None
    spend_df = None

    for sheet_name, df in workbook.items():
        cols_norm = _norm_cols(df)
        sheet_debug[sheet_name] = cols_norm[:10]  # first 10 columns for display

        # Sales detection
        if ("purchased on" in cols_norm and "discounted price" in cols_norm and "no of orders" in cols_norm):
            sales_df = df.copy()
            continue

        # Spend detection: look for 'date' and any spend-like column
        has_date = "date" in cols_norm
        # Look for spend column (common names)
        spend_col_candidates = ["spend", "cost", "ad spend", "ad cost", "spend_amount", "adspend"]
        spend_col = None
        for cand in spend_col_candidates:
            if cand in cols_norm:
                spend_col = cand
                break
        # Also fuzzy match
        if not spend_col:
            for c in cols_norm:
                if "spend" in c or "cost" in c:
                    spend_col = c
                    break
        if has_date and spend_col:
            # Ensure it's not the sales sheet
            if not ("discounted price" in cols_norm and "no of orders" in cols_norm):
                spend_df = df.copy()
                # Rename the detected spend column to 'spend'
                if spend_col != "spend":
                    spend_df.rename(columns={spend_col: "spend"}, inplace=True)
                break  # stop after first spend sheet

    if sales_df is None:
        return None, None, "Sales sheet not found.", sheet_debug

    if spend_df is None:
        spend_df = pd.DataFrame(columns=["date", "channel", "spend"])

    # ---------- Process sales ----------
    sales_df.columns = [str(c).strip().lower().replace(" ", "_") for c in sales_df.columns]
    sales_df["date"] = robust_parse_dates(sales_df["purchased_on"])
    sales_df = sales_df.dropna(subset=["date"])
    sales_df["revenue"] = pd.to_numeric(sales_df.get("discounted_price", 0), errors="coerce").fillna(0)
    sales_df["orders"] = pd.to_numeric(sales_df.get("no_of_orders", 0), errors="coerce").fillna(0)

    if "selling_commission" in sales_df.columns:
        sales_df["selling_commission"] = pd.to_numeric(sales_df["selling_commission"], errors="coerce").fillna(0)
    else:
        comm_candidates = [c for c in sales_df.columns if "commission" in c or "comm" in c]
        if comm_candidates:
            sales_df["selling_commission"] = pd.to_numeric(sales_df[comm_candidates[0]], errors="coerce").fillna(0)
        else:
            sales_df["selling_commission"] = 0.0

    sales_df["channel"] = sales_df["channel"].astype(str).str.strip()
    sales_df["type"] = sales_df["type"].astype(str).str.strip() if "type" in sales_df.columns else "Unknown"
    sales_df["Parent"] = sales_df["parent"].astype(str).str.strip() if "parent" in sales_df.columns else "Unknown"
    sales_df["SKU"] = sales_df["sku"].astype(str).str.strip() if "sku" in sales_df.columns else "Unknown"

    # ---------- Process spend ----------
    if len(spend_df) > 0:
        spend_df.columns = [str(c).strip().lower().replace(" ", "_") for c in spend_df.columns]
        # Ensure date column
        if "date" not in spend_df.columns:
            # try to find a date-like column
            for col in spend_df.columns:
                if "date" in col:
                    spend_df.rename(columns={col: "date"}, inplace=True)
                    break
        if "date" in spend_df.columns:
            spend_df["date"] = robust_parse_dates(spend_df["date"])
            spend_df = spend_df.dropna(subset=["date"])
        else:
            spend_df = pd.DataFrame(columns=["date", "channel", "spend"])

        if "spend" not in spend_df.columns:
            # find any numeric column that might be spend
            for col in spend_df.columns:
                if spend_df[col].dtype in ['float64', 'int64'] and col != "date":
                    spend_df.rename(columns={col: "spend"}, inplace=True)
                    break
            else:
                spend_df["spend"] = 0.0

        spend_df["spend"] = pd.to_numeric(spend_df["spend"], errors="coerce").fillna(0)

        if "channel" not in spend_df.columns:
            spend_df["channel"] = "All"
        else:
            spend_df["channel"] = spend_df["channel"].astype(str).str.strip()
            spend_df["channel"] = spend_df["channel"].replace({"": "All", "nan": "All", "None": "All"})
    else:
        spend_df = pd.DataFrame({"date": pd.to_datetime([]), "channel": [], "spend": []})

    return sales_df, spend_df, None, sheet_debug

# ---------- Load data ----------
with st.spinner("Loading data..."):
    sales_df, spend_df, error, sheet_debug = load_and_process_data()

if error:
    st.error(f"❌ {error}")
    st.stop()

if sales_df.empty:
    st.warning("No sales data found.")
    st.stop()

# ---------- Show debug info in sidebar (expandable) ----------
with st.sidebar.expander("🔍 Debug: Sheets & Columns", expanded=False):
    st.write("Detected sheets and first 10 columns:")
    for sheet, cols in sheet_debug.items():
        st.write(f"**{sheet}** → {cols}")
    st.write(f"**Sales data rows:** {len(sales_df)}")
    st.write(f"**Spend data rows:** {len(spend_df)}")
    if not spend_df.empty:
        st.write("Spend date range:", spend_df["date"].min(), "to", spend_df["date"].max())
        st.write("Spend sample:", spend_df.head(2).to_dict())

# ---------- Sidebar filters ----------
st.sidebar.title("🎛️ Control Panel")
min_date = sales_df["date"].min().date()
max_date = sales_df["date"].max().date()
st.sidebar.info(f"Available Data\n\n{min_date} → {max_date}")

default_end = max_date
default_start = max(min_date, max_date - timedelta(days=30))
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End Date", value=default_end, min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.error("Start date cannot be after End date")
    st.stop()

selected_channels = multiselect_with_all("📺 Marketplaces", sales_df["channel"].unique())

if "type" in sales_df.columns:
    def _remap_type(v):
        v = str(v).strip()
        if v in ("", "nan", "None", "Unknown"): return "Rings"
        m = {"ring":"Rings","rings":"Rings","pendant":"Pendants","pendants":"Pendants","earring":"Earrings","earrings":"Earrings","bracelet":"Bracelets","bracelets":"Bracelets"}
        return m.get(v.lower(), v)
    type_display = sales_df["type"].apply(_remap_type)
    type_opts = sorted(type_display.unique())
    selected_types_display = multiselect_with_all("🏷️ Product Types", type_opts)
    selected_types = sales_df.loc[type_display.isin(selected_types_display), "type"].unique().tolist()
else:
    selected_types = []

comparison_period = st.sidebar.selectbox("📊 Compare Against", ["Year over Year", "Month over Month"])

# ---------- Apply filters ----------
start_ts = pd.Timestamp(start_date).tz_localize(None)
end_ts = pd.Timestamp(end_date).tz_localize(None) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask_sales = (sales_df["date"] >= start_ts) & (sales_df["date"] <= end_ts) & (sales_df["channel"].isin(selected_channels))
if selected_types:
    mask_sales &= (sales_df["type"].isin(selected_types))
df_s = sales_df[mask_sales]

if spend_df.empty:
    df_sp = pd.DataFrame(columns=["date", "channel", "spend"])
else:
    mask_spend = (spend_df["date"] >= start_ts) & (spend_df["date"] <= end_ts)
    if "channel" in spend_df.columns:
        mask_spend &= (spend_df["channel"].isin(selected_channels))
    df_sp = spend_df[mask_spend]

# Show warning if spend data is missing for the period
if df_sp.empty and not spend_df.empty:
    st.warning(f"⚠️ Spend data exists in the sheet but none for the selected date range ({start_date} to {end_date}). Check that your spend sheet has dates within this range.")
elif df_sp.empty and spend_df.empty:
    st.warning("⚠️ No spend data loaded at all. Please ensure your Google Sheet has a tab with 'date' and 'spend' (or 'cost') columns.")

# ---------- Previous period for comparison ----------
if comparison_period == "Year over Year":
    start_ly = start_date - pd.DateOffset(years=1)
    end_ly = end_date - pd.DateOffset(years=1)
else:
    start_ly = start_date - pd.DateOffset(months=1)
    end_ly = end_date - pd.DateOffset(months=1)
start_ly_ts = pd.Timestamp(start_ly).tz_localize(None)
end_ly_ts = pd.Timestamp(end_ly).tz_localize(None) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask_sales_ly = (sales_df["date"] >= start_ly_ts) & (sales_df["date"] <= end_ly_ts) & (sales_df["channel"].isin(selected_channels))
if selected_types:
    mask_sales_ly &= (sales_df["type"].isin(selected_types))
df_s_ly = sales_df[mask_sales_ly]

if spend_df.empty:
    df_sp_ly = pd.DataFrame(columns=["date", "channel", "spend"])
else:
    mask_spend_ly = (spend_df["date"] >= start_ly_ts) & (spend_df["date"] <= end_ly_ts)
    if "channel" in spend_df.columns:
        mask_spend_ly &= (spend_df["channel"].isin(selected_channels))
    df_sp_ly = spend_df[mask_spend_ly]

# ---------- Metrics calculation ----------
def calc_metrics(sales, spend):
    rev = sales["revenue"].sum()
    comm = sales["selling_commission"].sum() if "selling_commission" in sales.columns else 0
    ads = spend["spend"].sum() if not spend.empty else 0
    orders = sales["orders"].sum()
    net = (rev * SAFE_MARGIN) - ads - comm
    roas = rev / ads if ads > 0 else 0
    acos = ads / rev * 100 if rev > 0 else 0
    aov = rev / orders if orders > 0 else 0
    return {"Revenue": rev, "Orders": orders, "Spend": ads, "Commission": comm,
            "Net": net, "ROAS": roas, "ACOS": acos, "AOV": aov}

curr = calc_metrics(df_s, df_sp)
prev = calc_metrics(df_s_ly, df_sp_ly)

def delta(k):
    if prev[k] == 0:
        return 0
    return ((curr[k] - prev[k]) / prev[k]) * 100

# ---------- Header ----------
c1, c2 = st.columns([3,1])
with c1:
    st.title("📊 Marketplace Business Insights")
    st.caption(f"Analyzing performance from {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')} • {comparison_period}")
with c2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ---------- KPI grid ----------
st.markdown('<div class="section-header">💎 Key Performance Indicators</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1: metric_card("Total Revenue", f"${curr['Revenue']:,.0f}", delta("Revenue"), prefix="$", color="blue", icon="💰")
with k2: metric_card("Total Orders", f"{curr['Orders']:,.0f}", delta("Orders"), color="cyan", icon="🛒")
with k3: metric_card("Average Order Value", f"${curr['AOV']:,.2f}", delta("AOV"), prefix="$", color="purple", icon="📊")
with k4: metric_card("Net Profit", f"${curr['Net']:,.0f}", delta("Net"), prefix="$", color="green", icon="💹")
st.markdown("")
k5, k6, k7, k8 = st.columns(4)
with k5: metric_card("Ad Spend", f"${curr['Spend']:,.0f}", delta("Spend"), prefix="$", color="orange", inverse=True, icon="📢")
with k6: metric_card("Selling Commission", f"${curr['Commission']:,.0f}", delta("Commission"), prefix="$", color="pink", inverse=True, icon="💳")
with k7: metric_card("ROAS", f"{curr['ROAS']:.2f}", delta("ROAS"), suffix="x", color="yellow", icon="🎯")
with k8: metric_card("ACOS", f"{curr['ACOS']:.1f}", delta("ACOS"), suffix="%", color="red", inverse=True, icon="📈")

# ---------- Tabs (only first tab shown for brevity; others unchanged) ----------
tabs = st.tabs(["🚀 Strategy & Recommendations", "📈 Performance Trends", "🛒 Marketplace Analysis", "🏷️ SKU Analysis", "📊 Profitability Deep Dive", "🔮 Forecasting & Predictions", "🧪 A/B Test Tracker", "📅 Weekly Reports", "📋 Data Explorer", "💎 Merchandising Intel"])

with tabs[0]:
    st.markdown('<div class="section-header">🧠 AI Strategic Insights</div>', unsafe_allow_html=True)
    ch_rev = df_s.groupby("channel")["revenue"].sum().reset_index()
    ch_sp = df_sp.groupby("channel")["spend"].sum().reset_index() if not df_sp.empty else pd.DataFrame(columns=["channel","spend"])
    ch_matrix = pd.merge(ch_rev, ch_sp, on="channel", how="outer").fillna(0)
    ch_matrix["roas"] = ch_matrix.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)
    recs = []
    if 'roas' in ch_matrix.columns:
        for _, row in ch_matrix[ch_matrix['roas'] >= 3.0].iterrows():
            recs.append({"type":"scale","title":f"🚀 Scale Up: {row['channel']}","msg":f"ROAS {row['roas']:.2f}x. Increase budget.","metric":f"{row['roas']:.2f}x"})
        for _, row in ch_matrix[(ch_matrix['roas'] < 1.5) & (ch_matrix['spend'] > 500)].iterrows():
            recs.append({"type":"crit","title":f"🛑 High Spend / Low Return: {row['channel']}","msg":f"Spent ${row['spend']:,.0f}, ROAS {row['roas']:.2f}x.","metric":f"${row['spend']:,.0f}"})
    if curr['Net'] < 0:
        recs.append({"type":"crit","title":"📉 Net Loss Alert","msg":"Operating at a net loss. Cut inefficient spend.","metric":f"${curr['Net']:,.0f}"})
    if recs:
        for rec in recs:
            icon = {"scale":"📈","crit":"🚨","warn":"⚠️","info":"💡"}.get(rec['type'],"💡")
            st.markdown(f"<div class='rec-card rec-{rec['type']}'><div class='rec-title'>{icon} {rec['title']}<span style='margin-left:auto;font-size:12px;opacity:0.8;background:rgba(255,255,255,0.1);padding:2px 8px;border-radius:10px;'>{rec['metric']}</span></div><div class='rec-body'>{rec['msg']}</div></div>", unsafe_allow_html=True)
    else:
        st.info("✅ Business looks stable. No critical alerts found.")

    # Projected outcome
    potential_savings = ch_matrix[ch_matrix['roas'] < 1.5]['spend'].sum() * 0.5
    potential_gain = ch_matrix[ch_matrix['roas'] >= 3.0]['revenue'].sum() * 0.2
    new_net = curr['Net'] + potential_savings + (potential_gain * 0.2)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Potential Wasted Ad Spend", f"${potential_savings:,.0f}")
        st.metric("Revenue Growth Opportunity", f"${potential_gain:,.0f}")
    with col2:
        st.markdown("**Projected Net Profit:**")
        st.markdown(f"<h2 style='color:#10b981'>${new_net:,.0f}</h2>", unsafe_allow_html=True)
        st.caption(f"Vs Current: ${curr['Net']:,.0f}")

# For the remaining tabs, you can either keep your original code or use placeholders.
# Since the user's main issue is spend data, I'll add simple placeholders for other tabs.
with tabs[1]:
    st.info("Performance Trends – your existing code goes here. No changes needed.")
with tabs[2]:
    st.info("Marketplace Analysis – your existing code goes here.")
with tabs[3]:
    st.info("SKU Analysis – your existing code (Supabase) goes here.")
with tabs[4]:
    st.info("Profitability Deep Dive – your existing code goes here.")
with tabs[5]:
    st.info("Forecasting & Predictions – requires forecast_engine.py")
with tabs[6]:
    st.info("A/B Test Tracker – your existing code goes here.")
with tabs[7]:
    st.info("Weekly Reports – your existing code goes here.")
with tabs[8]:
    st.info("Data Explorer – your existing code goes here.")
with tabs[9]:
    st.info("Merchandising Intel – your existing code goes here.")

st.markdown("---")
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Data points: {len(df_s):,}")
