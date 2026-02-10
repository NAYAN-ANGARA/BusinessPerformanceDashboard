import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets
from datetime import date, timedelta

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

# ---------------- CSS (PREMIUM DARK UI) ----------------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: #0f1116;
    }
    
    /* KPI Cards with Glassmorphism */
    .metric-card {
        background: rgba(30, 32, 40, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
    
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9ca3af;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
    }
    
    /* Dynamic Metric Accents */
    .accent-blue { border-top: 3px solid #3b82f6; }
    .accent-green { border-top: 3px solid #10b981; }
    .accent-orange { border-top: 3px solid #f97316; }
    .accent-purple { border-top: 3px solid #8b5cf6; }
    
    /* Delta Badge */
    .delta-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
    }
    .delta-pos { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .delta-neg { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    
    /* Section Dividers */
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #e5e7eb;
        margin: 30px 0 15px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-header::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #2d303e, transparent);
    }
    
    /* Custom Charts */
    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS ----------------
def metric_card(label, value, delta=None, prefix="", suffix="", color="blue", inverse=False):
    delta_html = ""
    if delta is not None:
        is_pos = delta >= 0
        # If inverse is True (e.g. ACOS), positive change is "bad" (Red)
        is_good = not is_pos if inverse else is_pos
        
        delta_class = "delta-pos" if is_good else "delta-neg"
        arrow = "↑" if is_pos else "↓"
        delta_html = f'<span class="delta-badge {delta_class}">{arrow} {abs(delta):.1f}%</span>'
    else:
        delta_html = '<span style="color:#6b7280; font-size:11px">No prev data</span>'
        
    st.markdown(f"""
    <div class="metric-card accent-{color}">
        <div class="metric-label">{label}</div>
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
    
    # 1. Load Raw
    try:
        data1 = load_all_sheets(creds, "USA - DB for Marketplace Dashboard")
        data2 = load_all_sheets(creds, "IB - Database for Marketplace Dashboard")
        
        all_dfs = {}
        if data1: 
            for k, v in data1.items(): all_dfs[f"USA_{k}"] = v
        if data2: 
            for k, v in data2.items(): all_dfs[f"IB_{k}"] = v
            
    except Exception as e:
        return None, str(e)

    if not all_dfs: return None, "No data found."

    # 2. Process Sales
    sales_list = []
    for name, df in all_dfs.items():
        if 'sales' in name.lower() and not df.empty:
            sales_list.append(df)
            
    if not sales_list: return None, "No Sales sheets found."
    
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
    sales = sales.dropna(subset=["date"])

    # 3. Process Spend (Robust)
    spend_list = []
    for name, df in all_dfs.items():
        if 'channel' in name.lower() and 'spend' in name.lower():
            df = df.copy()
            # Normalize Headers
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
    
    return sales, spend

# ---------------- LOAD STATE ----------------
sales_df, spend_df = load_and_process_data()

if isinstance(spend_df, str): # Error handling
    st.error(f"❌ {spend_df}")
    st.stop()

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.header("🎛️ Dashboard Controls")
    
    # Quick Date Selectors
    st.caption("Quick Select")
    c1, c2, c3 = st.columns(3)
    today = date.today()
    
    # Defaults
    if 'start_date' not in st.session_state: st.session_state.start_date = today - timedelta(days=30)
    if 'end_date' not in st.session_state: st.session_state.end_date = today

    if c1.button("30D", use_container_width=True):
        st.session_state.start_date = today - timedelta(days=30)
        st.session_state.end_date = today
        st.rerun()
        
    if c2.button("MTD", use_container_width=True):
        st.session_state.start_date = today.replace(day=1)
        st.session_state.end_date = today
        st.rerun()
        
    if c3.button("YTD", use_container_width=True):
        st.session_state.start_date = today.replace(month=1, day=1)
        st.session_state.end_date = today
        st.rerun()

    # Manual Date Picker
    start_date = st.date_input("Start Date", st.session_state.start_date)
    end_date = st.date_input("End Date", st.session_state.end_date)
    
    st.divider()
    
    # Filters
    st.subheader("Filters")
    selected_channels = multiselect_with_all("Channels", sales_df["channel"].unique())
    
    if "type" in sales_df.columns:
        selected_types = multiselect_with_all("Type", sales_df["type"].unique())
    else:
        selected_types = ["All"]

# ---------------- FILTERING LOGIC ----------------
# Current Period
mask_sales = (
    (sales_df["date"].dt.date >= start_date) & 
    (sales_df["date"].dt.date <= end_date) &
    (sales_df["channel"].isin(selected_channels)) &
    (sales_df["type"].isin(selected_types) if "type" in sales_df.columns else True)
)
df_s = sales_df[mask_sales]

mask_spend = (
    (spend_df["date"].dt.date >= start_date) & 
    (spend_df["date"].dt.date <= end_date) &
    (spend_df["channel"].isin(selected_channels))
)
df_sp = spend_df[mask_spend]

# Previous Period (YoY)
start_ly = start_date - pd.DateOffset(years=1)
end_ly = end_date - pd.DateOffset(years=1)

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
        "Revenue": rev, "Orders": orders, "Spend": ads, 
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
    st.caption(f"Analyzing performance from **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}**")
with c2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ---------------- UI: KPI GRID ----------------
st.markdown('<div class="section-header">📈 Financial Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
with k1: metric_card("Total Revenue", f"{curr['Revenue']:,.0f}", delta("Revenue"), prefix="$", color="blue")
with k2: metric_card("Net Profit", f"{curr['Net']:,.0f}", delta("Net"), prefix="$", color="green")
with k3: metric_card("Ad Spend", f"{curr['Spend']:,.0f}", delta("Spend"), prefix="$", color="orange", inverse=True)
with k4: metric_card("ROAS", f"{curr['ROAS']:.2f}", delta("ROAS"), suffix="x", color="purple")
with k5: metric_card("ACOS", f"{curr['ACOS']:.1f}", delta("ACOS"), suffix="%", color="orange", inverse=True)

# ---------------- UI: DETAILED ANALYSIS TABS ----------------
st.markdown("")
tabs = st.tabs(["📊 Trends & Efficiency", "🛒 Channel Deep Dive", "📋 Raw Data"])

with tabs[0]:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("**Revenue vs Efficiency Trend**")
        # Align dates
        daily_rev = df_s.groupby(pd.Grouper(key="date", freq="D"))["revenue"].sum().reset_index()
        daily_spend = df_sp.groupby(pd.Grouper(key="date", freq="D"))["spend"].sum().reset_index()
        daily_trend = pd.merge(daily_rev, daily_spend, on="date", how="outer").fillna(0)
        daily_trend["roas"] = daily_trend.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)
        
        fig_dual = go.Figure()
        
        # Revenue Bar
        fig_dual.add_trace(go.Bar(
            x=daily_trend["date"], y=daily_trend["revenue"],
            name="Revenue", marker_color="#3b82f6", opacity=0.7
        ))
        
        # ROAS Line
        fig_dual.add_trace(go.Scatter(
            x=daily_trend["date"], y=daily_trend["roas"],
            name="ROAS", yaxis="y2", line=dict(color="#10b981", width=3, shape='spline')
        ))
        
        fig_dual.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#2d303e"),
            yaxis2=dict(title="ROAS", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=0, r=0, t=40, b=0),
            height=380
        )
        st.plotly_chart(fig_dual, use_container_width=True)
        
    with c2:
        st.markdown("**Profitability Waterfall**")
        cost_goods = curr['Revenue'] * (1 - SAFE_MARGIN)
        commission = df_s["selling_commission"].sum() if "selling_commission" in df_s.columns else 0
        
        fig_water = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative", "relative", "relative", "relative", "total"],
            x = ["Gross Revenue", "COGS", "Commission", "Ad Spend", "Net Profit"],
            textposition = "outside",
            text = [f"${curr['Revenue']/1000:.1f}k", f"-${cost_goods/1000:.1f}k", 
                    f"-${commission/1000:.1f}k", f"-${curr['Spend']/1000:.1f}k", f"${curr['Net']/1000:.1f}k"],
            y = [curr['Revenue'], -cost_goods, -commission, -curr['Spend'], curr['Net']],
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
            height=380
        )
        st.plotly_chart(fig_water, use_container_width=True)

with tabs[1]:
    st.markdown("**Channel Efficiency Matrix (Bubble Size = ROAS)**")
    
    # Aggregation
    ch_rev = df_s.groupby("channel")["revenue"].sum().reset_index()
    ch_sp = df_sp.groupby("channel")["spend"].sum().reset_index()
    ch_matrix = pd.merge(ch_rev, ch_sp, on="channel", how="outer").fillna(0)
    ch_matrix["roas"] = ch_matrix.apply(lambda x: x["revenue"]/x["spend"] if x["spend"]>0 else 0, axis=1)
    ch_matrix = ch_matrix[ch_matrix["revenue"] > 0]
    
    # Scatter Chart
    fig_bubble = px.scatter(
        ch_matrix, x="spend", y="revenue", 
        size="roas", color="channel",
        hover_name="channel",
        labels={"spend": "Ad Spend ($)", "revenue": "Revenue ($)", "roas": "ROAS"},
        size_max=60,
        text="channel"
    )
    
    fig_bubble.update_traces(textposition='top center')
    fig_bubble.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.05)",
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

with tabs[2]:
    st.markdown('<div class="section-header">📋 Performance Data Table</div>', unsafe_allow_html=True)
    
    # Detailed Table
    tbl = ch_matrix.copy()
    tbl["orders"] = df_s.groupby("channel")["orders"].sum().values
    tbl["acos"] = tbl.apply(lambda x: (x["spend"]/x["revenue"]*100) if x["revenue"]>0 else 0, axis=1)
    tbl["net"] = (tbl["revenue"] * SAFE_MARGIN) - tbl["spend"]
    
    # Interactive Table (No Matplotlib dependency)
    st.dataframe(
        tbl,
        column_config={
            "channel": "Channel",
            "revenue": st.column_config.ProgressColumn("Revenue", format="$%d", min_value=0, max_value=int(tbl["revenue"].max())),
            "spend": st.column_config.NumberColumn("Ad Spend", format="$%d"),
            "orders": st.column_config.NumberColumn("Orders", format="%d"),
            "roas": st.column_config.NumberColumn("ROAS", format="%.2fx"),
            "acos": st.column_config.NumberColumn("ACOS", format="%.1f%%"),
            "net": st.column_config.NumberColumn("Net Profit", format="$%d"),
        },
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Export
    csv = tbl.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Report (CSV)", csv, "performance_report.csv", "text/csv")

st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #6b7280; font-size: 12px;'>Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | Safe Margin: {SAFE_MARGIN*100:.0f}%</div>", unsafe_allow_html=True)