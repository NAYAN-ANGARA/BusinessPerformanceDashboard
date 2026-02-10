import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets

# Configure Plotly
import plotly.io as pio
pio.renderers.default = "browser"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Business Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CONSTANTS ----------------
SAFE_MARGIN = 0.62

# ---------------- CSS ----------------
st.markdown("""
<style>
* { transition: all 0.3s ease; }

.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 24px;
    color: white;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(255,255,255,0.1);
}
.kpi-card:hover { 
    transform: translateY(-8px);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
}

.kpi-card.revenue {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.kpi-card.positive-metric {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}
.kpi-card.warning-metric {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}
.kpi-card.danger-metric {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

.kpi-title { 
    font-size: 12px; 
    color: rgba(255,255,255,0.85); 
    margin-bottom: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.kpi-value { 
    font-size: 36px; 
    font-weight: 800;
    letter-spacing: -1px;
}
.kpi-change { 
    font-size: 13px; 
    margin-top: 8px; 
    font-weight: 700;
}
.positive { color: #a7f3d0; }
.negative { color: #fecaca; }

.metric-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    margin-top: 8px;
}

.section-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 24px;
    margin: 32px 0 20px 0;
}

.insight-box {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border-left: 4px solid #667eea;
    padding: 16px;
    border-radius: 8px;
    margin: 16px 0;
}

.stat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(102, 126, 234, 0.2);
}

.chart-container {
    background: white;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(0,0,0,0.05);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

a { color: #667eea; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

def kpi(title, value, change=None, card_type="revenue"):
    change_html = ""
    if change is not None:
        color = "positive" if change >= 0 else "negative"
        arrow = "↑" if change >= 0 else "↓"
        change_html = f'<div class="kpi-change {color}">{arrow} {abs(change):.1f}% vs LY</div>'
    
    st.markdown(f"""
    <div class="kpi-card {card_type}">
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
    
    try:
        data1 = load_all_sheets(creds, "USA - DB for Marketplace Dashboard")
        if data1:
            for sheet_name, df in data1.items():
                all_data[f"USA_{sheet_name}"] = df
    except Exception as e:
        st.warning(f"⚠️ USA: {str(e)}")
    
    try:
        data2 = load_all_sheets(creds, "IB - Database for Marketplace Dashboard")
        if data2:
            for sheet_name, df in data2.items():
                all_data[f"IB_{sheet_name}"] = df
    except Exception as e:
        st.warning(f"⚠️ IB: {str(e)}")
    
    return all_data if all_data else None

# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([4, 1.5, 1])
with col1:
    st.title("📊 Business Performance Dashboard")
with col2:
    st.write("")
    st.write("")
    if st.button("📊 Run Analysis", use_container_width=True, help="Refresh data"):
        st.cache_data.clear()
        st.rerun()
with col3:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

data = load_data()
if not data:
    st.error("❌ No data loaded")
    st.stop()

# ---------------- COLLECT SALES ----------------
sales_sheets = [df.copy() for name, df in data.items() if 'sales' in name.lower() and not df.empty]
if not sales_sheets:
    st.error("❌ No sales data")
    st.stop()

sales = pd.concat(sales_sheets, ignore_index=True)

# ---------------- COLLECT SPEND ----------------
spend_sheets = []
for name in data.keys():
    lower = name.lower()
    # Match: "USA_channel_spend_data", "IB_Channel_spend_data" etc
    # Skip: "USA_Spend_data"
    if 'channel' in lower and 'spend' in lower:
        df = data[name].copy()
        if not df.empty:
            spend_sheets.append(df)

channel_spend = pd.concat(spend_sheets, ignore_index=True) if spend_sheets else pd.DataFrame()

# ---------------- NORMALIZE ----------------
def norm(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

sales = norm(sales)
if len(channel_spend) > 0:
    channel_spend = norm(channel_spend)

# Clean channels
for df in [sales, channel_spend]:
    if len(df) > 0 and "channel" in df.columns:
        df["channel"] = df["channel"].astype(str).str.strip()

# ---------------- PROCESS SALES ----------------
sales["purchased_on"] = pd.to_datetime(sales["purchased_on"], errors="coerce")
sales = sales.dropna(subset=["purchased_on"])

sales["no_of_orders"] = pd.to_numeric(sales["no_of_orders"], errors="coerce").fillna(0)

sales["discounted_price"] = sales["discounted_price"].astype(str).str.replace('[$,]', '', regex=True)
sales["discounted_price"] = pd.to_numeric(sales["discounted_price"], errors="coerce").fillna(0)
sales["revenue"] = sales["discounted_price"]

if "selling_commission" in sales.columns:
    sales["selling_commission"] = sales["selling_commission"].astype(str).str.replace('[$,]', '', regex=True)
    sales["selling_commission"] = pd.to_numeric(sales["selling_commission"], errors="coerce").fillna(0)
else:
    sales["selling_commission"] = 0

sales["type"] = sales.get("type", "Unknown")

# ---------------- PROCESS SPEND ----------------
if len(channel_spend) > 0:
    # Find spend column - check for both 'Spend' and 'Ad Spend' variations
    spend_col = None
    for col in channel_spend.columns:
        col_lower = col.lower().strip()
        # Match: "Spend", "Ad Spend", "ad_spend", "advertising_spend", etc.
        if 'spend' in col_lower:
            spend_col = col
            break
    
    # If found, create ad_spend column BEFORE normalization
    if spend_col:
        channel_spend["ad_spend"] = channel_spend[spend_col]
    
    # Date
    date_col = next((c for c in ["date", "Date", "purchased_on", "Purchased_on", "Purchased On"] if c in channel_spend.columns), None)
    if date_col:
        channel_spend["date"] = pd.to_datetime(channel_spend[date_col], errors="coerce")
        channel_spend = channel_spend.dropna(subset=["date"])
    
    # Ad spend - handle both $ and plain numbers
    if "ad_spend" in channel_spend.columns:
        # Remove currency symbols and commas, handle both formats
        channel_spend["ad_spend"] = channel_spend["ad_spend"].astype(str).str.replace('[$,₹£€]', '', regex=True).str.strip()
        channel_spend["ad_spend"] = pd.to_numeric(channel_spend["ad_spend"], errors="coerce").fillna(0)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📅 Date Range")
min_date = sales["purchased_on"].min().date()
max_date = sales["purchased_on"].max().date()

start_date = st.sidebar.date_input("Start", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End", value=max_date, min_value=min_date, max_value=max_date)

st.sidebar.header("🎯 Filters")
selected_channels = multiselect_with_all("Channels", sales["channel"].unique())
selected_types = multiselect_with_all("Types", sales["type"].unique())

# ---------------- FILTER ----------------
start_dt = pd.Timestamp(start_date)
end_dt_next = pd.Timestamp(end_date) + pd.Timedelta(days=1)

sales_f = sales[
    (sales["purchased_on"] >= start_dt) &
    (sales["purchased_on"] < end_dt_next) &
    (sales["channel"].isin(selected_channels)) &
    (sales["type"].isin(selected_types))
]

# ---------------- METRICS ----------------
total_rev = sales_f["revenue"].sum()
total_orders = sales_f["no_of_orders"].sum()
aov = total_rev / total_orders if total_orders > 0 else 0
total_commission = sales_f["selling_commission"].sum()

total_spend = 0
if len(channel_spend) > 0 and "date" in channel_spend.columns:
    spend_f = channel_spend[(channel_spend["date"] >= start_dt) & (channel_spend["date"] < end_dt_next)]
    if "channel" in channel_spend.columns:
        spend_f = spend_f[spend_f["channel"].isin(selected_channels)]
    total_spend = spend_f["ad_spend"].sum()

net_earning = (total_rev * SAFE_MARGIN) - total_spend - total_commission
acos = (total_spend / total_rev * 100) if total_rev > 0 else 0

# YoY
year_ago_start = start_dt - pd.DateOffset(years=1)
year_ago_end = end_dt_next - pd.DateOffset(years=1)

sales_ly = sales[
    (sales["purchased_on"] >= year_ago_start) &
    (sales["purchased_on"] < year_ago_end) &
    (sales["channel"].isin(selected_channels)) &
    (sales["type"].isin(selected_types))
]

ly_rev = sales_ly["revenue"].sum()
ly_orders = sales_ly["no_of_orders"].sum()
ly_commission = sales_ly["selling_commission"].sum()

ly_spend = 0
if len(channel_spend) > 0 and "date" in channel_spend.columns:
    spend_ly = channel_spend[(channel_spend["date"] >= year_ago_start) & (channel_spend["date"] < year_ago_end)]
    if "channel" in channel_spend.columns:
        spend_ly = spend_ly[spend_ly["channel"].isin(selected_channels)]
    ly_spend = spend_ly["ad_spend"].sum()

ly_net = (ly_rev * SAFE_MARGIN) - ly_spend - ly_commission

rev_change = ((total_rev - ly_rev) / ly_rev * 100) if ly_rev > 0 else 0
orders_change = ((total_orders - ly_orders) / ly_orders * 100) if ly_orders > 0 else 0
spend_change = ((total_spend - ly_spend) / ly_spend * 100) if ly_spend > 0 else 0
net_change = ((net_earning - ly_net) / ly_net * 100) if ly_net != 0 else 0

# Calculate additional metrics
roas = (total_rev / total_spend) if total_spend > 0 else 0
profit_margin = ((total_rev * SAFE_MARGIN - total_spend) / total_rev * 100) if total_rev > 0 else 0
roi = ((total_rev * SAFE_MARGIN - total_spend - total_commission) / (total_spend + total_commission) * 100) if (total_spend + total_commission) > 0 else 0

ly_roas = (ly_rev / ly_spend) if ly_spend > 0 else 0
ly_profit_margin = ((ly_rev * SAFE_MARGIN - ly_spend) / ly_rev * 100) if ly_rev > 0 else 0
ly_roi = ((ly_rev * SAFE_MARGIN - ly_spend - ly_commission) / (ly_spend + ly_commission) * 100) if (ly_spend + ly_commission) > 0 else 0

roas_change = ((roas - ly_roas) / ly_roas * 100) if ly_roas > 0 else 0
pm_change = (profit_margin - ly_profit_margin)
roi_change = ((roi - ly_roi) / ly_roi * 100) if ly_roi > 0 else 0

# Header with period info
st.markdown(f"<p style='color: #667eea; font-weight: 600; margin-top: -10px;'>📅 {start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}</p>", unsafe_allow_html=True)

# Health indicators
health_color = "positive-metric" if rev_change > 0 else "warning-metric"
spend_health = "warning-metric" if acos > 25 else "positive-metric"
roi_health = "positive-metric" if roi > 0 else "danger-metric"

# ---------------- KPIs ----------------
st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
cols = st.columns(8)

with cols[0]:
    kpi("Revenue", f"${total_rev:,.0f}", rev_change, card_type=health_color)
with cols[1]:
    kpi("Orders", f"{total_orders:,.0f}", orders_change, card_type="positive-metric")
with cols[2]:
    kpi("AOV", f"${aov:.2f}", card_type="revenue")
with cols[3]:
    kpi("Ad Spend", f"${total_spend:,.0f}", spend_change, card_type=spend_health)
with cols[4]:
    kpi("ROAS", f"{roas:.2f}x", roas_change, card_type="positive-metric")
with cols[5]:
    kpi("ACOS", f"{acos:.1f}%", card_type=spend_health)
with cols[6]:
    kpi("Profit %", f"{profit_margin:.1f}%", pm_change, card_type="positive-metric")
with cols[7]:
    kpi("Net Earning", f"${net_earning:,.0f}", net_change, card_type=roi_health)

st.markdown("---")

# Insights
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class="insight-box">
    <strong>💡 Top Performing Channel</strong><br>
    """, unsafe_allow_html=True)
    if len(sales_f) > 0:
        top_channel = sales_f.groupby("channel")["revenue"].sum().idxmax()
        top_rev = sales_f.groupby("channel")["revenue"].sum().max()
        st.write(f"**{top_channel}**: ${top_rev:,.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="insight-box">
    <strong>📊 Best Efficiency Metric</strong><br>
    """, unsafe_allow_html=True)
    if roas >= ly_roas:
        st.write(f"🔥 **ROAS Improved**: {roas:.2f}x (+{roas_change:.1f}%)")
    else:
        st.write(f"⚠️ **ROAS Declined**: {roas:.2f}x ({roas_change:.1f}%)")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="insight-box">
    <strong>🎯 Budget Efficiency</strong><br>
    """, unsafe_allow_html=True)
    if acos < 25:
        st.write(f"✅ **Excellent**: ACOS {acos:.1f}%")
    elif acos < 40:
        st.write(f"⚠️ **Good**: ACOS {acos:.1f}%")
    else:
        st.write(f"❌ **Needs Review**: ACOS {acos:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- CHARTS ----------------
st.markdown('<div class="section-header">📊 Detailed Analysis</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 📈 Revenue & Orders Trend")
    t = sales_f.groupby(pd.Grouper(key="purchased_on", freq="D")).agg({"revenue": "sum", "no_of_orders": "sum"}).reset_index()
    f = go.Figure()
    f.add_trace(go.Scatter(
        x=t["purchased_on"], y=t["revenue"], name="Revenue", 
        line=dict(color="#667eea", width=4), 
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
        hovertemplate="<b>Revenue</b><br>%{y:$,.0f}<extra></extra>"
    ))
    f.add_trace(go.Scatter(
        x=t["purchased_on"], y=t["no_of_orders"], name="Orders", 
        line=dict(color="#10b981", width=4), 
        yaxis="y2",
        hovertemplate="<b>Orders</b><br>%{y:,.0f}<extra></extra>"
    ))
    f.update_layout(
        yaxis=dict(title="Revenue ($)", titlefont=dict(color="#667eea")), 
        yaxis2=dict(title="Orders", overlaying="y", side="right", titlefont=dict(color="#10b981")), 
        hovermode="x unified", template="plotly_white", height=420,
        margin=dict(l=80, r=80, t=20, b=20)
    )
    st.plotly_chart(f, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 🛒 Revenue by Channel")
    cr = sales_f.groupby("channel").agg({"revenue": "sum"}).reset_index().sort_values("revenue", ascending=False)
    f2 = px.bar(
        cr, x="channel", y="revenue", 
        color="revenue", 
        color_continuous_scale=["#d1e7f0", "#667eea", "#764ba2"],
        hover_data={"revenue": ":.2f"}
    )
    f2.update_layout(
        showlegend=False, template="plotly_white", height=420,
        margin=dict(l=60, r=20, t=20, b=60),
        xaxis_title="Channel",
        yaxis_title="Revenue ($)"
    )
    f2.update_traces(hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>")
    st.plotly_chart(f2, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

c3, c4 = st.columns(2)

with c3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 💰 Ad Spend vs Revenue by Channel")
    cm = sales_f.groupby("channel").agg({"revenue": "sum"}).reset_index()
    if len(channel_spend) > 0 and "channel" in channel_spend.columns:
        sbc = channel_spend[(channel_spend["date"] >= start_dt) & (channel_spend["date"] < end_dt_next)].groupby("channel").agg({"ad_spend": "sum"}).reset_index()
        cm = cm.merge(sbc, on="channel", how="left")
        cm["ad_spend"] = cm["ad_spend"].fillna(0)
    else:
        cm["ad_spend"] = 0
    # Calculate ROAS per channel
    cm["roas"] = cm.apply(lambda r: r["revenue"] / r["ad_spend"] if r["ad_spend"] > 0 else 0, axis=1)
    cm = cm.sort_values("revenue", ascending=False)
    
    f3 = go.Figure()
    f3.add_trace(go.Bar(
        x=cm["channel"], y=cm["revenue"], name="Revenue", 
        marker_color="#667eea",
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
    ))
    f3.add_trace(go.Bar(
        x=cm["channel"], y=cm["ad_spend"], name="Ad Spend", 
        marker_color="#ef4444",
        hovertemplate="<b>%{x}</b><br>Spend: $%{y:,.0f}<extra></extra>"
    ))
    f3.update_layout(
        barmode="group", template="plotly_white", height=420,
        margin=dict(l=60, r=20, t=20, b=60),
        xaxis_title="Channel",
        yaxis_title="Amount ($)"
    )
    st.plotly_chart(f3, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 📦 Revenue Distribution by Type")
    tr = sales_f.groupby("type").agg({"revenue": "sum", "no_of_orders": "sum"}).reset_index().sort_values("revenue", ascending=False)
    f4 = px.pie(
        tr, values="revenue", names="type", 
        hole=0.35, 
        color_discrete_sequence=px.colors.sequential.Purples_r,
    )
    f4.update_traces(
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<extra></extra>",
        textposition="inside",
        textinfo="percent+label"
    )
    f4.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(f4, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TABLE ----------------
st.markdown("---")
st.markdown('<div class="section-header">📋 Channel Performance Summary</div>', unsafe_allow_html=True)

summ = sales_f.groupby("channel").agg({"no_of_orders": "sum", "revenue": "sum", "selling_commission": "sum"}).reset_index()

if len(channel_spend) > 0 and "channel" in channel_spend.columns:
    ss = channel_spend[(channel_spend["date"] >= start_dt) & (channel_spend["date"] < end_dt_next)].groupby("channel").agg({"ad_spend": "sum"}).reset_index()
    summ = summ.merge(ss, on="channel", how="left")
    summ["ad_spend"] = summ["ad_spend"].fillna(0)
else:
    summ["ad_spend"] = 0

summ["aov"] = summ["revenue"] / summ["no_of_orders"]
summ["net_earning"] = (summ["revenue"] * SAFE_MARGIN) - summ["ad_spend"] - summ["selling_commission"]
summ["roas"] = summ.apply(lambda r: r["revenue"] / r["ad_spend"] if r["ad_spend"] > 0 else 0, axis=1)
summ["acos"] = summ.apply(lambda r: (r["ad_spend"] / r["revenue"] * 100) if r["revenue"] > 0 else 0, axis=1)
summ = summ.sort_values("revenue", ascending=False)

summ.columns = ["Channel", "Orders", "Revenue", "Commission", "Ad Spend", "AOV", "Net Earning", "ROAS", "ACOS"]

# Format for display
display_df = summ.copy()

def format_dataframe(df):
    return df.style.format({
        "Orders": "{:.0f}", 
        "Revenue": "${:,.0f}", 
        "Commission": "${:,.0f}",
        "Ad Spend": "${:,.0f}", 
        "AOV": "${:,.2f}", 
        "Net Earning": "${:,.0f}",
        "ROAS": "{:.2f}x", 
        "ACOS": "{:.1f}%"
    }).background_gradient(
        subset=["Revenue", "ROAS"], 
        cmap="Greens", 
        vmin=0, 
        vmax=max(df["Revenue"].max(), 10000)
    ).background_gradient(
        subset=["ACOS"], 
        cmap="RdYlGn_r", 
        vmin=0, 
        vmax=50
    ).set_properties(**{
        'text-align': 'center',
        'padding': '10px'
    })

st.dataframe(
    format_dataframe(display_df),
    use_container_width=True, 
    height=300
)

# Summary stats
st.markdown('<div class="stat-header">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Channels", len(summ), delta=None, label_visibility="collapsed")
    st.write(f"**Active**: {len(summ[summ['Revenue'] > 0])}")

with col2:
    avg_aov = summ["AOV"].mean()
    st.metric("💵 Avg AOV", f"${avg_aov:.2f}", delta=None, label_visibility="collapsed")
    st.write(f"**Range**: ${summ['AOV'].min():.2f} - ${summ['AOV'].max():.2f}")

with col3:
    avg_roas = summ[summ["ROAS"] > 0]["ROAS"].mean()
    st.metric("📈 Avg ROAS", f"{avg_roas:.2f}x", delta=None, label_visibility="collapsed")
    best_channel = summ.loc[summ['ROAS'].idxmax()] if len(summ) > 0 else None
    if best_channel is not None:
        st.write(f"**Best**: {best_channel['Channel']}")

with col4:
    avg_acos = summ[summ["ACOS"] > 0]["ACOS"].mean()
    st.metric("🎯 Avg ACOS", f"{avg_acos:.1f}%", delta=None, label_visibility="collapsed")
    if avg_acos < 30:
        st.write("**✅ Excellent efficiency**")
    elif avg_acos < 40:
        st.write("**⚠️ Good efficiency**")
    else:
        st.write("**❌ Needs optimization**")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- EXPORT ----------------
st.markdown("---")
st.markdown('<div class="section-header">📥 Data Export & Reports</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    csv = summ.to_csv(index=False)
    st.download_button(
        "📊 Download Channel Summary (CSV)",
        csv,
        f"channel_summary_{start_date}_{end_date}.csv",
        "text/csv",
        use_container_width=True
    )

with col2:
    # Prepare detailed export with additional metrics
    detailed_export = sales_f.copy()
    detailed_export["purchased_on"] = detailed_export["purchased_on"].dt.strftime("%Y-%m-%d")
    csv_detailed = detailed_export.to_csv(index=False)
    st.download_button(
        "📈 Download Transaction Details (CSV)",
        csv_detailed,
        f"transactions_{start_date}_{end_date}.csv",
        "text/csv",
        use_container_width=True
    )

with col3:
    # Summary report
    report = f"""BUSINESS PERFORMANCE REPORT
Period: {start_date} to {end_date}

KEY METRICS:
- Total Revenue: ${total_rev:,.2f}
- Total Orders: {total_orders:,.0f}
- Avg Order Value: ${aov:.2f}
- Ad Spend: ${total_spend:,.2f}
- ROAS: {roas:.2f}x
- ACOS: {acos:.1f}%
- Net Earning: ${net_earning:,.2f}
- Profit Margin: {profit_margin:.1f}%

YoY COMPARISON:
- Revenue Change: {rev_change:+.1f}%
- Orders Change: {orders_change:+.1f}%
- Spend Change: {spend_change:+.1f}%
- Net Earning Change: {net_change:+.1f}%

CHANNEL BREAKDOWN:
"""
    for idx, row in summ.iterrows():
        report += f"\n{row['Channel']}: ${row['Revenue']:,.0f} | ROAS: {row['ROAS']:.2f}x | ACOS: {row['ACOS']:.1f}%"
    
    st.download_button(
        "📄 Download Report (TXT)",
        report,
        f"report_{start_date}_{end_date}.txt",
        "text/plain",
        use_container_width=True
    )

st.markdown("---")
st.caption(f"📊 Dashboard Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Safe Margin: {SAFE_MARGIN*100:.0f}% | Data Period: {(end_date - start_date).days} days")