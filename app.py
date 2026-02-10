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
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-5px); }
.kpi-title { font-size: 13px; color: rgba(255,255,255,0.9); margin-bottom: 8px; }
.kpi-value { font-size: 32px; font-weight: 700; }
.kpi-change { font-size: 12px; margin-top: 4px; font-weight: 600; }
.positive { color: #10b981; }
.negative { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

def kpi(title, value, change=None):
    change_html = ""
    if change is not None:
        color = "positive" if change >= 0 else "negative"
        arrow = "↑" if change >= 0 else "↓"
        change_html = f'<div class="kpi-change {color}">{arrow} {abs(change):.1f}%</div>'
    
    st.markdown(f"""
    <div class="kpi-card">
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
col1, col2 = st.columns([6, 1])
with col1:
    st.title("📊 Business Performance Dashboard")
with col2:
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
    # Show what columns we have BEFORE normalization
    st.sidebar.info(f"Spend columns before norm: {list(channel_spend.columns)}")
    
    # Find spend column - check BEFORE normalization for case sensitivity
    spend_col = None
    for col in channel_spend.columns:
        col_lower = col.lower().strip()
        if col_lower in ['ad_spend', 'spend', 'advertising_spend', 'ad spend']:
            spend_col = col
            st.sidebar.success(f"Found: '{col}'")
            break
    
    # If found, copy to ad_spend BEFORE normalization
    if spend_col:
        channel_spend["ad_spend"] = channel_spend[spend_col]
    
    # Date
    date_col = next((c for c in ["date", "Date", "purchased_on", "Purchased_on", "Purchased On"] if c in channel_spend.columns), None)
    if date_col:
        channel_spend["date"] = pd.to_datetime(channel_spend[date_col], errors="coerce")
        channel_spend = channel_spend.dropna(subset=["date"])
    
    # Ad spend - handle both $ and plain
    if "ad_spend" in channel_spend.columns:
        # Show sample before conversion
        st.sidebar.write("Sample ad_spend before:", channel_spend["ad_spend"].head(3).tolist())
        
        channel_spend["ad_spend"] = channel_spend["ad_spend"].astype(str).str.replace('[$,₹£€]', '', regex=True)
        channel_spend["ad_spend"] = pd.to_numeric(channel_spend["ad_spend"], errors="coerce").fillna(0)
        
        # Show sample after conversion
        st.sidebar.write("Sample ad_spend after:", channel_spend["ad_spend"].head(3).tolist())
        st.sidebar.success(f"Total: ${channel_spend['ad_spend'].sum():,.2f}")
    else:
        st.sidebar.error("❌ No ad_spend column created!")

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

# ---------------- KPIs ----------------
st.markdown("### 📈 Key Metrics")
cols = st.columns(7)

with cols[0]: kpi("Revenue", f"${total_rev:,.0f}", rev_change)
with cols[1]: kpi("Orders", f"{total_orders:,.0f}", orders_change)
with cols[2]: kpi("AOV", f"${aov:.2f}")
with cols[3]: kpi("Ad Spend", f"${total_spend:,.0f}", spend_change)
with cols[4]: kpi("Commission", f"${total_commission:,.0f}")
with cols[5]: kpi("Net Earning", f"${net_earning:,.0f}", net_change)
with cols[6]: kpi("ACOS", f"{acos:.1f}%")

st.markdown("---")

# ---------------- CHARTS ----------------
c1, c2 = st.columns(2)

with c1:
    st.markdown("### 📊 Revenue & Orders")
    t = sales_f.groupby(pd.Grouper(key="purchased_on", freq="D")).agg({"revenue": "sum", "no_of_orders": "sum"}).reset_index()
    f = go.Figure()
    f.add_trace(go.Scatter(x=t["purchased_on"], y=t["revenue"], name="Revenue", line=dict(color="#667eea", width=3), fill='tozeroy'))
    f.add_trace(go.Scatter(x=t["purchased_on"], y=t["no_of_orders"], name="Orders", line=dict(color="#10b981", width=3), yaxis="y2"))
    f.update_layout(yaxis=dict(title="Revenue"), yaxis2=dict(title="Orders", overlaying="y", side="right"), hovermode="x", template="plotly_white", height=400)
    st.plotly_chart(f, use_container_width=True, config={'displayModeBar': False})

with c2:
    st.markdown("### 🛒 Revenue by Channel")
    cr = sales_f.groupby("channel").agg({"revenue": "sum"}).reset_index().sort_values("revenue", ascending=False)
    f2 = px.bar(cr, x="channel", y="revenue", color="revenue", color_continuous_scale=["#667eea", "#764ba2"])
    f2.update_layout(showlegend=False, template="plotly_white", height=400)
    st.plotly_chart(f2, use_container_width=True, config={'displayModeBar': False})

c3, c4 = st.columns(2)

with c3:
    st.markdown("### 💰 Ad Spend vs Revenue")
    cm = sales_f.groupby("channel").agg({"revenue": "sum"}).reset_index()
    if len(channel_spend) > 0 and "channel" in channel_spend.columns:
        sbc = channel_spend[(channel_spend["date"] >= start_dt) & (channel_spend["date"] < end_dt_next)].groupby("channel").agg({"ad_spend": "sum"}).reset_index()
        cm = cm.merge(sbc, on="channel", how="left")
        cm["ad_spend"] = cm["ad_spend"].fillna(0)
    else:
        cm["ad_spend"] = 0
    cm = cm.sort_values("revenue", ascending=False)
    f3 = go.Figure()
    f3.add_trace(go.Bar(x=cm["channel"], y=cm["revenue"], name="Revenue", marker_color="#667eea"))
    f3.add_trace(go.Bar(x=cm["channel"], y=cm["ad_spend"], name="Ad Spend", marker_color="#ef4444"))
    f3.update_layout(barmode="group", template="plotly_white", height=400)
    st.plotly_chart(f3, use_container_width=True, config={'displayModeBar': False})

with c4:
    st.markdown("### 📦 Revenue by Type")
    tr = sales_f.groupby("type").agg({"revenue": "sum"}).reset_index()
    f4 = px.pie(tr, values="revenue", names="type", hole=0.4, color_discrete_sequence=px.colors.sequential.Purples_r)
    f4.update_layout(template="plotly_white", height=400)
    st.plotly_chart(f4, use_container_width=True, config={'displayModeBar': False})

# ---------------- TABLE ----------------
st.markdown("---")
st.markdown("### 📋 Detailed Data")

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

st.dataframe(
    summ.style.format({
        "Orders": "{:.0f}", "Revenue": "${:,.0f}", "Commission": "${:,.0f}",
        "Ad Spend": "${:,.0f}", "AOV": "${:,.2f}", "Net Earning": "${:,.0f}",
        "ROAS": "{:.2f}x", "ACOS": "{:.1f}%"
    }),
    use_container_width=True, height=400
)

# ---------------- EXPORT ----------------
st.markdown("---")
st.download_button(
    "📥 Export CSV",
    summ.to_csv(index=False),
    f"dashboard_{start_date}_{end_date}.csv",
    "text/csv"
)

st.caption(f"Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Margin: {SAFE_MARGIN*100:.0f}%")