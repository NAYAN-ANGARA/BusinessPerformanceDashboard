import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gsheets import load_all_sheets
import plotly.io as pio

pio.renderers.default = "browser"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Business Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAFE_MARGIN = 0.62  # 62% margin

# ---------------- HELPERS ----------------
def normalize_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def clean_channel(col):
    return (
        col.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"_ebay$", "_eBay", case=False, regex=True)
    )


# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=300)
def load_data():
    credentials_file = "secret-envoy-486405-j3-03851d061385.json"
    all_data = {}

    for prefix, sheet in [
        ("USA", "USA - DB for Marketplace Dashboard"),
        ("IB", "IB - Database for Marketplace Dashboard"),
    ]:
        data = load_all_sheets(credentials_file, sheet)
        for name, df in data.items():
            all_data[f"{prefix}_{name}"] = df

    return all_data


data = load_data()
if not data:
    st.stop()

# ---------------- SALES DATA ----------------
sales_frames = []
for name, df in data.items():
    if "sales" in name.lower() and not df.empty:
        sales_frames.append(df)

sales = pd.concat(sales_frames, ignore_index=True)
sales = normalize_columns(sales)

required_sales = {"purchased_on", "channel", "no_of_orders", "discounted_price"}
missing = required_sales - set(sales.columns)
if missing:
    st.error(f"Missing sales columns: {missing}")
    st.stop()

sales["purchased_on"] = pd.to_datetime(sales["purchased_on"], errors="coerce")
sales = sales.dropna(subset=["purchased_on"])
sales["no_of_orders"] = pd.to_numeric(sales["no_of_orders"], errors="coerce").fillna(0)

sales["discounted_price"] = (
    sales["discounted_price"].astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
sales["discounted_price"] = pd.to_numeric(sales["discounted_price"], errors="coerce").fillna(0)
sales["revenue"] = sales["discounted_price"]

sales["channel"] = clean_channel(sales["channel"])

if "selling_commission" not in sales.columns:
    sales["selling_commission"] = 0
else:
    sales["selling_commission"] = (
        sales["selling_commission"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    sales["selling_commission"] = pd.to_numeric(sales["selling_commission"], errors="coerce").fillna(0)

if "type" not in sales.columns:
    sales["type"] = "Unknown"

# ---------------- SPEND DATA (FIXED) ----------------
SPEND_KEYWORDS = [
    "spend", "ad", "ads", "advert", "marketing",
    "facebook", "google", "meta", "tiktok", "bing"
]

spend_frames = []

for name, df in data.items():
    if any(k in name.lower() for k in SPEND_KEYWORDS) and not df.empty:
        df = normalize_columns(df)
        df["_source"] = "USA" if name.startswith("USA_") else "IB"
        spend_frames.append(df)

if spend_frames:
    channel_spend = pd.concat(spend_frames, ignore_index=True)
else:
    channel_spend = pd.DataFrame()

if not channel_spend.empty:
    # Date
    for dcol in ["date", "purchased_on"]:
        if dcol in channel_spend.columns:
            channel_spend["date"] = pd.to_datetime(channel_spend[dcol], errors="coerce")
            break

    channel_spend = channel_spend.dropna(subset=["date"])

    # Spend
    if "ad_spend" not in channel_spend.columns:
        if "spend" in channel_spend.columns:
            channel_spend["ad_spend"] = channel_spend["spend"]
        else:
            st.error("Spend data missing ad_spend / spend column")
            st.stop()

    channel_spend["ad_spend"] = (
        channel_spend["ad_spend"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    channel_spend["ad_spend"] = pd.to_numeric(channel_spend["ad_spend"], errors="coerce").fillna(0)

    if "channel" not in channel_spend.columns:
        st.error("Spend data missing channel column")
        st.stop()

    channel_spend["channel"] = clean_channel(channel_spend["channel"])

# ---------------- SIDEBAR ----------------
st.sidebar.header("📅 Date Range")
min_date = sales["purchased_on"].min().date()
max_date = sales["purchased_on"].max().date()

start_date = st.sidebar.date_input("Start", min_date)
end_date = st.sidebar.date_input("End", max_date)

start_dt = pd.Timestamp(start_date)
end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

st.sidebar.header("🎯 Filters")
channels = sales["channel"].unique()
types = sales["type"].unique()

sel_channels = st.sidebar.multiselect("Channels", channels, default=list(channels))
sel_types = st.sidebar.multiselect("Types", types, default=list(types))

# ---------------- FILTER ----------------
sales_f = sales[
    (sales["purchased_on"] >= start_dt) &
    (sales["purchased_on"] < end_dt) &
    (sales["channel"].isin(sel_channels)) &
    (sales["type"].isin(sel_types))
]

if sales_f.empty:
    st.warning("No sales data for filters")
    st.stop()

if not channel_spend.empty:
    spend_f = channel_spend[
        (channel_spend["date"] >= start_dt) &
        (channel_spend["date"] < end_dt) &
        (channel_spend["channel"].isin(sel_channels))
    ]
else:
    spend_f = pd.DataFrame()

# ---------------- KPIs ----------------
total_rev = sales_f["revenue"].sum()
orders = sales_f["no_of_orders"].sum()
commission = sales_f["selling_commission"].sum()
ad_spend = spend_f["ad_spend"].sum() if not spend_f.empty else 0

net = (total_rev * SAFE_MARGIN) - ad_spend - commission
acos = (ad_spend / total_rev * 100) if total_rev else 0
roas = (total_rev / ad_spend) if ad_spend else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Revenue", f"${total_rev:,.0f}")
col2.metric("Orders", f"{orders:,.0f}")
col3.metric("Ad Spend", f"${ad_spend:,.0f}")
col4.metric("Commission", f"${commission:,.0f}")
col5.metric("Net Earning", f"${net:,.0f}")
col6.metric("ACOS", f"{acos:.1f}%")

# ---------------- DEBUG ----------------
with st.expander("🔍 Spend Debug"):
    if not channel_spend.empty:
        st.write(channel_spend.groupby("_source")["ad_spend"].sum())
        st.dataframe(spend_f.head(20))
    else:
        st.warning("No spend data loaded")

# ---------------- TABLE ----------------
summary = (
    sales_f.groupby("channel")
    .agg(
        Orders=("no_of_orders", "sum"),
        Revenue=("revenue", "sum"),
        Commission=("selling_commission", "sum"),
    )
    .reset_index()
)

if not spend_f.empty:
    spend_sum = spend_f.groupby("channel")["ad_spend"].sum().reset_index()
    summary = summary.merge(spend_sum, on="channel", how="left")

summary["ad_spend"] = summary["ad_spend"].fillna(0)
summary["Net Earning"] = (summary["Revenue"] * SAFE_MARGIN) - summary["ad_spend"] - summary["Commission"]
summary["ROAS"] = summary.apply(lambda r: r["Revenue"] / r["ad_spend"] if r["ad_spend"] else 0, axis=1)
summary["ACOS"] = summary.apply(lambda r: (r["ad_spend"] / r["Revenue"] * 100) if r["Revenue"] else 0, axis=1)

st.dataframe(
    summary.style.format({
        "Revenue": "${:,.0f}",
        "Commission": "${:,.0f}",
        "ad_spend": "${:,.0f}",
        "Net Earning": "${:,.0f}",
        "ROAS": "{:.2f}x",
        "ACOS": "{:.1f}%"
    }),
    use_container_width=True
)
