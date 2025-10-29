# dashboard/pages/1_Executive_Dashboard.py
import streamlit as st
from dashboard import utils
import plotly.express as px

st.set_page_config(page_title="Executive Summary", layout="wide")
st.title("Executive Summary — Key Business KPIs")

df = utils.load_data()

# Top row KPIs
kpis = utils.kpi_totals(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue (₹)", f"{kpis['total_revenue']:,.0f}")
col2.metric("Total Orders", f"{kpis['total_orders']:,}")
col3.metric("Active Customers", f"{kpis['active_customers']:,}")
col4.metric("Avg Order Value (AOV)", f"₹{kpis['aov']:,.2f}")

st.markdown("---")
# Revenue trend
st.subheader("Revenue Trend (2015-2025)")
fig = utils.line_chart_revenue(df)
st.plotly_chart(fig, use_container_width=True)

# Top categories & geographic snapshot
st.subheader("Top Categories")
top_cat = utils.top_categories(df, n=8)
st.table(top_cat[['category','revenue']].assign(revenue=lambda x: x['revenue'].map(lambda v: f"₹{v:,.0f}")))

st.subheader("Revenue by Year (Top 10 Cities)")
if 'customer_city' in df.columns:
    city_rev = df.groupby(['order_year','customer_city'], as_index=False).agg(rev=('final_amount_inr','sum'))
    latest = city_rev[city_rev['order_year']==city_rev['order_year'].max()]
    top_cities = latest.sort_values('rev', ascending=False).head(10)
    st.bar_chart(top_cities.set_index('customer_city')['rev'])
