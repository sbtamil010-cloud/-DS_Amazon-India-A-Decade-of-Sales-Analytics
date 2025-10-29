# dashboard/pages/2_Revenue_Analytics.py
import streamlit as st
from dashboard import utils
import plotly.express as px
import pandas as pd

st.title("Revenue Analytics")

df = utils.load_data()
years = sorted(df['order_year'].dropna().unique().tolist())
selected_year = st.sidebar.selectbox("Select Year (for month-level analysis)", ["All"] + years)

# Yearly revenue line
st.subheader("Yearly Revenue")
st.plotly_chart(utils.line_chart_revenue(df), use_container_width=True)

# Monthly heatmap (interactive)
st.subheader("Monthly Revenue Heatmap")
if selected_year == "All":
    pivot = df.groupby([df['order_date'].dt.year, df['order_date'].dt.month], as_index=False).agg(rev=('final_amount_inr','sum'))
    pivot.columns = ['year','month','rev']
    fig = px.imshow(
        pivot.pivot(index='year', columns='month', values='rev').fillna(0),
        labels=dict(x="Month", y="Year", color="Revenue"),
        aspect="auto", title="Revenue Heatmap (year vs month)"
    )
else:
    yr = int(selected_year)
    d = df[df['order_year'] == yr]
    monthly = d.groupby(d['order_date'].dt.month, as_index=False).agg(rev=('final_amount_inr','sum')).sort_values('order_date')
    fig = px.bar(monthly, x=monthly['order_date'].astype(str), y='rev', title=f"Monthly Revenue: {yr}")
st.plotly_chart(fig, use_container_width=True)

# Payment method evolution
st.subheader("Payment Method Evolution (2015-2025)")
if 'payment_method' in df.columns:
    pm = df.groupby(['order_year','payment_method'], as_index=False).agg(revenue=('final_amount_inr','sum'))
    fig2 = px.area(pm, x='order_year', y='revenue', color='payment_method', labels={'order_year':'Year','revenue':'Revenue'})
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("payment_method column not found in dataset.")
