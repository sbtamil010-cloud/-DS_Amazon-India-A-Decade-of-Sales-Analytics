# dashboard/pages/1b_RealTime_Monitor.py
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dashboard import utils
import plotly.express as px

st.set_page_config(page_title="Real-time Monitor", layout="wide")
st.title("Real-time Business Performance Monitor")

DB_URL = st.secrets.get("db_url", None)
engine = create_engine(DB_URL) if DB_URL else None

# Targets (could be dynamic loaded from config table)
targets = {
    "monthly_revenue_target": int(st.secrets.get("monthly_revenue_target", 10_000_000)),
    "monthly_orders_target": int(st.secrets.get("monthly_orders_target", 10_000))
}

if engine:
    with engine.connect() as conn:
        mtd_q = """
        SELECT 
          SUM(final_amount_inr) FILTER (WHERE order_date >= date_trunc('month', current_date)) as mtd_revenue,
          COUNT(*) FILTER (WHERE order_date >= date_trunc('month', current_date)) as mtd_orders,
          SUM(final_amount_inr) FILTER (WHERE order_date = current_date) as today_revenue,
          COUNT(*) FILTER (WHERE order_date = current_date) as today_orders
        FROM transactions;
        """
        row = conn.execute(text(mtd_q)).mappings().fetchone()
        mtd_revenue = float(row['mtd_revenue'] or 0)
        mtd_orders = int(row['mtd_orders'] or 0)
        today_revenue = float(row['today_revenue'] or 0)
        today_orders = int(row['today_orders'] or 0)
else:
    df = utils.load_data()
    today = pd.Timestamp.now().normalize()
    mtd_revenue = df[df['order_date'] >= today.replace(day=1)]['final_amount_inr'].sum()
    mtd_orders = len(df[df['order_date'] >= today.replace(day=1)])
    today_revenue = df[df['order_date'] == today]['final_amount_inr'].sum()
    today_orders = len(df[df['order_date'] == today])

col1, col2, col3, col4 = st.columns(4)
col1.metric("MTD Revenue (₹)", f"{mtd_revenue:,.0f}", delta=f"Target: ₹{targets['monthly_revenue_target']:,.0f}")
col2.metric("MTD Orders", f"{mtd_orders:,}", delta=f"Target: {targets['monthly_orders_target']:,}")
col3.metric("Today Revenue (₹)", f"{today_revenue:,.0f}")
col4.metric("Today Orders", f"{today_orders:,}")

# Simple alert rules
alerts = []
if mtd_revenue < targets['monthly_revenue_target'] * 0.7:
    alerts.append(("Revenue run-rate is below 70% of monthly target", "danger"))
if mtd_orders < targets['monthly_orders_target'] * 0.7:
    alerts.append(("Order volume is below 70% of monthly target", "warning"))

for message, level in alerts:
    if level == "danger":
        st.error(message)
    elif level == "warning":
        st.warning(message)

# Run-rate projection
days_in_month = pd.Timestamp.now().days_in_month
today_day = pd.Timestamp.now().day
projected = (mtd_revenue / max(1,today_day)) * days_in_month
st.subheader("Projected Month Revenue")
fig = px.bar(x=['Projected','Target'], y=[projected, targets['monthly_revenue_target']], labels={'x':'','y':'Revenue'}, title='Projected vs Target Month Revenue')
st.plotly_chart(fig, use_container_width=True)

# Top acquisition channels / payment method snapshot (today)
st.subheader("Today: Top Payment Methods")
if engine:
    q = """
    SELECT payment_method, SUM(final_amount_inr) AS revenue
    FROM transactions
    WHERE order_date = current_date
    GROUP BY payment_method ORDER BY revenue DESC LIMIT 10;
    """
    with engine.connect() as conn:
        pm = pd.read_sql(text(q), conn)
else:
    df = utils.load_data()
    pm = df[df['order_date'] == pd.Timestamp.now().normalize()].groupby('payment_method', as_index=False).agg(revenue=('final_amount_inr','sum')).sort_values('revenue', ascending=False)
st.table(pm)
