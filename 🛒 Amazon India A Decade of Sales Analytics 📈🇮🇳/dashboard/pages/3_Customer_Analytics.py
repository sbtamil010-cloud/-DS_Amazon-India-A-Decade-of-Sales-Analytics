# dashboard/pages/3_Customer_Analytics.py
import streamlit as st
from dashboard import utils
import pandas as pd
import plotly.express as px

st.title("Customer Analytics")

df = utils.load_data()

# RFM quick segmentation
st.subheader("RFM Snapshot")
if {'customer_id','order_date','final_amount_inr'}.issubset(df.columns):
    # Recency: days since last purchase (assuming max date in dataset as reference)
    ref_date = df['order_date'].max()
    rfm = df.groupby('customer_id').agg(
        Recency=('order_date', lambda x: (ref_date - x.max()).days),
        Frequency=('transaction_id' if 'transaction_id' in df.columns else 'order_date', 'count'),
        Monetary=('final_amount_inr','sum')
    ).reset_index()
    # Show top RFM customers by Monetary
    st.write("Top 10 customers by monetary value")
    st.table(rfm.sort_values('Monetary', ascending=False).head(10).style.format({'Monetary':'₹{:, .2f}'}))
    # Simple RFM scatter
    st.plotly_chart(px.scatter(rfm.sample(min(2000,len(rfm))), x='Recency', y='Monetary', size='Frequency', hover_data=['customer_id'], title='RFM sample'))
else:
    st.info("Required columns for RFM not present (customer_id, order_date, final_amount_inr).")

# Prime member analysis
st.subheader("Prime vs Non-Prime Behavior")
if 'is_prime_member' in df.columns:
    grp = df.groupby('is_prime_member', as_index=False).agg(aov=('final_amount_inr','mean'), orders=('transaction_id','count' if 'transaction_id' in df.columns else 'size'))
    st.table(grp.assign(aov=lambda x: x['aov'].map(lambda v: f"₹{v:,.2f}")))
else:
    st.info("is_prime_member column not found.")
