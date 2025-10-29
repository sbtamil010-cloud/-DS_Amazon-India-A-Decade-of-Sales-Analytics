# dashboard/pages/4_Product_Inventory.py
import streamlit as st
from dashboard import utils
import pandas as pd
import plotly.express as px

st.title("Product & Inventory Analytics")
df = utils.load_data()

st.subheader("Top Products by Revenue")
if {'product_id','product_name'}.issubset(df.columns):
    prod = df.groupby(['product_id','product_name'], as_index=False).agg(revenue=('final_amount_inr','sum'), units=('transaction_id','count' if 'transaction_id' in df.columns else 'size'))
    top = prod.sort_values('revenue', ascending=False).head(20)
    st.dataframe(top[['product_name','revenue','units']].assign(revenue=lambda x: x['revenue'].map(lambda v: f"₹{v:,.0f}")))
else:
    st.info("product_id/product_name columns missing.")
    
st.subheader("Category Performance")
if 'category' in df.columns:
    cat = df.groupby('category', as_index=False).agg(revenue=('final_amount_inr','sum'), orders=('transaction_id','count' if 'transaction_id' in df.columns else 'size'))
    fig = px.treemap(cat, path=['category'], values='revenue', title='Revenue by Category')
    st.plotly_chart(fig, use_container_width=True)
