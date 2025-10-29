# dashboard/pages/6_Advanced_Analytics.py
import streamlit as st
from dashboard import utils
import plotly.express as px
import pandas as pd

st.title("Advanced Analytics & Forecasting")
df = utils.load_data()

st.subheader("Simple Forecasting Input (Data Prep)")
if 'order_date' in df.columns:
    ts = df.groupby('order_date', as_index=True).agg(rev=('final_amount_inr','sum')).resample('M').sum().reset_index()
    st.write("Prepared monthly time series (sample):")
    st.dataframe(ts.tail(10))
    st.info("You can export this `ts` to run ARIMA/Prophet models outside Streamlit, or integrate forecasting code here.")
else:
    st.info("order_date / final_amount_inr columns required for time series.")
