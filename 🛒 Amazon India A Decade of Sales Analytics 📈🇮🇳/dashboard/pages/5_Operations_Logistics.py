# dashboard/pages/5_Operations_Logistics.py
import streamlit as st
from dashboard import utils
import plotly.express as px

st.title("Operations & Logistics")
df = utils.load_data()

# Delivery performance
st.subheader("Delivery Days Distribution")
if 'delivery_days' in df.columns:
    dd = df['delivery_days'].dropna()
    st.bar_chart(dd.value_counts().sort_index())
else:
    st.info("delivery_days column not found.")

# Returns & cancellations
st.subheader("Returns & Cancellation Overview")
if 'return_status' in df.columns:
    r = df['return_status'].value_counts().rename_axis('status').reset_index(name='count')
    st.table(r)
else:
    st.info("return_status column not found.")
