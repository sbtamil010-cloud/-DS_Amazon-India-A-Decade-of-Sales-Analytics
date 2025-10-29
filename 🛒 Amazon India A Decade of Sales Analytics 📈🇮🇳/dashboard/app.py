# dashboard/app.py
import streamlit as st

st.set_page_config(
    page_title="Amazon India: Sales Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 Amazon India — A Decade of Sales Analytics (2015-2025)")
st.markdown(
    "Multi-page Streamlit app. Use the left nav to open pages (or use the top menu 'Pages').\n\n"
    "Expect the cleaned dataset at: `data_cleaned/cleaned_amazon_india.csv`."
)

st.sidebar.header("Navigation")
st.sidebar.markdown(
    """
    Use the Pages menu at top-right (Streamlit native multipage) or click the links below:
    - Executive Dashboard
    - Revenue Analytics
    - Customer Analytics
    - Product & Inventory
    - Operations & Logistics
    - Advanced Analytics
    """
)

st.sidebar.info("If you prefer a single-file app, open `dashboard/app_singlefile.py` (not included).")
