# dashboard/utils.py
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.express as px

@st.cache_data(show_spinner=True)
def load_data(path='data_cleaned/cleaned_amazon_india.csv'):
    df = pd.read_csv(path, parse_dates=['order_date'], low_memory=False)
    # Ensure important columns exist and fillna safe defaults
    for c in ['final_amount_inr','original_price_inr','delivery_charges']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'order_year' not in df.columns and 'order_date' in df.columns:
        df['order_year'] = df['order_date'].dt.year
        df['order_month'] = df['order_date'].dt.to_period('M').astype(str)
    return df

def kpi_totals(df):
    total_revenue = df['final_amount_inr'].sum(min_count=1)
    total_orders = len(df)
    active_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else np.nan
    avg_order_value = total_revenue / total_orders if total_orders>0 else 0
    return {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'active_customers': active_customers,
        'aov': avg_order_value
    }

def yearly_revenue(df):
    grp = df.groupby('order_year', as_index=False).agg(year_revenue=('final_amount_inr','sum'))
    grp = grp.sort_values('order_year')
    return grp

def top_categories(df, n=10):
    if 'category' in df.columns:
        g = df.groupby('category', as_index=False).agg(revenue=('final_amount_inr','sum'), orders=('transaction_id','count' if 'transaction_id' in df.columns else 'size'))
        return g.sort_values('revenue', ascending=False).head(n)
    return pd.DataFrame()

def line_chart_revenue(df):
    yr = yearly_revenue(df)
    fig = px.line(yr, x='order_year', y='year_revenue', markers=True, title='Yearly Revenue')
    fig.update_layout(yaxis_title='Revenue (INR)')
    return fig

def monthly_heatmap(df, year=None):
    d = df.copy()
    if year:
        d = d[d['order_year'] == int(year)]
    d['month'] = d['order_date'].dt.month
    pivot = d.groupby(['month','order_year'], as_index=False).agg(rev=('final_amount_inr','sum'))
    return pivot
