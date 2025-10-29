# notebooks/EDA_Analysis.py
# EDA script for Amazon India (2015-2025) - 20 visualizations
# Run in Jupyter (recommended) or as a script.

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from datetime import datetime

# ---- Setup ----
DATA_PATH = "data_cleaned/cleaned_amazon_india.csv"
OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)

pd.options.display.float_format = "{:,.2f}".format

# load
df = pd.read_csv(DATA_PATH, parse_dates=["order_date"], low_memory=False)
# Ensure numeric columns
for c in ["original_price_inr", "final_amount_inr", "discount_percent", "delivery_charges", "delivery_days", "customer_rating"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Create derived columns if not present
if "order_year" not in df.columns:
    df["order_year"] = df["order_date"].dt.year
if "order_month" not in df.columns:
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
if "order_month_num" not in df.columns:
    df["order_month_num"] = df["order_date"].dt.month
if "order_quarter" not in df.columns:
    df["order_quarter"] = df["order_date"].dt.to_period("Q").astype(str)

# Helper to save plotly fig
def save_fig(fig, name):
    out = os.path.join(OUT_DIR, name + ".html")
    fig.write_html(out)
    try:
        png_out = os.path.join(OUT_DIR, name + ".png")
        fig.write_image(png_out, scale=2)
    except Exception:
        # write_image may require kaleido; ignore if not installed
        pass
    print("Saved:", out)

# ---- Q1: Yearly revenue trend w/ growth %, trendline, annotations ----
def q1_yearly_revenue(df):
    yearly = df.groupby("order_year", as_index=False).agg(revenue=("final_amount_inr","sum"), orders=("transaction_id","count" if "transaction_id" in df.columns else "size"))
    yearly = yearly.sort_values("order_year")
    yearly["pct_growth"] = yearly["revenue"].pct_change() * 100
    # Trendline using rolling or linear fit
    fig = px.line(yearly, x="order_year", y="revenue", markers=True, title="Yearly Revenue (2015-2025)")
    # add growth % as text
    for _, r in yearly.iterrows():
        fig.add_annotation(x=r["order_year"], y=r["revenue"],
                           text=f"{r['pct_growth']:.1f}%" if not np.isnan(r['pct_growth']) else "",
                           showarrow=False, yshift=15)
    # add linear trendline
    z = np.polyfit(yearly["order_year"].astype(int), yearly["revenue"], 1)
    p = np.poly1d(z)
    fig.add_traces(go.Scatter(x=yearly["order_year"], y=p(yearly["order_year"]), mode="lines", name="Linear Trend", line=dict(dash="dash")))
    save_fig(fig, "q1_yearly_revenue")
    return fig, yearly

fig1, yearly_df = q1_yearly_revenue(df)
fig1.show()

# ---- Q2: Seasonal patterns — monthly heatmap and compare categories ----
def q2_monthly_heatmap(df):
    # pivot: rows=year, cols=month
    df["month"] = df["order_date"].dt.month
    pivot = df.groupby([df["order_date"].dt.year, "month"], as_index=False).agg(revenue=("final_amount_inr","sum"))
    pivot_table = pivot.pivot(index="order_date", columns="month", values="revenue").fillna(0)
    # Plot heatmap with plotly
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=[datetime(2020, m, 1).strftime("%b") for m in pivot_table.columns],
        y=pivot_table.index.astype(str),
        colorscale="YlOrRd"
    ))
    fig.update_layout(title="Monthly Revenue Heatmap (Year vs Month)", xaxis_nticks=12)
    save_fig(fig, "q2_monthly_heatmap")
    return fig, pivot_table

fig2, pivot_table = q2_monthly_heatmap(df)
fig2.show()

# Seasonal patterns by category (compare top 4 categories)
def q2_category_seasonality(df):
    top_cats = df.groupby("category").agg(rev=("final_amount_inr","sum")).sort_values("rev", ascending=False).head(4).index.tolist()
    d = df[df["category"].isin(top_cats)].copy()
    d["month_str"] = d["order_date"].dt.strftime("%b")
    grp = d.groupby(["category", d["order_date"].dt.month], as_index=False).agg(rev=("final_amount_inr","sum"))
    fig = px.line(grp, x="order_date", y="rev", color="category", labels={"order_date":"Month (num)","rev":"Revenue"}, title="Seasonal trend by month (top categories)")
    save_fig(fig, "q2_seasonality_by_category")
    return fig

fig2b = q2_category_seasonality(df)
fig2b.show()

# ---- Q3: RFM segmentation ----
def q3_rfm(df, recency_ref=None):
    # Build RFM table
    if "customer_id" not in df.columns:
        raise ValueError("customer_id required for RFM")
    if recency_ref is None:
        recency_ref = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        Recency=("order_date", lambda x: (recency_ref - x.max()).days),
        Frequency=("transaction_id" if "transaction_id" in df.columns else "order_date", "count"),
        Monetary=("final_amount_inr","sum")
    ).reset_index()
    # Drop zeros / customers with 0 monetary
    rfm = rfm[rfm["Monetary"] > 0]
    # Score R,F,M into 1-5
    rfm["R_rank"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1]).astype(int)  # more recent gets higher rank
    rfm["F_rank"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_rank"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"] = rfm["R_rank"]*100 + rfm["F_rank"]*10 + rfm["M_rank"]
    # simple scatter for segmentation
    sample = rfm.sample(min(2000, len(rfm)), random_state=1)
    fig = px.scatter(sample, x="Recency", y="Monetary", size="Frequency", color="RFM_Score",
                     hover_data=["customer_id","R_rank","F_rank","M_rank"], title="RFM scatter (sample)")
    save_fig(fig, "q3_rfm_scatter")
    return fig, rfm

fig3, rfm_table = q3_rfm(df)
fig3.show()

# ---- Q4: Payment methods evolution 2015-2025 (stacked area) ----
def q4_payment_evolution(df):
    if "payment_method" not in df.columns:
        print("payment_method missing")
        return None
    pm = df.groupby(["order_year","payment_method"], as_index=False).agg(revenue=("final_amount_inr","sum"))
    fig = px.area(pm, x="order_year", y="revenue", color="payment_method", title="Payment Method Evolution (2015-2025)")
    save_fig(fig, "q4_payment_evolution")
    return fig

fig4 = q4_payment_evolution(df)
if fig4: fig4.show()

# ---- Q5: Category-wise performance (treemap, bar, pie) ----
def q5_category_performance(df):
    cat = df.groupby("category", as_index=False).agg(revenue=("final_amount_inr","sum"), orders=("transaction_id","count" if "transaction_id" in df.columns else "size"))
    cat = cat.sort_values("revenue", ascending=False)
    fig_tree = px.treemap(cat, path=["category"], values="revenue", title="Revenue by Category (Treemap)")
    fig_bar = px.bar(cat.head(20), x="category", y="revenue", title="Top Categories by Revenue")
    fig_pie = px.pie(cat, names="category", values="revenue", title="Revenue Share by Category")
    save_fig(fig_tree, "q5_treemap_category")
    save_fig(fig_bar, "q5_bar_category")
    save_fig(fig_pie, "q5_pie_category")
    return fig_tree, fig_bar, fig_pie

fig5_tree, fig5_bar, fig5_pie = q5_category_performance(df)
fig5_tree.show()

# ---- Q6: Prime membership impact ----
def q6_prime_impact(df):
    if "is_prime_member" not in df.columns:
        print("is_prime_member missing")
        return None
    grp = df.groupby("is_prime_member", as_index=False).agg(avg_order_value=("final_amount_inr","mean"), avg_freq=("transaction_id","count" if "transaction_id" in df.columns else "size"))
    fig = px.bar(grp, x="is_prime_member", y="avg_order_value", title="Avg Order Value: Prime vs Non-Prime")
    # Category preferences:
    cat_prime = df.groupby(["is_prime_member","category"], as_index=False).agg(revenue=("final_amount_inr","sum"))
    fig2 = px.bar(cat_prime, x="category", y="revenue", color="is_prime_member", barmode="group", title="Category preferences: Prime vs Non-Prime")
    save_fig(fig, "q6_aov_prime")
    save_fig(fig2, "q6_cat_pref_prime")
    return fig, fig2

fig6_aov, fig6_cat = q6_prime_impact(df)
if fig6_aov: fig6_aov.show()

# ---- Q7: Geographic analysis (choropleth + bar by tier) ----
def q7_geography(df):
    # We will do simple city & state revenue and a bar chart by city tier if 'city_tier' exists; otherwise create tiers heuristic by city population mapping (small)
    state_rev = df.groupby("customer_state", as_index=False).agg(revenue=("final_amount_inr","sum")).sort_values("revenue", ascending=False)
    fig_state = px.bar(state_rev.head(20), x="customer_state", y="revenue", title="Top States by Revenue")
    # city-level top 20
    if "customer_city" in df.columns:
        city_rev = df.groupby("customer_city", as_index=False).agg(revenue=("final_amount_inr","sum")).sort_values("revenue", ascending=False).head(20)
        fig_city = px.bar(city_rev, x="customer_city", y="revenue", title="Top 20 Cities by Revenue")
        save_fig(fig_city, "q7_top_cities")
        fig_city.show()
    save_fig(fig_state, "q7_top_states")
    return fig_state

fig7 = q7_geography(df)
fig7.show()

# ---- Q8: Festival sales impact (before/during/after) ----
def q8_festival_impact(df, festival_name="Diwali", window_days=15):
    # Find festival occurrences (we assume festival_name present in 'festival_name' column and is_festival_sale True for those days)
    if "festival_name" not in df.columns:
        print("festival_name column missing")
        return None
    fest_rows = df[df["festival_name"].str.contains(festival_name, na=False, case=False)]
    if fest_rows.empty:
        print("No festival rows found for", festival_name)
        return None
    fest_dates = fest_rows["order_date"].dt.date.unique()
    # For simplicity, take most common festival date (or analyze multiple)
    date = pd.to_datetime(pd.Series(fest_dates)).mode()[0]
    date = pd.to_datetime(date)
    window = pd.Timedelta(days=window_days)
    mask = (df["order_date"] >= (date - window)) & (df["order_date"] <= (date + window))
    ts = df[mask].groupby(df["order_date"].dt.date).agg(rev=("final_amount_inr","sum")).reset_index()
    fig = px.line(ts, x="order_date", y="rev", title=f"{festival_name} - revenue before/during/after")
    # annotate festival date
    fig.add_vline(x=date.date(), line_dash="dash", annotation_text=festival_name, annotation_position="top left")
    save_fig(fig, f"q8_festival_{festival_name}")
    return fig

fig8 = q8_festival_impact(df, festival_name="Diwali")
if fig8: fig8.show()

# ---- Q9: Age group behavior and preferences ----
def q9_agegroup_analysis(df):
    if "age_group" not in df.columns:
        print("age_group missing")
        return None
    grp = df.groupby("age_group", as_index=False).agg(revenue=("final_amount_inr","sum"), orders=("transaction_id","count" if "transaction_id" in df.columns else "size"))
    fig = px.bar(grp.sort_values("revenue", ascending=False), x="age_group", y="revenue", title="Revenue by Age Group")
    # category preferences per age group - heatmap
    cat_age = df.groupby(["age_group","category"], as_index=False).agg(rev=("final_amount_inr","sum"))
    pivot = cat_age.pivot(index="age_group", columns="category", values="rev").fillna(0)
    fig2 = px.imshow(pivot, title="Category preference heatmap by Age Group")
    save_fig(fig, "q9_agegroup_rev")
    save_fig(fig2, "q9_agegroup_cat_heatmap")
    return fig, fig2

fig9, fig9b = q9_agegroup_analysis(df)
if fig9: fig9.show()

# ---- Q10: Price vs Demand analysis (scatter + corr matrix) ----
def q10_price_vs_demand(df):
    # compute product-level avg price and units sold
    prod = df.groupby("product_id", as_index=False).agg(avg_price=("original_price_inr","mean"), units=("transaction_id","count" if "transaction_id" in df.columns else "size"), revenue=("final_amount_inr","sum"), category=("category", "first"))
    prod = prod.dropna(subset=["avg_price"])
    # scatter
    fig = px.scatter(prod.sample(min(2000,len(prod))), x="avg_price", y="units", color="category", hover_data=["product_id"], title="Price vs Units sold (product-level sample)")
    # correlation matrix between price, units, revenue, rating
    cols = ["avg_price","units","revenue"]
    corr = prod[cols].corr()
    fig2 = px.imshow(corr, text_auto=True, title="Correlation matrix (price,units,revenue)")
    save_fig(fig, "q10_price_vs_demand_scatter")
    save_fig(fig2, "q10_price_corr")
    return fig, fig2

fig10, fig10b = q10_price_vs_demand(df)
fig10.show()

# ---- Q11: Delivery performance ----
def q11_delivery_performance(df):
    if "delivery_days" not in df.columns:
        print("delivery_days missing")
        return None
    # distribution
    fig = px.histogram(df, x="delivery_days", nbins=30, title="Delivery days distribution")
    # on-time performance vs rating (assuming on-time <= promised days; if no promised, use <=5 as on-time)
    df["on_time"] = df["delivery_days"].apply(lambda x: True if (not pd.isna(x) and x <= 5) else False)
    if "customer_rating" in df.columns:
        grp = df.groupby("on_time", as_index=False).agg(avg_rating=("customer_rating","mean"))
        fig2 = px.bar(grp, x="on_time", y="avg_rating", title="Avg rating: On-time vs Late")
        save_fig(fig2, "q11_on_time_rating")
    save_fig(fig, "q11_delivery_dist")
    return fig

fig11 = q11_delivery_performance(df)
fig11.show()

# ---- Q12: Returns & customer satisfaction ----
def q12_returns_analysis(df):
    if "return_status" not in df.columns:
        print("return_status missing")
        return None
    rs = df["return_status"].value_counts(normalize=False).reset_index()
    rs.columns = ["status","count"]
    fig = px.pie(rs, names="status", values="count", title="Return & Cancellation distribution")
    # correlation: return vs rating/price
    if "customer_rating" in df.columns:
        df2 = df.copy()
        df2["is_returned"] = df2["return_status"].str.contains("return", case=False, na=False)
        agg = df2.groupby("is_returned", as_index=False).agg(avg_rating=("customer_rating","mean"), avg_price=("final_amount_inr","mean"))
        fig2 = px.bar(agg, x="is_returned", y=["avg_rating","avg_price"], title="Returned vs Not: Rating & Price (avg)")
        save_fig(fig2, "q12_return_corr")
    save_fig(fig, "q12_return_pie")
    return fig

fig12 = q12_returns_analysis(df)
fig12.show()

# ---- Q13: Brand performance & market share evolution ----
def q13_brand_performance(df):
    if "brand" not in df.columns:
        print("brand missing")
        return None
    brand_year = df.groupby(["order_year","brand"], as_index=False).agg(revenue=("final_amount_inr","sum"))
    top_brands = df.groupby("brand").agg(rev=("final_amount_inr","sum")).sort_values("rev", ascending=False).head(10).index.tolist()
    brand_year_top = brand_year[brand_year["brand"].isin(top_brands)]
    fig = px.line(brand_year_top, x="order_year", y="revenue", color="brand", title="Top brands: revenue trend")
    save_fig(fig, "q13_brand_trend")
    return fig

fig13 = q13_brand_performance(df)
if fig13: fig13.show()

# ---- Q14: CLV analysis using cohort/retention curves ----
def q14_clv_cohort(df):
    if "customer_id" not in df.columns:
        print("customer_id missing")
        return None
    # Cohort by first purchase month
    df["order_month_period"] = df["order_date"].dt.to_period("M")
    first_purchase = df.groupby("customer_id")["order_date"].min().dt.to_period("M").reset_index().rename(columns={"order_date":"first_month"})
    df_cp = df.merge(first_purchase, on="customer_id", how="left")
    cohorts = df_cp.groupby(["first_month", df_cp["order_date"].dt.to_period("M")]).agg(customers=("customer_id","nunique")).reset_index()
    cohorts["period_index"] = (cohorts["order_date"] - cohorts["first_month"]).apply(lambda x: x.n)
    cohort_pivot = cohorts.pivot_table(index="first_month", columns="period_index", values="customers")
    # retention rate
    cohort_size = cohort_pivot.iloc[:,0]
    retention = cohort_pivot.divide(cohort_size, axis=0)
    fig = px.imshow(retention.fillna(0), title="Cohort retention matrix (by month)")
    # simple avg CLV estimate = avg revenue per customer * expected lifetime months (here use mean retention horizon)
    avg_rev = df.groupby("customer_id").agg(mon=("final_amount_inr","sum")).mon.mean()
    approx_clv = avg_rev  # very rough
    save_fig(fig, "q14_cohort_retention")
    return fig

fig14 = q14_clv_cohort(df)
if fig14: fig14.show()

# ---- Q15: Discount & promotional effectiveness ----
def q15_discount_effect(df):
    if "discount_percent" not in df.columns:
        print("discount_percent missing")
        return None
    # Bin by discount
    df2 = df.dropna(subset=["discount_percent"])
    df2["disc_bin"] = pd.cut(df2["discount_percent"], bins=[-1,0,5,15,30,100], labels=["0%","1-5%","6-15%","16-30%","30%+"])
    grp = df2.groupby("disc_bin", as_index=False).agg(rev=("final_amount_inr","sum"), orders=("transaction_id","count" if "transaction_id" in df.columns else "size"))
    fig = px.bar(grp, x="disc_bin", y=["rev","orders"], title="Discount bucket: revenue & orders")
    save_fig(fig, "q15_discount_effect")
    return fig

fig15 = q15_discount_effect(df)
if fig15: fig15.show()

# ---- Q16: Ratings patterns & impact on sales ----
def q16_ratings_impact(df):
    if "customer_rating" not in df.columns:
        print("customer_rating missing")
        return None
    # rating distribution
    fig = px.histogram(df, x="customer_rating", nbins=20, title="Customer rating distribution")
    # rating vs revenue per product
    prod = df.groupby("product_id", as_index=False).agg(avg_rating=("customer_rating","mean"), revenue=("final_amount_inr","sum"), units=("transaction_id","count" if "transaction_id" in df.columns else "size"))
    fig2 = px.scatter(prod.sample(min(2000,len(prod))), x="avg_rating", y="revenue", size="units", title="Avg rating vs Product revenue (sample)")
    save_fig(fig, "q16_rating_dist")
    save_fig(fig2, "q16_rating_vs_revenue")
    return fig, fig2

fig16, fig16b = q16_ratings_impact(df)
fig16.show()

# ---- Q17: Customer journey (purchase frequency patterns, transitions) ----
def q17_customer_journey(df):
    # simple approach: sequence of categories per customer and build transition matrix for category changes
    seq = df.sort_values(["customer_id","order_date"]).groupby("customer_id")["category"].apply(list).reset_index()
    # build pairwise transitions
    from collections import Counter
    transitions = Counter()
    for row in seq["category"]:
        for a,b in zip(row, row[1:]):
            transitions[(a,b)] += 1
    # convert to DataFrame and matrix
    trans_df = pd.DataFrame(((a,b,c) for (a,b),c in transitions.items()), columns=["from","to","count"])
    # transition heatmap for top categories
    top_cats = df["category"].value_counts().head(10).index.tolist()
    mat = pd.DataFrame(0, index=top_cats, columns=top_cats)
    for _, r in trans_df.iterrows():
        if r["from"] in top_cats and r["to"] in top_cats:
            mat.at[r["from"], r["to"]] = r["count"]
    fig = px.imshow(mat, labels=dict(x="To", y="From"), title="Category transition matrix (top categories)")
    save_fig(fig, "q17_category_transitions")
    return fig

fig17 = q17_customer_journey(df)
fig17.show()

# ---- Q18: Inventory & product lifecycle patterns ----
def q18_product_lifecycle(df):
    # For each product, its first sale month and revenue trajectory
    prod_time = df.groupby(["product_id", df["order_date"].dt.to_period("M").astype(str)], as_index=False).agg(rev=("final_amount_inr","sum"))
    # for visualization show 10 random products lifecycle
    sample_prods = df["product_id"].drop_duplicates().sample(min(10, df["product_id"].nunique()), random_state=2).tolist()
    d = prod_time[prod_time["product_id"].isin(sample_prods)]
    fig = px.line(d, x="order_date", y="rev", color="product_id", title="Sample product lifecycle (monthly rev)")
    save_fig(fig, "q18_sample_product_lifecycle")
    return fig

fig18 = q18_product_lifecycle(df)
fig18.show()

# ---- Q19: Competitive pricing analysis (boxplots by category/brand) ----
def q19_pricing_analysis(df):
    if "original_price_inr" not in df.columns:
        print("original_price_inr missing")
        return None
    # boxplot for top categories
    cat = df[df["category"].notna()]
    top_cats = cat["category"].value_counts().head(6).index.tolist()
    d = cat[cat["category"].isin(top_cats)]
    fig = px.box(d, x="category", y="original_price_inr", title="Price distribution by top categories")
    save_fig(fig, "q19_price_box_by_category")
    return fig

fig19 = q19_pricing_analysis(df)
fig19.show()

# ---- Q20: Business health dashboard multi-panel summary (combine KPIs) ----
def q20_business_health(df):
    total_revenue = df["final_amount_inr"].sum()
    total_orders = len(df)
    active_customers = df["customer_id"].nunique() if "customer_id" in df.columns else np.nan
    aov = total_revenue / total_orders if total_orders>0 else 0
    yoy = None
    try:
        yrev = df.groupby("order_year").agg(revenue=("final_amount_inr","sum")).sort_index()
        yoy = yrev.pct_change().iloc[-1,0]*100
    except Exception:
        pass
    # create a small dashboard figure using plotly subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, specs=[[{"type":"indicator"},{"type":"xy"}],[{"type":"xy"},{"type":"xy"}]], subplot_titles=("Total Revenue","Yearly Trend","Top Categories","Customer Growth"))
    fig.add_trace(go.Indicator(mode="number", value=total_revenue, title={"text":"Total Revenue (₹)"}), row=1, col=1)
    # yearly trend
    yr = df.groupby("order_year", as_index=False).agg(revenue=("final_amount_inr","sum"))
    fig.add_trace(go.Scatter(x=yr["order_year"], y=yr["revenue"], name="Revenue"), row=1, col=2)
    # categories
    cat = df.groupby("category", as_index=False).agg(revenue=("final_amount_inr","sum")).sort_values("revenue", ascending=False).head(10)
    fig.add_trace(go.Bar(x=cat["category"], y=cat["revenue"], name="Top categories"), row=2, col=1)
    # customers over time
    if "customer_id" in df.columns:
        cust_agg = df.groupby(df["order_date"].dt.to_period("M")).agg(active_customers=("customer_id","nunique")).reset_index()
        cust_agg["order_date"] = cust_agg["order_date"].dt.to_timestamp()
        fig.add_trace(go.Scatter(x=cust_agg["order_date"], y=cust_agg["active_customers"], name="Active customers (monthly)"), row=2, col=2)
    fig.update_layout(height=900, title_text="Business Health: Executive Multi-panel")
    save_fig(fig, "q20_business_health")
    return fig

fig20 = q20_business_health(df)
fig20.show()

# ---- End: Save a small summary CSVs ----
# Top 20 products and top 20 customers for report
prod_summary = df.groupby(["product_id","product_name"], as_index=False).agg(revenue=("final_amount_inr","sum"), units=("transaction_id","count" if "transaction_id" in df.columns else "size")).sort_values("revenue", ascending=False).head(20)
cust_summary = df.groupby("customer_id", as_index=False).agg(revenue=("final_amount_inr","sum"), orders=("transaction_id","count" if "transaction_id" in df.columns else "size")).sort_values("revenue", ascending=False).head(20)
prod_summary.to_csv(os.path.join(OUT_DIR, "top_products.csv"), index=False)
cust_summary.to_csv(os.path.join(OUT_DIR, "top_customers.csv"), index=False)
print("Wrote top_products.csv and top_customers.csv to", OUT_DIR)

print("EDA complete. Interactive HTMLs and (where possible) PNGs are in", OUT_DIR)
