"""
Data cleaning script implementing the 10 practice challenges.

Usage:
    python scripts/data_cleaning_challenges.py --input data_raw/amazon_india_complete_2015_2025.csv \
        --output data_cleaned/cleaned_amazon_india.csv --dups_out data_cleaned/duplicates_review.csv
"""

import pandas as pd
import numpy as np
import argparse
from dateutil import parser
import re

# ---------------------------
# Helper cleaning functions
# ---------------------------

# 1) Date cleaning
def clean_order_date(x):
    """Parse many date formats and return ISO 'YYYY-MM-DD' or NaT on invalid."""
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    # quick reject obviously invalid (like month > 12 or day > 31) won't rely only on regex
    try:
        # allow fuzzy parsing and day-first (handles DD/MM/YYYY, DD-MM-YY)
        dt = parser.parse(s, dayfirst=True, yearfirst=False, fuzzy=True)
        # Accept only reasonable years (2010-2035 for this dataset)
        if 2010 <= dt.year <= 2035:
            return pd.Timestamp(dt.date())
        else:
            return pd.NaT
    except Exception:
        return pd.NaT

# 2) Price cleaning
def clean_price_inr(val):
    """
    Convert messy price strings like '₹1,25,000', '1,250.00', 'Price on Request' -> float or NaN.
    Strips currency symbols, commas, whitespace. Returns float in INR.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Common phrase -> treat as missing
    if re.search(r'price\s*on\s*request|por|n/a|na|unknown', s, flags=re.I):
        return np.nan
    # Remove currency symbols and non-numeric characters except dot and minus
    # Many Indian formats use ',', so remove them
    s = s.replace('\u202f','')  # narrow no-break space
    s = re.sub(r'[₹Rs\.\s,]', '', s, flags=re.I)
    # keep digits and dot
    s = re.sub(r'[^0-9.]', '', s)
    if s == '' or s == '.':
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

# 3) Rating normalization
def normalize_rating(val, fill_with=np.nan):
    """
    Normalize ratings like '4 stars', '3/5', '2.5/5.0', '5.0' -> float within 1.0-5.0.
    Missing values left as NaN unless fill_with provided.
    """
    if pd.isna(val):
        return fill_with
    s = str(val).strip()
    # If looks like fraction e.g., '3/5' or '2.5/5.0'
    frac = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)', s)
    if frac:
        try:
            num = float(frac.group(1)); den = float(frac.group(2))
            if den == 0:
                return np.nan
            scaled = (num/den) * 5.0
            return round(min(max(scaled, 1.0), 5.0), 2)
        except Exception:
            return np.nan
    # Otherwise extract first numeric token
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)', s)
    if m:
        valf = float(m.group(1))
        # If someone wrote '45' it's probably erroneous; cap by logic below
        if valf > 10:
            return np.nan
        # If value given on 10-point scale (like 8/10 with missing '/10'), try to scale if >5
        if valf > 5 and valf <= 10:
            return round((valf/10.0)*5.0, 2)
        return round(min(max(valf, 1.0), 5.0), 2)
    return np.nan

# 4) City normalization
def normalize_city(name):
    """
    Standardizes city names: handles Bangalore/Bengaluru, Bombay/Mumbai, Delhi/New Delhi,
    common misspellings, case variations. Returns Title-cased canonical name or NaN.
    Extend 'city_map' for more coverage.
    """
    if pd.isna(name):
        return np.nan
    s = str(name).strip().lower()
    # Remove punctuation
    s = re.sub(r'[^a-z0-9 /,-]', '', s)
    # split if contains separators like '/', ',', '-'
    parts = re.split(r'[\/,-]', s)
    # canonical mapping
    city_map = {
        'bangalore': 'Bengaluru',
        'bengaluru': 'Bengaluru',
        'bengalooru': 'Bengaluru',
        'bombay': 'Mumbai',
        'mumbai': 'Mumbai',
        'delhi': 'New Delhi',
        'new delhi': 'New Delhi',
        'nd': 'New Delhi',
        'calcutta': 'Kolkata',
        'kolkata': 'Kolkata',
        'madras': 'Chennai',
        'chennai': 'Chennai',
        'pune': 'Pune',
        'hyderabad': 'Hyderabad',
        'secunderabad': 'Hyderabad',
        'benglore': 'Bengaluru'
    }
    for p in parts:
        key = p.strip()
        if key in city_map:
            return city_map[key]
        # handle tokenized words, e.g., 'bangalore bengaluru' -> check each token
        toks = key.split()
        for t in toks:
            if t in city_map:
                return city_map[t]
    # fallback: title case cleaned first token
    first = parts[0].strip()
    if first == '':
        return np.nan
    return first.title()

# 5) Boolean normalization
def normalize_bool(val):
    """
    Convert mixed boolean-like values to True/False/NaN.
    Accepts booleans, 'Yes','No','Y','N','1','0', 1/0, 'true','false', etc.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ('true', 't', 'yes', 'y', '1', '1.0', 'yess'):
        return True
    if s in ('false', 'f', 'no', 'n', '0', '0.0', 'none'):
        return False
    return np.nan

# 6) Category normalization
def normalize_category(cat):
    """
    Normalize categories: collapse variants like 'Electronics', 'ELECTRONIC', 'Electronics & Accessories'
    to a canonical label. Extend mapping as needed.
    """
    if pd.isna(cat):
        return np.nan
    s = str(cat).strip().lower()
    s = s.replace('&', 'and')
    s = re.sub(r'[^a-z0-9 ]', ' ', s).strip()
    mapping = {
        'electronics': 'Electronics',
        'electronic': 'Electronics',
        'electronics and accessories': 'Electronics',
        'electronic accessories': 'Electronics',
        'fashion': 'Fashion',
        'clothing': 'Fashion',
        'home kitchen': 'Home & Kitchen',
        'home and kitchen': 'Home & Kitchen',
        'home & kitchen': 'Home & Kitchen',
        'beauty': 'Beauty',
        'books': 'Books',
        'grocery': 'Grocery',
        'toys': 'Toys',
        'sports': 'Sports'
    }
    for k,v in mapping.items():
        if k in s:
            return v
    # fallback to title case
    return s.title()

# 7) Delivery days cleaning
def clean_delivery_days(val):
    """
    Transform 'Same Day', '1-2 days', '3 days', negative values, unrealistic >30 days to numeric.
    Rules:
     - 'Same Day' -> 0
     - '1-2 days' -> average 1.5
     - Negative or >30 -> NaN (invalid)
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)) and not np.isnan(val):
        try:
            iv = int(round(val))
            if iv < 0 or iv > 30:
                return np.nan
            return iv
        except Exception:
            return np.nan
    s = str(val).strip().lower()
    if s == '':
        return np.nan
    if 'same' in s:
        return 0
    # range like '1-2 days'
    m = re.search(r'(\d+)\s*[-to]+\s*(\d+)', s)
    if m:
        a = int(m.group(1)); b = int(m.group(2))
        avg = int(round((a + b) / 2.0))
        if avg < 0 or avg > 30:
            return np.nan
        return avg
    # single integer
    m2 = re.search(r'(\d+)', s)
    if m2:
        v = int(m2.group(1))
        if v < 0 or v > 30:
            return np.nan
        return v
    return np.nan

# 8) Duplicate handling
def flag_and_handle_duplicates(df, key_cols=None):
    """
    Identify duplicates by key_cols (default: customer_id, product_id, order_date, final_amount_inr).
    Strategy:
      - Exact full-row duplicates -> drop (true duplicates).
      - Duplicates on key_cols:
         - If a 'quantity' or 'units' column exists and count>1, treat as plausible bulk orders -> keep.
         - If no quantity column and duplicates have identical transaction_id values -> drop duplicates by transaction_id.
         - Otherwise flag for manual review and keep for now.
    Returns (df_cleaned, duplicates_df) where duplicates_df contains flagged groups for review.
    """
    if key_cols is None:
        key_cols = [c for c in ['customer_id','product_id','order_date','final_amount_inr'] if c in df.columns]
    # Full-row duplicates -> remove
    before = len(df)
    df = df.drop_duplicates(keep='first')
    after = len(df)
    removed_full = before - after

    # Find groups with same key_cols appearing more than once
    if len(key_cols) == 0:
        return df, pd.DataFrame(columns=df.columns.tolist()+['dup_count','dup_reason'])
    grp = df.groupby(key_cols).size().reset_index(name='dup_count')
    dup_groups = grp[grp['dup_count'] > 1]
    flagged_rows = []
    for _, row in dup_groups.iterrows():
        cond = np.ones(len(df), dtype=bool)
        for c in key_cols:
            cond &= (df[c] == row[c])
        group = df[cond].copy()
        # If quantity column exists and sums indicate bulk order (quantity >1), we leave them
        if 'quantity' in df.columns:
            qsum = group['quantity'].sum()
            if qsum >= len(group):
                # treat as valid (bulk) orders -> do not drop, but note
                group['dup_reason'] = 'bulk_orders_possible'
            else:
                group['dup_reason'] = 'needs_review'
        else:
            # if transaction_id identical inside group -> true duplicate caused by ingestion -> drop later
            if group['transaction_id'].nunique() == 1:
                group['dup_reason'] = 'exact_duplicate_single_txn'
            else:
                # could be multiple transactions same customer/product/date/amount (could be error) - flag
                group['dup_reason'] = 'needs_review'
        flagged_rows.append(group)
    if flagged_rows:
        duplicates_df = pd.concat(flagged_rows, ignore_index=True)
    else:
        duplicates_df = pd.DataFrame(columns=df.columns.tolist()+['dup_reason'])
    # For 'exact_duplicate_single_txn' remove duplicates keeping first
    if not duplicates_df.empty:
        exact_dups = duplicates_df[duplicates_df['dup_reason']=='exact_duplicate_single_txn']
        if not exact_dups.empty:
            ids_to_drop = exact_dups.groupby('transaction_id').apply(lambda g: list(g.index)[1:]).sum()
            # The above yields index positions in extremes; better approach is to drop duplicates by subset
            df = df.drop_duplicates(subset=key_cols + ['transaction_id'], keep='first')
    return df, duplicates_df

# 9) Price outlier detection & correction
def fix_price_outliers(df, price_col='original_price_inr'):
    """
    Detect price outliers using IQR and domain rules; attempt auto-correction for probable decimal-shift errors.
    Method:
      - Compute IQR bounds; mark values outside bounds OR > median*20 as outliers.
      - For each outlier, try dividing by 10,100,1000,... and accept first candidate within median*5.
      - If no candidate fits, leave as is but flag.
    Returns (df_fixed, price_outliers_df).
    """
    s = df[price_col].dropna()
    if s.empty:
        return df, pd.DataFrame()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = max(q1 - 1.5 * iqr, 0)
    upper = q3 + 1.5 * iqr
    median = s.median()

    # Condition for outlier
    cond_out = (df[price_col].notna()) & ((df[price_col] < lower) | (df[price_col] > upper) | (df[price_col] > median*20))
    outliers = df[cond_out].copy()
    outliers['fixed_price'] = np.nan
    outliers['fix_action'] = ''

    def try_fix_value(v, med):
        if pd.isna(v) or v == 0:
            return None
        # try dividing by powers of 10
        for n in range(1,6):
            cand = v / (10**n)
            if cand <= med * 5:
                return round(cand, 2)
        return None

    for idx, row in outliers.iterrows():
        v = row[price_col]
        fixed = None
        fixed = try_fix_value(v, median)
        if fixed is not None:
            df.at[idx, price_col] = fixed
            outliers.at[idx, 'fixed_price'] = fixed
            outliers.at[idx, 'fix_action'] = 'divided_by_10_power'
        else:
            outliers.at[idx, 'fix_action'] = 'no_auto_fix'
    return df, outliers

# 10) Payment method normalization
def normalize_payment_method(pm):
    """
    Map strings like 'UPI/PhonePe/GooglePay' -> 'UPI', 'Credit Card/CREDIT_CARD/CC' -> 'Card', 'Cash on Delivery/COD' -> 'COD'
    Fallback: title case.
    """
    if pd.isna(pm):
        return np.nan
    s = str(pm).lower()
    if any(k in s for k in ['upi', 'phonepe', 'googlepay', 'gpay', 'paytm']):
        return 'UPI'
    if any(k in s for k in ['credit card', 'credit_card', 'cc', 'visa', 'mastercard', 'debit card', 'debit_card']):
        return 'Card'
    if any(k in s for k in ['cash on delivery', 'cod', 'c.o.d', 'cash']):
        return 'COD'
    if 'netbank' in s or 'net banking' in s:
        return 'NetBanking'
    if 'wallet' in s:
        return 'Wallet'
    return s.title()

# ---------------------------
# Main pipeline
# ---------------------------

def run(input_csv, output_csv, dups_out=None):
    print("Loading:", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    # Make a copy to avoid modifying original accidentally
    df = df.copy()

    # Question 1: order_date cleanup
    if 'order_date' in df.columns:
        print("Cleaning order_date...")
        df['order_date_orig'] = df['order_date']
        df['order_date'] = df['order_date'].apply(clean_order_date)
        # Optionally, create order_year/order_month
        df['order_year'] = df['order_date'].dt.year
        df['order_month'] = df['order_date'].dt.to_period('M').astype(str)
    else:
        print("Warning: 'order_date' column not found.")

    # Question 2: original_price_inr cleaning
    if 'original_price_inr' in df.columns:
        print("Cleaning original_price_inr...")
        df['original_price_inr_orig'] = df['original_price_inr']
        df['original_price_inr'] = df['original_price_inr'].apply(clean_price_inr)
    else:
        print("Warning: 'original_price_inr' missing.")

    # Also clean final_amount_inr and delivery_charges too (common)
    for col in ['final_amount_inr','delivery_charges']:
        if col in df.columns:
            df[col] = df[col].apply(clean_price_inr)

    # Question 3: ratings
    for col in ['customer_rating','product_rating']:
        if col in df.columns:
            print(f"Normalizing {col}...")
            # Fill missing with NaN initially, later we could impute median
            df[col] = df[col].apply(normalize_rating)
    # If any ratings still NaN, impute with median of that column (reasonable)
    for col in ['customer_rating','product_rating']:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med)

    # Question 4: normalize city
    if 'customer_city' in df.columns:
        print("Normalizing customer_city...")
        df['customer_city_orig'] = df['customer_city']
        df['customer_city'] = df['customer_city'].apply(normalize_city)

    # Question 5: booleans
    for bcol in ['is_prime_member','is_prime_eligible','is_festival_sale']:
        if bcol in df.columns:
            print(f"Normalizing boolean: {bcol}")
            df[bcol] = df[bcol].apply(normalize_bool)

    # Question 6: categories
    if 'category' in df.columns:
        print("Normalizing category...")
        df['category_orig'] = df['category']
        df['category'] = df['category'].apply(normalize_category)

    # Question 7: delivery_days
    if 'delivery_days' in df.columns:
        print("Cleaning delivery_days...")
        df['delivery_days_orig'] = df['delivery_days']
        df['delivery_days'] = df['delivery_days'].apply(clean_delivery_days)

    # Question 8: duplicates
    print("Flagging & handling duplicates...")
    df_after_dups, duplicates_df = flag_and_handle_duplicates(df)
    # keep df_after_dups as df for continued processing
    df = df_after_dups

    # Question 9: price outliers correction
    if 'original_price_inr' in df.columns:
        print("Detecting and attempting to fix price outliers...")
        df, outliers_df = fix_price_outliers(df, 'original_price_inr')
    else:
        outliers_df = pd.DataFrame()

    # Question 10: payment methods normalization
    if 'payment_method' in df.columns:
        print("Normalizing payment_method...")
        df['payment_method_orig'] = df['payment_method']
        df['payment_method'] = df['payment_method'].apply(normalize_payment_method)

    # Final housekeeping: ensure types, reorder, save
    # Convert order_date to ISO-format string for CSV consistency
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
    print("Writing cleaned CSV:", output_csv)
    df.to_csv(output_csv, index=False)

    if dups_out:
        print("Writing duplicates review CSV:", dups_out)
        # if duplicates_df empty, write empty df with columns
        duplicates_df.to_csv(dups_out, index=False)
    # Optionally write outliers details
    if not outliers_df.empty:
        outliers_out = dups_out.replace('.csv','_price_outliers.csv') if dups_out else 'price_outliers.csv'
        print("Writing price outliers log:", outliers_out)
        outliers_df.to_csv(outliers_out, index=False)

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/amazon_india_complete_2015_2025.csv")
    parser.add_argument("--output", default="data/processed/cleaned_amazon_india.csv")
    parser.add_argument('--dups_out', default='data_cleaned/duplicates_review.csv', help='Path to write duplicates flagged for review')
    args = parser.parse_args()
    run(args.input, args.output, args.dups_out)
