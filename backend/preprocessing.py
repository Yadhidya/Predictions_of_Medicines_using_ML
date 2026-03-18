# -----------------------------------------------
# PHASE 1 : DATA PREPROCESSING & FEATURE ENGINEERING
# Project: Predictive Modelling of Drug Consumption Pattern
# -----------------------------------------------

import pandas as pd
import numpy as np

# Load the dataset
file_path = "pharma-data.csv"   # Change path if necessary
df = pd.read_csv(file_path)

# -----------------------------
# STEP 1: BASIC INSPECTION
# -----------------------------
print("Shape of dataset:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing Values:\n", df.isnull().sum())

# Remove duplicate rows (if any)
df.drop_duplicates(inplace=True)

# -----------------------------
# STEP 2: CLEAN COLUMN NAMES
# -----------------------------
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# -----------------------------
# STEP 3: CREATE 'DATE' COLUMN
# -----------------------------
# Combine Year + Month → datetime (set day=1)
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")

# -----------------------------
# STEP 4: ADD 'SEASON' COLUMN
# -----------------------------
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df["season"] = df["date"].dt.month.apply(get_season)

def simplify_class(x):
    x = x.lower()
    if "antibiotic" in x:
        return "Infection"
    elif "analgesic" in x or "pain" in x:
        return "Pain/Fever"
    elif "antihistamine" in x or "cold" in x or "allergy" in x:
        return "Allergy/Cold"
    elif "antidepressant" in x or "mood" in x:
        return "Mental Health"
    else:
        return "Other"

df["disease_category"] = df["product_class"].apply(simplify_class)

df["city_class_key"] = df["city"] + "_" + df["product_class"]

drop_cols = ["manager", "sales_rep", "distributor", "customer_name", "sales_team"]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

numeric_cols = ["quantity", "price", "sales"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nFinal Columns:", list(df.columns))
print("\nData Types:\n", df.dtypes)
print("\nSample Data:\n", df.head(5))

df.to_csv("pharma_preprocessed.csv", index=False)
print("\n✅ Preprocessing complete! File saved as 'pharma_preprocessed.csv'")
