# -----------------------------------------------
# Data Extraction for Frontend Filters
# -----------------------------------------------

import pandas as pd
import json

# Load dataset
df = pd.read_csv("pharma_preprocessed.csv")

# --- Clean Columns ---
df.columns = df.columns.str.strip().str.lower()

# --- 1. Country → City Mapping ---
country_city_map = (
    df.groupby("country")["city"]
    .unique()
    .apply(list)
    .to_dict()
)

# --- 2. City → Product Class Mapping ---
city_productclass_map = (
    df.groupby("city")["product_class"]
    .unique()
    .apply(list)
    .to_dict()
)

# --- 3. Product Class → Medicines Mapping ---
productclass_product_map = (
    df.groupby("product_class")["product_name"]
    .unique()
    .apply(list)
    .to_dict()
)

# --- Save JSON Files ---
with open("country_city_map.json", "w") as f:
    json.dump(country_city_map, f, indent=4)

with open("city_productclass_map.json", "w") as f:
    json.dump(city_productclass_map, f, indent=4)

with open("productclass_product_map.json", "w") as f:
    json.dump(productclass_product_map, f, indent=4)

print("✅ Data extraction complete! Files created:")
print("1️⃣ country_city_map.json")
print("2️⃣ city_productclass_map.json")
print("3️⃣ productclass_product_map.json")
