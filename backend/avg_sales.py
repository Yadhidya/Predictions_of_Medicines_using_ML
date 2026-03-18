# compute_city_avg_sales.py

import pandas as pd
import json

# Load your dataset
df = pd.read_csv("pharma_preprocessed.csv")

# Ensure necessary columns exist
required_cols = {"city", "product_class", "sales"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns. Required: {required_cols}")

# Group by city and product class to get average sales
avg_sales = (
    df.groupby(["city", "product_class"])["sales"]
    .mean()
    .reset_index()
    .rename(columns={"sales": "avg_sales"})
)

# Convert to nested JSON structure:
# { "Chennai": {"Antibiotics": 923.5, "Painkillers": 789.2, ...}, "Mumbai": {...}, ... }
city_avg_map = {}
for _, row in avg_sales.iterrows():
    city = str(row["city"])
    pclass = str(row["product_class"])
    avg = float(row["avg_sales"])
    if city not in city_avg_map:
        city_avg_map[city] = {}
    city_avg_map[city][pclass] = avg

# Save the JSON file
with open("city_productclass_avg_sales.json", "w") as f:
    json.dump(city_avg_map, f, indent=4)

print("✅ Saved city-wise average sales for each product class → city_productclass_avg_sales.json")
