# -----------------------------------------------
# PHASE 2 : EXPLORATORY DATA ANALYSIS (EDA)
# Project: Predictive Modelling of Drug Consumption Pattern
# -----------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Configuration for visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the preprocessed data
df = pd.read_csv("pharma_preprocessed.csv")

# Ensure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'])


# =======================================================
# 1. Total Consumption Trend Over Time (High-Level)
# =======================================================
print("--- 1. Total Consumption Trend ---")

# Aggregate total quantity sold monthly
monthly_sales = df.groupby('date')['quantity'].sum().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_sales, x='date', y='quantity', color='darkorange', linewidth=2)
plt.title('Total Drug Quantity Sold Over Time (2018-2020)', fontsize=16)
plt.xlabel('Date (Monthly)', fontsize=12)
plt.ylabel('Total Quantity Sold', fontsize=12)
plt.show()


# =======================================================
# 2. Disease Category Seasonality (The Core Insight)
# =======================================================
print("\n--- 2. Disease Category Seasonality ---")

# Aggregate quantity by disease category and month
seasonal_disease = df.groupby([df['date'].dt.to_period('M'), 'disease_category'])['quantity'].sum().reset_index()
seasonal_disease['date'] = seasonal_disease['date'].dt.to_timestamp()

plt.figure(figsize=(16, 8))
sns.lineplot(
    data=seasonal_disease,
    x='date',
    y='quantity',
    hue='disease_category',
    palette='Spectral',
    linewidth=2
)
plt.title('Monthly Consumption Trends by Inferred Disease Category', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Quantity Sold', fontsize=12)
plt.legend(title='Disease Category')
plt.grid(True)
plt.show()

# Insight: Which months see the highest sales for each category?
monthly_avg_disease = df.groupby([df['date'].dt.month, 'disease_category'])['quantity'].sum().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(
    monthly_avg_disease.T,
    cmap="YlGnBu",
    annot=True,
    fmt=".0f",
    linewidths=.5,
    cbar_kws={'label': 'Total Quantity Sold'}
)
plt.title('Heatmap: Seasonality of Disease Categories (Quantity Sold)', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Disease Category', fontsize=12)
plt.show()


# =======================================================
# 3. Geographic Hotspot Analysis
# =======================================================
print("\n--- 3. Geographic Hotspot Analysis ---")

# Calculate average sales for each city
city_avg_sales = df.groupby(['city', 'latitude', 'longitude'])['sales'].sum().reset_index()
city_avg_sales.rename(columns={'sales': 'Total_Sales_Value'}, inplace=True)

# Create a base map (focused on Poland, as the data suggested)
center_lat, center_lon = 52.0, 19.0 # Center of Poland
m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

# Add circles to the map based on total sales value
max_radius = city_avg_sales['Total_Sales_Value'].max()

for index, row in city_avg_sales.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['Total_Sales_Value'] / (max_radius / 25), # Scale radius for visualization
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        tooltip=f"City: {row['city']}<br>Sales: {row['Total_Sales_Value']:,.0f}"
    ).add_to(m)

# NOTE: Since Folium maps cannot be displayed directly in text output,
# we save it as HTML. You will need to open this file in a browser.
map_file = "city_sales_map.html"
m.save(map_file)
print(f"✅ Geographic heatmap saved as: '{map_file}' (Open in browser)")


# =======================================================
# 4. Correlation Check
# =======================================================
print("\n--- 4. Correlation Check ---")

# Focus on numerical correlations
correlation_df = df[['quantity', 'price', 'sales']].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Quantity, Price, Sales)', fontsize=14)
plt.show()

print("\n✅ EDA Phase 2 Complete. Ready for Modeling.")