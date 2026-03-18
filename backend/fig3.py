import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("pharma_preprocessed.csv")

# Keep original city column
df_original = df.copy()

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder, StandardScaler
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -------------------------------
# Prepare data
# -------------------------------
X = df.drop(columns=['sales'])
y = df['sales']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, df.index, test_size=0.2, random_state=42
)

# -------------------------------
# Train best model (Random Forest)
# -------------------------------
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# -------------------------------
# Attach city info back
# -------------------------------
df_test = df_original.loc[idx_test].copy()
df_test["actual"] = y_test.values
df_test["predicted"] = y_pred

# -------------------------------
# Compute city-wise metrics
# -------------------------------
city_results = []

for city, group in df_test.groupby("city"):
    y_true = group["actual"]
    y_p = group["predicted"]

    if len(group) < 20:
        continue  # skip very small samples

    mae = mean_absolute_error(y_true, y_p)
    rmse = np.sqrt(mean_squared_error(y_true, y_p))
    r2 = r2_score(y_true, y_p)

    city_results.append([
        city,
        round(y_true.mean(), 2),
        round(y_p.mean(), 2),
        round(mae, 2),
        round(rmse, 2),
        round(r2, 3)
    ])

city_df = pd.DataFrame(city_results, columns=[
    "City", "Avg Actual", "Avg Predicted", "MAE", "RMSE", "R2"
])

# Sort by best R2
city_df = city_df.sort_values(by="R2", ascending=False)

# Show top 10 cities
print("\n📊 Table 11: Regional Prediction Accuracy\n")
print(city_df.head(10))