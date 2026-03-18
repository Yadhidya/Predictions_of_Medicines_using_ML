import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

MODEL_DIR = "temp_ml_model"

# Load trained pipeline
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_pipeline.joblib"))
metadata = joblib.load(os.path.join(MODEL_DIR, "rf_metadata.joblib"))

feature_columns = metadata["feature_columns"]
target_column = "sales"   # change to "quantity" if your model predicts quantity

# Load dataset
df = pd.read_csv("pharma_preprocessed.csv")

# Prepare features and target
X = df[feature_columns]
y_actual = df[target_column]

# Predictions
y_pred = rf_model.predict(X)

# Scatter plot
plt.figure(figsize=(7,6))
plt.scatter(y_actual, y_pred, alpha=0.5)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Figure 11. Predicted vs Actual Drug Demand")

# Ideal line
min_val = min(y_actual.min(), y_pred.min())
max_val = max(y_actual.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.tight_layout()
plt.show()