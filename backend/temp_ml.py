
import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from category_encoders.target_encoder import TargetEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)

MODEL_DIR = Path("temp_ml_model")
MODEL_DIR.mkdir(exist_ok=True)

print("⚙️  Loading dataset pharma_preprocessed.csv")
df = pd.read_csv("pharma_preprocessed.csv")

df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
df = df.dropna(subset=['sales']).reset_index(drop=True)

if df['month'].dtype == object:
    month_map = {
        'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
        'July':7,'August':8,'September':9,'October':10,'November':11,'December':12
    }
    df['month'] = df['month'].map(lambda x: month_map.get(str(x).strip(), x))
df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
df = df.dropna(subset=['month']).reset_index(drop=True)
df['month'] = df['month'].astype(int)

if 'year' not in df.columns:
    df['year'] = 0
else:
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

def get_season(m):
    if m in [12,1,2]: return 'Winter'
    if m in [3,4,5]: return 'Spring'
    if m in [6,7,8]: return 'Summer'
    return 'Autumn'
if 'season' not in df.columns:
    df['season'] = df['month'].apply(get_season)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

df['productclass_month'] = df['product_class'].astype(str) + "_" + df['month'].astype(str)

# Safe numeric conversions
if 'quantity' in df.columns:
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['quantity'].fillna(df['quantity'].median(), inplace=True)
    df['quantity'] = np.log1p(df['quantity'].clip(lower=0))
else:
    df['quantity'] = 0.0

if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(df['price'].median())
else:
    df['price'] = 0.0

df = df.sort_values(['city', 'product_class', 'year', 'month']).reset_index(drop=True)

df['pc_sales_lag1'] = (
    df.groupby(['city', 'product_class'])['sales']
      .shift(1)
)

df['pc_sales_roll3'] = (
    df.groupby(['city', 'product_class'])['sales']
      .apply(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
      .reset_index(level=[0, 1], drop=True)
)

group_mean = df.groupby(['city','product_class'])['sales'].transform('mean')
global_mean = df['sales'].mean()
df['pc_sales_lag1'] = df['pc_sales_lag1'].fillna(group_mean).fillna(global_mean)
df['pc_sales_roll3'] = df['pc_sales_roll3'].fillna(group_mean).fillna(global_mean)

df['pc_sales_lag1_log'] = np.log1p(df['pc_sales_lag1'].clip(lower=0))
df['pc_sales_roll3_log'] = np.log1p(df['pc_sales_roll3'].clip(lower=0))

print("📊 Computing historical totals and product shares...")
hist_tot = df.groupby(['city','product_class','month'])['sales'].sum().reset_index(name='total_sales')
hist_tot_avg = hist_tot.groupby(['city','product_class','month'])['total_sales'].mean().reset_index()

historical_totals = {}
for _, r in hist_tot_avg.iterrows():
    city = str(r['city'])
    pc = str(r['product_class'])
    m = int(r['month'])
    val = float(r['total_sales'])
    historical_totals.setdefault(city, {}).setdefault(pc, {})[m] = val

prod_month = df.groupby(['city','product_class','month','product_name'])['sales'].sum().reset_index()
class_month = prod_month.groupby(['city','product_class','month'])['sales'].sum().reset_index(name='class_total')
prod_month = prod_month.merge(class_month, on=['city','product_class','month'], how='left')
prod_month['share'] = prod_month['sales'] / prod_month['class_total']
prod_share = prod_month.groupby(['city','product_class','product_name'])['share'].mean().reset_index()

product_shares = {}
for _, r in prod_share.iterrows():
    city, pc, pname, s = str(r['city']), str(r['product_class']), str(r['product_name']), float(r['share'])
    product_shares.setdefault(city, {}).setdefault(pc, {})[pname] = s

mf = (
    df.groupby(['city','product_class','month'])['sales'].mean() /
    df.groupby(['city','product_class'])['sales'].mean()
).reset_index(name='month_factor')
mf['month_factor'] = mf['month_factor'].clip(lower=0.7, upper=1.3)
mf['month_factor'] = mf.groupby(['city','product_class'])['month_factor'].transform(lambda s: s.rolling(3, min_periods=1, center=True).mean())

month_factors = {}
for _, r in mf.iterrows():
    city, pc, m, val = str(r['city']), str(r['product_class']), int(r['month']), float(r['month_factor'])
    month_factors.setdefault(city, {}).setdefault(pc, {})[m] = val

with open(MODEL_DIR / "historical_totals.json", "w") as f:
    json.dump(historical_totals, f)
with open(MODEL_DIR / "product_shares.json", "w") as f:
    json.dump(product_shares, f)
with open(MODEL_DIR / "month_factors.json", "w") as f:
    json.dump(month_factors, f)
print("✅ Saved historical_totals.json, product_shares.json, month_factors.json")

# ---------- Prepare Training ----------
df['y'] = np.log1p(df['sales'].clip(lower=0))
feature_cols = [
    'country','city','product_class','product_name','month','year',
    'quantity','price',
    'channel' if 'channel' in df.columns else 'channel',
    'sub-channel' if 'sub-channel' in df.columns else 'sub-channel',
    'season','month_sin','month_cos','productclass_month',
    'pc_sales_lag1_log','pc_sales_roll3_log'
]
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0 if c in ['month','year','quantity','price','month_sin','month_cos','pc_sales_lag1_log','pc_sales_roll3_log'] else ""

X_all = df[feature_cols].copy()
y_all = df['y'].copy()

categorical_cols = X_all.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
high_card_cols = [c for c in categorical_cols if X_all[c].nunique() > 50]
low_card_cols = [c for c in categorical_cols if c not in high_card_cols]

print("🔹 High-card columns:", high_card_cols)
print("🔹 Low-card columns:", low_card_cols)
print("🔹 Numeric columns:", numeric_cols)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("low_cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_cols),
    ("high_card", TargetEncoder(), high_card_cols)
], remainder='drop')

rf = RandomForestRegressor(
    n_estimators=250,
    max_depth=14,
    min_samples_split=5,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)
pipeline = Pipeline([("preproc", preprocessor), ("rf", rf)])

# ---------- Train/Test Split ----------
df_for_split = df.sort_values(['year','month']).reset_index(drop=True)
split_idx = int(len(df_for_split) * 0.8)
train_idx = df_for_split.index[:split_idx]
test_idx = df_for_split.index[split_idx:]
X_train, y_train = X_all.loc[train_idx], y_all.loc[train_idx]
X_test, y_test = X_all.loc[test_idx], y_all.loc[test_idx]

print(f"📈 Training rows: {len(X_train)}, Testing rows: {len(X_test)}")
print("🚀 Training RandomForestRegressor...")
pipeline.fit(X_train, y_train)

# ---------- Evaluation ----------
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log).clip(min=0.0)
y_true = np.expm1(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
print(f"✅ Eval (original scale): MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

# ---------- Save Model ----------
joblib.dump(pipeline, MODEL_DIR / "rf_pipeline.joblib")
joblib.dump({
    "feature_columns": feature_cols,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "high_card_cols": high_card_cols,
    "low_card_cols": low_card_cols
}, MODEL_DIR / "rf_metadata.joblib")

# City × product_class average sales
city_pc_avg = (
    df.groupby(['city','product_class'])['sales'].mean()
    .reset_index()
    .pivot(index='city', columns='product_class', values='sales')
    .fillna(0)
    .to_dict()
)
with open(MODEL_DIR / "city_productclass_avg_sales.json", "w") as f:
    json.dump(city_pc_avg, f)

print("💾 Model, metadata and artifacts saved in temp_ml_model/")
