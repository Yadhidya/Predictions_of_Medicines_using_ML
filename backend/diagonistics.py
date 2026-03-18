# diagnostics.py
import joblib
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

# ---------- load model and data ----------
pipe = joblib.load("temp_ml_model/rf_pipeline.joblib")
meta = joblib.load("temp_ml_model/rf_metadata.joblib")
df = pd.read_csv("pharma_preprocessed.csv")

# normalize month column like you do in training
month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
if df['month'].dtype == object:
    df['month'] = df['month'].replace(month_map)
df['month'] = pd.to_numeric(df['month'], errors='coerce')
df = df.dropna(subset=['month'])
df['month'] = df['month'].astype(int)

# ---------- 1) check raw month-wise variance per product_class ----------
def month_stats_for_class(df, product_class):
    g = df[df['product_class']==product_class].groupby('month')['sales'].agg(['mean','std','count']).reset_index()
    print(f"\nMonth stats for product_class={product_class}")
    print(g)

# try a few classes of interest
for cls in df['product_class'].unique()[:6]:
    month_stats_for_class(df, cls)

# ---------- 2) overall month variation (global) ----------
global_month = df.groupby('month')['sales'].agg(['mean','std','count']).reset_index()
print("\nGlobal month stats:")
print(global_month)

# ---------- 3) feature importance from the trained RF inside pipeline ----------
# get rf and preproc
rf = pipe.named_steps['rf']
preproc = pipe.named_steps['preproc']

# get feature names (sklearn>=1.0)
try:
    feature_names = preproc.get_feature_names_out()
except Exception:
    # fallback: build feature names manually (approximate)
    feature_names = meta['feature_columns']

importances = rf.feature_importances_
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:50]
print("\nTop 40 feature importances (feature, importance):")
for f,i in feat_imp[:40]:
    print(f, round(i,6))

# ---------- 4) permutation importance for month & season (robust check) ----------
# build a small X_test sample
X = df.drop(columns=['sales']).sample(n=3000, random_state=42)
y = df.loc[X.index, 'sales']

# pre-transform X via pipeline.transform then run permutation on rf using transformed X
X_trans = preproc.transform(X)
perm = permutation_importance(rf, X_trans, y, n_repeats=10, random_state=42, n_jobs=4)
# map back names (order of feature_names corresponds to X_trans columns)
perm_idx_sorted = np.argsort(perm.importances_mean)[::-1]
print("\nTop 20 permutation importances (name, mean_importance):")
for idx in perm_idx_sorted[:20]:
    print(feature_names[idx], round(perm.importances_mean[idx],6))
