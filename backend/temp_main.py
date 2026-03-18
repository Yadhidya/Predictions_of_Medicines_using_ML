# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, os, json, numpy as np, pandas as pd, uvicorn

app = FastAPI(title="Drug Consumption Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = "temp_ml_model"
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_pipeline.joblib"))
metadata = joblib.load(os.path.join(MODEL_DIR, "rf_metadata.joblib"))
feature_columns = metadata["feature_columns"]

with open("productclass_product_map.json", "r") as f:
    product_class_map = json.load(f)
with open("city_productclass_avg_sales.json", "r") as f:
    city_avg_sales = json.load(f)

# loads created by temp_ml.py
with open(os.path.join(MODEL_DIR, "historical_totals.json"), "r") as f:
    historical_totals = json.load(f)
with open(os.path.join(MODEL_DIR, "product_shares.json"), "r") as f:
    product_shares = json.load(f)
with open(os.path.join(MODEL_DIR, "month_factors.json"), "r") as f:
    month_factors = json.load(f)

DISEASE_MAP = {
    "Analgesics": "Pain-related Disorders",
    "Antibiotics": "Bacterial Infections",
    "Antimalarial": "Parasitic Diseases (Malaria)",
    "Antipiretics": "Fever-related Illnesses",
    "Antiseptics": "Infection Prevention & Wound Care",
    "Mood Stabilizers": "Mental Health Disorders"
}
MONTH_TO_SEASON = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Summer",6:"Summer",7:"Monsoon",8:"Monsoon",9:"Autumn",10:"Autumn",11:"Winter",12:"Winter"}

class SalesRequest(BaseModel):
    country: str
    city: str
    product_class: str
    month: int
    year: int

def cyclic_month(month:int):
    return np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)

def get_month_factor(city, pc, month):
    try:
        return float(month_factors.get(city, {}).get(pc, {}).get(str(month), 1.0))
    except:
        return 1.0

def get_historical_total(city, pc, month):
    # fallback to city-product_class overall avg if exact month isn't present
    try:
        t = historical_totals.get(city, {}).get(pc, {}).get(str(month))
        if t is not None:
            return float(t)
        # fallback: average across months for this city/pc
        d = historical_totals.get(city, {}).get(pc, {})
        if d:
            return float(np.mean(list(d.values())))
    except:
        pass
    # ultimate fallback: city avg from other file or 0
    try:
        return float(city_avg_sales.get(city, {}).get(pc, 0) or 0)
    except:
        return 0.0

def get_product_share(city, pc, product_name, default_share=None):
    try:
        s = product_shares.get(city, {}).get(pc, {}).get(product_name)
        if s is not None:
            return float(s)
    except:
        pass
    # fallback to provided default or equal share sentinel
    return float(default_share) if default_share is not None else None

def prepare_input_row_for_product(medicine:str, req:SalesRequest, month_override=None):
    month = month_override if month_override is not None else req.month
    month_sin, month_cos = cyclic_month(month)
    season = MONTH_TO_SEASON.get(month, "Winter")
    data = {
        "country": req.country,
        "city": req.city,
        "product_class": req.product_class,
        "product_name": medicine,
        "month": month,
        "year": req.year,
        "quantity": np.log1p(10),
        "price": 100,
        "channel": "Retail",
        "sub-channel": "Pharmacy",
        "season": season,
        "disease_category": DISEASE_MAP.get(req.product_class, "General"),
        "city_class_key": "Urban",
        "month_sin": month_sin,
        "month_cos": month_cos,
        # placeholders; model expects these numeric fields; they will be overridden by historical-informed features later if needed
        "pc_sales_lag1": 0.0,
        "pc_sales_roll3": 0.0,
        "productclass_month": f"{req.product_class}_{month}"
    }
    df = pd.DataFrame([data])
    # ensure ordering & missing columns
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

@app.post("/predict_sales")
def predict_sales(req: SalesRequest):
    try:
        if req.product_class not in product_class_map:
            raise HTTPException(status_code=400, detail="Invalid product class.")
        meds = product_class_map[req.product_class]
        # 1) produce raw model preds for each product (using historical lag as pc_sales_lag1/roll3 if available)
        raw_preds = []
        for med in meds:
            # prepare input row
            X = prepare_input_row_for_product(med, req)
            # if hist lag available, set those numeric columns
            hist_total_for_month = get_historical_total(req.city, req.product_class, req.month)
            # Use historical per-product share to infer lag/roll if available:
            share = get_product_share(req.city, req.product_class, med, default_share=None)
            if share is not None and hist_total_for_month>0:
                inferred_lag = hist_total_for_month * share
                inferred_roll = hist_total_for_month * share
                # set into X (if those columns exist)
                if 'pc_sales_lag1' in X.columns:
                    X['pc_sales_lag1'] = inferred_lag
                if 'pc_sales_roll3' in X.columns:
                    X['pc_sales_roll3'] = inferred_roll
            # predict
            val = float(rf_model.predict(X)[0])
            raw_preds.append((med, val))
        # 2) aggregate raw model sum
        sum_raw = sum([v for (_,v) in raw_preds]) or 1.0
        # 3) get historical total for this city/pc/month
        hist_total = get_historical_total(req.city, req.product_class, req.month)
        # apply month_factor blending (conservative)
        month_factor = get_month_factor(req.city, req.product_class, req.month)
        # we will compute final_total = hist_total * month_factor_blend where blend is mild
        # but to avoid exploding, we blend: final_total = hist_total * (0.9 + 0.1*month_factor)
        final_total = hist_total * (0.9 + 0.1 * month_factor)
        # if hist_total is zero (no history), fallback to sum_raw (prevent zeroing)
        if final_total <= 0:
            final_total = sum_raw
        # 4) scale per-product raw preds proportionally to match final_total
        scaled_preds = []
        for med, raw in raw_preds:
            share = raw / sum_raw
            allocated = final_total * share
            scaled_preds.append((med, float(round(max(0.0, allocated),2))))
        total_predicted_sales = float(round(sum([v for (_,v) in scaled_preds]),2))
        average_sales = float(round(total_predicted_sales / max(1,len(scaled_preds)),2))
        city_data = city_avg_sales.get(req.city, {})
        city_avg = float(city_data.get(req.product_class, 0) or 0)
        ratio = average_sales / (city_avg + 1e-6)
        if ratio > 1.3:
            outbreak = "🚨 High Risk of Outbreak"
        elif ratio > 1.1:
            outbreak = "⚠️ Moderate Risk"
        else:
            outbreak = "🟢 Low Risk"
        preds_out = [{"product_name":m, "predicted_sales":v} for (m,v) in scaled_preds]
        return {
            "country": req.country,
            "city": req.city,
            "product_class": req.product_class,
            "disease_category": DISEASE_MAP.get(req.product_class, "General"),
            "month": req.month,
            "year": req.year,
            "total_predicted_sales": total_predicted_sales,
            "average_sales_per_product": average_sales,
            "historical_city_average": round(city_avg,2),
            "outbreak_alert": outbreak,
            "predictions": preds_out
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/seasonality_analysis")
def seasonality_analysis(req: SalesRequest):
    try:
        if req.product_class not in product_class_map:
            raise HTTPException(status_code=400, detail="Invalid product class.")

        city_pc_data = historical_totals.get(req.city, {}).get(req.product_class, {})

        if not city_pc_data:
            raise HTTPException(
                status_code=400,
                detail="Insufficient historical data for seasonality analysis."
            )

        base_monthly_avg = float(np.mean(list(city_pc_data.values())))
        BASE_YEAR = 2017
        ANNUAL_GROWTH = 0.02 

        year_diff = max(0, req.year - BASE_YEAR)
        year_factor = (1 + ANNUAL_GROWTH) ** year_diff

        results = []

        for month in range(1, 13):
            season = MONTH_TO_SEASON.get(month, "Winter")

            month_factor = (
                month_factors
                .get(req.city, {})
                .get(req.product_class, {})
                .get(str(month), 1.0)
            )

            predicted_sales = base_monthly_avg * month_factor * year_factor

            results.append({
                "month": month,
                "season": season,
                "predicted_sales": round(predicted_sales, 2)
            })

        return {
            "country": req.country,
            "city": req.city,
            "product_class": req.product_class,
            "disease_category": DISEASE_MAP.get(req.product_class),
            "year": req.year,
            "seasonality": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
