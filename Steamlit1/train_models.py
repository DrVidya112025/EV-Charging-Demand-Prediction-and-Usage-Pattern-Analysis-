"""Train the final models from the Step 3 files and save them for the Streamlit app.

Usage:  python train_models.py --data "D:\\HCL_PROJECTS\\EV_Charging_Demand_Analysis"
Reads train.csv, val.csv, test.csv; writes models/*.json, models/metadata.json, data/sample_sessions.csv
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

RS = 42
DROP = ["Month", "Month_sin", "Month_cos"]   # one year of data: month adds nothing (Step 3 / Step 4)
TARGET = "Demand"

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=".", help="folder with train.csv, val.csv, test.csv")
ap.add_argument("--raw", default=None, help="optional path to ChargingRecords_preprocessed.csv (for the Explore page sample)")
a = ap.parse_args()

train, val, test = [pd.read_csv(os.path.join(a.data, f"{n}.csv")) for n in ("train", "val", "test")]
FEATURES = [c for c in train.columns if c != TARGET and c not in DROP]
X = {k: d[FEATURES] for k, d in zip(("train", "val", "test"), (train, val, test))}
y = {k: d[TARGET] for k, d in zip(("train", "val", "test"), (train, val, test))}
THR = float(y["train"].median())
yc = {k: (v >= THR).astype(int) for k, v in y.items()}

reg = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=.8, colsample_bytree=.8, n_jobs=-1, random_state=RS)
reg.fit(X["train"], y["train"])
clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=.8, colsample_bytree=.8, n_jobs=-1, random_state=RS, eval_metric='logloss')
clf.fit(X["train"], yc["train"])

metrics = {}
for k in ("val", "test"):
    p, s = reg.predict(X[k]), clf.predict_proba(X[k])[:, 1]
    metrics[k] = {"reg_MAE": mean_absolute_error(y[k], p), "reg_R2": r2_score(y[k], p),
                  "clf_accuracy": accuracy_score(yc[k], s > .5), "clf_F1": f1_score(yc[k], s > .5), "clf_ROC_AUC": roc_auc_score(yc[k], s)}
    print(k, {m: round(v, 3) for m, v in metrics[k].items()})

hist = [c for c in FEATURES if c.startswith("hist_")]
meta = {
    "features": FEATURES, "threshold_kwh": THR, "metrics": metrics,
    "locations": [c.replace("Location_", "") for c in FEATURES if c.startswith("Location_")],
    "train_defaults": {c: float(train[c].median()) for c in hist},
    "train_mean_demand": float(y["train"].mean()),
    "hist_ranges": {c: [float(train[c].min()), float(train[c].quantile(.99))] for c in hist},
}
os.makedirs("models", exist_ok=True); os.makedirs("data", exist_ok=True)
# XGBoost native JSON format: works across xgboost versions (unlike pickle/joblib)
reg.save_model("models/regressor_xgboost.json")
clf.save_model("models/classifier_xgboost.json")
json.dump(meta, open("models/metadata.json", "w"), indent=2)

if a.raw:   # small sample of cleaned sessions for the Explore page
    raw = pd.read_csv(a.raw, usecols=["Location", "ChargerType", "ChargerCompany", "StartHour", "Weekday", "Duration", "Demand"])
    raw.sample(min(8000, len(raw)), random_state=RS).to_csv("data/sample_sessions.csv", index=False)
print("Saved models/, metadata.json", "and data/sample_sessions.csv" if a.raw else "")
