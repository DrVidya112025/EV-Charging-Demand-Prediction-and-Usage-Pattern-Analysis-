import json, os
import numpy as np, pandas as pd
import streamlit as st
from xgboost import XGBRegressor, XGBClassifier

st.set_page_config(page_title="EV Charging Demand", page_icon="⚡", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)


@st.cache_resource
def load():
    meta = json.load(open(P("models", "metadata.json")))
    reg = XGBRegressor(); reg.load_model(P("models", "regressor_xgboost.json"))
    clf = XGBClassifier(); clf.load_model(P("models", "classifier_xgboost.json"))
    return meta, reg, clf


@st.cache_data
def load_sample():
    f = P("data", "sample_sessions.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


try:
    meta, reg, clf = load()
except Exception as e:
    st.error("Model files not found. Run `python train_models.py --data <folder with train/val/test.csv>` first.")
    st.exception(e)
    st.stop()

FEATURES, THR, D = meta["features"], meta["threshold_kwh"], meta["train_defaults"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def tod_band(h):
    return "Night" if h <= 5 else "Morning" if h <= 11 else "Afternoon" if h <= 17 else "Evening"


def build_row(date, time, location, ctype, company, h):
    hour, dow = time.hour, date.weekday()
    row = dict.fromkeys(FEATURES, 0.0)
    row.update({"Hour": hour, "Day": date.day, "DayOfWeek": dow, "IsWeekend": int(dow >= 5),
                "Hour_sin": np.sin(2 * np.pi * hour / 24), "Hour_cos": np.cos(2 * np.pi * hour / 24),
                "DoW_sin": np.sin(2 * np.pi * dow / 7), "DoW_cos": np.cos(2 * np.pi * dow / 7),
                "ChargerType": ctype, "ChargerCompany": company, **h})
    row[f"Location_{location}"] = 1.0
    row[f"TimeOfDay_{tod_band(hour)}"] = 1.0
    return pd.DataFrame([row])[FEATURES]


page = st.sidebar.radio("Page", ["Predict a session", "Model performance", "Explore data", "About"])
st.sidebar.caption("Forecast made at plug-in: session length is not used.")

# ------------------------------------------------------------------ Predict
if page == "Predict a session":
    st.title("⚡ Predict EV charging demand")
    st.write("Enter what is known **when the session starts**. The model returns the expected energy (kWh) and whether it is likely to be a high-demand session.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Session")
        date = st.date_input("Start date", value=pd.Timestamp("2022-09-15"))
        time = st.time_input("Start time", value=pd.Timestamp("2022-09-15 18:00").time())
        location = st.selectbox("Location", sorted(meta["locations"]), index=sorted(meta["locations"]).index("public area") if "public area" in meta["locations"] else 0)
        ctype = st.radio("Charger type (code)", [0, 1], horizontal=True, help="Type 1 sessions are much shorter and deliver more energy per minute.")
        company = st.radio("Charger company (code)", [0, 1], horizontal=True)
    with c2:
        st.subheader("Charger history")
        has_hist = st.toggle("This charger has earlier sessions", value=True)
        if has_hist:
            n_prior = st.number_input("Earlier sessions on this charger", 0, 2000, int(D["hist_n_prior"]))
            mean_prior = st.slider("Mean demand of earlier sessions (kWh)", 0.0, 60.0, round(D["hist_mean_demand_prior"], 1), 0.1)
            n7 = st.number_input("Sessions that ended in the last 7 days", 0, 100, int(D["hist_n_7d"]))
            mean7 = st.slider("Mean demand of those sessions (kWh)", 0.0, 60.0, round(D["hist_mean_demand_7d"], 1), 0.1)
            util = st.slider("Share of last 24 h the charger was plugged in", 0.0, 1.0, round(D["hist_util_24h"], 2), 0.01)
            since = st.slider("Hours since the last session ended", 0.0, 720.0, round(min(D["hist_hours_since_last"], 720), 1), 0.5)
            h = {"hist_n_prior": n_prior, "hist_mean_demand_prior": mean_prior, "hist_n_7d": n7, "hist_mean_demand_7d": mean7 if n7 > 0 else meta["train_mean_demand"],
                 "hist_util_24h": util, "hist_hours_since_last": since, "has_history": 1}
        else:
            st.info("New charger: history values are filled the same way as in training (training mean demand, 720 hours).")
            h = {"hist_n_prior": 0, "hist_mean_demand_prior": meta["train_mean_demand"], "hist_n_7d": 0, "hist_mean_demand_7d": meta["train_mean_demand"],
                 "hist_util_24h": 0.0, "hist_hours_since_last": 720.0, "has_history": 0}

    X = build_row(date, time, location, ctype, company, h)
    pred = float(np.clip(reg.predict(X)[0], 0, None))
    p_high = float(clf.predict_proba(X)[0, 1])
    mae = meta["metrics"]["test"]["reg_MAE"]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected demand", f"{pred:.1f} kWh")
    m2.metric(f"P(high demand, ≥ {THR:.1f} kWh)", f"{p_high:.0%}")
    m3.metric("Typical error (test MAE)", f"± {mae:.1f} kWh")
    st.progress(min(max(p_high, 0.0), 1.0))
    st.success("Likely a HIGH-demand session") if p_high >= 0.5 else st.info("Likely a NORMAL-demand session")
    st.caption(f"Range to expect: about {max(pred - mae, 0):.1f} to {pred + mae:.1f} kWh. Test R² is {meta['metrics']['test']['reg_R2']:.2f}, so treat this as a rough forecast, not an exact figure.")
    with st.expander("Model inputs"):
        st.dataframe(X.T.rename(columns={0: "value"}))

# ------------------------------------------------------------------ Performance
elif page == "Model performance":
    st.title("Model performance")
    m = meta["metrics"]
    st.write("Models were fitted on the training period, compared on validation, and scored once on the test period (chronological split).")
    st.dataframe(pd.DataFrame({
        "Regression MAE (kWh)": [m["val"]["reg_MAE"], m["test"]["reg_MAE"]], "Regression R²": [m["val"]["reg_R2"], m["test"]["reg_R2"]],
        "Classification accuracy": [m["val"]["clf_accuracy"], m["test"]["clf_accuracy"]], "F1": [m["val"]["clf_F1"], m["test"]["clf_F1"]],
        "ROC-AUC": [m["val"]["clf_ROC_AUC"], m["test"]["clf_ROC_AUC"]]}, index=["Validation", "Test"]).round(3))
    st.subheader("Regression models compared (Step 4)")
    st.dataframe(pd.DataFrame({
        "Model": ["Mean baseline", "Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"],
        "Val RMSE": [13.995, 12.229, 12.291, 12.064, 12.095, 12.034, 12.047],
        "Test MAE": [10.735, 9.129, 9.275, 9.050, 8.992, 8.966, 8.935],
        "Test R²": [-0.001, 0.225, 0.161, 0.211, 0.224, 0.219, 0.223]}).set_index("Model"))
    st.subheader("Classification models compared (Step 4)")
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM", "XGBoost", "LightGBM"],
        "Val ROC-AUC": [0.757, 0.759, 0.778, 0.741, 0.767, 0.777, 0.775],
        "Test F1": [0.677, 0.685, 0.690, 0.666, 0.692, 0.698, 0.698],
        "Test ROC-AUC": [0.743, 0.747, 0.759, 0.722, 0.752, 0.759, 0.760]}).set_index("Model"))
    st.subheader("What the regressor uses most")
    imp = pd.Series(reg.feature_importances_, index=FEATURES).sort_values(ascending=False).head(12)
    st.bar_chart(imp)
    st.caption("Why R² is about 0.22: the biggest driver of demand, session length, is unknown at plug-in and is excluded on purpose. With Duration the cleaned-data model reaches R² 0.81, but that is not a forecast.")

# ------------------------------------------------------------------ Explore
elif page == "Explore data":
    st.title("Explore charging sessions")
    df = load_sample()
    up = st.file_uploader("Optional: upload ChargingRecords_preprocessed.csv", type="csv")
    if up is not None:
        df = pd.read_csv(up)
    if df is None:
        st.warning("No sample data found. Run train_models.py with --raw, or upload the cleaned CSV above.")
        st.stop()
    st.caption(f"{len(df):,} sessions shown" + (" (random sample)" if up is None else ""))
    locs = st.multiselect("Location", sorted(df["Location"].unique()), default=[])
    d = df[df["Location"].isin(locs)] if locs else df
    a, b = st.columns(2)
    with a:
        st.subheader("Sessions by start hour")
        st.bar_chart(d["StartHour"].value_counts().sort_index())
        st.subheader("Mean demand by location (kWh)")
        st.bar_chart(d.groupby("Location")["Demand"].mean().sort_values())
    with b:
        st.subheader("Mean demand by start hour (kWh)")
        st.line_chart(d.groupby("StartHour")["Demand"].mean())
        st.subheader("Demand vs Duration")
        st.scatter_chart(d.sample(min(3000, len(d)), random_state=1), x="Duration", y="Demand", color="ChargerType")
    st.dataframe(d.describe().T.round(2))

# ------------------------------------------------------------------ About
else:
    st.title("About")
    st.markdown(f"""
**Goal:** predict the energy demand (kWh) of an EV charging session at the moment it starts.

**Data:** about 72,000 cleaned sessions, 30 Sep 2021 to 30 Sep 2022.

**Models:** XGBoost regressor and XGBoost classifier (high demand = at least {THR:.2f} kWh, the training median).
Inputs are time features, location, charger type and company, and the charger's own history (earlier sessions, recent demand, utilisation).

**Leakage control:** session length (`Duration`) and end-time fields are not used. History features use only sessions that had already ended.

**Limits:** one year of data, so seasons are not learned. Test R² is about 0.22 and MAE about 9 kWh. UserID is not used.
""")
