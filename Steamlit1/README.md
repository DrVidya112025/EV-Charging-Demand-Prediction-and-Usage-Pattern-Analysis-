# EV Charging Demand App (Streamlit)

Predicts the energy demand (kWh) of a charging session at plug-in time, shows model performance, and lets you explore the sessions.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app (Predict, Model performance, Explore data, About) |
| `train_models.py` | Trains the final models from `train.csv`, `val.csv`, `test.csv` and saves them |
| `models/` | `regressor_xgboost.json`, `classifier_xgboost.json`, `metadata.json` |
| `data/sample_sessions.csv` | Sample of cleaned sessions for the Explore page (optional) |
| `requirements.txt` | Python packages |

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain (optional)
```bash
python train_models.py --data "D:\HCL_PROJECTS\EV_Charging_Demand_Analysis" --raw "D:\HCL_PROJECTS\EV_Charging_Demand_Analysis\ChargingRecords_preprocessed.csv"
```
Models are saved in XGBoost's native JSON format, so they load across xgboost versions. If a model file ever fails to load, rerun this command.

## Deploy
Push this folder to GitHub and deploy on Streamlit Community Cloud with `app.py` as the main file.
