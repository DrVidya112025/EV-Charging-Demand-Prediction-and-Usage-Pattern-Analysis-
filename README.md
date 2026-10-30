⚡ EV Charging Demand Prediction and Usage Pattern Analysis
Predicting and explaining electric vehicle charging demand using Machine Learning and an interactive Streamlit dashboard.

# 📋 Overview
This project analyzes over 72,000 EV charging sessions and provides:

- Regression Model to predict charging energy demand (kWh)
- Classification Model to identify High-Demand vs Normal-Demand sessions
- Interactive Streamlit Dashboard for visualization, prediction, and model explainability
- SHAP and LIME explanations for transparent AI insights

## 🚀 Project Highlights
### Dataset
- 72,031 EV charging sessions
- Time Period: September 2021 – September 2022
## Best Regression Model: XGBoost
- MAE: 3.32
- RMSE: 5.86
- R²: 0.811
### Best Classification Model: LightGBM
-Accuracy: 90.1%
-ROC-AUC: 0.962
### Dashboard Features
### Streamlit-based interactive dashboard
### SHAP and LIME explainability
### Demand forecasting and classification

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- LIME
- Plotly
- Streamlit
- Jupyter Notebook

## 📊 Key Business Insights
- Session duration is the strongest predictor of charging demand.
- Charger type significantly affects energy consumption.
- Demand varies across locations and charger companies.
- Tree-based models outperform linear models, indicating non-linear demand patterns.
- Accommodation sites, camping sites, and bus garages show higher average charging demand.

## 🎯 Applications
- EV Infrastructure Planning
- Smart Energy Management
- Charging Station Optimization

## 📁 Project Structure

```text
EV-Charging-Demand-Prediction-and-Usage-Pattern-Analysis/
│
├── README.md
├── ChargingRecords_preprocessed.csv
├── requirements.txt
│
├── notebooks/
│   ├── EV_Charging_Demand_Regression.ipynb
│   └── EV_Charging_HighDemand_Classification.ipynb
│
├── dashboard/
│   ├── app.py
│   ├── common.py
│   └── train_models.py
│
├── models/
└── data/
```
## 👩‍💻 Author
Vidya Priya V
- Machine Learning & Data Analytics Enthusiast
