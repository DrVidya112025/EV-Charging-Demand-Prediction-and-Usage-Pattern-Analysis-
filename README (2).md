# ⚡ EV Charging Demand Analysis

Predicting and explaining electric-vehicle charging demand with machine learning, delivered as an interactive Streamlit dashboard.

The project has three parts:

1. **Regression**: predict the energy demand (kWh) of a charging session.
2. **Classification**: label a session as **High Demand** or **Normal Demand**.
3. **Dashboard**: a Streamlit app that combines the analysis, both models and their explanations.

## Table of contents

- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Part 1: Regression](#part-1-regression)
- [Part 2: Classification](#part-2-classification)
- [Part 3: Streamlit dashboard](#part-3-streamlit-dashboard)
- [Key findings](#key-findings)
- [Applications](#applications)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Tech stack](#tech-stack)
- [Author](#author)

## Dataset

`ChargingRecords_preprocessed.csv`: **72,031 charging sessions** from 30 Sep 2021 to 30 Sep 2022.

| Column | Description |
|---|---|
| `UserID`, `ChargerID` | Identifiers |
| `ChargerCompany` | Operator (0 / 1) |
| `ChargerType` | Charger type (0 / 1) |
| `Location` | 14 site types (apartment, public area, resort, hotel, company, ...) |
| `StartTimestamp`, `EndTimestamp` | Session start and end |
| `Duration` | Session length in minutes |
| **`Demand`** | **Energy delivered in kWh (the regression target)** |
| `StartHour`, `DayOfWeek`, `IsWeekend`, `Month`, `CrossesMidnight` | Time features |
| `ElapsedMin`, `IdleMin`, `FlagIdleOver30`, `EndHour` | Post-session fields (not used as features) |

Cleaning applied to the raw file: duplicates removed, sessions with `Duration <= 0` removed, timestamps parsed.

### Features used by the models

| Group | Features |
|---|---|
| Charging | `Duration`, `ChargerType`, `ChargerCompany` |
| Location | `Location` (one-hot encoded) |
| Time | `StartHour` (plus sin/cos encoding), `DayOfWeek`, `IsWeekend`, `Month`, overnight flag |

**Deliberately excluded:**

- `UserID`, `ChargerID`: identifiers, not physical drivers of demand.
- `ElapsedMin`, `IdleMin`, `FlagIdleOver30`, `EndHour`: only known after the session ends, and `ElapsedMin` nearly duplicates `Duration`. Including them would inflate scores unfairly.

## Project structure

```text
.
├── README.md
├── notebooks/
│   ├── EV_Charging_Demand_Regression_preprocessed.ipynb   # Part 1
│   └── EV_Charging_HighDemand_Classification.ipynb        # Part 2
├── data/
│   └── ChargingRecords_preprocessed.csv
└── dashboard/                                             # Part 3
    ├── app.py             # Streamlit app
    ├── common.py          # shared data loading and feature engineering
    ├── train_models.py    # trains and saves the models used by the app
    ├── requirements.txt
    ├── .streamlit/config.toml
    ├── data/ChargingRecords_preprocessed.csv
    └── models/            # saved models, metrics, importances (created by train_models.py)
```

Adjust the paths above to match your repository layout. The notebooks read `ChargingRecords_preprocessed.csv` from their own folder.

## Quick start

**Requirements:** Python 3.10+

```bash
git clone <your-repo-url>
cd <your-repo>/dashboard
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

The saved models in `models/` were built with specific library versions. If they cannot be loaded on your machine, the app **retrains them automatically** on first start (about 3 to 5 minutes, once). You can also do it manually:

```bash
python train_models.py
```

To run the notebooks:

```bash
pip install jupyter xgboost lightgbm imbalanced-learn
jupyter notebook
```

Both notebooks have a `FAST = True` switch in the first cell (quick run, about 2 minutes). Set it to `False` for the full run with 5-fold cross-validation and hyper-parameter tuning (slow).

## Part 1: Regression

**Goal:** predict `Demand` (kWh) for a session.

**Models compared:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM. Metrics are MAE, MSE, RMSE and R², measured on a 20% hold-out set, with cross-validation on the training set.

| Model | MAE | RMSE | R² (test) |
|---|---|---|---|
| **XGBoost** | 3.32 | **5.86** | **0.811** |
| LightGBM | 3.35 | 5.88 | 0.809 |
| Random Forest | 3.28 | 5.96 | 0.805 |
| Gradient Boosting | 3.54 | 6.14 | 0.792 |
| Decision Tree | 3.56 | 6.28 | 0.783 |
| Linear Regression | 5.51 | 8.46 | 0.606 |

**Best validated model: XGBoost.** Cross-validation R² is about 0.80, close to the test score, so there is no sign of overfitting. Tuning XGBoost and LightGBM did not beat the untuned versions.

Notebook: `EV_Charging_Demand_Regression_preprocessed.ipynb`

## Part 2: Classification

**Goal:** classify sessions as **High Demand** (`Demand >=` median) or **Normal Demand** (`Demand <` median).

- Median threshold: **14.26 kWh**.
- **Class balance:** a median split gives about 50 / 50, so there is no real imbalance and resampling is not required.
- Random Oversampling, SMOTE and ADASYN were still tested (inside the training pipeline only, never on the test set). On the balanced target they change nothing meaningful. ADASYN cannot run on perfectly balanced classes.
- To show when resampling matters, the notebook also tests a top-10% target. There, Random Oversampling raised recall from 0.80 to 0.94 but dropped precision from 0.82 to 0.66. SMOTE and ADASYN changed little.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **LightGBM** | 0.901 | 0.872 | 0.939 | 0.904 | **0.962** |
| XGBoost | 0.901 | 0.870 | 0.942 | 0.905 | 0.962 |
| Random Forest | 0.900 | 0.866 | 0.946 | 0.904 | 0.959 |
| Decision Tree | 0.897 | 0.873 | 0.929 | 0.900 | 0.953 |
| SVM | 0.873 | 0.833 | 0.933 | 0.880 | 0.937 |
| KNN | 0.826 | 0.800 | 0.868 | 0.833 | 0.913 |
| Logistic Regression | 0.841 | 0.824 | 0.868 | 0.845 | 0.911 |

**Best validated model: LightGBM** (cross-validation ROC-AUC 0.961). SVM is trained on a 12,000-row sample in fast mode because it is slow on the full training set, and is still scored on the full test set.

Notebook: `EV_Charging_HighDemand_Classification.ipynb`

## Part 3: Streamlit dashboard

An interactive Streamlit app in the `dashboard/` folder that combines the data analysis, both models (regression and classification) and their explanations (SHAP and LIME).

```bash
cd dashboard
streamlit run app.py
```

## Key findings

- Session duration is the strongest predictor of charging demand.
- Charger type significantly affects energy consumption.
- Demand varies across locations and charger companies.
- Tree-based models outperform linear models, indicating non-linear demand patterns.
- Accommodation sites, camping sites, and bus garages show higher average charging demand.

## Applications

- EV Infrastructure Planning
- Smart Energy Management
- Charging Station Optimization

## Limitations

- **`Duration` is only known once a session ends.** The models therefore answer: *given a session of this length and context, how much energy will it use?* They are not plug-in-time forecasts. Without `Duration`, performance drops sharply (regression R² about 0.16 for the best model, classification ROC-AUC about 0.72). Both notebooks include a section showing this.
- Only one year of data is available, so month effects reflect a single season cycle.
- The saved model files depend on library versions (see [Troubleshooting](#troubleshooting)).

## Troubleshooting

| Problem | Fix |
|---|---|
| `xgboost ... input stream corrupted` when loading models | The saved models were built with different library versions. Use the latest `app.py` (it retrains automatically) or run `python train_models.py`. |
| `No module named 'shap'` / `'lime'` / `'lightgbm'` | `pip install -r requirements.txt` |
| `AttributeError: ... socketserver ... UnixStreamServer` | A broken optional `pyspark` install on Windows. The latest `app.py` works around it. You can also run `pip uninstall pyspark`. |
| `'streamlit' is not recognized` | Use `python -m streamlit run app.py` |
| Notebook stops with `NameError` on `X_train` or `RS` | An earlier cell failed. Use Kernel, Restart and Run All, and look at the first error. |

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP, LIME, Plotly, Matplotlib, Streamlit, Jupyter.

## Author

**Vidya Priya V**

- Machine Learning & Data Analytics Enthusiast
