# EV Charging Demand Analysis

Predicts the energy demand (kWh) of an EV charging session, as a regression and as a high/low classification, using about 72,000 charging sessions from 30 Sep 2021 to 30 Sep 2022.

## Headline results

| Task | Best model | Test result |
|---|---|---|
| Regression (forecast at plug-in) | XGBoost | MAE 8.97 kWh, R² 0.22 (mean baseline R² 0.00) |
| Classification (Demand ≥ train median, 14.37 kWh) | Random Forest | Accuracy 0.685, F1 0.690, ROC-AUC 0.759 |

R² is modest by design: session length (`Duration`) is the strongest driver of demand but is unknown when a session starts, so it is excluded to avoid leakage.

## Project structure

| Step | Notebook / file | Purpose |
|---|---|---|
| 1–2 | Cleaning notebooks | Produce `ChargingRecords_preprocessed.csv` (cleaned sessions) |
| 3 | `Step3 Feature Engineering.ipynb` | Time features, charger-history features, chronological split; writes `train.csv`, `val.csv`, `test.csv`, `feature_metadata.json` |
| 4 | `Step4_Regression_Classification_Forecast.ipynb` | Model comparison, class imbalance, SHAP and LIME on the Step 3 files |
| 4 (optional) | `Step 4 Regression Classification.ipynb` | Explanatory analysis on the cleaned CSV, with `Duration` |

## How to run

1. Install: `pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap lime matplotlib seaborn`
2. Run Step 3 to create `train.csv`, `val.csv`, `test.csv`.
3. In Step 4, set `FOLDER` (section 1) to the folder holding those three files.
4. Run all cells top to bottom (about 8 minutes on a 2-core machine).

## Method summary

- **Chronological split:** 70/15/15 by start time (train 50,391, validation 10,776, test 10,805 sessions). Sessions that straddle a boundary are purged (59 rows).
- **No leakage:** `Duration`, `ElapsedMin`, `IdleMin` and end-time columns are excluded. `UserID` and `ChargerID` are not used as features.
- **History features:** built only from sessions on the same charger that had already ended before the session started (count, mean demand, 7-day activity, 24-hour utilisation, hours since last session). Verified against a brute-force loop with 0 mismatches.
- **Fitted on train only:** fill values, encoders and the classification threshold.
- **Selection:** fit on train, choose on validation (lowest RMSE / highest ROC-AUC), report test once. Cross-validation uses `TimeSeriesSplit`.
- **Month features dropped:** one year of data means validation and test months are unseen in training.

## Results

### Regression

| Model | Val RMSE | Val R² | Test MAE | Test R² |
|---|---|---|---|---|
| Mean baseline | 13.995 | -0.001 | 10.735 | -0.001 |
| Linear Regression | 12.229 | 0.236 | 9.129 | 0.225 |
| Decision Tree | 12.291 | 0.228 | 9.275 | 0.161 |
| Random Forest | 12.064 | 0.256 | 9.050 | 0.211 |
| Gradient Boosting | 12.095 | 0.252 | 8.992 | 0.224 |
| **XGBoost** | **12.034** | 0.260 | 8.966 | 0.219 |
| LightGBM | 12.047 | 0.258 | 8.935 | 0.223 |

The top models are within about 0.03 RMSE of each other; Linear Regression is nearly as good.

### Classification

| Model | Val ROC-AUC | Test F1 | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.757 | 0.677 | 0.743 |
| Decision Tree | 0.759 | 0.685 | 0.747 |
| **Random Forest** | **0.778** | 0.690 | 0.759 |
| KNN | 0.741 | 0.666 | 0.722 |
| SVM | 0.767 | 0.692 | 0.752 |
| XGBoost | 0.777 | 0.698 | 0.759 |
| LightGBM | 0.775 | 0.698 | 0.760 |

### Class imbalance

The median target is balanced, so no resampling is needed there. On a separate 10% target (Demand ≥ train 90th percentile, 37.17 kWh), class weights and oversampling raise recall (for example Random Forest 0.07 to 0.65) and lower precision, with little change in ROC-AUC and PR-AUC.

### Explainability

SHAP ranks `hist_mean_demand_prior`, `hist_mean_demand_7d` and `Hour_cos` as the main drivers. LIME explains individual sessions. Removing the history features raises validation error by about 11% and roughly halves R².

## Two analyses, two questions

| | Cleaned data, with `Duration` | Step 3 files, forecast |
|---|---|---|
| Question | Energy for a session of known length | Energy given only what is known at plug-in |
| Regression R² / MAE | 0.811 / 3.31 kWh | 0.219 / 8.97 kWh |
| Classification ROC-AUC | 0.962 | 0.759 |

The 0.81 is explanatory, not a forecast.

## Limitations and next steps

- One year of data: seasonal effects cannot be learned or tested.
- `UserID` is unused (ID 0 covers 43% of sessions and looks like a placeholder); reliable user history could add signal.
- Hyperparameters were not tuned.
- Next: tune the top three models on validation, add user-level history, add a second year of data.
