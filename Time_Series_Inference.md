# Time Series Forecasting:Inference

Notebook: `Time_Series_Model.ipynb`. Data: `ChargingRecords_preprocessed.csv` (72,031 sessions, 30 Sep 2021 to 30 Sep 2022).

## Notes

### 1. Data preparation
- Each session's `Demand` (kWh) is placed at its `StartDatetime` and summed by hour, then by day. The result is **366 daily observations**.
- Demand is credited to the hour in which a session **starts**, not spread across the hours it charges.
- Missing hours or days are filled with 0 (no sessions means no demand).
- The split is **chronological 80/20**: 292 training days (to 18 Jul 2022) and 74 test days (19 Jul to 30 Sep 2022). Nothing is shuffled, so there is no look-ahead leakage.

### 2. Trend and seasonality
- Daily demand is decomposed additively with a **7-day period** into trend, weekly seasonal and residual parts.
- A weekly cycle is the natural pattern for charging behaviour (weekday against weekend).

### 3. Stationarity (ADF test)
- ADF statistic = **-1.87**, p-value = **0.347**.
- p > 0.05, so we cannot reject a unit root: the daily series is **non-stationary** (it has a trend or level shifts).
- This is why the ARIMA-type models use differencing (`d = 1`, seasonal `D = 1`).

### 4. Baseline models
| Model | MAE | RMSE |
|---|---|---|
| Moving Average (7-day) | 571.1 | 791.7 |
| Simple Exponential Smoothing | 572.8 | 792.7 |
| Holt-Winters (additive, period 7) | 797.1 | 1037.1 |

### 5. Advanced models
| Model | Setup | MAE | RMSE | WAPE |
|---|---|---|---|---|
| **XGBoost** | Lags 1, 2, 3, 7, 14, 28; rolling means (7, 14); calendar fields; recursive forecast | **584.5** | **791.2** | **13.8%** |
| Prophet | Weekly and yearly seasonality | 693.3 | 898.1 | 16.4% |
| SARIMA | (1,1,1)(1,1,1,7) | 778.6 | 1015.5 | 18.4% |
| SARIMAX | Same as SARIMA plus calendar regressors | 1650.4 | 2032.1 | 39.0% |

- The XGBoost forecast is **recursive**: each day's prediction feeds the next day's lags, and actual future demand is never used.
- Mean daily demand is about 4,200 kWh, so an MAE of about 585 kWh is an error of roughly 14%.

## Inference

1. **Best advanced model: XGBoost** (lowest RMSE 791.2 and MAE 584.5, WAPE 13.8%). Its forecasts are typically about 14% away from actual daily demand over a 10-week horizon.

2. **A plain 7-day moving average is just as good.** Its RMSE (791.7) matches XGBoost's, and its MAE (571.1) is slightly lower. The extra complexity of XGBoost buys essentially nothing on this data. For simple operational forecasts, the moving average is a strong and cheap baseline.

3. **SARIMAX is the worst model, so the calendar regressors hurt.** With only 292 training days, regressors such as `month`, `day_of_week` and the yearly sin/cos terms overfit, and they overlap with the weekly seasonality the model already has. Adding external variables does not automatically help.

4. **Holt-Winters and SARIMA underperform the simple baselines.** Both try to learn a trend plus seasonality, but the trend changes over the year, so extrapolating it over 74 days drifts away from reality. Level-based methods (moving average, SES) stay closer to actual demand.

5. **Demand is non-stationary** (ADF p = 0.347). Level and trend change over time, so models must difference the data or use recent history (lags, rolling means), which is what XGBoost does.

6. **Prophet is a solid middle option** (16.4% WAPE). It handles weekly patterns well, but its yearly component cannot be learned reliably from a single year of data.

## Practical takeaways
- **Infrastructure planning and energy management:** forecast daily demand with the XGBoost lag model or a 7-day moving average, and plan for roughly ±14% error.
- **Weekday against weekend patterns** are the main repeatable structure, and the lag-7 and rolling-mean features capture it.
- **Retrain regularly** (weekly or monthly) so the lag features and level stay current.

## Limitations
- **Only one year of data.** Yearly seasonality (summer against winter, holidays) cannot be validated, and the test window covers only Jul to Sep 2022.
- **One train/test split.** A rolling-origin (walk-forward) evaluation would give a more reliable ranking of the models.
- **Aggregate demand only.** The forecast is for total network demand. Location-level or charger-level forecasts (the main regression finding is that location matters) are not covered.
- **Recursive forecast error accumulates** over long horizons, so accuracy falls the further ahead you forecast.
- **The first day is partial** (30 Sep 2021 starts at 17:14), which slightly distorts the start of the series.

## Notebook housekeeping (suggested)
- Remove the `pip install` cell and the empty last cell.
- Section numbering restarts ("2. Prepare the time series") after the baseline section, and the data is reloaded and rebuilt. Renumber the sections and reuse the `daily_demand` series from the first part.
- Add the baseline results to the final comparison table, so the moving-average result is visible next to the advanced models.
- Keep the `evaluate` function in one place, since it is defined twice with different outputs.
