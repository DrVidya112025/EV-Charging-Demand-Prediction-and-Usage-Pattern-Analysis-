"""Train regression + classification models and save everything the dashboard needs.
Run:  python train_models.py        (about 3-4 minutes)"""
import json, joblib, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from common import *

df = load_data()
thr = float(df['Demand'].median()); df['HighDemand'] = (df['Demand'] >= thr).astype(int)
X = df[FEATURES]

def prep(scale=False):
    return ColumnTransformer([('num', StandardScaler() if scale else 'passthrough', NUM_COLS),
                              ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS)])

# ---------------- Regression ----------------
Xtr, Xte, ytr, yte = train_test_split(X, df['Demand'], test_size=0.2, random_state=RS)
reg = {
 'Linear Regression': Pipeline([('p', prep(True)), ('m', LinearRegression())]),
 'Decision Tree': Pipeline([('p', prep()), ('m', DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=RS))]),
 'Random Forest': Pipeline([('p', prep()), ('m', RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_samples=0.5, n_jobs=-1, random_state=RS))]),
 'Gradient Boosting': Pipeline([('p', prep()), ('m', GradientBoostingRegressor(n_estimators=100, subsample=0.5, max_depth=4, random_state=RS))]),
 'XGBoost': Pipeline([('p', prep()), ('m', XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=.8, colsample_bytree=.8, n_jobs=-1, random_state=RS))]),
 'LightGBM': Pipeline([('p', prep()), ('m', LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=.8, subsample_freq=1, colsample_bytree=.8, n_jobs=-1, random_state=RS, verbose=-1))]),
}
rows = []
for n, p in reg.items():
    p.fit(Xtr, ytr); pr = p.predict(Xte); mse = mean_squared_error(yte, pr)
    rows.append(dict(Model=n, MAE=mean_absolute_error(yte, pr), MSE=mse, RMSE=np.sqrt(mse), R2=r2_score(yte, pr)))
reg_res = pd.DataFrame(rows).sort_values('RMSE'); reg_res.to_csv('models/regression_metrics.csv', index=False)
best_reg = reg_res.iloc[0]['Model']; print('Best regressor:', best_reg); print(reg_res.round(4))
pi = permutation_importance(reg[best_reg], Xte, yte, n_repeats=5, random_state=RS, scoring='r2', n_jobs=-1)
pd.DataFrame({'Feature': FEATURES, 'Importance': pi.importances_mean, 'Std': pi.importances_std}).sort_values('Importance', ascending=False).to_csv('models/regression_importance.csv', index=False)
Xte.assign(Actual=yte, Predicted=reg[best_reg].predict(Xte)).to_csv('models/regression_test_predictions.csv', index=False)
joblib.dump(reg[best_reg], 'models/regressor.joblib')

# ---------------- Classification ----------------
Xtr, Xte, ytr, yte = train_test_split(X, df['HighDemand'], test_size=0.2, stratify=df['HighDemand'], random_state=RS)
clf = {
 'Logistic Regression': Pipeline([('p', prep(True)), ('m', LogisticRegression(max_iter=2000, random_state=RS))]),
 'Decision Tree': Pipeline([('p', prep()), ('m', DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=RS))]),
 'Random Forest': Pipeline([('p', prep()), ('m', RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_samples=0.5, n_jobs=-1, random_state=RS))]),
 'KNN': Pipeline([('p', prep(True)), ('m', KNeighborsClassifier(n_neighbors=25, n_jobs=-1))]),
 'SVM': Pipeline([('p', prep(True)), ('m', SVC(kernel='rbf', random_state=RS))]),
 'XGBoost': Pipeline([('p', prep()), ('m', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=.8, colsample_bytree=.8, n_jobs=-1, random_state=RS, eval_metric='logloss'))]),
 'LightGBM': Pipeline([('p', prep()), ('m', LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=.8, subsample_freq=1, colsample_bytree=.8, n_jobs=-1, random_state=RS, verbose=-1))]),
}
rows = []
for n, p in clf.items():
    if n == 'SVM': idx = Xtr.sample(12000, random_state=RS).index; p.fit(Xtr.loc[idx], ytr.loc[idx])  # SVM is slow: 12k-row sample
    else: p.fit(Xtr, ytr)
    pr = p.predict(Xte); s = p.predict_proba(Xte)[:,1] if hasattr(p, 'predict_proba') else p.decision_function(Xte)
    rows.append(dict(Model=n, Accuracy=accuracy_score(yte, pr), Precision=precision_score(yte, pr), Recall=recall_score(yte, pr), F1=f1_score(yte, pr), ROC_AUC=roc_auc_score(yte, s)))
clf_res = pd.DataFrame(rows).sort_values('ROC_AUC', ascending=False); clf_res.to_csv('models/classification_metrics.csv', index=False)
best_clf = clf_res.iloc[0]['Model']; print('Best classifier:', best_clf); print(clf_res.round(4))
pi = permutation_importance(clf[best_clf], Xte, yte, n_repeats=5, random_state=RS, scoring='roc_auc', n_jobs=-1)
pd.DataFrame({'Feature': FEATURES, 'Importance': pi.importances_mean, 'Std': pi.importances_std}).sort_values('Importance', ascending=False).to_csv('models/classification_importance.csv', index=False)
joblib.dump(clf[best_clf], 'models/classifier.joblib')
roc = {n: None for n in clf}
from sklearn.metrics import roc_curve
curves = {}
for n, p in clf.items():
    s = p.predict_proba(Xte)[:,1] if hasattr(p, 'predict_proba') else p.decision_function(Xte)
    fpr, tpr, _ = roc_curve(yte, s); k = np.linspace(0, len(fpr)-1, min(200, len(fpr))).astype(int); curves[n] = dict(fpr=fpr[k].tolist(), tpr=tpr[k].tolist())
json.dump(curves, open('models/roc_curves.json', 'w'))
json.dump(dict(threshold=thr, best_regressor=best_reg, best_classifier=best_clf, locations=sorted(df['Location'].unique().tolist()),
               duration_median=float(df.Duration.median())), open('models/meta.json', 'w'), indent=1)
print('Saved to models/')
