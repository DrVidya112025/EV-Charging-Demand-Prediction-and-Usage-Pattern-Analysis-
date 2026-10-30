import json, joblib, numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *

st.set_page_config(page_title='EV Charging Dashboard', page_icon='⚡', layout='wide')
TEAL, ORANGE, GREY = '#0F766E', '#D97706', '#94A3B8'
px.defaults.template = 'plotly_white'; px.defaults.color_discrete_sequence = [TEAL, ORANGE, '#2563EB', '#9333EA', '#DC2626']

# ------------------------------------------------------------------ data / models
@st.cache_data
def get_data(): return load_data()
@st.cache_resource(show_spinner='Loading models...')
def get_models():
    try:
        return joblib.load('models/regressor.joblib'), joblib.load('models/classifier.joblib'), json.load(open('models/meta.json'))
    except Exception as e:
        # Saved models were built with different library versions (or are missing/corrupt): retrain here, once.
        import subprocess, sys
        with st.spinner(f'Saved models are not compatible with this Python environment ({type(e).__name__}). Retraining them now - this takes about 3-5 minutes, once...'):
            r = subprocess.run([sys.executable, 'train_models.py'], capture_output=True, text=True)
        if r.returncode != 0:
            st.error('Retraining failed. Run `python train_models.py` in a terminal and send the error.'); st.code((r.stderr or r.stdout)[-2000:]); st.stop()
        return joblib.load('models/regressor.joblib'), joblib.load('models/classifier.joblib'), json.load(open('models/meta.json'))
@st.cache_data
def get_tables():
    return (pd.read_csv('models/regression_metrics.csv'), pd.read_csv('models/classification_metrics.csv'),
            pd.read_csv('models/regression_importance.csv'), pd.read_csv('models/classification_importance.csv'),
            json.load(open('models/roc_curves.json')), pd.read_csv('models/regression_test_predictions.csv'))

df_all = get_data(); reg, clf, meta = get_models()
reg_res, clf_res, reg_imp, clf_imp, roc, reg_pred = get_tables()
THR = meta['threshold']; LOCS = meta['locations']

# ------------------------------------------------------------------ explainability helpers
import sys
def import_shap():
    """Import shap safely. If a broken optional pyspark is installed (it crashes on Windows), hide it from shap."""
    try:
        import shap
    except Exception:
        for m in [k for k in sys.modules if k == 'pyspark' or k.startswith('pyspark.') or k == 'shap' or k.startswith('shap.')]: del sys.modules[m]
        sys.modules['pyspark'] = None   # makes "import pyspark" raise ImportError, which shap handles
        import shap
    return shap

def tx_names(pipe): return [n.split('__', 1)[1] for n in pipe.named_steps['p'].get_feature_names_out()]

def group_shap(vals, names):
    """Sum one-hot Location columns back into a single 'Location' feature."""
    out = {}
    for v, n in zip(vals, names): out['Location' if n.startswith('Location_') else n] = out.get('Location' if n.startswith('Location_') else n, 0) + v
    return pd.Series(out)

@st.cache_resource
def get_explainer(kind):
    shap = import_shap()
    pipe = reg if kind == 'reg' else clf
    return shap.TreeExplainer(pipe.named_steps['m'])

def shap_values(kind, X):
    pipe = reg if kind == 'reg' else clf
    Z = pipe.named_steps['p'].transform(X); Z = Z.toarray() if hasattr(Z, 'toarray') else np.asarray(Z)
    sv = get_explainer(kind).shap_values(Z)
    if isinstance(sv, list): sv = sv[1]
    return np.asarray(sv), Z

def local_bar(vals, names, title, unit):
    s = group_shap(vals, names); s = s.reindex(s.abs().sort_values().index).tail(8)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation='h', marker_color=[TEAL if v > 0 else ORANGE for v in s.values]))
    fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=40, b=10), xaxis_title=unit); return fig

@st.cache_resource
def get_lime(kind):
    from lime.lime_tabular import LimeTabularExplainer
    pipe = reg if kind == 'reg' else clf
    bg = df_all[FEATURES].sample(2000, random_state=RS); Z = pipe.named_steps['p'].transform(bg); Z = Z.toarray() if hasattr(Z, 'toarray') else np.asarray(Z)
    return LimeTabularExplainer(Z, feature_names=tx_names(pipe), mode='regression' if kind == 'reg' else 'classification',
                                class_names=['Normal', 'High'], discretize_continuous=True, random_state=RS)

@st.cache_data(show_spinner=False)
def imbalance_experiment(pct, model_name):
    """Train one model with each imbalance strategy on a 'demand >= percentile' target. Samplers touch training data only."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
    from imblearn.pipeline import Pipeline as ImbPipeline
    thr = df_all['Demand'].quantile(pct / 100); y = (df_all['Demand'] >= thr).astype(int)
    Xa, Xb, ya, yb = train_test_split(df_all[FEATURES], y, test_size=0.2, stratify=y, random_state=RS)
    scale = model_name == 'Logistic Regression'
    est = {'Logistic Regression': LogisticRegression(max_iter=2000, random_state=RS),
           'Random Forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_samples=0.5, n_jobs=-1, random_state=RS),
           'LightGBM': LGBMClassifier(n_estimators=300, learning_rate=0.05, subsample=.8, subsample_freq=1, colsample_bytree=.8, n_jobs=-1, random_state=RS, verbose=-1)}[model_name]
    strategies = {'None': None, 'Class weights': 'cw', 'Random Oversampling': RandomOverSampler(random_state=RS), 'SMOTE': SMOTE(random_state=RS), 'ADASYN': ADASYN(random_state=RS)}
    rows = []
    for sn, smp in strategies.items():
        e = clone(est); pre = ColumnTransformer([('num', StandardScaler() if scale else 'passthrough', NUM_COLS), ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS)])
        if smp == 'cw': e.set_params(class_weight='balanced'); steps = [('p', pre), ('m', e)]
        else: steps = [('p', pre)] + ([('s', smp)] if smp is not None else []) + [('m', e)]
        try:
            p = ImbPipeline(steps).fit(Xa, ya); pr = p.predict(Xb); sc = p.predict_proba(Xb)[:, 1]
            rows.append(dict(Strategy=sn, Accuracy=accuracy_score(yb, pr), Precision=precision_score(yb, pr, zero_division=0), Recall=recall_score(yb, pr), F1=f1_score(yb, pr), ROC_AUC=roc_auc_score(yb, sc), PR_AUC=average_precision_score(yb, sc)))
        except ValueError:
            rows.append(dict(Strategy=sn + ' (n/a: classes already balanced)'))
    return float(thr), float(y.mean()), pd.DataFrame(rows).set_index('Strategy')

# ------------------------------------------------------------------ sidebar filters
st.sidebar.title('⚡ Filters')
st.sidebar.caption('Apply to the analysis tabs and to global SHAP.')
sel_loc = st.sidebar.multiselect('Location', LOCS, default=LOCS)
sel_type = st.sidebar.multiselect('Charger type', [0, 1], default=[0, 1], format_func=lambda x: f'Type {x}')
sel_co = st.sidebar.multiselect('Charger company', [0, 1], default=[0, 1], format_func=lambda x: f'Company {x}')
dmin, dmax = df_all['Date'].min().date(), df_all['Date'].max().date()
dr = st.sidebar.date_input('Date range', (dmin, dmax), min_value=dmin, max_value=dmax)
hr = st.sidebar.slider('Start hour', 0, 23, (0, 23))
dur = st.sidebar.slider('Duration (min)', 0, int(df_all.Duration.max()), (0, int(df_all.Duration.max())))
d0, d1 = (dr if isinstance(dr, tuple) and len(dr) == 2 else (dmin, dmax))
df = df_all[df_all.Location.isin(sel_loc) & df_all.ChargerType.isin(sel_type) & df_all.ChargerCompany.isin(sel_co)
            & df_all.Date.between(pd.Timestamp(d0), pd.Timestamp(d1)) & df_all.StartHour.between(*hr) & df_all.Duration.between(*dur)].copy()
df['HighDemand'] = (df['Demand'] >= THR).astype(int)
st.sidebar.metric('Sessions selected', f'{len(df):,}', f'{len(df)/len(df_all):.0%} of all')

st.title('EV Charging Demand Dashboard')
st.caption(f'Sessions {dmin:%d %b %Y} – {dmax:%d %b %Y}  ·  High Demand = Demand ≥ median ({THR:.2f} kWh)')
tabs = st.tabs(['Overview', 'Demand analysis', 'Time trends', 'Predict', 'Model performance', 'Feature importance', 'SHAP explanations', 'Class imbalance'])

if df.empty:
    st.warning('No sessions match the current filters. Widen the filters in the sidebar.'); st.stop()

# ------------------------------------------------------------------ Overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric('Sessions', f'{len(df):,}'); c[1].metric('Total energy', f'{df.Demand.sum()/1000:,.0f} MWh')
    c[2].metric('Avg demand / session', f'{df.Demand.mean():.1f} kWh'); c[3].metric('Avg duration', f'{df.Duration.mean():.0f} min')
    c[4].metric('High-demand share', f'{df.HighDemand.mean():.0%}')
    c = st.columns(5)
    c[0].metric('Unique users', f'{df.UserID.nunique():,}'); c[1].metric('Unique chargers', f'{df.ChargerID.nunique():,}')
    c[2].metric('Locations', df.Location.nunique()); c[3].metric('Median demand', f'{df.Demand.median():.1f} kWh')
    c[4].metric('Avg energy rate', f'{df.Demand.sum()/df.Duration.sum()*60:.1f} kW')
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x='Demand', nbins=60, title='Distribution of session demand (kWh)').add_vline(x=THR, line_dash='dash', line_color=ORANGE, annotation_text='median'), width='stretch')
    b.plotly_chart(px.histogram(df[df.Duration <= df.Duration.quantile(.99)], x='Duration', nbins=60, title='Distribution of session duration (min, up to p99)'), width='stretch')

# ------------------------------------------------------------------ Demand analysis
with tabs[1]:
    a, b = st.columns(2)
    g = df.groupby('Location').agg(Sessions=('Demand', 'size'), Total=('Demand', 'sum'), Avg=('Demand', 'mean'), HighShare=('HighDemand', 'mean')).reset_index()
    a.plotly_chart(px.bar(g.sort_values('Avg'), x='Avg', y='Location', orientation='h', title='Average demand per session by location (kWh)', hover_data=['Sessions']), width='stretch')
    b.plotly_chart(px.bar(g.sort_values('Total'), x='Total', y='Location', orientation='h', title='Total energy by location (kWh)'), width='stretch')
    a, b = st.columns(2)
    t = df.groupby('ChargerTypeName').agg(Avg=('Demand', 'mean'), Sessions=('Demand', 'size')).reset_index()
    a.plotly_chart(px.bar(t, x='ChargerTypeName', y='Avg', text_auto='.1f', title='Average demand by charger type (kWh)', labels={'ChargerTypeName': '', 'Avg': 'kWh'}), width='stretch')
    b.plotly_chart(px.box(df, x='ChargerTypeName', y='Demand', color='CompanyName', title='Demand by charger type and company', labels={'ChargerTypeName': ''}), width='stretch')
    pv = df.pivot_table(index='Location', columns='ChargerTypeName', values='Demand', aggfunc='mean')
    st.plotly_chart(px.imshow(pv, text_auto='.1f', color_continuous_scale='Teal', title='Average demand: location × charger type (kWh)', aspect='auto'), width='stretch')
    st.plotly_chart(px.bar(g.sort_values('HighShare'), x='HighShare', y='Location', orientation='h', title='Share of high-demand sessions by location', labels={'HighShare': 'share'}).update_xaxes(tickformat='.0%'), width='stretch')

# ------------------------------------------------------------------ Time trends
with tabs[2]:
    gran = st.radio('Aggregate by', ['Day', 'Week', 'Month'], horizontal=True, index=1)
    rule = {'Day': 'D', 'Week': 'W', 'Month': 'MS'}[gran]
    ts = df.set_index('Start').resample(rule).agg(Sessions=('Demand', 'size'), Energy=('Demand', 'sum'), Avg=('Demand', 'mean')).reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.line(ts, x='Start', y='Energy', title=f'Total energy per {gran.lower()} (kWh)', labels={'Start': ''}), width='stretch')
    b.plotly_chart(px.line(ts, x='Start', y='Sessions', title=f'Sessions per {gran.lower()}', labels={'Start': ''}), width='stretch')
    a, b = st.columns(2)
    h = df.groupby('StartHour').agg(Avg=('Demand', 'mean'), Sessions=('Demand', 'size')).reset_index()
    a.plotly_chart(px.bar(h, x='StartHour', y='Sessions', title='Sessions by start hour'), width='stretch')
    b.plotly_chart(px.line(h, x='StartHour', y='Avg', markers=True, title='Average demand by start hour (kWh)'), width='stretch')
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hm = df.pivot_table(index='Weekday', columns='StartHour', values='Demand', aggfunc='size').reindex(order)
    st.plotly_chart(px.imshow(hm, color_continuous_scale='Teal', aspect='auto', title='Session count: weekday × start hour'), width='stretch')

# ------------------------------------------------------------------ Predict
with tabs[3]:
    st.subheader('Predict a charging session')
    st.caption(f'Regression model: **{meta["best_regressor"]}**  ·  Classification model: **{meta["best_classifier"]}**')
    c = st.columns(4)
    p_dur = c[0].number_input('Duration (min)', 1, 1500, 120, step=10)
    p_hour = c[1].slider('Start hour', 0, 23, 18)
    p_dow = c[2].selectbox('Day of week', range(7), index=2, format_func=lambda i: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i])
    p_month = c[3].selectbox('Month', range(1, 13), index=5, format_func=lambda i: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i - 1])
    c = st.columns(4)
    p_loc = c[0].selectbox('Location', LOCS, index=LOCS.index('apartment') if 'apartment' in LOCS else 0)
    p_type = c[1].selectbox('Charger type', [0, 1], format_func=lambda x: f'Type {x}')
    p_co = c[2].selectbox('Charger company', [0, 1], index=1, format_func=lambda x: f'Company {x}')
    p_on = c[3].checkbox('Session crosses midnight', value=(p_hour + p_dur / 60) >= 24)
    row = make_row(p_dur, p_hour, p_dow, p_month, p_on, p_type, p_co, p_loc)
    pred_kwh = float(reg.predict(row)[0]); prob = float(clf.predict_proba(row)[0, 1])
    a, b, c3 = st.columns(3)
    a.metric('Predicted demand', f'{pred_kwh:.1f} kWh', f'{pred_kwh - THR:+.1f} vs median')
    b.metric('P(High Demand)', f'{prob:.0%}'); c3.metric('Predicted class', 'High Demand' if prob >= .5 else 'Normal Demand')
    st.progress(min(max(prob, 0.0), 1.0), text=f'High-demand probability {prob:.0%}')
    st.info('Duration is only known once a session ends, so this tool answers: *given a session of this length and context, how much energy will it use?*')
    st.markdown('##### Why this prediction? (SHAP)')
    a, b = st.columns(2)
    sv, _ = shap_values('reg', row); a.plotly_chart(local_bar(sv[0], tx_names(reg), 'Regression: contribution to predicted kWh', 'kWh'), width='stretch')
    sv, _ = shap_values('clf', row); b.plotly_chart(local_bar(sv[0], tx_names(clf), 'Classification: contribution (log-odds)', 'log-odds'), width='stretch')
    st.caption('Teal bars push the prediction up, orange bars push it down, relative to the average session.')
    with st.expander('LIME explanation (local surrogate model)'):
        kind = st.radio('Model to explain', ['Regression', 'Classification'], horizontal=True)
        k = 'reg' if kind == 'Regression' else 'clf'; pipe = reg if k == 'reg' else clf
        Z = pipe.named_steps['p'].transform(row); Z = Z.toarray() if hasattr(Z, 'toarray') else np.asarray(Z)
        fn = (lambda a: pipe.named_steps['m'].predict(a)) if k == 'reg' else (lambda a: pipe.named_steps['m'].predict_proba(a))
        with st.spinner('Running LIME...'):
            ex = get_lime(k).explain_instance(Z[0], fn, num_features=8, num_samples=1500)
        items = ex.as_list(label=1) if k == 'clf' else ex.as_list()
        ldf = pd.DataFrame(items, columns=['Condition', 'Weight']).iloc[::-1]
        st.plotly_chart(go.Figure(go.Bar(x=ldf.Weight, y=ldf.Condition, orientation='h', marker_color=[TEAL if v > 0 else ORANGE for v in ldf.Weight])).update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10)), width='stretch')

# ------------------------------------------------------------------ Model performance
with tabs[4]:
    st.subheader('Regression — predicting demand (kWh)')
    st.caption('Hold-out test set (20%). Lower is better for MAE / MSE / RMSE; higher for R².')
    st.dataframe(reg_res.set_index('Model').style.format('{:.3f}').highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='#CCFBF1').highlight_max(subset=['R2'], color='#CCFBF1'), width='stretch')
    a, b = st.columns(2)
    a.plotly_chart(px.bar(reg_res.sort_values('RMSE', ascending=False), x='RMSE', y='Model', orientation='h', title='RMSE by model'), width='stretch')
    sm = reg_pred.sample(min(4000, len(reg_pred)), random_state=1)
    fg = px.scatter(sm, x='Actual', y='Predicted', opacity=.35, title=f'Actual vs predicted — {meta["best_regressor"]}'); fg.add_shape(type='line', x0=0, y0=0, x1=sm.Actual.max(), y1=sm.Actual.max(), line=dict(color=ORANGE, dash='dash')); b.plotly_chart(fg, width='stretch')
    st.subheader('Classification — High vs Normal Demand')
    st.dataframe(clf_res.set_index('Model').style.format('{:.3f}').highlight_max(color='#CCFBF1'), width='stretch')
    a, b = st.columns(2)
    m = clf_res.melt(id_vars='Model', var_name='Metric', value_name='Score'); a.plotly_chart(px.bar(m, x='Model', y='Score', color='Metric', barmode='group', title='Classification metrics').update_yaxes(range=[.5, 1]), width='stretch')
    fg = go.Figure()
    for n, c in roc.items(): fg.add_trace(go.Scatter(x=c['fpr'], y=c['tpr'], mode='lines', name=n))
    fg.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color=GREY)); fg.update_layout(title='ROC curves', xaxis_title='False positive rate', yaxis_title='True positive rate'); b.plotly_chart(fg, width='stretch')

# ------------------------------------------------------------------ Feature importance
with tabs[5]:
    st.caption('Permutation importance on the hold-out set: how much the score drops when a feature is shuffled.')
    a, b = st.columns(2)
    for col, tbl, title, unit in [(a, reg_imp, f'Regression ({meta["best_regressor"]})', 'Drop in R²'), (b, clf_imp, f'Classification ({meta["best_classifier"]})', 'Drop in ROC-AUC')]:
        t = tbl.sort_values('Importance'); col.plotly_chart(px.bar(t, x='Importance', y='Feature', orientation='h', error_x='Std', title=title, labels={'Importance': unit}), width='stretch')
    st.markdown('**Reading it:** session **duration** dominates, followed by **charger type**; location and charger company matter a little; weekday and weekend add almost nothing.')

# ------------------------------------------------------------------ SHAP
with tabs[6]:
    shap = import_shap()
    st.caption('Global SHAP explanations on a sample of the currently filtered sessions.')
    n = min(1500, len(df)); samp = df[FEATURES].sample(n, random_state=RS)
    kind = st.radio('Model', ['Regression', 'Classification'], horizontal=True, key='gk'); k = 'reg' if kind == 'Regression' else 'clf'
    sv, Z = shap_values(k, samp); names = tx_names(reg if k == 'reg' else clf)
    g = pd.DataFrame([group_shap(r, names) for r in sv]); mean_abs = g.abs().mean().sort_values()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(x=mean_abs.values, y=mean_abs.index, orientation='h', title='Mean |SHAP| by feature', labels={'x': 'mean |SHAP|', 'y': ''}), width='stretch')
    with b:
        fig = plt.figure(figsize=(6, 4.5)); shap.summary_plot(sv, Z, feature_names=names, show=False, max_display=10); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    top = st.selectbox('Dependence plot: feature', [f for f in ['Duration', 'StartHour', 'Month', 'ChargerType', 'ChargerCompany'] if f in names])
    fig = px.scatter(x=samp[top].values, y=g.loc[:, top].values, labels={'x': top, 'y': 'SHAP value'}, opacity=.4, title=f'How {top} changes the prediction'); st.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------------ Class imbalance
with tabs[7]:
    st.subheader('Class imbalance and resampling')
    st.caption('Uses the full dataset (sidebar filters do not apply). Resampling is applied to the training data only; results are on an untouched 20% test set.')
    pos = st.slider('Define "high demand" as demand at or above this percentile', 50, 95, 90, step=5, help='50 = median split (balanced). 90 = top 10% of sessions (imbalanced).')
    share = 1 - pos / 100 if pos != 50 else 0.5
    a, b = st.columns([1, 2])
    a.metric('Positive (high-demand) share', f'{share:.0%}'); a.metric('Majority : minority', f'{(1 - share) / share:.1f} : 1' if share < .5 else '1 : 1')
    b.plotly_chart(px.bar(x=['Normal', 'High demand'], y=[1 - share, share], text_auto='.0%', title='Class balance', labels={'x': '', 'y': 'share'}).update_yaxes(tickformat='.0%', range=[0, 1]), width='stretch')
    if pos == 50: st.success('The median split is balanced (50 / 50), so resampling is not needed. ADASYN cannot run on balanced classes.')
    elif pos >= 85: st.warning('Imbalanced target: accuracy is misleading here. Compare Recall, F1 and PR-AUC.')
    mname = st.selectbox('Model', ['LightGBM', 'Random Forest', 'Logistic Regression'])
    try:
        import imblearn
        if st.button('Run comparison', type='primary'):
            with st.spinner('Training with each strategy (about 20-60 seconds)...'):
                st.session_state['imb'] = (pos, mname) + imbalance_experiment(pos, mname)
        res = st.session_state.get('imb')
        if res:
            rp, rm, thr_v, pshare, tbl = res
            st.markdown(f'**Results:** {rm}, high demand = demand ≥ {thr_v:.2f} kWh (top {100 - rp}% , positive share {pshare:.0%})')
            st.dataframe(tbl.style.format('{:.3f}', na_rep='-').highlight_max(subset=[c for c in tbl.columns if c != 'Accuracy'], color='#CCFBF1'), width='stretch')
            m = tbl.reset_index().melt(id_vars='Strategy', value_vars=['Precision', 'Recall', 'F1', 'PR_AUC'], var_name='Metric', value_name='Score')
            st.plotly_chart(px.bar(m, x='Strategy', y='Score', color='Metric', barmode='group', title='Effect of each strategy').update_yaxes(range=[0, 1]), width='stretch')
            st.info('Oversampling and class weights usually raise **recall** (more high-demand sessions found) but lower **precision** (more false alarms). Ranking quality (ROC-AUC, PR-AUC) changes little. Choose by the cost of a missed high-demand session versus a false alarm.')
        else: st.caption('Pick a percentile and model, then click **Run comparison**.')
    except ImportError:
        st.error('This tab needs the `imbalanced-learn` package. Run `pip install imbalanced-learn` and restart the app.')
