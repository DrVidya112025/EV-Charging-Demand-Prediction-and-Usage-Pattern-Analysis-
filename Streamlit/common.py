"""Shared feature engineering used by training and the dashboard."""
import numpy as np, pandas as pd

NUM_COLS = ['Duration','StartHour','HourSin','HourCos','DayOfWeek','IsWeekend','Month','Overnight','ChargerType','ChargerCompany']
CAT_COLS = ['Location']
FEATURES = NUM_COLS + CAT_COLS
DATA_PATH = 'data/ChargingRecords_preprocessed.csv'
RS = 42

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop_duplicates(); df = df[df['Duration'] > 0].copy()
    df['Start'] = pd.to_datetime(df['StartTimestamp'])
    df['Date'] = df['Start'].dt.normalize()
    h = df['Start'].dt.hour + df['Start'].dt.minute/60
    df['HourSin'] = np.sin(2*np.pi*h/24); df['HourCos'] = np.cos(2*np.pi*h/24)
    df['IsWeekend'] = df['IsWeekend'].astype(int); df['Overnight'] = df['CrossesMidnight'].astype(int)
    df['CompanyName'] = df['ChargerCompany'].map({0:'Company 0', 1:'Company 1'})
    df['ChargerTypeName'] = df['ChargerType'].map({0:'Type 0', 1:'Type 1'})
    return df

def make_row(duration, start_hour, day_of_week, month, overnight, charger_type, company, location):
    h = start_hour
    return pd.DataFrame([{
        'Duration': duration, 'StartHour': int(h), 'HourSin': np.sin(2*np.pi*h/24), 'HourCos': np.cos(2*np.pi*h/24),
        'DayOfWeek': day_of_week, 'IsWeekend': int(day_of_week >= 5), 'Month': month, 'Overnight': int(overnight),
        'ChargerType': charger_type, 'ChargerCompany': company, 'Location': location}])[FEATURES]
