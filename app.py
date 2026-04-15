import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(os.getenv("ASSET_ROOT", str(PROJECT_ROOT / "assets")))

MODEL_DIR = ASSET_ROOT / "models"
REF_DIR = ASSET_ROOT / "reference"

MODEL_OUT_PATH = MODEL_DIR / "xgb_trips_out_model.json"
MODEL_IN_PATH = MODEL_DIR / "xgb_trips_in_model.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
STATIONS_PATH = REF_DIR / "stations_catalog.csv"
LAG_DEFAULTS_PATH = REF_DIR / "station_hour_lag_defaults.csv"

# imports ...

PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(os.getenv("ASSET_ROOT", str(PROJECT_ROOT / "assets")))
MODEL_DIR = ASSET_ROOT / "models"
REF_DIR = ASSET_ROOT / "reference"

MODEL_OUT_PATH = MODEL_DIR / "xgb_trips_out_model.json"
MODEL_IN_PATH = MODEL_DIR / "xgb_trips_in_model.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
STATIONS_PATH = REF_DIR / "stations_catalog.csv"
LAG_DEFAULTS_PATH = REF_DIR / "station_hour_lag_defaults.csv"

for p in [MODEL_OUT_PATH, MODEL_IN_PATH, METRICS_PATH, STATIONS_PATH, LAG_DEFAULTS_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

# Business rule mapping (edit if your operation meaning is reversed)
# send_bike   -> bikes needed to be sent out from station (uses trips_out)
# remove_bike -> bikes expected to come in / removed from circulation at station side (uses trips_in)
ACTION_TO_TARGET = {
    'send_bike': 'pred_trips_out',
    'remove_bike': 'pred_trips_in',
}

# Team info shown in UI header.
# Update these values as needed.
GROUP_NAME = 'Pandas'
GROUP_MEMBERS = [
    'Thet Htun Swe (ST126391)',
    'Rabin (ST125993)',
]


@st.cache_resource
def load_models():
    m_out = xgb.Booster()
    m_in = xgb.Booster()
    m_out.load_model(str(MODEL_OUT_PATH))
    m_in.load_model(str(MODEL_IN_PATH))
    return m_out, m_in


@st.cache_data
def load_meta():
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    feature_columns = metrics.get('feature_columns', [])
    stations = pd.read_csv(STATIONS_PATH)
    lag_defaults = pd.read_csv(LAG_DEFAULTS_PATH)

    stations['station'] = stations['station'].astype(str)
    lag_defaults['station'] = lag_defaults['station'].astype(str)
    lag_defaults['hour'] = lag_defaults['hour'].astype(int)

    station_lag_mean = lag_defaults.groupby('station', as_index=False).mean(numeric_only=True)
    global_lag_mean = lag_defaults.mean(numeric_only=True).to_dict()

    return feature_columns, stations, lag_defaults, station_lag_mean, global_lag_mean


def cyc_features(date_value, hour_value: int):
    dt = pd.Timestamp(date_value) + pd.Timedelta(hours=hour_value)
    day_of_week = int(dt.dayofweek)
    is_weekend = int(day_of_week >= 5)

    hour_sin = float(np.sin(2 * np.pi * hour_value / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour_value / 24.0))
    day_sin = float(np.sin(2 * np.pi * day_of_week / 7.0))
    day_cos = float(np.cos(2 * np.pi * day_of_week / 7.0))

    return {
        'datetime_hour': dt,
        'hour': hour_value,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
    }


def lookup_lag_features(station: str, hour: int, lag_defaults: pd.DataFrame, station_lag_mean: pd.DataFrame, global_lag_mean: dict):
    row = lag_defaults[(lag_defaults['station'] == station) & (lag_defaults['hour'] == hour)]
    if len(row):
        r = row.iloc[0]
        return {
            'trips_out_lag_1h': float(r['trips_out_lag_1h']),
            'trips_out_lag_2h': float(r['trips_out_lag_1h']),
            'trips_out_lag_3h': float(r['trips_out_lag_1h']),
            'trips_out_lag_24h': float(r['trips_out_lag_24h']),
            'trips_out_rolling_mean_3h': float(r['trips_out_rolling_mean_3h']),
            'trips_in_lag_1h': float(r['trips_in_lag_1h']),
            'trips_in_lag_2h': float(r['trips_in_lag_1h']),
            'trips_in_lag_3h': float(r['trips_in_lag_1h']),
            'trips_in_lag_24h': float(r['trips_in_lag_24h']),
            'trips_in_rolling_mean_3h': float(r['trips_in_rolling_mean_3h']),
        }

    srow = station_lag_mean[station_lag_mean['station'] == station]
    if len(srow):
        s = srow.iloc[0]
        return {
            'trips_out_lag_1h': float(s.get('trips_out_lag_1h', 0.0)),
            'trips_out_lag_2h': float(s.get('trips_out_lag_1h', 0.0)),
            'trips_out_lag_3h': float(s.get('trips_out_lag_1h', 0.0)),
            'trips_out_lag_24h': float(s.get('trips_out_lag_24h', 0.0)),
            'trips_out_rolling_mean_3h': float(s.get('trips_out_rolling_mean_3h', 0.0)),
            'trips_in_lag_1h': float(s.get('trips_in_lag_1h', 0.0)),
            'trips_in_lag_2h': float(s.get('trips_in_lag_1h', 0.0)),
            'trips_in_lag_3h': float(s.get('trips_in_lag_1h', 0.0)),
            'trips_in_lag_24h': float(s.get('trips_in_lag_24h', 0.0)),
            'trips_in_rolling_mean_3h': float(s.get('trips_in_rolling_mean_3h', 0.0)),
        }

    return {
        'trips_out_lag_1h': float(global_lag_mean.get('trips_out_lag_1h', 0.0)),
        'trips_out_lag_2h': float(global_lag_mean.get('trips_out_lag_1h', 0.0)),
        'trips_out_lag_3h': float(global_lag_mean.get('trips_out_lag_1h', 0.0)),
        'trips_out_lag_24h': float(global_lag_mean.get('trips_out_lag_24h', 0.0)),
        'trips_out_rolling_mean_3h': float(global_lag_mean.get('trips_out_rolling_mean_3h', 0.0)),
        'trips_in_lag_1h': float(global_lag_mean.get('trips_in_lag_1h', 0.0)),
        'trips_in_lag_2h': float(global_lag_mean.get('trips_in_lag_1h', 0.0)),
        'trips_in_lag_3h': float(global_lag_mean.get('trips_in_lag_1h', 0.0)),
        'trips_in_lag_24h': float(global_lag_mean.get('trips_in_lag_24h', 0.0)),
        'trips_in_rolling_mean_3h': float(global_lag_mean.get('trips_in_rolling_mean_3h', 0.0)),
    }


def build_feature_row(station_row: pd.Series, date_value, hour: int, feature_columns, lag_defaults, station_lag_mean, global_lag_mean):
    row = {
        'station_id': int(station_row['station_id']),
        'lat': float(station_row['lat']),
        'lng': float(station_row['lng']),
    }
    row.update(cyc_features(date_value, hour))
    row.update(lookup_lag_features(str(station_row['station']), hour, lag_defaults, station_lag_mean, global_lag_mean))

    row['lag_net_flow_1h'] = row['trips_in_lag_1h'] - row['trips_out_lag_1h']
    row['lag_net_flow_24h'] = row['trips_in_lag_24h'] - row['trips_out_lag_24h']

    return {k: float(row[k]) for k in feature_columns}


def predict_counts(feature_row: dict, model_out, model_in):
    x = pd.DataFrame([feature_row], dtype='float32')
    dmat = xgb.DMatrix(x.to_numpy(dtype='float32'))
    # Model was saved from sklearn wrapper; raw Booster can have inconsistent
    # stored feature-name metadata across environments. We enforce exact feature
    # order from metrics.json and disable strict name validation here.
    pred_out = float(np.clip(model_out.predict(dmat, validate_features=False)[0], 0, None))
    pred_in = float(np.clip(model_in.predict(dmat, validate_features=False)[0], 0, None))
    pred_net = float(pred_in - pred_out)
    return pred_out, pred_in, pred_net


def main():
    st.set_page_config(page_title='Citi Bike Simple Inference', layout='centered')
    st.title('Citi Bike Prediction (Simple)')
    st.markdown(f"**Group:** {GROUP_NAME}")
    st.markdown('**Members:** ' + ', '.join(GROUP_MEMBERS))
    # st.caption('Inputs: station, date, period. Action selects which prediction to return.')

    feature_columns, stations, lag_defaults, station_lag_mean, global_lag_mean = load_meta()
    model_out, model_in = load_models()

    required = {'station', 'lat', 'lng', 'station_id'}
    if not required.issubset(stations.columns):
        st.error(f'stations_catalog.csv missing columns: {sorted(required - set(stations.columns))}')
        return

    station_names = stations['station'].dropna().astype(str).tolist()
    station = st.selectbox('Station Name', options=station_names, index=0)

    col1, col2 = st.columns(2)
    with col1:
        date_value = st.date_input('Date')
    with col2:
        period = st.selectbox('Period (1-hour step)', options=[f'{h:02d}:00' for h in range(24)], index=8)

    action = st.selectbox('Action', options=['send_bike', 'remove_bike'], index=0)

    if st.button('Predict bikes', type='primary'):
        station_row = stations.loc[stations['station'] == station].iloc[0]
        hour = int(period.split(':')[0])

        feature_row = build_feature_row(
            station_row=station_row,
            date_value=date_value,
            hour=hour,
            feature_columns=feature_columns,
            lag_defaults=lag_defaults,
            station_lag_mean=station_lag_mean,
            global_lag_mean=global_lag_mean,
        )

        pred_out, pred_in, pred_net = predict_counts(feature_row, model_out, model_in)

        selected_field = ACTION_TO_TARGET[action]
        selected_value = pred_out if selected_field == 'pred_trips_out' else pred_in

        st.subheader('Prediction Result')
        st.metric('Predicted bikes', f'{selected_value:.2f}')

        c1, c2, c3 = st.columns(3)
        c1.metric('trips_out', f'{pred_out:.2f}')
        c2.metric('trips_in', f'{pred_in:.2f}')
        c3.metric('net_flow (in-out)', f'{pred_net:.2f}')


if __name__ == '__main__':
    main()
