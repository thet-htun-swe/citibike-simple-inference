# Simple Inference UI (New)

This is a new minimal UI, separate from `inference_app`.

## Inputs
- Station name (searchable dropdown)
- Date
- Period (hour list, 1-hour step)
- Action (`send_bike` or `remove_bike`)

## Data sources
- Station catalog (name, lat, lng, station_id):
  - `artifacts/inference/reference/stations_catalog.csv`
- Lag defaults by station-hour:
  - `artifacts/inference/reference/station_hour_lag_defaults.csv`
- Trained models:
  - `artifacts/model_training/xgb_two_target_count/v1_0/xgb_trips_out_model.json`
  - `artifacts/model_training/xgb_two_target_count/v1_0/xgb_trips_in_model.json`

## Run (Local)
```bash
pip install -r simple_inference_ui/requirements.txt
streamlit run simple_inference_ui/app.py
```

## Deploy To Streamlit Community Cloud
1. Push this repository to GitHub (including `simple_inference_ui/assets/...` files).
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set:
   - Main file path: `streamlit_app.py`
   - Python version: `3.11` or `3.12` (recommended)
4. Deploy.

The root `requirements.txt` already points to `simple_inference_ui/requirements.txt`.

## Notes
- UI computes time features from date + period.
- `lat/lng` are resolved from station catalog.
- Lag/rolling features are auto-filled from reference defaults.
- Action mapping can be changed in `ACTION_TO_TARGET`.
