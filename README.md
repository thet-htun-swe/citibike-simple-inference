# Inference Web App (API + UI)

This folder provides a deployable inference stack:
- FastAPI backend for model inference
- Streamlit frontend for interactive predictions and visualization

## Expected Artifacts
Default configuration assumes two-target XGBoost count artifacts:
- `artifacts/model_training/xgb_two_target_count/v1_0/xgb_trips_out_model.json`
- `artifacts/model_training/xgb_two_target_count/v1_0/xgb_trips_in_model.json`
- `artifacts/model_training/xgb_two_target_count/v1_0/metrics.json`
- Feature dataset:
  - `data/proceed/micro_mobility_two_targets_training_data_2025_v1.csv`

## Install
```bash
pip install -r inference_app/requirements.txt
```

## Configure
Copy `.env.example` values into your runtime environment (or set environment variables directly):
- `PROJECT_ROOT`
- `MODEL_MODE` (`two_target_count` or `single_net`)
- `ARTIFACT_DIR`
- `METRICS_PATH`
- `FEATURE_DATA_PATH`

## Run Backend
From repo root:
```bash
uvicorn inference_app.api.main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health`
- `GET /model-info`
- `GET /stations`
- `POST /predict`
- `POST /predict-batch`
- `GET /predict-from-reference?station=...&datetime_hour=...`

## Run UI
From repo root:
```bash
streamlit run inference_app/ui/app.py
```

Set API URL in the sidebar (default: `http://localhost:8000`).

## Deployment Notes
- Backend: deploy as a standard ASGI service (Render/Railway/Cloud Run/etc.)
- UI: deploy Streamlit service separately, pointing to backend URL
- Persist the model artifact directory as mounted volume or baked image assets
