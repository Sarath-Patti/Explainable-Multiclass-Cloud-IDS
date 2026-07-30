# Explainable Multiclass Cloud IDS - Backend API

Production-ready FastAPI backend for the Explainable Multiclass Cloud Intrusion Detection System.

## Architecture

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── health.py    # GET /api/v1/health status endpoint
│   │       │   ├── predict.py   # POST /api/v1/predict batch CSV inference
│   │       │   └── explain.py   # POST /api/v1/explain SHAP feature attributions
│   │       └── router.py        # v1 Router
│   ├── core/
│   │   └── config.py            # Environment configuration & artifact paths
│   ├── models/                  # Data models
│   ├── services/
│   │   ├── model_loader.py      # Singleton ML binary & label mapping loader
│   │   ├── predictor.py         # Batch feature validation & model inference
│   │   └── shap_service.py      # TreeExplainer single-instance SHAP service
│   ├── schemas/
│   │   ├── health.py            # Health Pydantic response schema
│   │   ├── predict.py           # Prediction request & response schemas
│   │   └── explain.py           # SHAP explanation request & response schemas
│   ├── utils/                   # Helpers
│   └── main.py                  # FastAPI application entrypoint
├── requirements.txt
└── README.md
```

## Setup & Running Locally

1. **Activate Virtual Environment**:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Development Server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

4. **API Endpoints**:
   - Health Status: `GET http://localhost:8000/api/v1/health`
   - Batch Inference: `POST http://localhost:8000/api/v1/predict` (upload CSV as `file` form-data)
   - SHAP Explanation: `POST http://localhost:8000/api/v1/explain` (JSON payload)
   - Interactive Swagger UI: `http://localhost:8000/api/v1/docs`

## SHAP Explanation API Usage

Example `curl` request:
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/explain' \
  -H 'Content-Type: application/json' \
  -d '{
    "row": 0,
    "features": {
      "Destination Port": 80,
      "Init_Win_bytes_forward": 29200
    }
  }'
```

Example JSON Response:
```json
{
  "prediction": "DDoS",
  "confidence": 0.9984,
  "base_value": 0.05,
  "top_features": [
    {
      "feature": "Destination Port",
      "value": 80.0,
      "shap_value": 4.82
    }
  ]
}
```
