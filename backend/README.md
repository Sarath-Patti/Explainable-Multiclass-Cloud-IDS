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
│   │       │   └── predict.py   # POST /api/v1/predict batch CSV inference
│   │       └── router.py        # v1 Router
│   ├── core/
│   │   └── config.py            # Environment configuration & artifact paths
│   ├── models/                  # Data models
│   ├── services/
│   │   ├── model_loader.py      # Singleton ML binary & label mapping loader
│   │   └── predictor.py         # Batch feature validation & model inference
│   ├── schemas/
│   │   ├── health.py            # Health Pydantic response schema
│   │   └── predict.py           # Prediction request & response schemas
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
   - Interactive Swagger UI: `http://localhost:8000/api/v1/docs`

## Batch Prediction API Usage

Example `curl` request:
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_sample.csv;type=text/csv'
```

Example JSON Response:
```json
{
  "summary": {
    "total_samples": 1000,
    "predicted_attacks": 213,
    "predicted_benign": 787
  },
  "predictions": [
    {
      "row": 0,
      "prediction": "BENIGN",
      "confidence": 0.9984
    }
  ]
}
```
