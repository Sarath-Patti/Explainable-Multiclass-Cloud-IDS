# Explainable Multiclass Cloud IDS - Backend API

Production-ready FastAPI backend for the Explainable Multiclass Cloud Intrusion Detection System.

## Architecture

```
backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   └── health.py    # Health status endpoint
│   │       └── router.py        # v1 Router
│   ├── core/
│   │   └── config.py            # Environment configuration
│   ├── models/                  # Data models
│   ├── services/                # Business & inference logic
│   ├── schemas/                 # Pydantic request/response schemas
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

4. **Interactive Documentation**:
   - Swagger UI: `http://localhost:8000/api/v1/docs`
   - ReDoc: `http://localhost:8000/api/v1/redoc`
   - Health Endpoint: `http://localhost:8000/api/v1/health`
