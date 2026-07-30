# Explainable Multiclass Cloud IDS - Frontend

Production-ready React + TypeScript + Vite + Tailwind CSS frontend interface for Explainable Multiclass Cloud Intrusion Detection System.

## Architecture

```
frontend/
├── src/
│   ├── assets/        # Static media assets
│   ├── components/    # Reusable UI components
│   │   ├── Upload/            # CSV Drag & Drop, picker & validation component
│   │   ├── SummaryCards/      # Total, Benign, Attack & Class metric summary cards
│   │   ├── PredictionTable/   # Searchable, sortable, paginated results table
│   │   ├── Loading/           # Upload & inference progress state indicator
│   │   ├── ErrorAlert/        # Structured error message alerts (missing features)
│   │   ├── Navbar/            # Navigation bar with live API connection indicator
│   │   ├── Footer.tsx         # Footer bar component
│   │   └── HealthStatus.tsx   # FastAPI backend telemetry monitor
│   ├── pages/         # Page views (Home, Dashboard, Upload)
│   ├── services/      # Axios HTTP client, POST /api/v1/predict API & CSV exporter
│   ├── hooks/         # Custom React hooks (useHealth)
│   ├── types/         # TypeScript interface definitions (PredictionResponse, etc.)
│   ├── App.tsx        # Main application layout & React Router config
│   ├── main.tsx       # React DOM entrypoint
│   └── index.css      # Tailwind directives & glassmorphism theme
├── public/            # Public static assets
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
└── README.md
```

## Features (v1.2)

- **CSV Upload Workflow**: Drag-and-drop file upload with format/size validation and real-time upload progress tracking.
- **Batch Multiclass Classification**: Integrated with `POST /api/v1/predict` backend service.
- **Prediction Summary Dashboard**: Metric cards displaying total evaluated flows, benign count, attack count, and detected attack vectors.
- **Interactive Prediction Table**: Search by class name/row index, sort by column, filter by category (All/Attacks/Benign), client-side pagination, and risk confidence progress bars.
- **CSV Results Export**: Download batch prediction outputs as a `.csv` file (`row,prediction,confidence`).
- **Resilient Error Alerts**: User-friendly alerts displaying exact missing feature column names or network errors.

## Setup & Execution Instructions

1. **Navigate to Frontend Directory**:
   ```bash
   cd frontend
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Start Development Server**:
   ```bash
   npm run dev
   ```

   The application will be accessible at `http://localhost:5173`.

4. **Build Production Bundle**:
   ```bash
   npm run build
   ```
