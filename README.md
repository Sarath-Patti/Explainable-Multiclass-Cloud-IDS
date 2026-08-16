# Explainable-Multiclass-Cloud-IDS

An Explainable Multiclass Cloud Intrusion Detection System (IDS) designed to identify, classify, and explain malicious network traffic and security events in cloud environments.

---

## Project Overview

The **Explainable-Multiclass-Cloud-IDS** repository encompasses a complete, production-grade cloud security framework developed during **Summer 2026**, alongside an ongoing **Autumn 2026 Research Extension** that investigates risk-aware decision making, explainable incident response, and feedback-driven adaptation.

### System Evolution

```
┌────────────────────────────────────────────────────────────────────────┐
│                   COMPLETED WORK (Summer 2026)                         │
│             Explainable Multiclass Cloud IDS Platform                  │
│                                                                        │
│  • Automated Data Pipeline (Profiling, Cleaning, Preprocessing)        │
│  • Cost-Sensitive Multiclass Models (Random Forest & XGBoost)          │
│  • Global & Local SHAP Explainability & Top-14 Feature Selection       │
│  • Decoupled Full-Stack Architecture (FastAPI Backend + React UI)      │
│  • Dockerized Deployment & Automated GitHub Actions CI/CD Pipeline     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼ (Extends via Research)
┌────────────────────────────────────────────────────────────────────────┐
│                 RESEARCH EXTENSION (Autumn 2026)                       │
│                     Status: Ongoing Research                           │
│                                                                        │
│  1. Risk-Aware Security Decisions (Severity & Confidence Weighting)    │
│  2. Explainable Security Response (Attribution-to-Remediation Mapping) │
│  3. Feedback-Driven Adaptation (Analyst Verification Loop)             │
│  4. Comprehensive Empirical Evaluation vs. Summer 2026 Baseline        │
│  5. Applied Cloud Security Decision-Support Direction                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Objectives

- **Multiclass Threat Detection**: Classify network flows into distinct attack categories (e.g., DoS/DDoS, PortScan, Brute Force, Web Attacks, Botnet, Infiltration) distinguishing them from normal benign cloud traffic.
- **Explainability (XAI)**: Utilize feature attribution methods (SHAP TreeExplainer) to explain model decisions, giving security analysts transparency into anomalous feature contributions.
- **Production-Ready Architecture**: Provide a modular, containerized full-stack platform comprising a Python FastAPI backend and a React dashboard orchestrated via Docker Compose and validated through CI/CD.
- **Advanced Decision Research (Ongoing)**: Extend the detection framework into a risk-aware, response-oriented, and feedback-adaptive decision-support system.

---

## Research Extension — Autumn 2026

**Status**: Ongoing Research  
**Research Title**: *"Risk-Aware and Explainable Machine Learning for Cloud Intrusion Detection and Response"*

The Autumn 2026 research program extends the existing Summer 2026 IDS baseline into an advanced decision-support and response framework. Rather than replacing the established multiclass classification and SHAP attribution pipeline, this research builds layered capabilities on top of it.

### 1. Risk-Aware Security Decisions
- **Motivation**: Raw attack class predictions and probability scores do not directly convey organizational operational risk. A low-confidence probe may require less urgency than an active data infiltration.
- **Research Scope**:
  - Investigating a risk-aware decision layer that synthesizes model prediction confidence, attack severity metrics, and feature-level anomaly magnitudes.
  - Designing multi-factor risk scoring models to categorize detected network events into actionable risk tiers (e.g., *Low*, *Medium*, *High*, *Critical*).
  - Investigating dynamic alert prioritization to help Security Operations Center (SOC) teams focus on high-impact cloud security incidents.

### 2. Explainable Security Response
- **Motivation**: Current XAI implementations primarily provide visual attributions (e.g., SHAP force and summary plots) without linking feature attributions directly to remediation actions.
- **Research Scope**:
  - Investigating how local feature attributions (such as abnormal packet length distributions or suspicious destination port patterns) can be translated into targeted response recommendations.
  - Formulating rule- and policy-guided response mapping algorithms (e.g., dynamic firewall rule suggestions, rate-limiting policies, or host isolation flags) based on key SHAP contributor features.
  - Evaluating operator interpretability and decision confidence when presented with paired explanations and response recommendations.

### 3. Feedback-Driven Adaptation
- **Motivation**: Static machine learning models suffer from concept drift and recurring false positives in evolving cloud environments.
- **Research Scope**:
  - Investigating an analyst-in-the-loop feedback mechanism to capture analyst verifications (confirmations, reclassifications, or false-positive tags).
  - Developing lightweight adaptation strategies to incorporate analyst feedback to refine decision thresholds and reduce repeated false alerts without necessitating immediate, full-scale model retraining.
  - Assessing the trade-offs between rapid local adaptation and global model stability.

### 4. Experimental Evaluation
- **Methodology**: The proposed research extensions will be evaluated systematically against the completed Summer 2026 IDS baseline.
- **Evaluation Criteria**:
  - **Detection Performance**: Macro-F1, per-class Precision/Recall, and False Positive Rate (FPR).
  - **Risk Prioritization Quality**: Alert triage efficiency, high-severity detection coverage, and ranking accuracy.
  - **Response Quality**: Appropriateness and consistency of generated response recommendations.
  - **Computational Efficiency**: Latency overhead introduced by risk scoring and response mapping layers.
- *Note*: Empirical evaluation and benchmarking are actively in progress; formal quantitative results will be documented upon experimental completion.

### 5. Application and Commercialization Direction
- **Orientation**: This research is application-oriented, exploring how an explainable, risk-aware IDS can evolve into a practical cloud security monitoring and decision-support platform.
- **Target Applications**:
  - Continuous cloud infrastructure telemetry monitoring.
  - Automated alert triage and incident prioritization for SOC analysts.
  - Decision-support interfaces integrating real-time telemetry, model confidence, SHAP attributions, and response guidance.
- *Scope Clarification*: This section outlines planned research investigations and technical directions; it does not claim an active commercial product, commercial deployment, or fully autonomous remediation system.

---

## Folder Structure

```
Explainable-Multiclass-Cloud-IDS/
├── backend/           # FastAPI REST API backend (Python 3.11+)
│   ├── app/
│   │   ├── api/v1/    # Versioned API endpoints (/health, /predict, /explain)
│   │   ├── core/      # Configuration, settings & CORS management
│   │   ├── models/    # Domain data structures
│   │   ├── schemas/   # Pydantic request/response validation schemas
│   │   ├── services/  # PredictionService & SHAPService singletons
│   │   └── main.py    # FastAPI application entrypoint
│   ├── Dockerfile     # Production Python 3.11-slim backend container
│   ├── .dockerignore  # Backend Docker build exclusion list
│   └── requirements.txt
├── frontend/          # React + TypeScript + Vite + Tailwind CSS frontend
│   ├── src/
│   │   ├── components/# Modular UI components (PredictionTable, ExplainDrawer, SummaryCards)
│   │   ├── pages/     # Page views (Home, Dashboard, Upload)
│   │   ├── services/  # Axios API client (/api/v1 integration)
│   │   ├── hooks/     # Custom React hooks (useHealth)
│   │   ├── types/     # TypeScript interfaces and API models
│   │   ├── App.tsx    # Layout shell and React Router configuration
│   │   └── main.tsx   # React DOM entrypoint
│   ├── nginx.conf     # Production Nginx reverse proxy and SPA routing configuration
│   ├── Dockerfile     # Multi-stage Node.js build & Nginx production container
│   ├── .dockerignore  # Frontend Docker build exclusion list
│   └── package.json
├── .github/
│   └── workflows/
│       └── ci-cd.yml  # GitHub Actions CI/CD pipeline (Test, Buildx, GHCR publish)
├── docker-compose.yml # Multi-container orchestration with health checks
├── configs/           # Configuration files
├── data/
│   ├── raw/           # Original, unmodified datasets
│   ├── merged/        # Datasets combined from multiple sources
│   └── processed/     # Cleaned datasets and train/validation/test splits
├── src/
│   ├── data/          # Preprocessing, profiling & cleaning pipeline scripts
│   ├── analysis/      # EDA & feature selection analysis scripts
│   ├── models/        # Model architectures, RF & XGBoost training scripts
│   ├── explainability/# Offline SHAP explainability generation scripts
│   └── utils/         # Shared utilities and logging configuration
├── models/            # Serialized models and label artifacts (Top-14 XGBoost, encoders)
├── outputs/           # Generated figures, reports, metrics, and execution logs
├── requirements.txt   # Core Python dependencies
└── README.md          # Comprehensive project documentation
```

---

## Installation & Setup

### 1. ML Pipeline & Core Dependencies
```bash
# Clone the repository
git clone https://github.com/Sarath-Patti/Explainable-Multiclass-Cloud-IDS.git
cd Explainable-Multiclass-Cloud-IDS

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 2. Dockerized Deployment (v1.4 Production Setup)

#### Prerequisites
- [Docker Engine](https://docs.docker.com/get-docker/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/install/) v2.0+

#### Build & Launch Multi-Container Stack
```bash
# Build and launch both frontend and backend containers in detached mode
docker compose up --build -d

# View container status & health checks
docker compose ps

# View live application logs
docker compose logs -f
```

#### Stop & Remove Stack
```bash
docker compose down
```

#### Container Access URLs
- **React Web Dashboard UI**: `http://localhost` or `http://localhost:5173`
- **FastAPI OpenAPI Swagger Docs**: `http://localhost:8000/api/v1/docs`
- **Backend Health Check**: `http://localhost:8000/api/v1/health`

#### Troubleshooting Container Deployment
- **Port Conflict (80 or 8000 already in use)**: Stop local Uvicorn/Vite processes before running `docker compose up`.
- **Backend Model Artifact Missing**: Ensure `models/xgboost_shap_selected.pkl` exists in the repository before building the backend Docker image.
- **Frontend Nginx Proxy Timeout**: Adjust `client_max_body_size` in `frontend/nginx.conf` if uploading large batch CSV files (>100MB).

### 3. Production CI/CD Pipeline (v1.5)

The repository includes an automated GitHub Actions workflow (`.github/workflows/ci-cd.yml`):

#### Automated Checks (On Pull Requests & Pushes)
- **Frontend CI**: Executes `npm ci` and `npm run build` using Node 20 with `npm` dependency caching.
- **Backend CI**: Installs Python 3.11 requirements with `pip` caching and verifies module compilation (`py_compile`).
- **Docker Compose Spec**: Runs `docker compose config` validation.

#### Registry Publishing (On `main` Branch Pushes & Version Tags)
- **Docker Buildx & Layer Caching**: Utilizes `docker/setup-buildx-action@v3` with GitHub Actions cache (`type=gha`) for fast layer reuse.
- **GHCR Image Publishing**: Automatically builds and pushes production images to GitHub Container Registry:
  - Frontend: `ghcr.io/<owner>/explainable-multiclass-cloud-ids-frontend:latest`
  - Backend: `ghcr.io/<owner>/explainable-multiclass-cloud-ids-backend:latest`
- **Required Secrets**: Authenticates securely using GitHub's built-in `${{ secrets.GITHUB_TOKEN }}`.

```bash
# Pull published production images from GHCR
docker pull ghcr.io/<owner>/explainable-multiclass-cloud-ids-frontend:latest
docker pull ghcr.io/<owner>/explainable-multiclass-cloud-ids-backend:latest
```

### 4. Local Development Setup (Without Docker)

#### FastAPI Backend Server
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### React Frontend Development Server
```bash
cd frontend
npm install
npm run dev
```

---

## Usage Instructions
Run the modules in sequence to ingest, profile, clean, analyze, preprocess the dataset, train baseline models, and compute SHAP explanations:

```bash
# 1. Merge and standardize raw dataset files
python3 src/data/data_merger.py

# 2. Profile and validate the merged dataset
python3 src/data/data_profiler.py

# 3. Clean the dataset (remove rare classes, duplicates, missing/infinities, and constant features)
python3 src/data/data_cleaner.py

# 4. Perform Exploratory Data Analysis (EDA)
python3 src/analysis/eda.py

# 5. Preprocess and prepare data splits for training
python3 src/data/preprocessor.py

# 6. Train and evaluate Random Forest baseline model
python3 src/models/random_forest.py

# 7. Train and evaluate XGBoost baseline model
python3 src/models/xgboost_model.py

# 8. Perform model comparison and comparative benchmark analysis
python3 src/analysis/model_comparison.py

# 9. Compute SHAP explainability and model interpretability
python3 src/explainability/shap_explainer.py

# 10. Perform SHAP-guided feature selection and optimization
python3 src/analysis/shap_feature_selection.py
```

---

## Feature Engineering & Preprocessing Pipeline
The preprocessing pipeline separates predictor features from target classes, encodes target labels, splits the dataset, and serializes all artifacts.

### Key Preprocessing Design Decisions

#### 1. Omission of Feature Scaling
Feature scaling (such as MinMaxScaler or StandardScaler) is intentionally omitted from the preprocessing pipeline:
- **Tree-Based Model Invariance**: The models targeted for this system (Random Forest and XGBoost) are invariant to monotonic feature scaling.
- **Explainability (SHAP) & Interpretability**: Scaling distorts natural network feature units (e.g., bytes, packet counts, durations). Keeping features in their raw physical units ensures that SHAP force plots, summary plots, and decision trees remain intuitive and directly actionable for security analysts.

#### 2. Class Weighting vs. Resampling
Resampling methods (such as SMOTE or undersampling) are deferred in favor of class-weighted learning:
- **Data Integrity Preservation**: Undersampling discards legitimate normal and attack samples, while oversampling synthesizes artificial samples that can introduce non-existent patterns or inflate false alarm rates.
- **Cost-Sensitive Class-Weighted Learning**: By computing balanced class weights exclusively from the training partition and passing them directly to the loss function, minority attack classes (e.g., Web Attacks, Botnet traffic) receive appropriate penalty weights without distorting real-world traffic distributions.

---

## Random Forest Baseline Model Pipeline (v0.5)
The Random Forest baseline pipeline trains a tree ensemble on the preprocessed training dataset incorporating class weighting and stratified subset hyperparameter tuning.

### Model Training & Tuning Setup
- **Stratified Subset Tuning**: Tuning is executed on a representative stratified subset of 250,000 training samples using `RandomizedSearchCV` with 3-fold `StratifiedKFold` cross-validation (5 iterations, optimizing for **Macro F1**).
- **Full-Dataset Retraining**: After selecting optimal hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`), the final model is retrained on the complete training dataset (1,764,525 samples).
- **Cost-Sensitive Learning**: Configured with pre-calculated balanced `class_weight` dictionary to penalize minority attack errors.
- **Out-of-Bag Validation**: `oob_score=True` to compute unbiased generalized error during ensemble construction.

### Evaluation & Generated Artifacts
- **Serialized Model**: `models/random_forest.pkl`
- **Metrics & Reports**:
  - `outputs/metrics/rf_metrics.json` & `rf_best_params.json`
  - `outputs/metrics/rf_classification_report.csv` & `rf_confusion_matrix.csv`
  - `outputs/reports/random_forest_report.txt`
- **Visualizations**:
  - `outputs/plots/rf_confusion_matrix.png`
  - `outputs/plots/rf_feature_importance.png` (Top 20 Gini feature importances)
  - `outputs/plots/rf_roc_curves.png` (One-vs-Rest ROC curves per class)
  - `outputs/plots/rf_precision_recall.png` (Precision-Recall curves per class)

---

## XGBoost Baseline Model Pipeline (v0.6)
The XGBoost baseline pipeline trains a histogram-based Gradient Boosted Decision Tree (GBDT) ensemble mirroring the Random Forest setup for direct benchmark comparison.

### Model Training & Tuning Setup
- **Hyperparameter Optimization**: `RandomizedSearchCV` on a representative 250,000 sample stratified tuning subset with 3-fold `StratifiedKFold` cross-validation (5 iterations, optimizing for **Macro F1**).
- **Search Space**: Tunes `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.
- **Histogram-based Training**: Utilizes `tree_method="hist"`, `objective="multi:softprob"`, `eval_metric="mlogloss"`, and sample weighting for fast, scalable multiclass training.
- **Full-Dataset Retraining**: Final model is retrained on the complete training dataset (1,764,525 samples).

### Evaluation & Generated Artifacts
- **Serialized Model**: `models/xgboost_model.pkl`
- **Metrics & Reports**:
  - `outputs/metrics/xgboost_metrics.json` & `xgb_best_params.json`
  - `outputs/metrics/xgb_classification_report.csv` & `xgb_confusion_matrix.csv`
  - `outputs/reports/xgboost_report.txt`
- **Visualizations**:
  - `outputs/plots/xgb_confusion_matrix.png`
  - `outputs/plots/xgb_feature_importance.png` (Top 20 feature importances)
  - `outputs/plots/xgb_roc_curves.png` (One-vs-Rest ROC curves per class)
  - `outputs/plots/xgb_precision_recall.png` (Precision-Recall curves per class)

---

## Model Comparison & Benchmark Analysis (v0.7)
The model comparison module evaluates Random Forest and XGBoost side-by-side across overall test performance, class-wise detection rates, and computational latencies.

### Generated Artifacts & Visualizations
- **Comparative Reports & Metrics**:
  - `outputs/reports/model_comparison_report.txt`
  - `outputs/metrics/model_comparison.json`
- **Visualizations**:
  - `outputs/plots/model_comparison_metrics.png` (Overall metric side-by-side comparison)
  - `outputs/plots/per_class_f1_comparison.png` (Per-class F1-score comparison)
  - `outputs/plots/per_class_recall_comparison.png` (Per-class Recall comparison)
  - `outputs/plots/training_inference_time.png` (Training and inference latency trade-offs)

---

## SHAP Explainability & Model Interpretability (v0.8)
The SHAP explainability module utilizes `shap.TreeExplainer` on a representative stratified test sample (1,000 instances) to extract global, local, and class-wise feature attributions for the trained XGBoost model.

### Key Explainability Features & Outputs
- **Global Interpretability**:
  - `outputs/explainability/global_summary.png` (Multiclass stacked feature impact summary)
  - `outputs/explainability/global_beeswarm.png` (Aggregated feature impact beeswarm distribution)
  - `outputs/explainability/global_bar.png` (Mean absolute SHAP feature importance bar plot)
  - `outputs/explainability/feature_importance.csv` (Ranked global feature importance table)
- **Local Instance Interpretability**:
  - `outputs/explainability/waterfall_<class>.png` (Local waterfall plots per attack class instance)
  - `outputs/explainability/decision_<class>.png` (Decision path plots per attack class instance)
  - `outputs/explainability/force_<class>.html` (Interactive D3 force plots per attack class instance)
- **Class-wise Interpretability & Research Reports**:
  - `outputs/explainability/class_feature_importance.csv` (Class-specific feature rankings)
  - `outputs/explainability/shap_values.pkl` & `shap_metrics.json` (Serialized SHAP objects and metrics JSON)
  - `outputs/reports/shap_report.txt` (Comprehensive report covering Bot and Web Attack deep dives, comparison with Gini importance, and SOC operational implications)

---

## SHAP-Guided Feature Selection & Optimization (v0.9)
The SHAP-guided feature selection module systematically evaluates XGBoost models across top SHAP feature subsets (Top 70, 50, 40, 30, 20, 15, 10) to determine the smallest feature space that preserves $\ge 99\%$ of baseline Macro F1 performance.

### Key Optimization Outputs & Artifacts
- **Metrics & Reports**:
  - `outputs/metrics/shap_feature_selection.json`
  - `outputs/reports/shap_feature_selection_report.txt`
  - `outputs/metrics/timing_statistics.json` (Multi-run inference latency and throughput statistics)
  - `outputs/reports/shap_feature_selection_timing_refinement_report.txt` (Timing refinement & prediction throughput validation report)
- **Trade-off Figures**:
  - `outputs/plots/feature_count_vs_f1.png` (Feature Count vs Macro F1)
  - `outputs/plots/feature_count_vs_accuracy.png` (Feature Count vs Accuracy)
  - `outputs/plots/feature_count_vs_auc.png` (Feature Count vs Macro ROC-AUC)
  - `outputs/plots/feature_count_vs_training_time.png` (Feature Count vs Training Time)
  - `outputs/plots/feature_count_vs_inference_time.png` (Feature Count vs Test Inference Time)

### Independent Timing Benchmark Execution
To re-run only the multi-run inference timing & throughput benchmark independently without retraining models:
```bash
python3 src/analysis/shap_feature_selection.py --timing-only
```

---

## Roadmap

### Completed Milestones (Summer 2026 Core Platform)
- [x] **v0.1**: Project Foundation (Directory layout, initial configurations, skeleton scripts)
- [x] **v0.2**: Dataset Ingestion & Standardization (`data_merger.py`)
- [x] **v0.25**: Dataset Profiling & Validation (`data_profiler.py`)
- [x] **v0.3**: Data Cleaning Pipeline (`data_cleaner.py`)
- [x] **v0.35**: Exploratory Data Analysis (EDA) (`eda.py`)
- [x] **v0.4**: Feature Engineering & Preprocessing (`preprocessor.py`)
- [x] **v0.5**: Random Forest Baseline (`random_forest.py`)
- [x] **v0.6**: XGBoost Model Training & Benchmark Comparison (`xgboost_model.py`)
- [x] **v0.7**: Comparative Model Evaluation & Benchmark Analysis (`model_comparison.py`)
- [x] **v0.8**: Explainable AI Integration (`shap_explainer.py`)
- [x] **v0.9**: SHAP-Guided Feature Selection & Timing Validation (`shap_feature_selection.py`)
- [x] **v1.0**: React + FastAPI Foundation (Decoupled architecture, CORS, `/api/v1/health` status, React Router & Dashboard UI)
- [x] **v1.1**: Model Inference API (Top-14 XGBoost model singleton loader, CSV validation, `POST /api/v1/predict` endpoint)
- [x] **v1.2**: Frontend Prediction Workflow (Drag-and-drop CSV upload, progress indicator, summary cards, prediction table, CSV export)
- [x] **v1.3**: Interactive SHAP Explainability Dashboard (TreeExplainer backend service, `POST /api/v1/explain`, sliding `ExplainDrawer`, SHAP contribution charts & tables)
- [x] **v1.4**: Dockerized Deployment (Multi-stage React Nginx container, lightweight Python FastAPI container, Docker Compose orchestration, health checks)
- [x] **v1.5**: Production CI/CD (GitHub Actions workflow, Docker Buildx GHA caching, GHCR container registry publishing)

### Research Roadmap (Autumn 2026 Extensions — Ongoing & Planned)
- [ ] **v2.0-R1 (Risk Decision Layer)**: Formulation of multi-factor risk scoring models combining prediction confidence, threat severity weighting, and feature anomaly magnitudes.
- [ ] **v2.0-R2 (Explainable Response Mapping)**: Investigation of algorithms mapping SHAP local feature attributions directly to actionable security-response policies (firewall rules, rate limiting, host isolation).
- [ ] **v2.0-R3 (Feedback-Driven Adaptation)**: Design and evaluation of analyst verification loops to capture feedback and mitigate recurring false positives without full retraining.
- [ ] **v2.0-R4 (Empirical Evaluation & Benchmarking)**: Comprehensive comparative evaluation against the Summer 2026 baseline measuring Macro-F1, FPR, triage coverage, response consistency, and latency overhead.
- [ ] **v2.0-R5 (Decision Support Framework)**: Synthesis of risk scoring, explainable remediation guidance, and analyst feedback into an integrated decision-support platform architecture.
