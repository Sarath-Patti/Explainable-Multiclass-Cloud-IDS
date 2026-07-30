# Explainable-Multiclass-Cloud-IDS

An Explainable Multiclass Cloud Intrusion Detection System (IDS) designed to identify, classify, and explain malicious network traffic and security events in cloud environments.

## Project Overview
The Explainable-Multiclass-Cloud-IDS is a framework for developing machine learning models to detect various types of network-based attacks (e.g., DDoS, Brute Force, Infiltration) in cloud infrastructure. By integrating advanced machine learning techniques with explainable AI (XAI) tools like SHAP, the project aims to provide security analysts with transparent, interpretable, and actionable insights into the detected threats.

## Objectives
- **Multiclass Detection**: Classify network events into multiple security categories, distinguishing between normal traffic and specific types of cloud attacks.
- **Explainability (XAI)**: Utilize feature attribution methods (such as SHAP) to explain model decisions and help security operators trust and act upon the alerts.
- **Production-Ready Architecture**: Design a clean, modular repository structure supporting independent development of data preprocessing, modeling, visualization, and application services.

## Folder Structure
The repository is organized as follows:

```
Explainable-Multiclass-Cloud-IDS/
├── backend/           # FastAPI REST API backend (Python 3.11+)
│   ├── app/
│   │   ├── api/v1/    # Versioned API routes & endpoints (/health)
│   │   ├── core/      # Application settings & CORS configuration
│   │   ├── models/    # Data domain models
│   │   ├── schemas/   # Pydantic request/response schemas
│   │   ├── services/  # Inference & business logic
│   │   └── main.py    # FastAPI main entrypoint
│   └── requirements.txt
├── frontend/          # React + TypeScript + Vite + Tailwind CSS frontend
│   ├── src/
│   │   ├── components/# Reusable UI components (Navbar, Footer, HealthStatus)
│   │   ├── pages/     # Page views (Home, Dashboard, Upload)
│   │   ├── services/  # Axios API client (/api/v1)
│   │   ├── hooks/     # Custom React hooks (useHealth)
│   │   ├── types/     # TypeScript interfaces
│   │   ├── App.tsx    # Application layout & router configuration
│   │   └── main.tsx   # React DOM entrypoint
│   └── package.json
├── configs/           # Configuration files
├── data/
│   ├── raw/           # Original, unmodified datasets
│   ├── merged/        # Datasets combined from multiple sources
│   └── processed/     # Cleaned datasets and train/validation/test splits
├── src/
│   ├── data/          # Preprocessing & cleaning pipeline scripts
│   ├── analysis/      # EDA & feature selection analysis scripts
│   ├── models/        # Model architectures, RF & XGBoost training scripts
│   ├── explainability/# SHAP explainability implementation scripts
│   └── utils/         # Helper functions and shared utilities
├── models/            # Serialized models and preprocessing artifacts
├── outputs/           # Generated figures, reports, metrics, and execution logs
├── requirements.txt   # Core Python dependencies
├── README.md          # Project documentation
└── main.py            # Application entrypoint
```

## Installation & Setup

### 1. ML Pipeline & Core Dependencies
```bash
# Clone the repository
git clone https://github.com/<username>/Explainable-Multiclass-Cloud-IDS.git
cd Explainable-Multiclass-Cloud-IDS

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 2. FastAPI Backend Server (v1.0)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# OpenAPI Docs: http://localhost:8000/api/v1/docs
```

### 3. React Frontend Dashboard (v1.0)
```bash
cd frontend
npm install
npm run dev
# Dashboard UI: http://localhost:5173
```

## Usage Instructions
Run the modules in sequence to ingest, profile, clean, analyze, preprocess the dataset, and train the baseline model:

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

## Feature Engineering & Preprocessing Pipeline
The preprocessing pipeline separates the predictor features from target classes, encodes the target labels, splits the dataset, and serializes all artifacts.

### Key Preprocessing Design Decisions

#### 1. Omission of Feature Scaling
Feature scaling (such as MinMaxScaler, StandardScaler, RobustScaler, or PowerTransformer) is intentionally omitted from the preprocessing pipeline:
- **Tree-Based Model Invariance**: The models targeted for this system (Random Forest and XGBoost) are invariant to monotonic transformations of the features. Scaling features yields no performance benefits for these algorithms.
- **Explainability (SHAP) & Interpretability**: Scaling distorts the natural values of network features (e.g., bytes, packets, durations). Keeping features in their raw physical units ensures that SHAP force plots, summary plots, and decision trees remain highly readable and actionable for security analysts.

#### 2. Class Weighting vs. Resampling
Resampling methods (such as SMOTE, oversampling, or undersampling) are deferred in favor of class-weighted learning:
- **Data Integrity Preservation**: Undersampling discards valuable normal and attack samples, while oversampling generates synthetic samples that may introduce non-existent patterns or inflate false positive rates in real-world network traffic.
- **Cost-Sensitive Class-Weighted Learning**: By computing balanced class weights exclusively from the training partition and passing them directly to the model's loss function, we preserve all original data structures while ensuring that minority attack classes (e.g., Web Attacks, Botnet traffic) receive appropriate penalty weights during training.

## Random Forest Baseline Model Pipeline (v0.5)
The Random Forest baseline pipeline trains a robust tree ensemble on the preprocessed training dataset while incorporating class weighting and efficient hyperparameter tuning.

### Model Training & Tuning Setup
- **Stratified Subset Tuning**: Hyperparameter tuning is executed on a representative stratified subset of 250,000 training samples using `RandomizedSearchCV` with 3-fold `StratifiedKFold` cross-validation (5 iterations, optimizing for **Macro F1**). This significantly reduces computation time while maintaining statistical rigor and avoiding data leakage.
- **Full-Dataset Retraining**: After selecting the optimal hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`), the final Random Forest model is retrained on the complete training dataset (1,764,525 samples).
- **Cost-Sensitive Learning**: Configured with pre-calculated balanced `class_weight` dictionary to penalize minority attack classification errors without synthetic resampling.
- **Out-of-Bag Validation**: `oob_score=True` to compute unbiased generalized error during ensemble construction.

### Evaluation & Generated Artifacts
- **Serialized Model**: `models/random_forest.pkl`
- **Metrics & Reports**:
  - `outputs/metrics/rf_metrics.json` & `rf_best_params.json`
  - `outputs/metrics/rf_classification_report.csv` & `rf_confusion_matrix.csv`
  - `outputs/reports/random_forest_report.txt`
- **Publication-Quality Visualizations**:
  - `outputs/plots/rf_confusion_matrix.png`
  - `outputs/plots/rf_feature_importance.png` (Top 20 Gini feature importances)
  - `outputs/plots/rf_roc_curves.png` (One-vs-Rest ROC curves per class)
  - `outputs/plots/rf_precision_recall.png` (Precision-Recall curves per class)

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
- **Publication-Quality Visualizations**:
  - `outputs/plots/xgb_confusion_matrix.png`
  - `outputs/plots/xgb_feature_importance.png` (Top 20 feature importances)
  - `outputs/plots/xgb_roc_curves.png` (One-vs-Rest ROC curves per class)
  - `outputs/plots/xgb_precision_recall.png` (Precision-Recall curves per class)

## Model Comparison & Benchmark Analysis (v0.7)
The model comparison module evaluates Random Forest and XGBoost side-by-side across overall test performance, class-wise detection rates, and computational latencies.

### Generated Artifacts & Visualizations
- **Comparative Reports & Metrics**:
  - `outputs/reports/model_comparison_report.txt`
  - `outputs/metrics/model_comparison.json`
- **Publication-Quality Visualizations**:
  - `outputs/plots/model_comparison_metrics.png` (Overall metric side-by-side comparison)
  - `outputs/plots/per_class_f1_comparison.png` (Per-class F1-score comparison)
  - `outputs/plots/per_class_recall_comparison.png` (Per-class Recall comparison)
  - `outputs/plots/training_inference_time.png` (Training and inference latency trade-offs)

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
  - `outputs/reports/shap_report.txt` (Comprehensive research report covering Bot and Web Attack deep dives, comparison with built-in XGBoost Gini importance, and SOC operational implications)

## SHAP-Guided Feature Selection & Optimization (v0.9)
The SHAP-guided feature selection module systematically evaluates XGBoost models across top SHAP feature subsets (Top 70, 50, 40, 30, 20, 15, 10) to determine the smallest feature space that preserves $\ge 99\%$ of baseline Macro F1 performance.

### Key Optimization Outputs & Artifacts
- **Metrics & Reports**:
  - `outputs/metrics/shap_feature_selection.json`
  - `outputs/reports/shap_feature_selection_report.txt`
  - `outputs/metrics/timing_statistics.json` (Multi-run inference latency and throughput statistics)
  - `outputs/reports/shap_feature_selection_timing_refinement_report.txt` (Timing refinement & prediction throughput validation report)
- **Publication-Quality Trade-off Figures**:
  - `outputs/plots/feature_count_vs_f1.png` (Feature Count vs Macro F1)
  - `outputs/plots/feature_count_vs_accuracy.png` (Feature Count vs Accuracy)
  - `outputs/plots/feature_count_vs_auc.png` (Feature Count vs Macro ROC-AUC)
  - `outputs/plots/feature_count_vs_training_time.png` (Feature Count vs Training Time)
  - `outputs/plots/feature_count_vs_inference_time.png` (Feature Count vs Test Inference Time)

### Independent Timing Benchmark Execution
To re-run only the multi-run inference timing & throughput benchmark independently without retraining models or re-evaluating feature selection:
```bash
python3 src/analysis/shap_feature_selection.py --timing-only
```

## Roadmap
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
- [x] **v0.9**: SHAP-Guided Feature Selection (`shap_feature_selection.py`)
- [x] **v1.0**: React + FastAPI Foundation (Decoupled architecture, CORS, /api/v1/health status, React Router & Security Operations Dashboard UI)
- [x] **v1.1**: Model Inference API (Top-14 XGBoost model singleton loader, CSV feature validation, POST /api/v1/predict batch endpoint)
- [x] **v1.2**: Frontend Prediction Workflow (Drag-and-drop CSV upload, progress indicator, summary cards, searchable/sortable/paginated prediction table, CSV results export)
