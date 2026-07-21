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
├── configs/           # Configuration files
├── data/
│   ├── raw/           # Original, unmodified datasets
│   ├── merged/        # Datasets combined from multiple sources
│   └── processed/     # Cleaned and engineered features ready for ML
├── src/
│   ├── data/          # Modules for ingestion, cleaning, and preprocessing
│   ├── analysis/      # Exploratory data analysis (EDA) utilities
│   ├── models/        # Model architectures, training, and evaluation scripts
│   ├── explainability/# Explainable AI (SHAP) implementation scripts
│   ├── visualization/ # Plotting and data visualization functions
│   └── utils/         # Helper functions and shared utilities
├── app/               # Flask application and REST API endpoints
├── tests/             # Unit and integration test suites
├── notebooks/         # Jupyter notebooks for experimentation and prototyping
├── outputs/
│   ├── plots/         # Generated figures and charts
│   ├── reports/       # Generated PDF/HTML reports
│   ├── metrics/       # Model evaluation logs and performance metrics
│   └── logs/          # Module execution log files
├── requirements.txt   # Package dependencies
├── .gitignore         # Version control exclusion file
├── README.md          # Project documentation
└── main.py            # Application entrypoint
```

## Installation
To set up the development environment, clone this repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/<username>/Explainable-Multiclass-Cloud-IDS.git
cd Explainable-Multiclass-Cloud-IDS

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## Usage Instructions
Run the modules in sequence to ingest, profile, clean, and analyze the dataset:

```bash
# 1. Merge and standardize raw dataset files
python3 src/data/data_merger.py

# 2. Profile and validate the merged dataset
python3 src/data/data_profiler.py

# 3. Clean the dataset (remove rare classes, duplicates, missing/infinities, and constant features)
python3 src/data/data_cleaner.py

# 4. Perform Exploratory Data Analysis (EDA)
python3 src/analysis/eda.py
```

## Roadmap
- [x] **v0.1**: Project Foundation (Directory layout, initial configurations, skeleton scripts)
- [x] **v0.2**: Dataset Ingestion & Standardization (`data_merger.py`)
- [x] **v0.25**: Dataset Profiling & Validation (`data_profiler.py`)
- [x] **v0.3**: Data Cleaning Pipeline (`data_cleaner.py`)
- [x] **v0.35**: Exploratory Data Analysis (EDA) (`eda.py`)
- [ ] **v0.4**: Model Training & Multiclass Classification (XGBoost, Scikit-learn)
- [ ] **v0.5**: Explainable AI Integration (SHAP explanations and visualizations)
- [ ] **v0.6**: Flask API & Web Dashboard Interface
- [ ] **v0.7**: Report Generation (PDF reports using ReportLab) & Model Monitoring
- [ ] **v1.0**: Production Release, Optimization, & Comprehensive Testing

