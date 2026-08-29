# Real-Time Performance & Scalability Benchmark Suite

This directory contains an automated, reproducible benchmark suite for evaluating the real-time inference latency, throughput, concurrency scaling, and pipeline component breakdown of the **Explainable-Multiclass-Cloud-IDS**.

---

## Benchmark Objectives & Domain Alignment

In real-time cloud security monitoring (such as Juspay / JusTrust risk management applications), intrusion detection systems must deliver:
1. **Low-Latency Detection (<100 ms Target)**: Real-time traffic classification without creating operational bottlenecks.
2. **Deterministic & Explainable AI**: Clear separation between high-speed prediction paths and deep SHAP feature attribution paths.
3. **Controlled Concurrency Scaling**: Reliable throughput performance under concurrent threat request loads.

---

## Architectural Separation: Detection vs. Explanation

The benchmark suite explicitly distinguishes between two independent execution paths:

```
                          ┌───────────────────────────┐
                          │   Client / Test Runner    │
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           │                                                         │
           ▼                                                         ▼
 ┌───────────────────┐                                     ┌───────────────────┐
 │ POST /predict     │                                     │ POST /explain     │
 │ (Detection Path)  │                                     │ (XAI Path)        │
 ├───────────────────┤                                     ├───────────────────┤
 │ • CSV Upload      │                                     │ • Single-Instance │
 │ • Fast Pandas     │                                     │ • TreeExplainer   │
 │ • XGBoost Model   │                                     │ • Base Value      │
 │ • Batch Response  │                                     │ • SHAP Ranking    │
 └───────────────────┘                                     └───────────────────┘
```

- **`POST /api/v1/predict` (Detection Path)**: Evaluates end-to-end network flow classification latency and throughput over CSV batches (100, 500, 1,000, 5,000 rows).
- **`POST /api/v1/explain` (Explanation Path)**: Evaluates local SHAP feature attribution latency for individual incident investigations. SHAP is not executed on the prediction critical path.

---

## Benchmark Suite Architecture

```
benchmarks/
├── README.md                          # Methodology, results summary & reproduction guide
├── benchmark_api.py                   # Live HTTP container API benchmark (Predict & Explain)
├── benchmark_batches.py               # Isolated component-level pipeline profiler
└── results/
    ├── .gitkeep                       # Placeholder for result artifacts
    ├── api_benchmark_results.json     # Complete structured JSON benchmark output
    ├── prediction_batch_results.csv   # Batch size latency & throughput summary CSV
    └── pipeline_profiling_results.json # Component breakdown metrics JSON
```

---

## Measured Baseline Benchmark Results

*Evaluated on the running Docker container stack (`http://localhost:8000`) using representative CICIDS2017 test traffic (`data/processed/X_test_sample.csv`).*

### 1. End-to-End Prediction Benchmark (`POST /api/v1/predict`)

| Batch Size | Repetitions | p50 Latency | p95 Latency | p99 Latency | Mean Latency | Per-Row Latency | Throughput (rows/sec) | Error Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 rows** | 10 | **9.93 ms** | **24.41 ms** | **30.44 ms** | 12.49 ms | 0.1249 ms/row | **8,003.48** | 0.0% |
| **500 rows** | 10 | **14.95 ms** | **23.09 ms** | **27.74 ms** | 16.26 ms | 0.0325 ms/row | **30,759.04** | 0.0% |
| **1,000 rows** | 10 | **23.62 ms** | **37.98 ms** | **44.26 ms** | 26.39 ms | 0.0264 ms/row | **37,892.74** | 0.0% |
| **5,000 rows** | 10 | **89.14 ms** | **114.38 ms** | **121.27 ms** | 93.87 ms | 0.0188 ms/row | **53,267.68** | 0.0% |

---

### 2. Concurrency Load Benchmark (`POST /api/v1/predict`, 500-Row Batches)

| Concurrency (Workers) | Total Requests | p50 Latency | p95 Latency | p99 Latency | Mean Latency | Aggregate Throughput | Error Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 20 | **15.22 ms** | **24.25 ms** | **27.79 ms** | 16.59 ms | **29,640.23 rows/sec** | 0.0% |
| **5** | 20 | **61.61 ms** | **63.24 ms** | **66.92 ms** | 56.91 ms | **39,341.00 rows/sec** | 0.0% |
| **10** | 20 | **118.50 ms** | **127.01 ms** | **127.10 ms** | 98.17 ms | **39,385.80 rows/sec** | 0.0% |
| **25** | 20 | **151.81 ms** | **255.74 ms** | **264.99 ms** | 147.25 ms | **36,176.39 rows/sec** | 0.0% |
| **50** | 20 | **140.86 ms** | **244.85 ms** | **253.97 ms** | 146.25 ms | **38,267.26 rows/sec** | 0.0% |

---

### 3. SHAP Explanation Benchmark (`POST /api/v1/explain`)

| Target Endpoint | Request Type | Evaluated Requests | p50 Latency | p95 Latency | p99 Latency | Mean Latency | Error Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `/api/v1/explain` | Single-Instance SHAP | 50 | **20.29 ms** | **22.24 ms** | **27.87 ms** | 20.69 ms | 0.0% |

---

### 4. Component-Level Pipeline Breakdown (`benchmark_batches.py`)

| Batch Size | CSV Bytes Gen | CSV Parsing (`pd.read_csv`) | Feature Slicing | Value Cleaning | XGBoost Execution | Response Formatting | Total Pipeline |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 rows** | 1.41 ms (21.0%) | 0.88 ms (13.1%) | 0.29 ms (4.3%) | 0.08 ms (1.1%) | **3.83 ms (57.1%)** | 0.22 ms (3.3%) | **6.71 ms** |
| **500 rows** | 6.20 ms (37.6%) | 2.02 ms (12.3%) | 0.34 ms (2.1%) | 0.08 ms (0.5%) | **6.72 ms (40.8%)** | 1.12 ms (6.8%) | **16.47 ms** |
| **1,000 rows** | 12.31 ms (40.5%) | 3.68 ms (12.1%) | 0.43 ms (1.4%) | 0.08 ms (0.3%) | **11.81 ms (38.8%)** | 2.12 ms (7.0%) | **30.43 ms** |
| **5,000 rows** | 65.00 ms (45.1%) | 19.72 ms (13.7%) | 0.54 ms (0.4%) | 0.12 ms (0.1%) | **47.18 ms (32.7%)** | 11.56 ms (8.0%) | **144.12 ms** |

---

### 5. Response Formatting Optimization Microbenchmark (`benchmark_formatting.py`)

To validate the efficiency of vectorizing prediction label resolution and confidence rounding in `PredictorService.predict_dataframe`, an isolated microbenchmark evaluated the formatting stage independently over 50 repetitions:

| Batch Size | Original Mean Latency | Optimized Mean Latency | Absolute Improvement | Percentage Improvement |
| :---: | :---: | :---: | :---: | :---: |
| **100 rows** | 0.2528 ms | **0.0833 ms** | **+0.1695 ms** | **+67.06%** |
| **500 rows** | 1.6577 ms | **0.4072 ms** | **+1.2505 ms** | **+75.44%** |
| **1,000 rows** | 3.1179 ms | **1.1882 ms** | **+1.9297 ms** | **+61.89%** |
| **5,000 rows** | 18.4892 ms | **8.0494 ms** | **+10.4399 ms** | **+56.46%** |

*Conclusion*: Vectorizing label lookups and confidence rounding reduces Python object formatting overhead by **56.5% to 75.4%**, saving **~10.44 ms** per 5,000-row prediction batch without altering API contracts or output behavior.

---

### 6. Server-Side CSV Parsing Optimization (`benchmark_csv_parsing.py`)

To optimize server-side CSV parsing in `backend/app/api/v1/endpoints/predict.py`, `pd.read_csv` was configured with single-pass `usecols` set-filtering (`usecols=lambda c: c in expected_set`). An isolated microbenchmark evaluated parsing strategies across 50 repetitions:

| Batch Size | Baseline Parse Mean | Optimized Parse Mean | Absolute Improvement | Percentage Improvement |
| :---: | :---: | :---: | :---: | :---: |
| **100 rows** | 0.7901 ms | **0.3659 ms** | **+0.4242 ms** | **+53.70%** |
| **500 rows** | 1.8807 ms | **1.0207 ms** | **+0.8600 ms** | **+45.73%** |
| **1,000 rows** | 3.7956 ms | **2.1140 ms** | **+1.6816 ms** | **+44.31%** |
| **5,000 rows** | 18.6650 ms | **10.2970 ms** | **+8.3680 ms** | **+44.83%** |

*Conclusion*: Filtering CSV parsing to the target feature set cuts server-side parsing overhead by **~44% to 53%** (saving **~8.37 ms** on 5,000 rows), while preserving 100% exact missing-feature validation (`MissingFeaturesError` returning `HTTP 400`).

---

## Real-Time <100 ms Target Analysis

- **Batch Size Compliance**: For batch sizes of **100, 500, and 1,000 rows**, single-request **p95 and p99 latencies are strictly under <100 ms** (e.g., 500 rows: p95 = 23.09 ms, p99 = 27.74 ms; 1,000 rows: p95 = 37.98 ms, p99 = 44.26 ms).
- **SHAP Compliance**: Single-instance SHAP explanations (`POST /api/v1/explain`) achieve **p95 = 22.24 ms**, satisfying the <100 ms target for analyst investigations.
- **Large Batch Trade-off**: At **5,000 rows**, p95 latency reaches **114.38 ms** due to CSV payload size and CPU execution matrix scaling.
- **Concurrency Scaling**: Under single-worker Uvicorn deployment, concurrency beyond 10 workers increases request queuing latency, maintaining a maximum aggregate throughput of **~39,000 rows/sec**.

---

## Reproduction Instructions

1. **Start the Dockerized Container Stack**:
   ```bash
   docker compose up --build -d
   ```

2. **Verify Backend Health**:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

3. **Run Pipeline Component Profiler**:
   ```bash
   python3 benchmarks/benchmark_batches.py
   ```

4. **Run Live HTTP API Benchmark Suite**:
   ```bash
   python3 benchmarks/benchmark_api.py
   ```
