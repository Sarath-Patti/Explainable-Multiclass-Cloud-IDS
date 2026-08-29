"""Component-level pipeline profiler for prediction inference path in Explainable-Multiclass-Cloud-IDS."""

import io
import json
import time
import pathlib
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "X_test_sample.csv"
RESULTS_DIR = BASE_DIR / "benchmarks" / "results"

# Import backend prediction service components for isolated profiling
import sys
sys.path.insert(0, str(BASE_DIR / "backend"))
from app.services.model_loader import model_loader


def profile_prediction_pipeline(df_input: pd.DataFrame, num_runs: int = 10) -> Dict[str, Any]:
    """Profiles individual component execution times for prediction pipeline."""
    model_loader.load_artifacts()
    expected_features = model_loader.expected_features
    model = model_loader.model
    index_to_label = model_loader.index_to_label

    t_csv_bytes: List[float] = []
    t_csv_parse: List[float] = []
    t_feat_slice: List[float] = []
    t_clean: List[float] = []
    t_model_exec: List[float] = []
    t_format: List[float] = []
    t_total: List[float] = []

    # Warm-up run
    csv_bytes = df_input.to_csv(index=False).encode("utf-8")
    df_parsed = pd.read_csv(io.BytesIO(csv_bytes))
    X_sliced = df_parsed.loc[:, expected_features].copy()
    X_cleaned = X_sliced.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    _ = model.predict_proba(X_cleaned)

    for _ in range(num_runs):
        t0 = time.perf_counter()

        # Step 1: CSV Serialization
        t1_start = time.perf_counter()
        csv_data = df_input.to_csv(index=False).encode("utf-8")
        t1_end = time.perf_counter()

        # Step 2: CSV Parsing
        t2_start = time.perf_counter()
        df = pd.read_csv(io.BytesIO(csv_data))
        t2_end = time.perf_counter()

        # Step 3: Feature Validation & Slicing
        t3_start = time.perf_counter()
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X = df.loc[:, expected_features].copy()
        t3_end = time.perf_counter()

        # Step 4: Value Cleaning (NaN/Inf)
        t4_start = time.perf_counter()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        t4_end = time.perf_counter()

        # Step 5: Model Inference
        t5_start = time.perf_counter()
        probabilities = model.predict_proba(X)
        t5_end = time.perf_counter()

        # Step 6: Response Extraction & Formatting
        t6_start = time.perf_counter()
        pred_indices = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        items = []
        benign_cnt = 0
        for i in range(len(X)):
            idx = int(pred_indices[i])
            lbl = index_to_label.get(idx, f"Unknown_{idx}")
            conf = float(round(confidences[i], 4))
            if lbl.upper() == "BENIGN":
                benign_cnt += 1
            items.append({"row": i, "prediction": lbl, "confidence": conf})
        t6_end = time.perf_counter()

        t_end = time.perf_counter()

        t_csv_bytes.append((t1_end - t1_start) * 1000)
        t_csv_parse.append((t2_end - t2_start) * 1000)
        t_feat_slice.append((t3_end - t3_start) * 1000)
        t_clean.append((t4_end - t4_start) * 1000)
        t_model_exec.append((t5_end - t5_start) * 1000)
        t_format.append((t6_end - t6_start) * 1000)
        t_total.append((t_end - t0) * 1000)

    rows = len(df_input)
    return {
        "num_rows": rows,
        "num_runs": num_runs,
        "mean_csv_bytes_ms": float(np.mean(t_csv_bytes)),
        "mean_csv_parse_ms": float(np.mean(t_csv_parse)),
        "mean_feat_slice_ms": float(np.mean(t_feat_slice)),
        "mean_clean_ms": float(np.mean(t_clean)),
        "mean_model_exec_ms": float(np.mean(t_model_exec)),
        "mean_format_ms": float(np.mean(t_format)),
        "mean_total_ms": float(np.mean(t_total)),
        "per_row_ms": float(np.mean(t_total) / rows),
        "throughput_rows_sec": float(rows / (np.mean(t_total) / 1000)),
    }


def main():
    print("=" * 80)
    print(" EXPLAINABLE-MULTICLASS-CLOUD-IDS: PIPELINE PROFILING BENCHMARK")
    print("=" * 80)

    if not DATA_PATH.exists():
        print(f"Error: Sample data file not found at {DATA_PATH}")
        return

    df_full = pd.read_csv(DATA_PATH)
    total_available = len(df_full)
    print(f"Loaded source dataset: {DATA_PATH.name} ({total_available} total rows)")

    batch_sizes = [100, 500, 1000, 5000]
    batch_sizes = [b for b in batch_sizes if b <= total_available]

    profiling_results = []

    for b_size in batch_sizes:
        print(f"\n--- Profiling Pipeline for Batch Size: {b_size} rows ---")
        df_batch = df_full.iloc[:b_size].copy()
        res = profile_prediction_pipeline(df_batch, num_runs=15)
        profiling_results.append(res)

        print(f"  Total Pipeline Time (mean):  {res['mean_total_ms']:.2f} ms")
        print(f"  Per-Row Latency (mean):       {res['per_row_ms']:.4f} ms/row")
        print(f"  Throughput:                   {res['throughput_rows_sec']:.2f} rows/sec")
        print("  Component Breakdown:")
        print(f"    - CSV Bytes Generation:     {res['mean_csv_bytes_ms']:.2f} ms ({res['mean_csv_bytes_ms']/res['mean_total_ms']*100:.1f}%)")
        print(f"    - CSV Parsing (pd.read_csv):{res['mean_csv_parse_ms']:.2f} ms ({res['mean_csv_parse_ms']/res['mean_total_ms']*100:.1f}%)")
        print(f"    - Feature Slicing:          {res['mean_feat_slice_ms']:.2f} ms ({res['mean_feat_slice_ms']/res['mean_total_ms']*100:.1f}%)")
        print(f"    - Value Cleaning (NaN/Inf): {res['mean_clean_ms']:.2f} ms ({res['mean_clean_ms']/res['mean_total_ms']*100:.1f}%)")
        print(f"    - XGBoost Model Execution:  {res['mean_model_exec_ms']:.2f} ms ({res['mean_model_exec_ms']/res['mean_total_ms']*100:.1f}%)")
        print(f"    - Response Formatting:      {res['mean_format_ms']:.2f} ms ({res['mean_format_ms']/res['mean_total_ms']*100:.1f}%)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "pipeline_profiling_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profiling_results, f, indent=2)

    print(f"\nProfiling results successfully saved to: {out_json}")


if __name__ == "__main__":
    main()
