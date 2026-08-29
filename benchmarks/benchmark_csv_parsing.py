"""Isolated microbenchmark for server-side CSV parsing strategies in PredictorService."""

import io
import time
import json
import pathlib
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "X_test_sample.csv"
RESULTS_DIR = BASE_DIR / "benchmarks" / "results"

import sys
sys.path.insert(0, str(BASE_DIR / "backend"))
from app.services.model_loader import model_loader
from app.services.predictor import MissingFeaturesError


def parse_strategy_0_baseline(content: bytes, expected_features: List[str]) -> pd.DataFrame:
    """Strategy 0 (Current Baseline): Standard pd.read_csv(io.BytesIO(content))."""
    df = pd.read_csv(io.BytesIO(content))
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise MissingFeaturesError(missing)
    return df


def parse_strategy_1_c_engine(content: bytes, expected_features: List[str]) -> pd.DataFrame:
    """Strategy 1: pd.read_csv with engine='c' and low_memory=False."""
    df = pd.read_csv(io.BytesIO(content), engine="c", low_memory=False)
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise MissingFeaturesError(missing)
    return df


def parse_strategy_2_usecols_header_check(content: bytes, expected_features: List[str]) -> pd.DataFrame:
    """Strategy 2: Header check via nrows=0, then read_csv with usecols=expected_features."""
    # 1. Read header line to validate missing columns
    header_df = pd.read_csv(io.BytesIO(content), nrows=0)
    missing = [f for f in expected_features if f not in header_df.columns]
    if missing:
        raise MissingFeaturesError(missing)
    # 2. Parse only expected columns
    df = pd.read_csv(io.BytesIO(content), usecols=expected_features, engine="c")
    return df


def parse_strategy_3_usecols_lambda(content: bytes, expected_features: List[str]) -> pd.DataFrame:
    """Strategy 3: Single-pass read_csv using set lookup for usecols."""
    expected_set = set(expected_features)
    df = pd.read_csv(io.BytesIO(content), usecols=lambda c: c in expected_set, engine="c")
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise MissingFeaturesError(missing)
    return df


def calculate_stats(times_ms: List[float]) -> Dict[str, float]:
    """Calculates latency statistics."""
    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def benchmark_csv_parsing(batch_sizes: List[int] = [100, 500, 1000, 5000], repetitions: int = 50) -> List[Dict[str, Any]]:
    """Runs microbenchmark comparing CSV parsing strategies on identical CSV byte streams."""
    model_loader.load_artifacts()
    expected_features = model_loader.expected_features

    df_full = pd.read_csv(DATA_PATH)
    results = []

    print("=" * 95)
    print(" MICROBENCHMARK: SERVER-SIDE CSV PARSING STRATEGIES")
    print("=" * 95)

    for b_size in batch_sizes:
        if b_size > len(df_full):
            continue

        df_batch = df_full.iloc[:b_size].copy()
        csv_bytes = df_batch.to_csv(index=False).encode("utf-8")

        # Correctness check: ensure all strategies return equivalent DataFrames for expected_features
        df0 = parse_strategy_0_baseline(csv_bytes, expected_features)
        df1 = parse_strategy_1_c_engine(csv_bytes, expected_features)
        df2 = parse_strategy_2_usecols_header_check(csv_bytes, expected_features)
        df3 = parse_strategy_3_usecols_lambda(csv_bytes, expected_features)

        for feat in expected_features:
            assert np.allclose(df0[feat].values, df1[feat].values, equal_nan=True)
            assert np.allclose(df0[feat].values, df2[feat].values, equal_nan=True)
            assert np.allclose(df0[feat].values, df3[feat].values, equal_nan=True)

        # Warm-up (10 runs)
        for _ in range(10):
            _ = parse_strategy_0_baseline(csv_bytes, expected_features)
            _ = parse_strategy_1_c_engine(csv_bytes, expected_features)
            _ = parse_strategy_2_usecols_header_check(csv_bytes, expected_features)
            _ = parse_strategy_3_usecols_lambda(csv_bytes, expected_features)

        # Run benchmarks
        strategies = {
            "Strategy 0 (Baseline)": lambda b=csv_bytes: parse_strategy_0_baseline(b, expected_features),
            "Strategy 1 (C Engine)": lambda b=csv_bytes: parse_strategy_1_c_engine(b, expected_features),
            "Strategy 2 (Usecols Header Check)": lambda b=csv_bytes: parse_strategy_2_usecols_header_check(b, expected_features),
            "Strategy 3 (Usecols Lambda)": lambda b=csv_bytes: parse_strategy_3_usecols_lambda(b, expected_features),
        }

        batch_metrics = {}
        for strat_name, strat_fn in strategies.items():
            times: List[float] = []
            for _ in range(repetitions):
                t0 = time.perf_counter()
                _ = strat_fn()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            batch_metrics[strat_name] = calculate_stats(times)

        entry = {"batch_size": b_size, "repetitions": repetitions, "metrics": batch_metrics}
        results.append(entry)

        base_mean = batch_metrics["Strategy 0 (Baseline)"]["mean_ms"]
        print(f"\nBatch Size: {b_size:5d} rows ({repetitions} repetitions)")
        for name, m in batch_metrics.items():
            delta = base_mean - m["mean_ms"]
            pct = (delta / base_mean) * 100 if base_mean > 0 else 0.0
            print(f"  {name:<35s}: mean = {m['mean_ms']:7.4f} ms | p50 = {m['p50_ms']:7.4f} ms | p95 = {m['p95_ms']:7.4f} ms | delta = {delta:+7.4f} ms ({pct:+6.2f}%)")

    # Save summary table JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "csv_parsing_microbenchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nCSV Parsing microbenchmark results saved to: {out_json}")
    return results


if __name__ == "__main__":
    benchmark_csv_parsing()
