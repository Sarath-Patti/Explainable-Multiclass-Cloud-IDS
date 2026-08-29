"""Microbenchmark comparing Original vs. Optimized response formatting implementations in PredictorService."""

import time
import json
import pathlib
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Base paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "X_test_sample.csv"
RESULTS_DIR = BASE_DIR / "benchmarks" / "results"

# Import schemas and model loader
import sys
sys.path.insert(0, str(BASE_DIR / "backend"))
from app.services.model_loader import model_loader
from app.schemas.predict import PredictionItem, PredictionSummary, PredictionResponse


def format_original(pred_indices: np.ndarray, confidences: np.ndarray, index_to_label: Dict[int, str], n_samples: int) -> PredictionResponse:
    """Original response formatting implementation (sequential loop with item appends)."""
    prediction_items: List[PredictionItem] = []
    benign_count = 0

    for i in range(n_samples):
        idx = int(pred_indices[i])
        label = index_to_label.get(idx, f"Unknown_{idx}")
        conf = float(round(confidences[i], 4))

        if label.upper() == "BENIGN":
            benign_count += 1

        prediction_items.append(
            PredictionItem(
                row=i,
                prediction=label,
                confidence=conf
            )
        )

    total_samples = n_samples
    attack_count = total_samples - benign_count

    summary = PredictionSummary(
        total_samples=total_samples,
        predicted_attacks=attack_count,
        predicted_benign=benign_count
    )
    return PredictionResponse(summary=summary, predictions=prediction_items)


def format_optimized(pred_indices: np.ndarray, confidences: np.ndarray, index_to_label: Dict[int, str], n_samples: int) -> PredictionResponse:
    """Current optimized response formatting implementation (vectorized comprehensions)."""
    labels = [index_to_label.get(int(idx), f"Unknown_{idx}") for idx in pred_indices]
    rounded_confidences = np.round(confidences, 4)

    benign_count = sum(1 for lbl in labels if lbl.upper() == "BENIGN")

    prediction_items = [
        PredictionItem(
            row=i,
            prediction=labels[i],
            confidence=float(rounded_confidences[i])
        )
        for i in range(n_samples)
    ]

    total_samples = n_samples
    attack_count = total_samples - benign_count

    summary = PredictionSummary(
        total_samples=total_samples,
        predicted_attacks=attack_count,
        predicted_benign=benign_count
    )
    return PredictionResponse(summary=summary, predictions=prediction_items)


def calculate_stats(times_ms: List[float]) -> Dict[str, float]:
    """Calculates summary statistics for a list of timing measurements in milliseconds."""
    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def benchmark_formatting(batch_sizes: List[int] = [100, 500, 1000, 5000], repetitions: int = 50) -> List[Dict[str, Any]]:
    """Runs isolated microbenchmark comparing Original vs. Optimized formatting on identical model output arrays."""
    model_loader.load_artifacts()
    expected_features = model_loader.expected_features
    model = model_loader.model
    index_to_label = model_loader.index_to_label

    df_full = pd.read_csv(DATA_PATH)
    comparison_results = []

    print("=" * 90)
    print(" MICROBENCHMARK: RESPONSE FORMATTING (ORIGINAL vs. OPTIMIZED)")
    print("=" * 90)

    for b_size in batch_sizes:
        if b_size > len(df_full):
            continue

        # Generate exact model outputs for this batch size
        df_batch = df_full.iloc[:b_size].copy()
        X = df_batch.loc[:, expected_features].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        probabilities = model.predict_proba(X)
        pred_indices = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        # Verify equivalence of outputs from both implementations
        resp_orig = format_original(pred_indices, confidences, index_to_label, b_size)
        resp_opt = format_optimized(pred_indices, confidences, index_to_label, b_size)

        assert resp_orig.summary.total_samples == resp_opt.summary.total_samples
        assert resp_orig.summary.predicted_benign == resp_opt.summary.predicted_benign
        assert resp_orig.summary.predicted_attacks == resp_opt.summary.predicted_attacks
        assert len(resp_orig.predictions) == len(resp_opt.predictions)
        for idx in range(b_size):
            assert resp_orig.predictions[idx].prediction == resp_opt.predictions[idx].prediction
            assert resp_orig.predictions[idx].confidence == resp_opt.predictions[idx].confidence

        # Warm-up phase (10 iterations)
        for _ in range(10):
            _ = format_original(pred_indices, confidences, index_to_label, b_size)
            _ = format_optimized(pred_indices, confidences, index_to_label, b_size)

        # Benchmark Original Implementation
        times_orig: List[float] = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            _ = format_original(pred_indices, confidences, index_to_label, b_size)
            t1 = time.perf_counter()
            times_orig.append((t1 - t0) * 1000)

        # Benchmark Optimized Implementation
        times_opt: List[float] = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            _ = format_optimized(pred_indices, confidences, index_to_label, b_size)
            t1 = time.perf_counter()
            times_opt.append((t1 - t0) * 1000)

        stats_orig = calculate_stats(times_orig)
        stats_opt = calculate_stats(times_opt)

        abs_diff_ms = stats_orig["mean_ms"] - stats_opt["mean_ms"]
        pct_diff = (abs_diff_ms / stats_orig["mean_ms"]) * 100 if stats_orig["mean_ms"] > 0 else 0.0

        per_row_orig_ms = stats_orig["mean_ms"] / b_size
        per_row_opt_ms = stats_opt["mean_ms"] / b_size

        res_entry = {
            "batch_size": b_size,
            "repetitions": repetitions,
            "original": stats_orig,
            "optimized": stats_opt,
            "original_per_row_ms": round(per_row_orig_ms, 6),
            "optimized_per_row_ms": round(per_row_opt_ms, 6),
            "improvement_ms": round(abs_diff_ms, 4),
            "percent_improvement": round(pct_diff, 2),
        }
        comparison_results.append(res_entry)

        print(f"\nBatch Size: {b_size:5d} rows ({repetitions} repetitions)")
        print(f"  Original  Mean: {stats_orig['mean_ms']:7.4f} ms | p50: {stats_orig['p50_ms']:7.4f} ms | p95: {stats_orig['p95_ms']:7.4f} ms | p99: {stats_orig['p99_ms']:7.4f} ms")
        print(f"  Optimized Mean: {stats_opt['mean_ms']:7.4f} ms | p50: {stats_opt['p50_ms']:7.4f} ms | p95: {stats_opt['p95_ms']:7.4f} ms | p99: {stats_opt['p99_ms']:7.4f} ms")
        print(f"  Delta:          {abs_diff_ms:+7.4f} ms ({pct_diff:+6.2f}%)")

    # Output Summary Table
    print("\n" + "=" * 90)
    print(" SUMMARY TABLE: RESPONSE FORMATTING PERFORMANCE COMPARISON")
    print("=" * 90)
    print(f"| {'Batch Size':^10} | {'Original Mean (ms)':^18} | {'Optimized Mean (ms)':^20} | {'Improvement (ms)':^18} | {'% Improvement':^15} |")
    print(f"|{'-'*12}|{'-'*20}|{'-'*22}|{'-'*20}|{'-'*17}|")
    for r in comparison_results:
        print(f"| {r['batch_size']:^10d} | {r['original']['mean_ms']:^18.4f} | {r['optimized']['mean_ms']:^20.4f} | {r['improvement_ms']:^+18.4f} | {r['percent_improvement']:^+15.2f}% |")
    print("=" * 90)

    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "formatting_microbenchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nMicrobenchmark results saved to: {out_json}")

    return comparison_results


if __name__ == "__main__":
    benchmark_formatting()
