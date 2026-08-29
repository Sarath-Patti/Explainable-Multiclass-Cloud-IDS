"""HTTP API Benchmark Suite for Explainable-Multiclass-Cloud-IDS running in Docker.

Evaluates:
  1. POST /api/v1/predict (Batch prediction latency, throughput, concurrency)
  2. POST /api/v1/explain (Single-instance SHAP explanation latency)
"""

import io
import json
import time
import pathlib
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

# Base URLs and paths
API_BASE_URL = "http://localhost:8000/api/v1"
PREDICT_URL = f"{API_BASE_URL}/predict"
EXPLAIN_URL = f"{API_BASE_URL}/explain"
HEALTH_URL = f"{API_BASE_URL}/health"

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "X_test_sample.csv"
RESULTS_DIR = BASE_DIR / "benchmarks" / "results"


def check_api_health() -> bool:
    """Verifies that the target FastAPI backend container is running and healthy."""
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "healthy"
    except Exception as e:
        print(f"API Health Check Failed: {e}")
        return False
    return False


def post_multipart_csv(url: str, csv_bytes: bytes, filename: str = "batch.csv") -> Tuple[int, float, int]:
    """Posts CSV data to multipart form endpoint and returns (status_code, latency_ms, bytes_received)."""
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: text/csv\r\n\r\n")
    body.extend(csv_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            t1 = time.perf_counter()
            content = resp.read()
            return resp.status, (t1 - t0) * 1000, len(content)
    except urllib.error.HTTPError as e:
        t1 = time.perf_counter()
        content = e.read()
        return e.code, (t1 - t0) * 1000, len(content)
    except Exception as e:
        t1 = time.perf_counter()
        return 500, (t1 - t0) * 1000, 0


def post_json(url: str, payload: Dict[str, Any]) -> Tuple[int, float, int]:
    """Posts JSON payload to REST endpoint and returns (status_code, latency_ms, bytes_received)."""
    json_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=json_bytes, method="POST")
    req.add_header("Content-Type", "application/json")

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            t1 = time.perf_counter()
            content = resp.read()
            return resp.status, (t1 - t0) * 1000, len(content)
    except urllib.error.HTTPError as e:
        t1 = time.perf_counter()
        content = e.read()
        return e.code, (t1 - t0) * 1000, len(content)
    except Exception:
        t1 = time.perf_counter()
        return 500, (t1 - t0) * 1000, 0


def warmup_api(csv_bytes_small: bytes, feature_sample: Dict[str, Any]):
    """Executes unmeasured warm-up requests to avoid startup cold-start distortion."""
    print("Executing API warm-up requests...")
    for _ in range(3):
        post_multipart_csv(PREDICT_URL, csv_bytes_small)
        post_json(EXPLAIN_URL, {"row": 0, "features": feature_sample})
    time.sleep(1)


def calculate_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculates min, max, mean, p50, p95, and p99 latency statistics."""
    arr = np.array(latencies_ms)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def benchmark_prediction_batch_sizes(df_full: pd.DataFrame, batch_sizes: List[int], repetitions: int = 10) -> List[Dict[str, Any]]:
    """Evaluates prediction latency and throughput across different batch sizes."""
    results = []
    print("\n" + "=" * 80)
    print(" 1. END-TO-END PREDICTION BENCHMARK (/api/v1/predict)")
    print("=" * 80)

    for b_size in batch_sizes:
        df_batch = df_full.iloc[:b_size].copy()
        csv_bytes = df_batch.to_csv(index=False).encode("utf-8")

        latencies: List[float] = []
        status_codes: Dict[int, int] = {}
        successes = 0
        failures = 0
        total_resp_bytes = 0

        for _ in range(repetitions):
            code, lat, bytes_len = post_multipart_csv(PREDICT_URL, csv_bytes, filename=f"batch_{b_size}.csv")
            latencies.append(lat)
            status_codes[code] = status_codes.get(code, 0) + 1
            if code == 200:
                successes += 1
                total_resp_bytes += bytes_len
            else:
                failures += 1

        stats = calculate_latency_stats(latencies)
        mean_lat_sec = stats["mean"] / 1000.0
        throughput = b_size / mean_lat_sec
        per_row_lat = stats["mean"] / b_size

        entry = {
            "batch_size": b_size,
            "repetitions": repetitions,
            "successes": successes,
            "failures": failures,
            "error_rate": failures / repetitions,
            "throughput_rows_sec": round(throughput, 2),
            "per_row_latency_ms": round(per_row_lat, 4),
            "p50_ms": round(stats["p50"], 2),
            "p95_ms": round(stats["p95"], 2),
            "p99_ms": round(stats["p99"], 2),
            "mean_ms": round(stats["mean"], 2),
            "min_ms": round(stats["min"], 2),
            "max_ms": round(stats["max"], 2),
        }
        results.append(entry)

        print(f"\nBatch Size: {b_size:5d} rows | Repetitions: {repetitions}")
        print(f"  Request Latency (ms):  p50: {stats['p50']:7.2f} | p95: {stats['p95']:7.2f} | p99: {stats['p99']:7.2f} | mean: {stats['mean']:7.2f}")
        print(f"  Per-Row Latency:       {per_row_lat:7.4f} ms/row")
        print(f"  Throughput:            {throughput:7.2f} rows/sec")
        print(f"  Success / Failure:     {successes} / {failures} (Error Rate: {failures/repetitions*100:.1f}%)")

    return results


def benchmark_prediction_concurrency(df_full: pd.DataFrame, batch_size: int = 500, concurrency_levels: List[int] = [1, 5, 10, 25, 50], requests_per_level: int = 20) -> List[Dict[str, Any]]:
    """Evaluates prediction latency and throughput under controlled concurrent load."""
    results = []
    df_batch = df_full.iloc[:batch_size].copy()
    csv_bytes = df_batch.to_csv(index=False).encode("utf-8")

    print("\n" + "=" * 80)
    print(f" 2. CONCURRENCY / LOAD BENCHMARK (/api/v1/predict, Batch Size: {batch_size} rows)")
    print("=" * 80)

    for conc in concurrency_levels:
        latencies: List[float] = []
        successes = 0
        failures = 0

        t_start_all = time.perf_counter()

        def worker(_):
            return post_multipart_csv(PREDICT_URL, csv_bytes, filename=f"conc_{conc}.csv")

        with ThreadPoolExecutor(max_workers=conc) as executor:
            futures = [executor.submit(worker, i) for i in range(requests_per_level)]
            for fut in as_completed(futures):
                code, lat, _ = fut.result()
                latencies.append(lat)
                if code == 200:
                    successes += 1
                else:
                    failures += 1

        t_end_all = time.perf_counter()
        total_time_sec = t_end_all - t_start_all

        stats = calculate_latency_stats(latencies)
        total_rows_processed = successes * batch_size
        overall_throughput = total_rows_processed / total_time_sec if total_time_sec > 0 else 0

        entry = {
            "concurrency": conc,
            "total_requests": requests_per_level,
            "batch_size": batch_size,
            "successes": successes,
            "failures": failures,
            "error_rate": failures / requests_per_level,
            "overall_throughput_rows_sec": round(overall_throughput, 2),
            "p50_ms": round(stats["p50"], 2),
            "p95_ms": round(stats["p95"], 2),
            "p99_ms": round(stats["p99"], 2),
            "mean_ms": round(stats["mean"], 2),
            "min_ms": round(stats["min"], 2),
            "max_ms": round(stats["max"], 2),
        }
        results.append(entry)

        print(f"\nConcurrency Level: {conc:2d} workers | Total Requests: {requests_per_level}")
        print(f"  Request Latency (ms):  p50: {stats['p50']:7.2f} | p95: {stats['p95']:7.2f} | p99: {stats['p99']:7.2f} | mean: {stats['mean']:7.2f}")
        print(f"  Aggregate Throughput:  {overall_throughput:7.2f} rows/sec")
        print(f"  Success / Failure:     {successes} / {failures} (Error Rate: {failures/requests_per_level*100:.1f}%)")

    return results


def benchmark_shap_explanation(df_full: pd.DataFrame, num_samples: int = 50) -> Dict[str, Any]:
    """Evaluates single-instance SHAP explanation request latency (/api/v1/explain)."""
    print("\n" + "=" * 80)
    print(" 3. SHAP EXPLANATION BENCHMARK (/api/v1/explain)")
    print("=" * 80)

    latencies: List[float] = []
    successes = 0
    failures = 0

    for i in range(num_samples):
        row_idx = i % len(df_full)
        row_dict = df_full.iloc[row_idx].to_dict()
        payload = {"row": row_idx, "features": row_dict}

        code, lat, _ = post_json(EXPLAIN_URL, payload)
        latencies.append(lat)
        if code == 200:
            successes += 1
        else:
            failures += 1

    stats = calculate_latency_stats(latencies)
    results = {
        "endpoint": "/api/v1/explain",
        "num_requests": num_samples,
        "successes": successes,
        "failures": failures,
        "error_rate": failures / num_samples,
        "p50_ms": round(stats["p50"], 2),
        "p95_ms": round(stats["p95"], 2),
        "p99_ms": round(stats["p99"], 2),
        "mean_ms": round(stats["mean"], 2),
        "min_ms": round(stats["min"], 2),
        "max_ms": round(stats["max"], 2),
    }

    print(f"Evaluated {num_samples} single-instance SHAP requests:")
    print(f"  SHAP Latency (ms):     p50: {stats['p50']:7.2f} | p95: {stats['p95']:7.2f} | p99: {stats['p99']:7.2f} | mean: {stats['mean']:7.2f}")
    print(f"  Success / Failure:     {successes} / {failures}")

    return results


def main():
    print("=" * 80)
    print(" EXPLAINABLE-MULTICLASS-CLOUD-IDS: REAL-TIME API BENCHMARK SUITE")
    print("=" * 80)

    if not check_api_health():
        print(f"Error: Unable to connect to healthy backend container at {HEALTH_URL}")
        print("Please ensure Docker container stack is running (docker compose up -d).")
        return

    if not DATA_PATH.exists():
        print(f"Error: Sample data file not found at {DATA_PATH}")
        return

    df_full = pd.read_csv(DATA_PATH)
    print(f"Target Container: {API_BASE_URL}")
    print(f"Dataset Loaded:   {DATA_PATH.name} ({len(df_full)} rows)")

    # 0. Warm-Up
    sample_small = df_full.iloc[:10].to_csv(index=False).encode("utf-8")
    sample_feat = df_full.iloc[0].to_dict()
    warmup_api(sample_small, sample_feat)

    # 1. Batch Prediction Latency & Throughput Benchmark
    batch_sizes = [100, 500, 1000, 5000]
    batch_sizes = [b for b in batch_sizes if b <= len(df_full)]
    batch_results = benchmark_prediction_batch_sizes(df_full, batch_sizes, repetitions=10)

    # 2. Concurrency Load Benchmark
    concurrency_results = benchmark_prediction_concurrency(df_full, batch_size=500, concurrency_levels=[1, 5, 10, 25, 50], requests_per_level=20)

    # 3. SHAP Explanation Benchmark
    shap_results = benchmark_shap_explanation(df_full, num_samples=50)

    # Save Results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_batch_benchmarks": batch_results,
        "concurrency_benchmarks": concurrency_results,
        "shap_explanation_benchmark": shap_results,
    }

    out_json = RESULTS_DIR / "api_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # Also output batch results CSV
    out_csv = RESULTS_DIR / "prediction_batch_results.csv"
    pd.DataFrame(batch_results).to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETED SUCCESSFULLY")
    print(f"  - Summary JSON saved to: {out_json}")
    print(f"  - Batch CSV saved to:   {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
