"""
Storage Layout & Compression Benchmark for FDIC Financials Panel.
Course: BFT65B - Big Data Analytics in Finance
Subagent 1: Data Pipeline Engineer

Supports two modes:
1. --mode prototype (default): Benchmarks 4 quarters of 2023 across 6 layouts (5 iterations).
2. --mode full: Benchmarks entire 170-quarter panel (~1.68M rows) across 5 layouts:
   - CSV (Uncompressed, single file)
   - CSV.gz (Gzip compressed, single file)
   - Parquet (Uncompressed, partitioned by REPDTE)
   - Parquet (Snappy, partitioned by REPDTE)
   - Parquet (ZSTD, partitioned by REPDTE)
   (1 warmup + 3 timed iterations, predicate: REPDTE == '20091231')

Measures:
- Disk footprint (MB & compression ratio vs CSV)
- Write time (median & mean ms)
- Full scan read time (Pandas & Polars, ms)
- Column projection pushdown (5 CAMELS cols: CERT, REPDTE, ASSET, EQ, ROA, ms)
- Predicate pushdown + projection (ms)
- In-memory footprint (Pandas & Polars, MB)

Saves to:
- reports/storage_benchmark_prototype.{json,csv} OR reports/storage_benchmark_full.{json,csv}
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import DATA_RAW_FIN, REPORTS_DIR, FIN_FIELDS
from src.data.acquire import FINANCIALS_SCHEMA

sys.stdout.reconfigure(encoding="utf-8")

BENCHMARK_TMP = Path("data/benchmark_tmp")
PROJECTION_COLS = ["CERT", "REPDTE", "ASSET", "EQ", "ROA"]


def get_dir_size_bytes(path: Path) -> int:
    """Recursively calculate the total byte size of a file or directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def time_execution(fn: Callable[[], Any], warmup: int = 1, iterations: int = 3) -> tuple[float, float, Any]:
    """Execute fn with warmup and return (median_ms, mean_ms, last_result)."""
    for _ in range(warmup):
        res = fn()
    durations = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = fn()
        durations.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(durations)), float(np.mean(durations)), res


def run_benchmark(mode: str = "full") -> pd.DataFrame:
    print("===================================================================")
    print(f"      STORAGE LAYOUT & COMPRESSION BENCHMARK (MODE: {mode.upper()})     ")
    print("===================================================================\n")

    is_full = (mode == "full")
    iterations = 3 if is_full else 5
    predicate_quarter = "20091231" if is_full else "20230331"

    source_files = sorted(list(DATA_RAW_FIN.glob("REPDTE=*.parquet")))
    if not source_files:
        raise FileNotFoundError(f"No parquet files found in {DATA_RAW_FIN}")

    if not is_full:
        # Filter to 2023 only for prototype mode
        source_files = [f for f in source_files if "2023" in f.stem]

    print(f"Loading {len(source_files)} source partitions into memory...")
    dfs_by_quarter: dict[str, pd.DataFrame] = {}
    for f in source_files:
        q_name = f.stem.split("=")[-1]
        df_q = pd.read_parquet(f)
        dfs_by_quarter[q_name] = df_q

    full_df = pd.concat(list(dfs_by_quarter.values()), ignore_index=True)
    full_table = pa.Table.from_pandas(full_df, schema=FINANCIALS_SCHEMA, preserve_index=False)
    quarters = sorted(list(dfs_by_quarter.keys()))

    print(f"Full panel shape: {full_df.shape} ({len(quarters)} quarters, {full_df['CERT'].nunique():,} distinct banks)")
    print(f"Warmup: 1, Iterations: {iterations}, Predicate Target: REPDTE == '{predicate_quarter}'")

    if BENCHMARK_TMP.exists():
        shutil.rmtree(BENCHMARK_TMP)
    BENCHMARK_TMP.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    p_csv_single = BENCHMARK_TMP / "panel_single.csv"
    p_csvgz_single = BENCHMARK_TMP / "panel_single.csv.gz"
    p_csv_part = BENCHMARK_TMP / "csv_partitioned"
    p_pq_uncompressed = BENCHMARK_TMP / "parquet_uncompressed"
    p_pq_snappy = BENCHMARK_TMP / "parquet_snappy"
    p_pq_zstd = BENCHMARK_TMP / "parquet_zstd"

    # ------------------------------------------------------------------------
    # 1. CSV (Uncompressed, single file)
    # ------------------------------------------------------------------------
    print("\n--- Benchmarking: CSV (Uncompressed, single file) ---")
    def write_csv_single():
        full_df.to_csv(p_csv_single, index=False)
    w_med, w_mean, _ = time_execution(write_csv_single, warmup=1, iterations=iterations)
    disk_bytes = get_dir_size_bytes(p_csv_single)
    base_csv_bytes = disk_bytes

    def read_pd_csv_single():
        return pd.read_csv(p_csv_single)
    r_pd_med, r_pd_mean, df_pd_csv = time_execution(read_pd_csv_single, warmup=1, iterations=iterations)
    mem_pd_mb = df_pd_csv.memory_usage(deep=True).sum() / (1024 * 1024)

    def read_pl_csv_single():
        return pl.read_csv(p_csv_single)
    r_pl_med, r_pl_mean, df_pl_csv = time_execution(read_pl_csv_single, warmup=1, iterations=iterations)
    mem_pl_mb = df_pl_csv.estimated_size() / (1024 * 1024)

    def proj_pl_csv_single():
        return pl.scan_csv(p_csv_single).select(PROJECTION_COLS).collect()
    proj_med, proj_mean, _ = time_execution(proj_pl_csv_single, warmup=1, iterations=iterations)

    def pred_proj_pl_csv_single():
        return (
            pl.scan_csv(p_csv_single)
            .filter(pl.col("REPDTE").cast(pl.String) == predicate_quarter)
            .select(PROJECTION_COLS)
            .collect()
        )
    pred_med, pred_mean, _ = time_execution(pred_proj_pl_csv_single, warmup=1, iterations=iterations)

    results.append({
        "layout": "CSV (Uncompressed, single file)",
        "disk_size_bytes": disk_bytes,
        "disk_size_kb": disk_bytes / 1024,
        "disk_size_mb": disk_bytes / (1024 * 1024),
        "compression_ratio_vs_csv": base_csv_bytes / disk_bytes,
        "write_time_median_ms": w_med,
        "write_time_mean_ms": w_mean,
        "full_read_pandas_median_ms": r_pd_med,
        "full_read_pandas_mean_ms": r_pd_mean,
        "full_read_polars_median_ms": r_pl_med,
        "full_read_polars_mean_ms": r_pl_mean,
        "projection_read_polars_median_ms": proj_med,
        "projection_read_polars_mean_ms": proj_mean,
        "predicate_projection_read_polars_median_ms": pred_med,
        "predicate_projection_read_polars_mean_ms": pred_mean,
        "in_memory_pandas_mb": mem_pd_mb,
        "in_memory_polars_mb": mem_pl_mb,
    })

    # Optional: CSV partitioned (only in prototype mode)
    if not is_full:
        print("--- Benchmarking: CSV (Uncompressed, partitioned by quarter) ---")
        p_csv_part.mkdir(parents=True, exist_ok=True)
        def write_csv_part():
            for q, df_q in dfs_by_quarter.items():
                df_q.to_csv(p_csv_part / f"REPDTE={q}.csv", index=False)
        w_med, w_mean, _ = time_execution(write_csv_part, warmup=1, iterations=iterations)
        disk_bytes = get_dir_size_bytes(p_csv_part)

        csv_part_files = sorted(list(p_csv_part.glob("*.csv")))
        def read_pd_csv_part():
            return pd.concat([pd.read_csv(f) for f in csv_part_files], ignore_index=True)
        r_pd_med, r_pd_mean, df_pd = time_execution(read_pd_csv_part, warmup=1, iterations=iterations)
        mem_pd_mb = df_pd.memory_usage(deep=True).sum() / (1024 * 1024)

        part_csv_glob = str(p_csv_part / "*.csv")
        def read_pl_csv_part():
            return pl.read_csv(part_csv_glob)
        r_pl_med, r_pl_mean, df_pl = time_execution(read_pl_csv_part, warmup=1, iterations=iterations)
        mem_pl_mb = df_pl.estimated_size() / (1024 * 1024)

        def proj_pl_csv_part():
            return pl.scan_csv(part_csv_glob).select(PROJECTION_COLS).collect()
        proj_med, proj_mean, _ = time_execution(proj_pl_csv_part, warmup=1, iterations=iterations)

        def pred_proj_pl_csv_part():
            return (
                pl.scan_csv(part_csv_glob)
                .filter(pl.col("REPDTE").cast(pl.String) == predicate_quarter)
                .select(PROJECTION_COLS)
                .collect()
            )
        pred_med, pred_mean, _ = time_execution(pred_proj_pl_csv_part, warmup=1, iterations=iterations)

        results.append({
            "layout": "CSV (Uncompressed, partitioned by quarter)",
            "disk_size_bytes": disk_bytes,
            "disk_size_kb": disk_bytes / 1024,
            "disk_size_mb": disk_bytes / (1024 * 1024),
            "compression_ratio_vs_csv": base_csv_bytes / disk_bytes,
            "write_time_median_ms": w_med,
            "write_time_mean_ms": w_mean,
            "full_read_pandas_median_ms": r_pd_med,
            "full_read_pandas_mean_ms": r_pd_mean,
            "full_read_polars_median_ms": r_pl_med,
            "full_read_polars_mean_ms": r_pl_mean,
            "projection_read_polars_median_ms": proj_med,
            "projection_read_polars_mean_ms": proj_mean,
            "predicate_projection_read_polars_median_ms": pred_med,
            "predicate_projection_read_polars_mean_ms": pred_mean,
            "in_memory_pandas_mb": mem_pd_mb,
            "in_memory_polars_mb": mem_pl_mb,
        })

    # ------------------------------------------------------------------------
    # 2. CSV.gz (Gzip compressed, single file)
    # ------------------------------------------------------------------------
    print("--- Benchmarking: CSV.gz (Gzip compressed, single file) ---")
    def write_csvgz_single():
        full_df.to_csv(p_csvgz_single, compression="gzip", index=False)
    w_med, w_mean, _ = time_execution(write_csvgz_single, warmup=1, iterations=iterations)
    disk_bytes = get_dir_size_bytes(p_csvgz_single)

    def read_pd_csvgz_single():
        return pd.read_csv(p_csvgz_single, compression="gzip")
    r_pd_med, r_pd_mean, df_pd = time_execution(read_pd_csvgz_single, warmup=1, iterations=iterations)
    mem_pd_mb = df_pd.memory_usage(deep=True).sum() / (1024 * 1024)

    def read_pl_csvgz_single():
        return pl.read_csv(p_csvgz_single)
    r_pl_med, r_pl_mean, df_pl = time_execution(read_pl_csvgz_single, warmup=1, iterations=iterations)
    mem_pl_mb = df_pl.estimated_size() / (1024 * 1024)

    def proj_pl_csvgz_single():
        return pl.scan_csv(p_csvgz_single).select(PROJECTION_COLS).collect()
    proj_med, proj_mean, _ = time_execution(proj_pl_csvgz_single, warmup=1, iterations=iterations)

    def pred_proj_pl_csvgz_single():
        return (
            pl.scan_csv(p_csvgz_single)
            .filter(pl.col("REPDTE").cast(pl.String) == predicate_quarter)
            .select(PROJECTION_COLS)
            .collect()
        )
    pred_med, pred_mean, _ = time_execution(pred_proj_pl_csvgz_single, warmup=1, iterations=iterations)

    results.append({
        "layout": "CSV.gz (Gzip compressed, single file)",
        "disk_size_bytes": disk_bytes,
        "disk_size_kb": disk_bytes / 1024,
        "disk_size_mb": disk_bytes / (1024 * 1024),
        "compression_ratio_vs_csv": base_csv_bytes / disk_bytes,
        "write_time_median_ms": w_med,
        "write_time_mean_ms": w_mean,
        "full_read_pandas_median_ms": r_pd_med,
        "full_read_pandas_mean_ms": r_pd_mean,
        "full_read_polars_median_ms": r_pl_med,
        "full_read_polars_mean_ms": r_pl_mean,
        "projection_read_polars_median_ms": proj_med,
        "projection_read_polars_mean_ms": proj_mean,
        "predicate_projection_read_polars_median_ms": pred_med,
        "predicate_projection_read_polars_mean_ms": pred_mean,
        "in_memory_pandas_mb": mem_pd_mb,
        "in_memory_polars_mb": mem_pl_mb,
    })

    # ------------------------------------------------------------------------
    # Helper for Parquet Layouts (Partitioned by REPDTE)
    # ------------------------------------------------------------------------
    def benchmark_parquet_layout(name: str, target_dir: Path, codec: str | None):
        print(f"--- Benchmarking: {name} ---")
        target_dir.mkdir(parents=True, exist_ok=True)

        def write_pq():
            for q, df_q in dfs_by_quarter.items():
                t = pa.Table.from_pandas(df_q, schema=FINANCIALS_SCHEMA, preserve_index=False)
                pq.write_table(t, target_dir / f"REPDTE={q}.parquet", compression=codec)

        w_med, w_mean, _ = time_execution(write_pq, warmup=1, iterations=iterations)
        disk_bytes = get_dir_size_bytes(target_dir)

        def read_pd_pq():
            return pd.read_parquet(target_dir)
        r_pd_med, r_pd_mean, df_pd = time_execution(read_pd_pq, warmup=1, iterations=iterations)
        mem_pd_mb = df_pd.memory_usage(deep=True).sum() / (1024 * 1024)

        pq_glob = str(target_dir / "*.parquet")
        def read_pl_pq():
            return pl.read_parquet(pq_glob)
        r_pl_med, r_pl_mean, df_pl = time_execution(read_pl_pq, warmup=1, iterations=iterations)
        mem_pl_mb = df_pl.estimated_size() / (1024 * 1024)

        def proj_pl_pq():
            return pl.scan_parquet(pq_glob).select(PROJECTION_COLS).collect()
        proj_med, proj_mean, _ = time_execution(proj_pl_pq, warmup=1, iterations=iterations)

        def pred_proj_pl_pq():
            return (
                pl.scan_parquet(pq_glob)
                .filter(pl.col("REPDTE") == predicate_quarter)
                .select(PROJECTION_COLS)
                .collect()
            )
        pred_med, pred_mean, _ = time_execution(pred_proj_pl_pq, warmup=1, iterations=iterations)

        return {
            "layout": name,
            "disk_size_bytes": disk_bytes,
            "disk_size_kb": disk_bytes / 1024,
            "disk_size_mb": disk_bytes / (1024 * 1024),
            "compression_ratio_vs_csv": base_csv_bytes / disk_bytes,
            "write_time_median_ms": w_med,
            "write_time_mean_ms": w_mean,
            "full_read_pandas_median_ms": r_pd_med,
            "full_read_pandas_mean_ms": r_pd_mean,
            "full_read_polars_median_ms": r_pl_med,
            "full_read_polars_mean_ms": r_pl_mean,
            "projection_read_polars_median_ms": proj_med,
            "projection_read_polars_mean_ms": proj_mean,
            "predicate_projection_read_polars_median_ms": pred_med,
            "predicate_projection_read_polars_mean_ms": pred_mean,
            "in_memory_pandas_mb": mem_pd_mb,
            "in_memory_polars_mb": mem_pl_mb,
        }

    # 3. Parquet (Uncompressed, partitioned by REPDTE)
    results.append(benchmark_parquet_layout("Parquet (Uncompressed, partitioned by REPDTE)", p_pq_uncompressed, None))

    # 4. Parquet (Snappy, partitioned by REPDTE)
    results.append(benchmark_parquet_layout("Parquet (Snappy, partitioned by REPDTE)", p_pq_snappy, "snappy"))

    # 5. Parquet (ZSTD, partitioned by REPDTE)
    results.append(benchmark_parquet_layout("Parquet (ZSTD, partitioned by REPDTE)", p_pq_zstd, "zstd"))

    # Build DataFrame
    res_df = pd.DataFrame(results)

    # Save to reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "full" if is_full else "prototype"
    json_path = REPORTS_DIR / f"storage_benchmark_{suffix}.json"
    csv_path = REPORTS_DIR / f"storage_benchmark_{suffix}.csv"

    res_df.to_json(json_path, orient="records", indent=2)
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to:\n- {json_path}\n- {csv_path}")

    # Cleanup temporary files
    print("\nCleaning up temporary benchmark directory (data/benchmark_tmp)...")
    shutil.rmtree(BENCHMARK_TMP)
    print("Cleanup complete.")

    return res_df


def main():
    parser = argparse.ArgumentParser(description="Storage Layout & Compression Benchmark")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "prototype"],
        help="Benchmark mode: full (all quarters) or prototype (2023 only)",
    )
    args = parser.parse_args()

    df = run_benchmark(mode=args.mode)
    print("\n" + "=" * 95)
    print(f"               STORAGE BENCHMARK RESULTS SUMMARY (MODE: {args.mode.upper()})                ")
    print("=" * 95)
    display_cols = [
        "layout",
        "disk_size_mb",
        "compression_ratio_vs_csv",
        "write_time_median_ms",
        "full_read_pandas_median_ms",
        "full_read_polars_median_ms",
        "projection_read_polars_median_ms",
        "predicate_projection_read_polars_median_ms",
        "in_memory_pandas_mb",
        "in_memory_polars_mb",
    ]
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 1000)
    pd.set_option("display.precision", 2)
    print(df[display_cols].to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    main()
