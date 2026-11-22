"""
Independent End-to-End Verification & Leakage Audit Suite (2nd Audit).
Course: BFT65B - Big Data Analytics in Finance
Subagent 4: Independent Leakage & Rubric Auditor (Gatekeeper)

Executes a complete, adversarial verification from disk across Parts 1 to 5:
- Part 1: Raw Bronze completeness (170 quarters, 1,682,616 rows, 0 duplicates,
          4,117 failures, 27,835 institutions, PyArrow schemas, storage benchmark).
- Part 2: Silver Panel invariants (1,681,431 rows, 0 non-positive ASSET, 14,605 EQ < 0
          retained, multi-event ASSISTANCE vs terminal FAILURE handling incl. CERT=1722,
          0 mergers labeled as failure, null target after 2025-06-30, _meta_* isolation).
- Part 3: Scalable processing invariants (3-way live equivalence Pandas == Polars Eager ==
          Polars Lazy + 4-way PySpark verification across all 1,681,431 rows, Polars AST
          pushdowns & CSE, PySpark Catalyst plans, Amdahl's Law & Snappy Scale-Up).
- Part 4: Gold Feature Table & Zero-Leakage Gate (1,663,902 rows x 57 cols, 0 null labels,
          strict contiguous lag verification across all 32 gapped banks, manual formula
          spot-checks, sentinel 0.0 decoding for RBC1AAJ/RBCRWAJ, zero target overlap
          between dev and test across 2007 embargo, and live mathematical perturbation
          proof on build_preprocessing_pipeline yielding max_diff == 0.0).
- Part 5: Modelling & Validation Gate (4-fold Purged Expanding-Window CV with strict 4Q
          purge gaps in 1987/1990/1993/2000 and zero label overlap, dev-only OOF tuning
          and cost-driven threshold optimization tau* = 0.0401, artifact cross-checks).

Outputs structured audit certificate to: reports/final_audit_certificate.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from src.configs.config import (
    DATA_FEATURES,
    DATA_PROCESSED,
    DATA_RAW,
    DATA_RAW_FIN,
    FIN_FIELDS,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src.features.build_features import (
    ALL_MODEL_FEATURES,
    CAMELS_FEATURE_MAP,
    CATEGORICAL_FEATURES,
    KEY_COLUMNS,
    NUMERIC_FEATURES,
)
from src.features.preprocessing_pipeline import build_preprocessing_pipeline
from src.processing.pandas_baseline import build_pandas_pipeline
from src.processing.polars_pipeline import build_polars_pipeline

sys.stdout.reconfigure(encoding="utf-8")


def audit_part1() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("  [AUDIT PART 1] DATA ACQUISITION, SCHEMAS & STORAGE BENCHMARK INVARIANTS")
    print("=" * 78)

    fin_files = sorted(list(DATA_RAW_FIN.glob("REPDTE=*.parquet")))
    n_quarters = len(fin_files)
    earliest_q = fin_files[0].stem.split("=")[-1] if fin_files else None
    latest_q = fin_files[-1].stem.split("=")[-1] if fin_files else None

    # Check schema of first and last partition
    schema_first = pq.read_schema(fin_files[0])
    schema_last = pq.read_schema(fin_files[-1])
    schema_match = (schema_first == schema_last) and (len(schema_first.names) == 33)

    # Scan all raw financials
    lf_fin = pl.scan_parquet(str(DATA_RAW_FIN / "REPDTE=*.parquet"))
    total_fin_rows = int(lf_fin.select(pl.len()).collect().item())
    distinct_certs = int(lf_fin.select(pl.col("CERT").n_unique()).collect().item())
    dup_count = int(
        lf_fin.group_by(["CERT", "REPDTE"])
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") > 1)
        .select(pl.len())
        .collect()
        .item()
    )

    # Check manifests
    manifest_fin = json.loads((DATA_RAW / "manifest_financials.json").read_text(encoding="utf-8"))
    manifest_tab = json.loads((DATA_RAW / "manifest_tables.json").read_text(encoding="utf-8"))
    incomplete_q = [q for q, m in manifest_fin.items() if not m.get("complete", False)]
    manifest_rows_sum = sum(m["rows"] for m in manifest_fin.values())
    manifest_api_sum = sum(m["api_total"] for m in manifest_fin.values())

    # Reference tables
    df_fail = pl.read_parquet(DATA_RAW / "failures.parquet")
    df_inst = pl.read_parquet(DATA_RAW / "institutions.parquet")
    n_fail_rows = len(df_fail)
    n_inst_rows = len(df_inst)
    n_inst_unique_cert = int(df_inst["CERT"].n_unique())

    # Storage benchmark verification
    bench_full = json.loads((REPORTS_DIR / "storage_benchmark_full.json").read_text(encoding="utf-8"))
    layouts = {r["layout"]: r for r in bench_full}
    csv_mb = layouts["CSV (Uncompressed, single file)"]["disk_size_mb"]
    snappy_mb = layouts["Parquet (Snappy, partitioned by REPDTE)"]["disk_size_mb"]
    zstd_mb = layouts["Parquet (ZSTD, partitioned by REPDTE)"]["disk_size_mb"]
    snappy_pred_ms = layouts["Parquet (Snappy, partitioned by REPDTE)"]["predicate_projection_read_polars_median_ms"]
    csv_pred_ms = layouts["CSV (Uncompressed, single file)"]["predicate_projection_read_polars_median_ms"]

    # Check if benchmark_tmp exists (adversarial check)
    bench_tmp_dir = PROJECT_ROOT / "data" / "benchmark_tmp"
    bench_tmp_exists = bench_tmp_dir.exists()
    bench_tmp_contents = [str(p.relative_to(PROJECT_ROOT)) for p in bench_tmp_dir.rglob("*")] if bench_tmp_exists else []

    checks = {
        "quarters_count_is_170": n_quarters == 170,
        "quarter_range_1984Q1_to_2026Q2": (earliest_q == "19840331") and (latest_q == "20260630"),
        "financial_rows_is_1682616": total_fin_rows == 1_682_616,
        "distinct_certs_is_24064": distinct_certs == 24_064,
        "zero_duplicate_cert_repdte": dup_count == 0,
        "manifest_100pct_complete": (len(incomplete_q) == 0) and (manifest_rows_sum == 1_682_616) and (manifest_api_sum == 1_682_616),
        "pyarrow_schema_33_cols_consistent": schema_match,
        "failures_rows_is_4117": (n_fail_rows == 4_117) and (manifest_tab["failures"]["api_total"] == 4_117),
        "institutions_rows_is_27835_unique": (n_inst_rows == 27_835) and (n_inst_unique_cert == 27_835) and (manifest_tab["institutions"]["api_total"] == 27_835),
        "storage_benchmark_5_layouts_valid": (len(bench_full) == 5) and (zstd_mb < snappy_mb < csv_mb) and (snappy_pred_ms < csv_pred_ms / 10.0),
    }

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        assert v, f"Part 1 invariant failed: {k}"

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "invariants": checks,
        "metrics": {
            "quarters_count": n_quarters,
            "earliest_quarter": earliest_q,
            "latest_quarter": latest_q,
            "financial_rows": total_fin_rows,
            "distinct_certs": distinct_certs,
            "duplicate_cert_repdte": dup_count,
            "failures_rows": n_fail_rows,
            "institutions_rows": n_inst_rows,
            "csv_disk_mb": round(csv_mb, 2),
            "parquet_snappy_disk_mb": round(snappy_mb, 2),
            "parquet_zstd_disk_mb": round(zstd_mb, 2),
            "parquet_snappy_predicate_pushdown_ms": round(snappy_pred_ms, 2),
            "residual_benchmark_tmp_exists": bench_tmp_exists,
            "residual_benchmark_tmp_contents": bench_tmp_contents,
        },
    }


def audit_part2() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("  [AUDIT PART 2] DATA QUALITY & POINT-IN-TIME SILVER PANEL INVARIANTS")
    print("=" * 78)

    silver_path = DATA_PROCESSED / "bank_quarter_panel.parquet"
    df_silver = pl.read_parquet(silver_path)
    n_silver = len(df_silver)
    n_silver_certs = int(df_silver["CERT"].n_unique())

    # 1. Duplicate & ASSET checks
    n_dups = int(df_silver.select(["CERT", "REPDTE"]).is_duplicated().sum())
    n_bad_asset = int(df_silver.filter(pl.col("ASSET").is_null() | (pl.col("ASSET") <= 0)).height)

    # 2. Negative equity (EQ < 0) preserved
    n_neg_eq_silver = int(df_silver.filter(pl.col("EQ") < 0).height)
    raw_neg_eq = int(
        pl.scan_parquet(str(DATA_RAW_FIN / "REPDTE=*.parquet"))
        .filter(pl.col("EQ") < 0)
        .select(pl.len())
        .collect()
        .item()
    )

    # 3. Multi-event bank verification (ASSISTANCE vs terminal FAILURE, e.g. CERT=1722)
    df_fail = (
        pl.read_parquet(DATA_RAW / "failures.parquet")
        .filter(pl.col("CERT").is_not_null())
        .with_columns([
            pl.col("CERT").cast(pl.Int64),
            pl.col("FAILDATE").str.to_date("%m/%d/%Y", strict=False).alias("fail_date"),
            pl.col("COST").cast(pl.Float64).fill_null(0.0).alias("cost_val"),
            pl.col("RESTYPE").cast(pl.String),
        ])
        .filter(pl.col("fail_date").is_not_null())
    )
    multi_event_certs = (
        df_fail.group_by("CERT").agg(pl.len().alias("n_events")).filter(pl.col("n_events") > 1)
    )
    n_multi_event_certs = len(multi_event_certs)

    # Check CERT = 1722 (First City, Texas-Houston, N.A.: ASSISTANCE on 1988-04-20, FAILURE on 1992-10-30)
    c1722 = df_silver.filter(pl.col("CERT") == 1722).sort("repdte_date")
    c1722_rows = len(c1722)
    c1722_max_date = str(c1722["repdte_date"].max())
    c1722_assist_ones = int(
        c1722.filter((pl.col("repdte_date") >= date(1987, 6, 30)) & (pl.col("repdte_date") <= date(1988, 3, 31)))["target_fail_4q"].sum()
    )
    c1722_mid_zeros = int(
        c1722.filter((pl.col("repdte_date") >= date(1988, 6, 30)) & (pl.col("repdte_date") <= date(1991, 9, 30)))["target_fail_4q"].sum()
    )
    c1722_fail_ones = int(
        c1722.filter((pl.col("repdte_date") >= date(1991, 12, 31)) & (pl.col("repdte_date") <= date(1992, 9, 30)))["target_fail_4q"].sum()
    )
    c1722_assist_cost = float(
        c1722.filter(pl.col("repdte_date") == date(1988, 3, 31))["_meta_fail_cost"].item()
    )

    # Check if any multi-event bank has 2 FAILURE events and whether rows between FAILURE_1 and FAILURE_2 exist
    two_failure_certs = (
        df_fail.filter(pl.col("RESTYPE") == "FAILURE")
        .group_by("CERT")
        .agg([pl.len().alias("n_fail"), pl.col("fail_date").min().alias("first_fail"), pl.col("fail_date").max().alias("last_fail")])
        .filter(pl.col("n_fail") > 1)
    )
    n_two_failure_certs = len(two_failure_certs)
    rows_after_first_failure_in_two_fail_certs = int(
        df_silver.join(two_failure_certs, on="CERT", how="inner")
        .filter(pl.col("repdte_date") >= pl.col("first_fail"))
        .height
    )

    # 4. Merger / voluntary exit check:
    # Any bank NOT in failures.parquet must have target_fail_4q == 0 everywhere (before 2025-06-30)
    failed_cert_set = df_fail.select("CERT").unique()
    non_failed_silver = df_silver.join(failed_cert_set, on="CERT", how="anti")
    mergers_labeled_one = int(non_failed_silver.filter(pl.col("target_fail_4q") == 1).height)
    distinct_merged_or_exited_non_fail = int(
        non_failed_silver.filter(pl.col("_meta_inst_active") == 0)["CERT"].n_unique()
    )

    # 5. Label cutoff check (strictly null after 2025-06-30, strictly non-null on/before 2025-06-30)
    after_cutoff = df_silver.filter(pl.col("repdte_date") > date(2025, 6, 30))
    before_cutoff = df_silver.filter(pl.col("repdte_date") <= date(2025, 6, 30))
    non_null_after_cutoff = int(after_cutoff.filter(pl.col("target_fail_4q").is_not_null()).height)
    null_before_cutoff = int(before_cutoff.filter(pl.col("target_fail_4q").is_null()).height)
    total_target_ones = int(before_cutoff["target_fail_4q"].sum())

    # 6. Static 2026 snapshot isolation under _meta_*
    non_meta_cols = [c for c in df_silver.columns if not c.startswith("_meta_")]
    forbidden_leaks = [c for c in ["ACTIVE", "CHANGEC1", "ENDEFYMD", "FAILDATE", "COST", "RESTYPE"] if c in non_meta_cols]

    checks = {
        "silver_rows_is_1681431": n_silver == 1_681_431,
        "silver_distinct_certs_is_24062": n_silver_certs == 24_062,
        "zero_duplicate_cert_repdte": n_dups == 0,
        "zero_null_or_nonpositive_asset": n_bad_asset == 0,
        "negative_equity_preserved_14605_rows": (n_neg_eq_silver == 14_605) and (raw_neg_eq == 15_659),
        "cert_1722_multi_event_assistance_then_failure_exact": (
            c1722_rows == 35
            and c1722_max_date == "1992-09-30"
            and c1722_assist_ones == 4
            and c1722_mid_zeros == 0
            and c1722_fail_ones == 4
            and abs(c1722_assist_cost - 79114.98) < 0.01
        ),
        "zero_mergers_or_voluntary_exits_labeled_1": mergers_labeled_one == 0,
        "target_strictly_null_after_2025_06_30": (non_null_after_cutoff == 0) and (len(after_cutoff) == 17_529),
        "target_strictly_complete_on_or_before_2025_06_30": (null_before_cutoff == 0) and (total_target_ones == 12_459),
        "static_2026_snapshot_isolated_under_meta_prefix": len(forbidden_leaks) == 0,
    }

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        assert v, f"Part 2 invariant failed: {k}"

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "invariants": checks,
        "metrics": {
            "silver_rows": n_silver,
            "silver_distinct_certs": n_silver_certs,
            "raw_rows_dropped_post_terminal": 1_682_616 - n_silver,
            "negative_equity_rows_raw": raw_neg_eq,
            "negative_equity_rows_silver": n_neg_eq_silver,
            "multi_event_certs_count": n_multi_event_certs,
            "two_failure_certs_count": n_two_failure_certs,
            "rows_after_first_failure_in_two_fail_certs": rows_after_first_failure_in_two_fail_certs,
            "cert_1722_rows": c1722_rows,
            "cert_1722_assist_4q_ones": c1722_assist_ones,
            "cert_1722_terminal_4q_ones": c1722_fail_ones,
            "distinct_merged_non_failed_banks": distinct_merged_or_exited_non_fail,
            "mergers_mislabeled_as_failure": mergers_labeled_one,
            "incomplete_label_rows_after_2025Q2": len(after_cutoff),
            "total_4q_failure_events": total_target_ones,
        },
    }


def audit_part3() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("  [AUDIT PART 3] SCALABLE PROCESSING, 4-WAY PARITY & QUERY PLAN INVARIANTS")
    print("=" * 78)

    # 1. Live 3-way execution & equivalence check (Pandas vs Polars Eager vs Polars Lazy)
    t0 = time.perf_counter()
    pl_lazy_panel_lf, pl_lazy_peer_lf = build_polars_pipeline(is_lazy=True)
    pl_lazy_df, pl_lazy_peer = pl.collect_all([pl_lazy_panel_lf, pl_lazy_peer_lf])
    pl_lazy_df = pl_lazy_df.sort(["CERT", "q_idx"])
    pl_lazy_peer = pl_lazy_peer.sort(["REPDTE", "STNAME"])
    t_pl_lazy = time.perf_counter() - t0

    t0 = time.perf_counter()
    pl_eager_df, pl_eager_peer = build_polars_pipeline(is_lazy=False)
    pl_eager_df = pl_eager_df.sort(["CERT", "q_idx"])
    pl_eager_peer = pl_eager_peer.sort(["REPDTE", "STNAME"])
    t_pl_eager = time.perf_counter() - t0

    t0 = time.perf_counter()
    pd_df, pd_peer = build_pandas_pipeline()
    pd_df = pd_df.sort_values(["CERT", "q_idx"]).reset_index(drop=True)
    pd_peer = pd_peer.sort_values(["REPDTE", "STNAME"]).reset_index(drop=True)
    t_pd = time.perf_counter() - t0

    eng_cols = [
        "netinc_q",
        "eq_to_asset",
        "npl_to_loans",
        "reserve_to_loans",
        "liquid_to_asset",
        "bro_to_dep",
        "asset_growth_4q",
        "eq_to_asset_diff_4q",
        "roa_rolling_4q_mean",
        "roa_rolling_4q_std",
    ]

    row_match_3way = (len(pl_lazy_df) == 1_681_431) and (len(pl_eager_df) == 1_681_431) and (len(pd_df) == 1_681_431)
    peer_match_3way = (len(pl_lazy_peer) == 9_605) and (len(pl_eager_peer) == 9_605) and (len(pd_peer) == 9_605)

    max_diff_eager_lazy = 0.0
    max_diff_pandas_lazy = 0.0
    null_mask_all_match = True

    for c in eng_cols:
        arr_lazy = pl_lazy_df[c].to_numpy()
        arr_eager = pl_eager_df[c].to_numpy()
        arr_pd = pd_df[c].to_numpy(dtype=np.float64)

        null_lazy = pl_lazy_df[c].is_null().to_numpy()
        null_eager = pl_eager_df[c].is_null().to_numpy()
        null_pd = np.isnan(arr_pd)

        if not (np.array_equal(null_lazy, null_eager) and np.array_equal(null_lazy, null_pd)):
            null_mask_all_match = False

        valid = ~null_lazy
        if np.any(valid):
            d_el = float(np.max(np.abs(arr_lazy[valid] - arr_eager[valid])))
            d_pl = float(np.max(np.abs(arr_lazy[valid] - arr_pd[valid])))
            max_diff_eager_lazy = max(max_diff_eager_lazy, d_el)
            max_diff_pandas_lazy = max(max_diff_pandas_lazy, d_pl)

    # 2. Check PySpark 4th-way verification artifact
    spark_ver = json.loads((REPORTS_DIR / "part3_spark_verification.json").read_text(encoding="utf-8"))
    spark_rows_match = (spark_ver["panel_row_count"] == 1_681_431) and (spark_ver["peer_row_count"] == 9_605)
    spark_max_diff = max(m["max_abs_diff"] for m in spark_ver["column_metrics"].values())
    spark_nulls_match = all(m["null_mask_match"] for m in spark_ver["column_metrics"].values())

    # 3. Check Polars Query Plan file
    pl_plans_txt = (REPORTS_DIR / "polars_query_plans.txt").read_text(encoding="utf-8")
    has_polars_pred_pushdown = 'SELECTION: ((col("ASSET").is_not_null() & (col("ASSET") > 0.0)) & (col("REPDTE") >= "20080101"))' in pl_plans_txt
    has_polars_proj_pushdown = "PROJECT 5/33 COLUMNS" in pl_plans_txt
    has_polars_cse = "__POLARS_CSER_" in pl_plans_txt

    # 4. Check PySpark Catalyst Plan file
    sp_plans_txt = (REPORTS_DIR / "pyspark_catalyst_plans.txt").read_text(encoding="utf-8")
    has_spark_smj = "SortMergeJoin" in sp_plans_txt and "Exchange hashpartitioning" in sp_plans_txt
    has_spark_bhj = "BroadcastHashJoin" in sp_plans_txt and "BroadcastExchange" in sp_plans_txt
    has_spark_window = "Window [lag(" in sp_plans_txt
    has_spark_hashagg = "HashAggregate" in sp_plans_txt

    # 5. Check Amdahl & Scale-Up artifacts
    amdahl_data = json.loads((REPORTS_DIR / "part3_amdahl_scaling.json").read_text(encoding="utf-8"))
    p_hat = amdahl_data["fitted_parallel_fraction_p"]
    s_max = amdahl_data["max_theoretical_speedup"]
    amdahl_math_ok = abs(s_max - (1.0 / (1.0 - p_hat))) < 0.05

    scaleup_data = json.loads((REPORTS_DIR / "part3_scaleup_stress.json").read_text(encoding="utf-8"))
    disk_sizes = [r["dataset_disk_mb"] for r in scaleup_data["stress_test_table"]]
    scaleup_monotonic = disk_sizes[0] < disk_sizes[1] < disk_sizes[2] < disk_sizes[3]

    checks = {
        "panel_and_peer_rows_match_3way_live": row_match_3way and peer_match_3way,
        "null_masks_match_100pct_3way_live": null_mask_all_match,
        "polars_eager_vs_lazy_exact_0_diff": max_diff_eager_lazy == 0.0,
        "pandas_vs_polars_lazy_diff_lt_1e12": max_diff_pandas_lazy < 1e-12,
        "pyspark_4th_way_full_1681431_rows_verified": spark_rows_match and spark_nulls_match and (spark_max_diff < 1e-12),
        "polars_plan_predicate_and_projection_pushdown_verified": has_polars_pred_pushdown and has_polars_proj_pushdown and has_polars_cse,
        "pyspark_catalyst_plan_smj_bhj_window_hashagg_verified": has_spark_smj and has_spark_bhj and has_spark_window and has_spark_hashagg,
        "amdahl_law_fit_and_snappy_scaleup_consistent": amdahl_math_ok and scaleup_monotonic,
    }

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        assert v, f"Part 3 invariant failed: {k}"

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "invariants": checks,
        "metrics": {
            "live_polars_lazy_sec": round(t_pl_lazy, 3),
            "live_polars_eager_sec": round(t_pl_eager, 3),
            "live_pandas_sec": round(t_pd, 3),
            "max_diff_polars_eager_vs_lazy": max_diff_eager_lazy,
            "max_diff_pandas_vs_polars_lazy": max_diff_pandas_lazy,
            "max_diff_pyspark_vs_polars_lazy": spark_max_diff,
            "amdahl_parallel_fraction_p": p_hat,
            "amdahl_max_theoretical_speedup": s_max,
            "scaleup_disk_mb_progression": disk_sizes,
        },
    }


def audit_part4() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("  [AUDIT PART 4] GOLD FEATURE TABLE & ZERO-LEAKAGE PREPROCESSING GATE")
    print("=" * 78)

    gold_path = DATA_FEATURES / "gold_feature_table.parquet"
    df_gold = pl.read_parquet(gold_path)
    n_rows, n_cols = df_gold.shape

    # 1. Shape & Label completeness
    n_null_labels = int(df_gold["target_fail_4q"].is_null().sum())
    n_incomplete_split = int(df_gold.filter(pl.col("split_regime") == "incomplete_label").height)
    total_fails = int(df_gold["target_fail_4q"].sum())

    # 2. Verify Contiguous Quarter Lag Arithmetic on ALL banks (including 32 gapped banks)
    df_sorted = df_gold.sort(["CERT", "q_idx"])
    q_diff_1 = (pl.col("q_idx") - pl.col("q_idx").shift(1).over("CERT"))
    q_diff_2 = (pl.col("q_idx") - pl.col("q_idx").shift(2).over("CERT"))
    q_diff_3 = (pl.col("q_idx") - pl.col("q_idx").shift(3).over("CERT"))
    q_diff_4 = (pl.col("q_idx") - pl.col("q_idx").shift(4).over("CERT"))

    # Any row where q_diff_1 != 1 (and quarter != 1) must have null YTD de-cumulated flow features
    non_contig_1q_q234 = df_sorted.filter((pl.col("quarter") != 1) & (q_diff_1.is_null() | (q_diff_1 != 1)))
    ytd_flow_features = [
        "cost_to_income_ratio",
        "overhead_to_assets_q",
        "netinc_q_to_assets",
        "provision_to_loans_q",
        "net_chargeoff_to_loans_q",
    ]
    ytd_gap_leaks = sum(int(non_contig_1q_q234[c].is_not_null().sum()) for c in ytd_flow_features)

    # Any row where q_diff_4 != 4 must have null 4Q lag/growth features
    non_contig_4q = df_sorted.filter(q_diff_4.is_null() | (q_diff_4 != 4))
    lag_4q_features = [
        "asset_growth_4q",
        "eq_to_asset_diff_4q",
        "npl_to_loans_diff_4q",
        "liquid_to_asset_diff_4q",
        "bro_to_dep_diff_4q",
    ]
    lag4_gap_leaks = sum(int(non_contig_4q[c].is_not_null().sum()) for c in lag_4q_features)

    # Any row where 4Q rolling window is not contiguous (q_diff_1 != 1 or q_diff_2 != 2 or q_diff_3 != 3)
    non_contig_roll = df_sorted.filter(
        q_diff_1.is_null() | (q_diff_1 != 1)
        | q_diff_2.is_null() | (q_diff_2 != 2)
        | q_diff_3.is_null() | (q_diff_3 != 3)
    )
    roll_gap_leaks = int(non_contig_roll["roa_rolling_4q_mean"].is_not_null().sum()) + int(
        non_contig_roll["roa_rolling_4q_std"].is_not_null().sum()
    )

    # 3. Verify Contemporaneous State-Quarter Peer Feature (zero cross-quarter bleed)
    peer_check = df_gold.with_columns(
        (pl.col("eq_to_asset") - pl.col("eq_to_asset").mean().over(["REPDTE", "STNAME"])).alias("manual_eq_vs_peer")
    ).filter(pl.col("eq_vs_state_peer").is_not_null())
    max_peer_diff = float((peer_check["eq_vs_state_peer"] - peer_check["manual_eq_vs_peer"]).abs().max())

    # 4. Verify Sentinel 0.0 decoding for RBC1AAJ and RBCRWAJ against Silver Panel
    df_cap_audit = (
        pl.read_parquet(DATA_PROCESSED / "bank_quarter_panel.parquet")
        .filter(pl.col("_meta_split") != "incomplete_label")
        .select(["CERT", "REPDTE", "RBCRWAJ", "RBC1AAJ", "EQ"])
        .join(
            df_gold.select(["CERT", "REPDTE", "total_rbc_ratio", "tier1_leverage_ratio", "basel_missing_flag"]),
            on=["CERT", "REPDTE"],
            how="inner",
        )
    )
    sentinel_rbcrw_df = df_cap_audit.filter(
        (pl.col("RBCRWAJ") == 0.0) & (pl.col("EQ").is_null() | (pl.col("EQ") != 0.0))
    )
    n_sentinel_rbcrw = int(sentinel_rbcrw_df.height)
    undecoded_rbcrw = int(sentinel_rbcrw_df["total_rbc_ratio"].is_not_null().sum())
    basel_flag_mismatch = int(
        df_cap_audit.filter(pl.col("total_rbc_ratio").is_null() != (pl.col("basel_missing_flag") == 1.0)).height
    )

    sentinel_rbc1_df = df_cap_audit.filter(
        (pl.col("RBC1AAJ") == 0.0) & (pl.col("EQ").is_null() | (pl.col("EQ") != 0.0))
    )
    n_sentinel_rbc1 = int(sentinel_rbc1_df.height)
    undecoded_rbc1 = int(sentinel_rbc1_df["tier1_leverage_ratio"].is_not_null().sum())

    # 5. Verify Zero Target Window Overlap between DEV (<= 2006Q4) and TEST (>= 2008Q1)
    df_dev = df_gold.filter(pl.col("split_regime") == "dev")
    df_embargo = df_gold.filter(pl.col("split_regime") == "embargo")
    df_test = df_gold.filter(pl.col("split_regime") == "test")

    dev_max_repdte = str(df_dev["repdte_date"].max())
    test_min_repdte = str(df_test["repdte_date"].min())
    # For all dev rows with target_fail_4q == 1, check max(_meta_fail_date)
    dev_pos_max_fail_date = str(
        df_dev.filter(pl.col("target_fail_4q") == 1)["_meta_fail_date"].max()
    )
    dev_target_bleed_into_test = int(
        df_dev.filter((pl.col("target_fail_4q") == 1) & (pl.col("_meta_fail_date") >= "2008-01-01")).height
    )

    # 6. Live End-to-End Silver -> Gold -> Preprocessor Perturbation Proof
    X_dev_pd = df_dev.select(ALL_MODEL_FEATURES).to_pandas()
    y_dev_np = df_dev["target_fail_4q"].to_numpy()

    pipe1 = build_preprocessing_pipeline()
    pipe1.fit(X_dev_pd, y_dev_np)
    num1 = pipe1.named_steps["preprocessor"].named_transformers_["num"]
    w1_low = num1.named_steps["winsorizer"].lower_bounds_.copy()
    w1_high = num1.named_steps["winsorizer"].upper_bounds_.copy()
    s1_center = num1.named_steps["scaler"].center_.copy()
    s1_scale = num1.named_steps["scaler"].scale_.copy()
    i1_stat = num1.named_steps["imputer"].statistics_.copy()

    # Corrupt all post-2006 (year > 2006) rows in the Silver panel BEFORE feature engineering
    from src.features.build_features import build_gold_feature_table
    corrupt_fin_cols = ["ASSET", "DEP", "EQ", "NETINC", "LNLSNET", "NCLNLS", "ROA", "ROE", "RBC1AAJ", "RBCRWAJ"]
    silver_corrupt_lf = pl.scan_parquet(str(DATA_PROCESSED / "bank_quarter_panel.parquet")).with_columns([
        pl.when(pl.col("year") > 2006)
        .then(pl.col(c) * 1_000_000.0 + 999_999.0)
        .otherwise(pl.col(c))
        .alias(c)
        for c in corrupt_fin_cols
    ])
    df_gold_corrupt = build_gold_feature_table(input_lf=silver_corrupt_lf, write_output=False)
    X_dev_corrupt_pd = df_gold_corrupt.filter(pl.col("split_regime") == "dev").select(ALL_MODEL_FEATURES).to_pandas()

    pipe2 = build_preprocessing_pipeline()
    pipe2.fit(X_dev_corrupt_pd, y_dev_np)
    num2 = pipe2.named_steps["preprocessor"].named_transformers_["num"]
    max_perturbation_diff = float(
        max(
            np.max(np.abs(w1_low - num2.named_steps["winsorizer"].lower_bounds_)),
            np.max(np.abs(w1_high - num2.named_steps["winsorizer"].upper_bounds_)),
            np.max(np.abs(s1_center - num2.named_steps["scaler"].center_)),
            np.max(np.abs(s1_scale - num2.named_steps["scaler"].scale_)),
            np.max(np.abs(i1_stat - num2.named_steps["imputer"].statistics_)),
        )
    )

    # Check non-degenerate IQRs on high-missingness features (pastdue30_89_to_loans, total_rbc_ratio)
    idx_p3 = NUMERIC_FEATURES.index("pastdue30_89_to_loans")
    idx_rbc = NUMERIC_FEATURES.index("total_rbc_ratio")
    iqr_p3 = float(s1_scale[idx_p3])
    iqr_rbc = float(s1_scale[idx_rbc])

    checks = {
        "gold_shape_is_1663902_x_57": (n_rows == 1_663_902) and (n_cols == 57),
        "zero_null_labels_and_12459_positives": (n_null_labels == 0) and (n_incomplete_split == 0) and (total_fails == 12_459),
        "contiguous_ytd_decumulation_zero_gap_bleed": ytd_gap_leaks == 0,
        "contiguous_4q_lags_zero_gap_bleed": lag4_gap_leaks == 0,
        "contiguous_4q_rolling_zero_gap_bleed": roll_gap_leaks == 0,
        "contemporaneous_peer_feature_exact": max_peer_diff < 1e-12,
        "sentinel_0_decoded_for_rbc1aaj_and_rbcrwaj": (
            undecoded_rbcrw == 0 and undecoded_rbc1 == 0 and basel_flag_mismatch == 0 and n_sentinel_rbcrw > 350_000 and n_sentinel_rbc1 > 0
        ),
        "zero_target_window_overlap_dev_to_test_via_2007_embargo": (
            dev_max_repdte == "2006-12-31"
            and dev_pos_max_fail_date <= "2007-12-31"
            and test_min_repdte == "2008-03-31"
            and dev_target_bleed_into_test == 0
        ),
        "perturbation_proof_max_diff_is_exact_zero": max_perturbation_diff < 1e-12,
        "robust_scaler_before_imputer_non_degenerate_iqr": (0.01 < iqr_p3 < 0.05) and (5.0 < iqr_rbc < 15.0),
    }

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        assert v, f"Part 4 invariant failed: {k}"

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "invariants": checks,
        "metrics": {
            "gold_rows": n_rows,
            "gold_cols": n_cols,
            "ytd_gap_leaks": ytd_gap_leaks,
            "lag4_gap_leaks": lag4_gap_leaks,
            "rolling_gap_leaks": roll_gap_leaks,
            "sentinel_zeros_decoded_rbcrwaj": n_sentinel_rbcrw,
            "sentinel_zeros_decoded_rbc1aaj": n_sentinel_rbc1,
            "dev_max_repdte": dev_max_repdte,
            "dev_positive_label_max_fail_date": dev_pos_max_fail_date,
            "test_min_repdte": test_min_repdte,
            "dev_to_test_label_overlap_rows": dev_target_bleed_into_test,
            "perturbation_proof_max_abs_diff": max_perturbation_diff,
            "pastdue30_89_fitted_iqr": round(iqr_p3, 6),
            "total_rbc_ratio_fitted_iqr": round(iqr_rbc, 6),
        },
    }


def audit_part5() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("  [AUDIT PART 5] PURGED CV, COST-DRIVEN THRESHOLD & MODELLING GATE")
    print("=" * 78)

    df_gold = pl.read_parquet(DATA_FEATURES / "gold_feature_table.parquet")
    df_dev = df_gold.filter(pl.col("split_regime") == "dev")
    df_test = df_gold.filter(pl.col("split_regime") == "test")

    # 1. Verify all 4 CV folds enforce a strict 4-quarter (1-year) purge gap with 0 label overlap
    fold_defs = [
        (1, pl.col("year") <= 1986, pl.col("year") == 1987, (pl.col("year") >= 1988) & (pl.col("year") <= 1989)),
        (2, pl.col("year") <= 1989, pl.col("year") == 1990, (pl.col("year") >= 1991) & (pl.col("year") <= 1992)),
        (3, pl.col("year") <= 1992, pl.col("year") == 1993, (pl.col("year") >= 1994) & (pl.col("year") <= 1999)),
        (4, pl.col("year") <= 1999, pl.col("year") == 2000, (pl.col("year") >= 2001) & (pl.col("year") <= 2006)),
    ]

    cv_purge_audit = []
    zero_cv_overlap = True
    for f_id, tr_c, pg_c, vl_c in fold_defs:
        tr = df_dev.filter(tr_c)
        pg = df_dev.filter(pg_c)
        vl = df_dev.filter(vl_c)

        tr_max_date = str(tr["repdte_date"].max())
        tr_max_fail_date = str(tr.filter(pl.col("target_fail_4q") == 1)["_meta_fail_date"].max())
        vl_min_date = str(vl["repdte_date"].min())
        pg_quarters = int(pg["REPDTE"].n_unique())

        # Any train positive label whose fail_date reaches into val_min_date?
        bleed_rows = int(
            tr.filter((pl.col("target_fail_4q") == 1) & (pl.col("_meta_fail_date") >= vl_min_date)).height
        )
        if bleed_rows > 0 or pg_quarters != 4:
            zero_cv_overlap = False

        cv_purge_audit.append({
            "fold": f_id,
            "train_rows": len(tr),
            "train_max_repdte": tr_max_date,
            "train_positive_max_fail_date": tr_max_fail_date,
            "purge_rows": len(pg),
            "purge_quarters": pg_quarters,
            "val_rows": len(vl),
            "val_fails": int(vl["target_fail_4q"].sum()),
            "val_min_repdte": vl_min_date,
            "train_to_val_label_overlap_rows": bleed_rows,
        })

    # 2. Verify C_FN empirical estimation strictly on DEV
    dev_fail_costs = df_dev.filter((pl.col("target_fail_4q") == 1) & (pl.col("_meta_fail_cost") > 0))["_meta_fail_cost"]
    annual_med_cost = float(dev_fail_costs.median())
    quarterly_med_cost = annual_med_cost / 4.0
    c_fp = 100.0
    tau_bayes = round(c_fp / (c_fp + quarterly_med_cost), 4)

    # 3. Verify Part 5 CSV & JSON artifacts consistency
    comp_json = json.loads((REPORTS_DIR / "part5_model_comparison.json").read_text(encoding="utf-8"))
    comp_csv = pd.read_csv(REPORTS_DIR / "part5_model_comparison.csv")
    cv_csv = pd.read_csv(REPORTS_DIR / "part5_cv_fold_metrics.csv")
    thresh_csv = pd.read_csv(REPORTS_DIR / "part5_threshold_optimization.csv")
    fi_csv = pd.read_csv(REPORTS_DIR / "part5_feature_importance.csv")
    mllib_json = json.loads((REPORTS_DIR / "part5_spark_mllib_results.json").read_text(encoding="utf-8"))

    # Check if stale part5_cv_fold_metrics.json differs from part5_cv_fold_metrics.csv
    cv_json_path = REPORTS_DIR / "part5_cv_fold_metrics.json"
    cv_json_stale_diff = False
    if cv_json_path.exists():
        cv_json = pd.read_json(cv_json_path)
        max_cv_csv_vs_json_diff = float(np.max(np.abs(cv_csv["pr_auc"].to_numpy() - cv_json["pr_auc"].to_numpy())))
        cv_json_stale_diff = max_cv_csv_vs_json_diff > 1e-6

    # Check Champion LightGBM test metrics
    lgb_test_row = comp_csv[(comp_csv["model"] == "LightGBM_CAMELS") & (comp_csv["partition"].str.contains("TEST"))].iloc[0]
    base_test_row = comp_csv[(comp_csv["model"] == "Supervisory_Rule_Baseline") & (comp_csv["partition"].str.contains("TEST"))].iloc[0]

    lgb_test_prauc = float(lgb_test_row["pr_auc"])
    lgb_test_rocauc = float(lgb_test_row["roc_auc"])
    lgb_test_brier = float(lgb_test_row["brier_score"])
    base_test_prauc = float(base_test_row["pr_auc"])

    # Check threshold optimization metrics
    opt_tau = comp_json["cost_driven_threshold_optimization"]["dev_learned_optimal_tau"]
    verdict = comp_json["cost_driven_threshold_optimization"]["test_verdict_at_optimal_tau"]
    cm_sum = verdict["tp"] + verdict["fp"] + verdict["fn"] + verdict["tn"]
    pos_sum = verdict["tp"] + verdict["fn"]

    # Check Feature Importance sum of gain shares
    gain_share_sum = float(fi_csv["lgb_gain_share_pct"].sum())
    n_ohe_features = len(fi_csv)

    # Check Spark MLlib results
    mllib_gbt = mllib_json["models"]["Spark_MLlib_GBTClassifier"]
    mllib_lr = mllib_json["models"]["Spark_MLlib_LogisticRegression"]
    mllib_valid = (
        mllib_json["dataset_summary"]["train_dev_rows"] == 1_193_702
        and mllib_json["dataset_summary"]["test_rows"] == 212_645
        and abs(mllib_gbt["test_pr_auc_spark"] - mllib_gbt["test_pr_auc_sklearn"]) < 0.005
    )

    checks = {
        "all_4_cv_folds_have_strict_4q_purge_and_zero_overlap": zero_cv_overlap,
        "empirical_c_fn_derived_strictly_from_dev": (abs(annual_med_cost - 10399.0) < 0.01) and (abs(quarterly_med_cost - 2599.75) < 0.01) and (tau_bayes == 0.0370),
        "hyperparameter_tuning_validated_on_dev_oof": (
            comp_json["purged_cv_expanding_window"]["hyperparameter_check"]["lightgbm_tuned_prauc"] == 0.482977
            and comp_json["purged_cv_expanding_window"]["hyperparameter_check"]["lightgbm_default_prauc"] == 0.332513
        ),
        "optimal_threshold_tau_00401_learned_on_dev_oof": opt_tau == 0.0401,
        "champion_lightgbm_outperforms_baseline_on_test": (lgb_test_prauc == 0.64537) and (lgb_test_prauc > base_test_prauc) and (lgb_test_rocauc == 0.982803),
        "test_confusion_matrix_sums_to_212645_rows_and_2052_fails": (cm_sum == 212_645) and (pos_sum == 2_052),
        "feature_importance_49_features_sum_to_100pct": (n_ohe_features == 49) and (abs(gain_share_sum - 100.0) < 0.01),
        "spark_mllib_distributed_demo_verified": mllib_valid,
        "cv_fold_metrics_csv_and_json_synchronized": not cv_json_stale_diff,
    }

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        assert v, f"Part 5 invariant failed: {k}"

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "invariants": checks,
        "metrics": {
            "cv_folds_purge_audit": cv_purge_audit,
            "dev_annual_median_dif_loss_k": annual_med_cost,
            "dev_quarterly_median_dif_loss_k": quarterly_med_cost,
            "theoretical_tau_bayes": tau_bayes,
            "dev_learned_optimal_tau": opt_tau,
            "champion_test_pr_auc": lgb_test_prauc,
            "champion_test_roc_auc": lgb_test_rocauc,
            "champion_test_brier_score": lgb_test_brier,
            "test_confusion_matrix_at_tau_star": {
                "tp": verdict["tp"],
                "fp": verdict["fp"],
                "fn": verdict["fn"],
                "tn": verdict["tn"],
                "recall": verdict["recall"],
                "dollar_recall": verdict["dollar_recall"],
                "precision": verdict["precision"],
                "net_savings_million": verdict["net_savings_million"],
            },
            "spark_mllib_gbt_test_pr_auc": mllib_gbt["test_pr_auc_spark"],
            "spark_mllib_lr_test_pr_auc": mllib_lr["test_pr_auc_spark"],
            "stale_cv_fold_metrics_json_detected": cv_json_stale_diff,
        },
    }


def main() -> None:
    t0 = time.time()
    print("==============================================================================")
    print("  BFT65B INDEPENDENT END-TO-END VERIFICATION & ANTI-LEAKAGE AUDIT SUITE       ")
    print("==============================================================================")

    p1 = audit_part1()
    p2 = audit_part2()
    p3 = audit_part3()
    p4 = audit_part4()
    p5 = audit_part5()

    all_passed = all(p["status"] == "PASS" for p in [p1, p2, p3, p4, p5])

    certificate = {
        "audit_title": "BFT65B Independent End-to-End Verification & Zero-Leakage Audit Certificate",
        "auditor": "Subagent 4 - Independent Leakage & Rubric Auditor (Gatekeeper)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": round(time.time() - t0, 2),
        "overall_audit_status": "PASS (ZERO DATA LEAKAGE VERIFIED)" if all_passed else "FAIL",
        "leakage_gate_triggered": False,
        "parts": {
            "part1_acquisition_and_storage": p1,
            "part2_data_quality_and_silver_panel": p2,
            "part3_scalable_processing_and_spark": p3,
            "part4_feature_engineering_and_leakage_gate": p4,
            "part5_modelling_purged_cv_and_evaluation": p5,
        },
        "adversarial_findings_and_discrepancies": [
            {
                "id": "FINDING-1",
                "severity": "LOW (Artifact Sync)",
                "status": "RESOLVED",
                "location": "reports/part5_cv_fold_metrics.json vs reports/part5_cv_fold_metrics.csv (src/models/train_evaluate.py:374)",
                "description": (
                    "train_evaluate.py now writes both part5_cv_fold_metrics.csv and part5_cv_fold_metrics.json simultaneously, "
                    "ensuring 100% synchronization across CSV and JSON artifacts."
                ),
            },
            {
                "id": "FINDING-2",
                "severity": "LOW (Documentation Sync)",
                "status": "RESOLVED",
                "location": "reports/PROJECT_LOG.md & reports/part4_feature_summary.json vs reports/part5_feature_importance.csv",
                "description": (
                    "Synchronized PROJECT_LOG.md and part4_feature_summary.json with exact post-fix LightGBM split counts, "
                    "Logistic Regression standardized coefficients, and 49 OHE features (39 numeric + 10 BKCLASS categories)."
                ),
            },
            {
                "id": "FINDING-3",
                "severity": "LOW (Floating-Point Grid Rounding)",
                "status": "RESOLVED",
                "location": "src/models/train_evaluate.py:595",
                "description": (
                    "Standardized threshold rounding tau_r = round(float(tau), 4) (0.0401) across both DEV grid search "
                    "and single-shot TEST evaluation so test_verdict_at_optimal_tau and sensitivity_table match identically (FP=2,909)."
                ),
            },
            {
                "id": "FINDING-4",
                "severity": "LOW (Cleanup & Portability)",
                "status": "RESOLVED",
                "location": "src/models/spark_mllib_demo.py:36, 240",
                "description": (
                    "Replaced hardcoded Windows path with PROJECT_ROOT / 'data' / 'benchmark_tmp' / 'hadoop' and added "
                    "automatic shutil.rmtree cleanup after spark.stop()."
                ),
            },
            {
                "id": "FINDING-5",
                "severity": "LOW (Narrative String Interpolation)",
                "status": "RESOLVED",
                "location": "src/data/quality_audit.py:340 & reports/data_quality_report.json",
                "description": (
                    "Updated quality_audit.py to explicitly report both RBC1AAJ (19840331) and RBCRWAJ (19900331 Basel I) "
                    "first appearance quarters and added PROJECT_ROOT to sys.path across all src/data/ scripts."
                ),
            },
            {
                "id": "FINDING-6",
                "severity": "MEDIUM (Reproducibility & Pinned Dependencies)",
                "status": "RESOLVED",
                "location": "requirements.txt",
                "description": (
                    "Pinned all 14 core packages (including psutil==7.2.2, polars==2.1.0, pyspark==4.3.0, lightgbm==4.7.0, "
                    "scikit-learn==1.8.0) with exact '==' versions per Section 4.2 of the BFT65B exam specification."
                ),
            },
        ],
    }

    out_path = REPORTS_DIR / "final_audit_certificate.json"
    out_path.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"  FINAL AUDIT STATUS: {certificate['overall_audit_status']}")
    print(f"  Saved machine-readable audit certificate to: {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
