"""
Clean Point-in-Time Silver Panel Construction for FDIC Early Warning System.
Course: BFT65B - Big Data Analytics in Finance
Subagent 1: Data Pipeline Engineer

Rules:
1. Zero Survivorship Bias: Retains all banks (active, merged, liquidated, failed)
   across every historical quarter they reported.
2. Filter Drops:
   - Drops rows where ASSET is null or ASSET <= 0 (unusable / defunct).
   - Drops residual post-failure filings where repdte_date >= fail_date (bank already failed).
   - Keeps negative equity (EQ < 0) as valid technical insolvencies.
3. Anti-Leakage Feature Joins:
   - Joins static birth charter class BKCLASS from institutions.parquet.
   - Strictly isolates 2026 snapshot features (ACTIVE, CHANGEC1, ENDEFYMD) under _meta_ prefix.
4. Point-in-Time Label Construction (Section 2.3):
   - Computes exact forward 1-year horizon: (repdte_date, repdte_date + 4Q].
   - Matches earliest future failure date: min(fail_date > repdte_date).
   - Sets target_fail_4q = 1 if failure occurs in horizon, else 0 (mergers/exits = 0).
   - For repdte_date > 2025-06-30: sets target_fail_4q = null, _meta_label_complete = False.
   - Sets _meta_split: 'dev', 'embargo', 'test', 'post_test', 'incomplete_label'.
   - Attaches _meta_fail_cost for Part 5 cost-weighted evaluation.

Saves:
- data/processed/bank_quarter_panel.parquet
- reports/silver_panel_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import (
    DATA_RAW,
    DATA_RAW_FIN,
    DATA_PROCESSED,
    REPORTS_DIR,
    FIN_FIELDS,
    DEV_START_DATE,
    DEV_END_DATE,
    EMBARGO_START_DATE,
    EMBARGO_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    LABEL_CUTOFF_DATE,
)

sys.stdout.reconfigure(encoding="utf-8")


def build_silver_panel() -> dict[str, Any]:
    print("===================================================================")
    print("      PART 2B: POINT-IN-TIME SILVER PANEL CONSTRUCTION             ")
    print("===================================================================\n")
    t0 = time.time()

    # Verify input files
    fin_files = sorted(list(DATA_RAW_FIN.glob("REPDTE=*.parquet")))
    fail_path = DATA_RAW / "failures.parquet"
    inst_path = DATA_RAW / "institutions.parquet"

    if not fin_files:
        raise FileNotFoundError(f"No financials files found in {DATA_RAW_FIN}")
    if not fail_path.exists():
        raise FileNotFoundError(f"Failures file not found: {fail_path}")
    if not inst_path.exists():
        raise FileNotFoundError(f"Institutions file not found: {inst_path}")

    print(f"Loading raw panel partitions: {len(fin_files)} quarters from {DATA_RAW_FIN}...")
    pq_glob = str(DATA_RAW_FIN / "REPDTE=*.parquet")
    raw_df = pl.read_parquet(pq_glob)
    n_raw_rows = len(raw_df)
    n_raw_certs = raw_df["CERT"].n_unique()
    print(f"Raw Panel Loaded: {n_raw_rows:,} rows, {n_raw_certs:,} distinct CERTs.")

    # ------------------------------------------------------------------------
    # 1. Parse Dates and Calendar Components
    # ------------------------------------------------------------------------
    print("\n--- 1. Parsing Temporal Keys ---")
    df = raw_df.with_columns([
        pl.col("REPDTE").str.to_date("%Y%m%d").alias("repdte_date"),
        pl.col("REPDTE").str.slice(0, 4).cast(pl.Int16).alias("year"),
        pl.col("REPDTE").str.slice(4, 2).cast(pl.Int32).alias("month"),
        pl.col("REPDTE").str.slice(6, 2).cast(pl.Int32).alias("day"),
    ]).with_columns(
        (pl.col("month") // 3).cast(pl.Int8).alias("quarter"),
        pl.date(pl.col("year") + 1, pl.col("month"), pl.col("day")).alias("horizon_4q_end"),
    )

    # ------------------------------------------------------------------------
    # 2. Join Failures and Filter Post-Mortem Rows
    # ------------------------------------------------------------------------
    print("\n--- 2. Joining Failures Table & Applying Anti-Leakage Filters ---")
    raw_failures = pl.read_parquet(fail_path).filter(pl.col("CERT").is_not_null())
    clean_failures = (
        raw_failures.with_columns([
            pl.col("CERT").cast(pl.Int64),
            pl.col("FAILDATE").str.to_date("%m/%d/%Y", strict=False).alias("fail_date"),
            pl.col("COST").cast(pl.Float64).fill_null(0.0).alias("cost_val"),
            pl.col("RESTYPE").cast(pl.String),
        ])
        .filter(pl.col("fail_date").is_not_null())
        .select(["CERT", "fail_date", "cost_val", "RESTYPE"])
    )

    # Define terminal closure per CERT:
    # Banks with ASSISTANCE remain operating institutions!
    # Drop ONLY rows on/after a terminal 'FAILURE' closure.
    terminal_closures = (
        clean_failures
        .sort(["CERT", "fail_date"])
        .group_by("CERT")
        .agg([
            pl.col("fail_date").last().alias("last_fail_date"),
            pl.col("RESTYPE").last().alias("last_restype"),
        ])
        .with_columns(
            pl.when(pl.col("last_restype") == "FAILURE")
            .then(pl.col("last_fail_date"))
            .otherwise(pl.lit(None, dtype=pl.Date))
            .alias("terminal_fail_date")
        )
        .select(["CERT", "terminal_fail_date"])
    )
    df = df.join(terminal_closures, on="CERT", how="left")

    # Filter 1: Drop non-positive ASSET
    cond_valid_asset = pl.col("ASSET").is_not_null() & (pl.col("ASSET") > 0)
    n_drop_asset = len(df.filter(~cond_valid_asset))
    df = df.filter(cond_valid_asset)

    # Filter 2: Drop post-terminal failure residual filings (repdte_date >= terminal_fail_date)
    # Retains 100% of post-ASSISTANCE operating bank-quarters!
    cond_not_post_mortem = pl.col("terminal_fail_date").is_null() | (pl.col("repdte_date") < pl.col("terminal_fail_date"))
    n_drop_post_mortem = len(df.filter(~cond_not_post_mortem))
    df = df.filter(cond_not_post_mortem).drop("terminal_fail_date")

    print(f"Filter Drop 1 (ASSET <= 0 or null)   : {n_drop_asset:,} rows dropped")
    print(f"Filter Drop 2 (repdte_date >= terminal failure) : {n_drop_post_mortem:,} residual post-mortem rows dropped")
    print(f"Cleaned Living Bank-Quarters        : {len(df):,} rows remaining")

    # ------------------------------------------------------------------------
    # 3. Exact Forward 1-Year Failure Horizon Matching
    # ------------------------------------------------------------------------
    print("\n--- 3. Constructing Point-in-Time 4-Quarter Failure Target ---")
    # For each row, find earliest future failure where fail_date > repdte_date
    df_with_fails = (
        df.select(["CERT", "repdte_date", "horizon_4q_end"])
        .join(clean_failures, on="CERT", how="inner")
        .filter(pl.col("fail_date") > pl.col("repdte_date"))
        .group_by(["CERT", "repdte_date"])
        .agg([
            pl.col("fail_date").min().alias("next_fail_date"),
            pl.col("cost_val").sort_by("fail_date").first().alias("next_fail_cost"),
            pl.col("RESTYPE").sort_by("fail_date").first().alias("next_fail_restype"),
        ])
    )

    # Join future failure back to panel
    df = df.join(df_with_fails, on=["CERT", "repdte_date"], how="left")

    # Define binary failure event within 1-year horizon:
    # 1 if next_fail_date is not null and next_fail_date <= horizon_4q_end
    # else 0
    cutoff_d = pd.to_datetime(LABEL_CUTOFF_DATE).date()
    df = df.with_columns([
        pl.when(pl.col("next_fail_date").is_not_null() & (pl.col("next_fail_date") <= pl.col("horizon_4q_end")))
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("raw_target_fail_4q"),

        pl.when(pl.col("next_fail_date").is_not_null() & (pl.col("next_fail_date") <= pl.col("horizon_4q_end")))
        .then(pl.col("next_fail_cost"))
        .otherwise(pl.lit(0.0))
        .alias("_meta_fail_cost"),

        pl.when(pl.col("next_fail_date").is_not_null() & (pl.col("next_fail_date") <= pl.col("horizon_4q_end")))
        .then(pl.col("next_fail_restype"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("_meta_fail_restype"),

        pl.col("next_fail_date").cast(pl.String).alias("_meta_fail_date"),
    ]).drop(["next_fail_date", "next_fail_cost", "next_fail_restype", "month", "day", "horizon_4q_end"])

    # Apply 2025Q2 label cutoff (repdte_date > 2025-06-30 => target is null, _meta_label_complete = False)
    df = df.with_columns([
        pl.when(pl.col("repdte_date") <= cutoff_d)
        .then(pl.col("raw_target_fail_4q"))
        .otherwise(pl.lit(None, dtype=pl.Int8))
        .alias("target_fail_4q"),

        (pl.col("repdte_date") <= cutoff_d).alias("_meta_label_complete"),
    ]).drop("raw_target_fail_4q")

    # ------------------------------------------------------------------------
    # 4. Partition Splits Definition (_meta_split)
    # ------------------------------------------------------------------------
    print("\n--- 4. Assigning Temporal Splits (_meta_split) ---")
    dev_end_d = pd.to_datetime(DEV_END_DATE).date()
    embargo_start_d = pd.to_datetime(EMBARGO_START_DATE).date()
    embargo_end_d = pd.to_datetime(EMBARGO_END_DATE).date()
    test_start_d = pd.to_datetime(TEST_START_DATE).date()
    test_end_d = pd.to_datetime(TEST_END_DATE).date()

    df = df.with_columns(
        pl.when(pl.col("repdte_date") <= dev_end_d)
        .then(pl.lit("dev"))
        .when((pl.col("repdte_date") >= embargo_start_d) & (pl.col("repdte_date") <= embargo_end_d))
        .then(pl.lit("embargo"))
        .when((pl.col("repdte_date") >= test_start_d) & (pl.col("repdte_date") <= test_end_d))
        .then(pl.lit("test"))
        .when((pl.col("repdte_date") > test_end_d) & (pl.col("repdte_date") <= cutoff_d))
        .then(pl.lit("post_test"))
        .otherwise(pl.lit("incomplete_label"))
        .alias("_meta_split")
    )

    # ------------------------------------------------------------------------
    # 5. Join Static Institutions Metadata (Anti-Leakage Architecture)
    # ------------------------------------------------------------------------
    print("\n--- 5. Joining Institutions Static Data (Anti-Leakage) ---")
    df_inst = (
        pl.read_parquet(inst_path)
        .select(["CERT", "BKCLASS", "ACTIVE", "CHANGEC1", "ENDEFYMD"])
        .with_columns([
            pl.col("CERT").cast(pl.Int64),
            pl.col("BKCLASS").cast(pl.String),
            pl.col("ACTIVE").cast(pl.Int64).alias("_meta_inst_active"),
            pl.col("CHANGEC1").cast(pl.Float64).alias("_meta_inst_changec1"),
            pl.col("ENDEFYMD").cast(pl.String).alias("_meta_inst_enddate"),
        ])
        .drop(["ACTIVE", "CHANGEC1", "ENDEFYMD"])
    )

    df = df.join(df_inst, on="CERT", how="left")

    # ------------------------------------------------------------------------
    # 6. Column Ordering & Final Schema Structuring
    # ------------------------------------------------------------------------
    key_cols = ["CERT", "REPDTE", "repdte_date", "year", "quarter", "STNAME", "BKCLASS"]
    fin_cols = [c for c in FIN_FIELDS if c not in ("CERT", "REPDTE", "STNAME")]
    target_cols = ["target_fail_4q"]
    meta_cols = [c for c in df.columns if c.startswith("_meta_")]

    all_ordered_cols = key_cols + fin_cols + target_cols + meta_cols
    df = df.select(all_ordered_cols).sort(["CERT", "REPDTE"])

    # ------------------------------------------------------------------------
    # 7. Write Processed Silver Panel
    # ------------------------------------------------------------------------
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_parquet_path = DATA_PROCESSED / "bank_quarter_panel.parquet"
    print(f"\nWriting clean Silver panel to {out_parquet_path}...")
    df.write_parquet(out_parquet_path, compression="snappy")
    file_size_mb = out_parquet_path.stat().st_size / (1024 * 1024)

    # ------------------------------------------------------------------------
    # 8. Compute Summary Audit Statistics
    # ------------------------------------------------------------------------
    print("\n--- 8. Computing Silver Panel Summary Statistics ---")
    split_counts = (
        df.group_by("_meta_split")
        .agg([
            pl.len().alias("total_rows"),
            pl.col("target_fail_4q").sum().alias("fail_4q_events"),
            pl.col("CERT").n_unique().alias("distinct_certs"),
            (pl.col("target_fail_4q").sum() * 100.0 / pl.len()).alias("failure_rate_pct"),
            pl.col("_meta_fail_cost").sum().alias("total_fail_cost_k"),
        ])
        .sort("total_rows", descending=True)
        .to_dicts()
    )

    total_silver_rows = len(df)
    total_distinct_certs = df["CERT"].n_unique()
    total_failures_flagged = df["target_fail_4q"].sum()
    failed_banks_distinct = (
        df.filter(pl.col("target_fail_4q") == 1)["CERT"].n_unique()
    )

    print(f"Total Silver Rows             : {total_silver_rows:>10,}")
    print(f"Distinct Banks (CERT)         : {total_distinct_certs:>10,}")
    print(f"Total 4Q Failure Events Flagged: {total_failures_flagged:>10,}")
    print(f"Distinct Failed Banks in Horizon: {failed_banks_distinct:>10,}")
    print(f"Silver Parquet Size           : {file_size_mb:>10.2f} MB")

    for s in split_counts:
        print(f"  Split {s['_meta_split']:<18}: {s['total_rows']:>8,} rows | "
              f"{s['fail_4q_events']:>5} fail events ({s['failure_rate_pct']:.3f}%) | "
              f"{s['distinct_certs']:>6,} distinct banks")

    summary_data = {
        "silver_panel_path": str(out_parquet_path),
        "file_size_mb": round(file_size_mb, 2),
        "generation_time_seconds": round(time.time() - t0, 2),
        "raw_rows_input": n_raw_rows,
        "rows_dropped_asset_non_positive": n_drop_asset,
        "rows_dropped_post_mortem": n_drop_post_mortem,
        "total_silver_rows": total_silver_rows,
        "distinct_certs": total_distinct_certs,
        "total_4q_failure_events": int(total_failures_flagged) if total_failures_flagged is not None else 0,
        "distinct_failed_banks": int(failed_banks_distinct),
        "splits_breakdown": split_counts,
        "anti_leakage_guarantees": [
            "Contemporaneous features contain strictly historical Call Report financial data and static birth BKCLASS.",
            "Institutions snapshot variables (ACTIVE, CHANGEC1, ENDEFYMD) strictly isolated under _meta_ prefix.",
            "Forward failure matching is strictly point-in-time: earliest future failure where fail_date > repdte_date.",
            "Labels after 2025-06-30 are null-masked (_meta_label_complete = False) to prevent truncated-horizon bias.",
            "All voluntary exits, mergers, and absorptions receive target_fail_4q = 0 (only FDIC-assisted receiverships/closures are 1).",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "silver_panel_summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSaved Silver Panel Summary to: {summary_path}")

    return summary_data


def main():
    build_silver_panel()


if __name__ == "__main__":
    main()
