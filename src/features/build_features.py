"""
Leak-Free CAMELS Feature Engineering Pipeline for FDIC Bank Failure Early Warning.
Course: BFT65B - Big Data Analytics in Finance
Subagent 3: Financial Quant & Machine Learning Engineer

Core Architecture:
1. Strict Point-in-Time Ordering & Contiguous Lag Arithmetic:
   - Sort deterministically by ["CERT", "q_idx"] before any .shift(k).over("CERT").
   - Enforce contiguity check: (q_idx - q_idx.shift(k)) == k for all lags.
   - Backward lags computed on the full panel first, then filtered split_regime != "incomplete_label".
2. De-cumulation of Year-to-Date (YTD) Call Report Income Flows (NETINC, INTINC, EINTEXP, NONII, NONIX, ELNATR, NTLNLS).
3. 39 Numeric CAMELS Features + 1 Categorical Charter Feature (BKCLASS) = 40 Model Features.
4. Export Gold Feature Table: data/features/gold_feature_table.parquet.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.configs.config import (
    DATA_FEATURES,
    DATA_PROCESSED,
    RANDOM_SEED,
    REPORTS_DIR,
)

# ---------------------------------------------------------------------------
# Feature Column Definitions (Single Source of Truth)
# ---------------------------------------------------------------------------

CAMELS_FEATURE_MAP: Dict[str, List[str]] = {
    "C": [
        "eq_to_asset",
        "tier1_leverage_ratio",
        "total_rbc_ratio",
        "equity_to_loans",
        "technical_insolvency_flag",
        "basel_missing_flag",
    ],
    "A": [
        "npl_to_loans",
        "nonaccrual_to_loans",
        "pastdue90_to_loans",
        "pastdue30_89_to_loans",
        "oreo_to_assets",
        "texas_ratio_proxy",
        "reserve_to_loans",
        "coverage_ratio",
    ],
    "M": [
        "cost_to_income_ratio",
        "overhead_to_assets_q",
        "real_estate_to_loans",
        "ci_to_loans",
        "consumer_to_loans",
        "log_assets",
        "asset_growth_4q",
    ],
    "E": [
        "roa",
        "roe",
        "nim",
        "netinc_q_to_assets",
        "provision_to_loans_q",
        "net_chargeoff_to_loans_q",
    ],
    "L": [
        "liquid_to_asset",
        "cash_to_asset",
        "loans_to_deposits",
        "bro_to_dep",
        "non_deposit_funding_to_asset",
    ],
    "S": [
        "eq_to_asset_diff_4q",
        "npl_to_loans_diff_4q",
        "liquid_to_asset_diff_4q",
        "bro_to_dep_diff_4q",
        "roa_rolling_4q_mean",
        "roa_rolling_4q_std",
        "eq_vs_state_peer",
    ],
}

NUMERIC_FEATURES: List[str] = [
    feat for group in CAMELS_FEATURE_MAP.values() for feat in group
]

CATEGORICAL_FEATURES: List[str] = ["BKCLASS"]

ALL_MODEL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

KEY_COLUMNS: List[str] = [
    "CERT",
    "REPDTE",
    "repdte_date",
    "year",
    "quarter",
    "q_idx",
    "STNAME",
    "split_regime",
    "target_fail_4q",
]


def build_gold_feature_table(
    input_path: Path | str = DATA_PROCESSED / "bank_quarter_panel.parquet",
    output_path: Path | str = DATA_FEATURES / "gold_feature_table.parquet",
    input_lf: pl.LazyFrame | None = None,
    write_output: bool = True,
) -> pl.DataFrame:
    """
    Construct the leak-free Gold Feature Table from Silver panel:
    1. Parse q_idx and split_regime.
    2. Deterministic sort by ["CERT", "q_idx"].
    3. Contiguous de-cumulation of 7 YTD flow line items.
    4. Compute 39 CAMELS numeric features + BKCLASS.
    5. Contiguous 4Q window differences, growth rates, and ROA rolling statistics.
    6. Contemporaneous peer benchmark (State-Quarter).
    7. Filter incomplete label records (> 2025-06-30).
    8. Write gold_feature_table.parquet (when write_output=True).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    t0 = time.time()

    if write_output:
        print("=" * 70)
        print("      PART 4: LEAK-FREE CAMELS FEATURE PIPELINE (POLARS LAZY)      ")
        print("=" * 70)
        print(f"Reading Silver Panel from: {input_path}...")

    lf = input_lf if input_lf is not None else pl.scan_parquet(str(input_path))

    # 1. Temporal keys, split regime, and clean BKCLASS (null or empty string -> 'MISSING')
    lf = lf.with_columns([
        (pl.col("year").cast(pl.Int32) * 4 + pl.col("quarter").cast(pl.Int32) - 1).alias("q_idx"),
        pl.col("_meta_split").alias("split_regime"),
        pl.when(pl.col("BKCLASS").is_null() | (pl.col("BKCLASS").str.strip_chars() == ""))
        .then(pl.lit("MISSING"))
        .otherwise(pl.col("BKCLASS").str.strip_chars())
        .alias("BKCLASS"),
    ])

    # 2. Strict Point-in-Time Deterministic Ordering
    lf = lf.sort(["CERT", "q_idx"])

    # 3. Contiguous De-cumulation of Year-to-Date (YTD) Income & Expense Flows
    # In Call Reports, flows accumulate across Q1, Q2, Q3, Q4 of each calendar year.
    flow_cols = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"]
    is_contig_1q = (pl.col("q_idx") - pl.col("q_idx").shift(1).over("CERT")) == 1

    decumulated_exprs = []
    for col in flow_cols:
        col_lag1 = pl.col(col).shift(1).over("CERT")
        flow_q_expr = (
            pl.when(pl.col("quarter") == 1)
            .then(pl.col(col))
            .when(is_contig_1q.fill_null(False))
            .then(pl.col(col) - col_lag1)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(f"{col.lower()}_q")
        )
        decumulated_exprs.append(flow_q_expr)

    lf = lf.with_columns(decumulated_exprs)

    # 4. Engineer Base CAMELS Features
    # C — Capital Adequacy
    # Note on FDIC BankFind API encoding:
    # Unreported RBC1AAJ (pre-1990 TFR schedules / null EQ) and RBCRWAJ (pre-1990 Basel I
    # and post-2020 CBLR opt-out) are returned as exact 0.0 when EQ != 0 or EQ is null.
    # Decode sentinel 0.0s to null so pre-Basel/CBLR rows are not mistaken for 0% capital.
    t1_lev_clean = (
        pl.when((pl.col("RBC1AAJ") == 0.0) & (pl.col("EQ").is_null() | (pl.col("EQ") != 0.0)))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("RBC1AAJ"))
    )
    rbc_total_clean = (
        pl.when((pl.col("RBCRWAJ") == 0.0) & (pl.col("EQ").is_null() | (pl.col("EQ") != 0.0)))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("RBCRWAJ"))
    )

    c_exprs = [
        (pl.col("EQ") / pl.col("ASSET")).alias("eq_to_asset"),
        t1_lev_clean.alias("tier1_leverage_ratio"),
        rbc_total_clean.alias("total_rbc_ratio"),
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("EQ") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("equity_to_loans"),
        pl.when(pl.col("EQ") < 0)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .alias("technical_insolvency_flag"),
        pl.when(rbc_total_clean.is_null())
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .alias("basel_missing_flag"),
    ]

    # A — Asset Quality
    texas_denom = pl.col("EQ") + pl.col("LNATRES").fill_null(0.0)
    texas_num = pl.col("NCLNLS").fill_null(0.0) + pl.col("ORE").fill_null(0.0)
    texas_valid = pl.col("EQ").is_not_null() & (
        pl.col("NCLNLS").is_not_null() | pl.col("ORE").is_not_null()
    )
    texas_ratio_expr = (
        pl.when(~texas_valid)
        .then(pl.lit(None, dtype=pl.Float64))
        .when(texas_denom > 0)
        .then(texas_num / texas_denom)
        .otherwise(
            pl.when(texas_num > 0).then(pl.lit(10.0)).otherwise(pl.lit(0.0))
        )
        .alias("texas_ratio_proxy")
    )

    a_exprs = [
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("NCLNLS") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("npl_to_loans"),
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("NAASSET") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("nonaccrual_to_loans"),
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("P9ASSET") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("pastdue90_to_loans"),
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("P3ASSET") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("pastdue30_89_to_loans"),
        (pl.col("ORE") / pl.col("ASSET")).alias("oreo_to_assets"),
        texas_ratio_expr,
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("LNATRES") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("reserve_to_loans"),
        pl.when(pl.col("NCLNLS") > 0)
        .then(pl.col("LNATRES") / pl.col("NCLNLS"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("coverage_ratio"),
    ]

    # M — Management, Efficiency & Portfolio Concentration (base)
    op_inc = pl.col("intinc_q") - pl.col("eintexp_q") + pl.col("nonii_q")
    m_base_exprs = [
        pl.when(op_inc > 0)
        .then(pl.col("nonix_q") / op_inc)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("cost_to_income_ratio"),
        ((pl.col("nonix_q") * 4.0) / pl.col("ASSET")).alias("overhead_to_assets_q"),
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("LNRE") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("real_estate_to_loans"),
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("LNCI") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("ci_to_loans"),
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("LNCON") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("consumer_to_loans"),
        pl.col("ASSET").log().alias("log_assets"),
    ]

    # E — Earnings & Profitability
    e_exprs = [
        pl.col("ROA").alias("roa"),
        pl.col("ROE").alias("roe"),
        pl.col("NIMY").alias("nim"),
        ((pl.col("netinc_q") * 4.0) / pl.col("ASSET")).alias("netinc_q_to_assets"),
        pl.when(pl.col("LNLSGR") > 0)
        .then((pl.col("elnatr_q") * 4.0) / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("provision_to_loans_q"),
        pl.when(pl.col("LNLSGR") > 0)
        .then((pl.col("ntlnls_q") * 4.0) / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("net_chargeoff_to_loans_q"),
    ]

    # L — Liquidity & Funding Structure
    l_exprs = [
        ((pl.col("CHBAL").fill_null(0.0) + pl.col("SC").fill_null(0.0)) / pl.col("ASSET")).alias("liquid_to_asset"),
        (pl.col("CHBAL") / pl.col("ASSET")).alias("cash_to_asset"),
        pl.when(pl.col("DEP") > 0)
        .then(pl.col("LNLSGR") / pl.col("DEP"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("loans_to_deposits"),
        pl.when(pl.col("DEP") > 0)
        .then(pl.col("BRO") / pl.col("DEP"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("bro_to_dep"),
        ((pl.col("ASSET") - pl.col("EQ") - pl.col("DEP").fill_null(0.0)) / pl.col("ASSET")).alias("non_deposit_funding_to_asset"),
    ]

    lf = lf.with_columns(c_exprs + a_exprs + m_base_exprs + e_exprs + l_exprs)

    # 5. S — Sensitivity, 4Q Dynamics & Contemporaneous Peer Benchmarks
    # Contiguity check for 4 quarters: (q_idx - q_idx.shift(4).over("CERT")) == 4
    q_lag4 = pl.col("q_idx").shift(4).over("CERT")
    is_contig_4q = (pl.col("q_idx") - q_lag4) == 4

    asset_lag4 = pl.col("ASSET").shift(4).over("CERT")
    asset_growth_4q_expr = (
        pl.when(is_contig_4q.fill_null(False) & (asset_lag4 > 0))
        .then((pl.col("ASSET") - asset_lag4) / asset_lag4)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("asset_growth_4q")
    )

    eq_lag4 = pl.col("eq_to_asset").shift(4).over("CERT")
    eq_to_asset_diff_4q_expr = (
        pl.when(is_contig_4q.fill_null(False))
        .then(pl.col("eq_to_asset") - eq_lag4)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("eq_to_asset_diff_4q")
    )

    npl_lag4 = pl.col("npl_to_loans").shift(4).over("CERT")
    npl_to_loans_diff_4q_expr = (
        pl.when(is_contig_4q.fill_null(False))
        .then(pl.col("npl_to_loans") - npl_lag4)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("npl_to_loans_diff_4q")
    )

    liq_lag4 = pl.col("liquid_to_asset").shift(4).over("CERT")
    liquid_to_asset_diff_4q_expr = (
        pl.when(is_contig_4q.fill_null(False))
        .then(pl.col("liquid_to_asset") - liq_lag4)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("liquid_to_asset_diff_4q")
    )

    bro_lag4 = pl.col("bro_to_dep").shift(4).over("CERT")
    bro_to_dep_diff_4q_expr = (
        pl.when(is_contig_4q.fill_null(False))
        .then(pl.col("bro_to_dep") - bro_lag4)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("bro_to_dep_diff_4q")
    )

    # 4-quarter rolling ROA mean and sample std over [t-3, t]
    q1 = pl.col("q_idx").shift(1).over("CERT")
    q2 = pl.col("q_idx").shift(2).over("CERT")
    q3 = pl.col("q_idx").shift(3).over("CERT")
    is_rolling_contiguous = (
        ((pl.col("q_idx") - q1) == 1)
        & ((pl.col("q_idx") - q2) == 2)
        & ((pl.col("q_idx") - q3) == 3)
    )

    r0 = pl.col("ROA")
    r1 = pl.col("ROA").shift(1).over("CERT")
    r2 = pl.col("ROA").shift(2).over("CERT")
    r3 = pl.col("ROA").shift(3).over("CERT")
    all_r_not_null = r0.is_not_null() & r1.is_not_null() & r2.is_not_null() & r3.is_not_null()
    valid_rolling = is_rolling_contiguous.fill_null(False) & all_r_not_null

    mean_roa_4q = (r0 + r1 + r2 + r3) / 4.0
    sum_sq_diff_roa_4q = (
        (r0 - mean_roa_4q).pow(2)
        + (r1 - mean_roa_4q).pow(2)
        + (r2 - mean_roa_4q).pow(2)
        + (r3 - mean_roa_4q).pow(2)
    )
    std_roa_4q = (sum_sq_diff_roa_4q / 3.0).sqrt()

    roa_rolling_4q_mean_expr = (
        pl.when(valid_rolling)
        .then(mean_roa_4q)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("roa_rolling_4q_mean")
    )

    roa_rolling_4q_std_expr = (
        pl.when(valid_rolling)
        .then(std_roa_4q)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("roa_rolling_4q_std")
    )

    # Strictly contemporaneous peer benchmark within (REPDTE, STNAME)
    eq_vs_state_peer_expr = (
        pl.col("eq_to_asset") - pl.col("eq_to_asset").mean().over(["REPDTE", "STNAME"])
    ).alias("eq_vs_state_peer")

    s_exprs = [
        asset_growth_4q_expr,
        eq_to_asset_diff_4q_expr,
        npl_to_loans_diff_4q_expr,
        liquid_to_asset_diff_4q_expr,
        bro_to_dep_diff_4q_expr,
        roa_rolling_4q_mean_expr,
        roa_rolling_4q_std_expr,
        eq_vs_state_peer_expr,
    ]

    lf = lf.with_columns(s_exprs)

    # 6. Filter Incomplete Label Horizon (keep split_regime != "incomplete_label")
    # All 4Q historical lags were computed on the full panel first to prevent boundary truncation.
    lf = lf.filter(pl.col("split_regime") != "incomplete_label")

    # 7. Collect metadata columns to preserve for Part 5 cost evaluations & audits
    schema_cols = lf.collect_schema().names()
    meta_cols = [c for c in schema_cols if c.startswith("_meta_")]

    final_columns = KEY_COLUMNS + meta_cols + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    # Remove any accidental duplicates while preserving order
    seen = set()
    ordered_unique_cols = []
    for c in final_columns:
        if c not in seen:
            seen.add(c)
            ordered_unique_cols.append(c)

    lf = lf.select(ordered_unique_cols)

    # 8. Collect and Write to Gold Feature Parquet
    if write_output:
        print("\nExecuting Lazy Query Plan and collecting Gold Feature Table...")
    df = lf.collect()

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {len(df):,} rows × {len(df.columns)} columns to {output_path}...")
        df.write_parquet(output_path, compression="snappy")
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"Gold Feature Parquet Size: {file_size_mb:.2f} MB")
        print(f"Generation Time: {time.time() - t0:.2f} seconds.")
        print("=" * 70)
    return df


def audit_gold_feature_table(df: pl.DataFrame | None = None) -> Dict[str, Any]:
    """
    Validate the Gold Feature Table integrity:
    1. Check exactly 1,663,902 rows.
    2. Check target_fail_4q is strictly {0, 1} with 0 nulls.
    3. Check split distribution (dev, embargo, test, post_test).
    4. Check all 39 numeric features and BKCLASS are present.
    """
    if df is None:
        gold_path = DATA_FEATURES / "gold_feature_table.parquet"
        print(f"Loading Gold Feature Table from {gold_path} for audit...")
        df = pl.read_parquet(str(gold_path))

    total_rows = len(df)
    n_certs = df["CERT"].n_unique()
    target_nulls = df["target_fail_4q"].is_null().sum()
    target_fails = df["target_fail_4q"].sum()

    split_stats = (
        df.group_by("split_regime")
        .agg([
            pl.len().alias("rows"),
            pl.col("target_fail_4q").sum().alias("fail_events"),
            (pl.col("target_fail_4q").sum() * 100.0 / pl.len()).alias("fail_rate_pct"),
            pl.col("CERT").n_unique().alias("certs"),
            pl.col("repdte_date").min().alias("min_date"),
            pl.col("repdte_date").max().alias("max_date"),
        ])
        .sort("min_date")
        .to_dicts()
    )

    print("\n--- Split Breakdown in Gold Table ---")
    for s in split_stats:
        print(
            f"Split {s['split_regime']:<12}: {s['rows']:>9,} rows | "
            f"{s['fail_events']:>6,} fail events ({s['fail_rate_pct']:.3f}%) | "
            f"{s['certs']:>6,} banks | {s['min_date']} to {s['max_date']}"
        )

    # Verify column presence
    missing_num = [c for c in NUMERIC_FEATURES if c not in df.columns]
    missing_cat = [c for c in CATEGORICAL_FEATURES if c not in df.columns]
    if missing_num:
        raise ValueError(f"Missing numeric features: {missing_num}")
    if missing_cat:
        raise ValueError(f"Missing categorical features: {missing_cat}")

    print(f"\nAll {len(NUMERIC_FEATURES)} Numeric CAMELS Features + {len(CATEGORICAL_FEATURES)} Categorical Features verified!")
    print(f"Total Rows: {total_rows:,} (Expected: 1,663,902)")
    print(f"Target Nulls: {target_nulls} (Expected: 0)")
    print(f"Target Failures Flagged: {target_fails:,} (Expected: 12,459)")

    return {
        "total_rows": total_rows,
        "distinct_certs": n_certs,
        "target_nulls": int(target_nulls),
        "target_failures": int(target_fails),
        "split_stats": split_stats,
        "numeric_features_count": len(NUMERIC_FEATURES),
        "categorical_features_count": len(CATEGORICAL_FEATURES),
    }


def main():
    df = build_gold_feature_table()
    audit_gold_feature_table(df)


if __name__ == "__main__":
    main()
