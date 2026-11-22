"""
Polars High-Performance Scalable Processing Pipeline for FDIC Bank Panel.
Course: BFT65B - Big Data Analytics in Finance
Subagent 2: Scalable Processing Specialist

Core Features:
1. Lazy execution with pl.scan_parquet for predicate & projection pushdown.
2. Contiguous quarterly lag arithmetic for YTD de-cumulation and 4Q window features.
3. State-quarter peer aggregations (group_by(["REPDTE", "STNAME"])).
4. Query plan extraction (unoptimized vs. optimized AST).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl

from src.configs.config import (
    DATA_RAW,
    DATA_RAW_FIN,
    REPORTS_DIR,
)


def get_terminal_failures_polars(
    fail_path: Union[str, Path] = DATA_RAW / "failures.parquet",
    is_lazy: bool = True,
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    Extract terminal failure dates per CERT:
    - Retains ASSISTANCE records as operating institutions.
    - Flags terminal_fail_date ONLY when the last event is RESTYPE == 'FAILURE'.
    """
    fail_path = Path(fail_path)
    base = pl.scan_parquet(str(fail_path)) if is_lazy else pl.read_parquet(str(fail_path))
    return (
        base
        .filter(pl.col("CERT").is_not_null())
        .with_columns([
            pl.col("CERT").cast(pl.Int64),
            pl.col("FAILDATE").str.to_date("%m/%d/%Y", strict=False).alias("fail_date"),
            pl.col("RESTYPE").cast(pl.String),
        ])
        .filter(pl.col("fail_date").is_not_null())
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


def get_institutions_polars(
    inst_path: Union[str, Path] = DATA_RAW / "institutions.parquet",
    is_lazy: bool = True,
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    Extract bank charter class BKCLASS per CERT from institutions.parquet.
    """
    inst_path = Path(inst_path)
    base = pl.scan_parquet(str(inst_path)) if is_lazy else pl.read_parquet(str(inst_path))
    return (
        base
        .select(["CERT", "BKCLASS"])
        .with_columns([
            pl.col("CERT").cast(pl.Int64),
            pl.col("BKCLASS").cast(pl.String),
        ])
    )


def build_polars_pipeline(
    fin_glob: Union[str, Path] = str(DATA_RAW_FIN / "REPDTE=*.parquet"),
    inst_path: Union[str, Path] = DATA_RAW / "institutions.parquet",
    fail_path: Union[str, Path] = DATA_RAW / "failures.parquet",
    is_lazy: bool = True,
) -> Tuple[Union[pl.LazyFrame, pl.DataFrame], Union[pl.LazyFrame, pl.DataFrame]]:
    """
    Build the full Polars transformation pipeline:
    a. Filter valid ASSET > 0, parse calendar keys (year, month, quarter, q_idx).
    b. Join institutions (BKCLASS) and filter post-terminal failures.
    c. De-cumulate YTD Income Flows (netinc_q).
    d. Core CAMELS ratios (eq_to_asset, npl_to_loans, reserve_to_loans, liquid_to_asset, bro_to_dep).
    e. 4-Quarter Window Transformations (asset_growth_4q, eq_to_asset_diff_4q, roa_rolling_4q_mean, roa_rolling_4q_std).
    f. State-Quarter Peer Aggregation (peer_bank_count, peer_total_assets, peer_mean_roa, peer_mean_eq_to_asset).

    Returns:
        (transformed_panel, peer_summary) — LazyFrames when is_lazy=True, eager DataFrames when is_lazy=False.
    """
    # 1. Base Scan (Lazy) vs Eager Read (DataFrame) of Financials
    fin_lf = pl.scan_parquet(str(fin_glob)) if is_lazy else pl.read_parquet(str(fin_glob))

    # Parse temporal keys & filter positive assets
    df_lf = (
        fin_lf
        .filter(pl.col("ASSET").is_not_null() & (pl.col("ASSET") > 0))
        .with_columns([
            pl.col("REPDTE").str.to_date("%Y%m%d").alias("repdte_date"),
            pl.col("REPDTE").str.slice(0, 4).cast(pl.Int32).alias("year"),
            pl.col("REPDTE").str.slice(4, 2).cast(pl.Int32).alias("month"),
        ])
        .with_columns([
            (pl.col("month") // 3).cast(pl.Int32).alias("quarter"),
        ])
        .with_columns([
            (pl.col("year") * 4 + pl.col("quarter") - 1).cast(pl.Int32).alias("q_idx"),
        ])
    )

    # 2. Join Failures & Institutions
    term_lf = get_terminal_failures_polars(fail_path, is_lazy=is_lazy)
    inst_lf = get_institutions_polars(inst_path, is_lazy=is_lazy)

    df_lf = (
        df_lf
        .join(term_lf, on="CERT", how="left")
        .filter(pl.col("terminal_fail_date").is_null() | (pl.col("repdte_date") < pl.col("terminal_fail_date")))
        .drop("terminal_fail_date")
        .join(inst_lf, on="CERT", how="left")
    )

    # Sort deterministically by bank and quarter index before window operations
    df_lf = df_lf.sort(["CERT", "q_idx"])

    # 3. De-cumulate YTD Income Flows
    # Single quarter net income:
    # If quarter == 1: NETINC
    # Else if (q_idx - q_idx.shift(1)) == 1: NETINC - NETINC.shift(1)
    # Else: null
    q_lag1 = pl.col("q_idx").shift(1).over("CERT")
    netinc_lag1 = pl.col("NETINC").shift(1).over("CERT")
    is_contig_1q = (pl.col("q_idx") - q_lag1) == 1

    netinc_q_expr = (
        pl.when(pl.col("quarter") == 1)
        .then(pl.col("NETINC"))
        .when(is_contig_1q)
        .then(pl.col("NETINC") - netinc_lag1)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("netinc_q")
    )

    # 4. Core CAMELS Ratios (Safe divisions with null-handling)
    eq_to_asset_expr = (pl.col("EQ") / pl.col("ASSET")).alias("eq_to_asset")

    npl_to_loans_expr = (
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("NCLNLS") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("npl_to_loans")
    )

    reserve_to_loans_expr = (
        pl.when(pl.col("LNLSGR") > 0)
        .then(pl.col("LNATRES") / pl.col("LNLSGR"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("reserve_to_loans")
    )

    liquid_to_asset_expr = (
        (pl.col("CHBAL").fill_null(0.0) + pl.col("SC").fill_null(0.0)) / pl.col("ASSET")
    ).alias("liquid_to_asset")

    bro_to_dep_expr = (
        pl.when(pl.col("DEP") > 0)
        .then(pl.col("BRO") / pl.col("DEP"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("bro_to_dep")
    )

    df_lf = df_lf.with_columns([
        netinc_q_expr,
        eq_to_asset_expr,
        npl_to_loans_expr,
        reserve_to_loans_expr,
        liquid_to_asset_expr,
        bro_to_dep_expr,
    ])

    # 5. Point-in-Time 4-Quarter Window Transformations (Deterministic Lag Arithmetic)
    q_lag4 = pl.col("q_idx").shift(4).over("CERT")
    is_4q_contiguous = (pl.col("q_idx") - q_lag4) == 4
    asset_lag4 = pl.col("ASSET").shift(4).over("CERT")
    eq_to_asset_lag4 = pl.col("eq_to_asset").shift(4).over("CERT")

    asset_growth_4q_expr = (
        pl.when(is_4q_contiguous & (asset_lag4 > 0))
        .then((pl.col("ASSET") - asset_lag4) / asset_lag4)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("asset_growth_4q")
    )

    eq_to_asset_diff_4q_expr = (
        pl.when(is_4q_contiguous)
        .then(pl.col("eq_to_asset") - eq_to_asset_lag4)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("eq_to_asset_diff_4q")
    )

    # 4-quarter rolling ROA mean and std over [t-3, t]
    r0 = pl.col("ROA")
    r1 = pl.col("ROA").shift(1).over("CERT")
    r2 = pl.col("ROA").shift(2).over("CERT")
    r3 = pl.col("ROA").shift(3).over("CERT")

    q1 = pl.col("q_idx").shift(1).over("CERT")
    q2 = pl.col("q_idx").shift(2).over("CERT")
    q3 = pl.col("q_idx").shift(3).over("CERT")

    is_rolling_contiguous = (
        ((pl.col("q_idx") - q1) == 1)
        & ((pl.col("q_idx") - q2) == 2)
        & ((pl.col("q_idx") - q3) == 3)
    )
    all_r_not_null = r0.is_not_null() & r1.is_not_null() & r2.is_not_null() & r3.is_not_null()
    valid_rolling = is_rolling_contiguous & all_r_not_null

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

    df_lf = df_lf.with_columns([
        asset_growth_4q_expr,
        eq_to_asset_diff_4q_expr,
        roa_rolling_4q_mean_expr,
        roa_rolling_4q_std_expr,
    ])

    # 6. State-Quarter Peer Aggregation
    peer_lf = (
        df_lf
        .group_by(["REPDTE", "STNAME"])
        .agg([
            pl.len().alias("peer_bank_count"),
            pl.col("ASSET").sum().alias("peer_total_assets"),
            pl.col("ROA").mean().alias("peer_mean_roa"),
            pl.col("eq_to_asset").mean().alias("peer_mean_eq_to_asset"),
        ])
        .sort(["REPDTE", "STNAME"])
    )

    return df_lf, peer_lf


def build_selective_polars_query(
    fin_glob: Union[str, Path] = str(DATA_RAW_FIN / "REPDTE=*.parquet"),
    min_year: int = 2008,
    is_lazy: bool = True,
) -> Tuple[Union[pl.LazyFrame, pl.DataFrame], Union[pl.LazyFrame, pl.DataFrame]]:
    """
    Demonstrate Predicate Pushdown and Projection Pushdown:
    Filters REPDTE >= '{min_year}0101' (direct Parquet row-group statistics predicate pushdown)
    and projects only 6 columns: [CERT, REPDTE, STNAME, ASSET, EQ, ROA].
    When is_lazy=False, executes eagerly via pl.read_parquet (no scan-level pushdown).
    """
    min_repdte = f"{min_year:04d}0101"
    base = pl.scan_parquet(str(fin_glob)) if is_lazy else pl.read_parquet(str(fin_glob))

    selective_lf = (
        base
        .filter(
            pl.col("ASSET").is_not_null()
            & (pl.col("ASSET") > 0)
            & (pl.col("REPDTE") >= min_repdte)
        )
        .select(["CERT", "REPDTE", "STNAME", "ASSET", "EQ", "ROA"])
        .with_columns([
            (pl.col("EQ") / pl.col("ASSET")).alias("eq_to_asset"),
        ])
    )

    peer_lf = (
        selective_lf
        .group_by(["REPDTE", "STNAME"])
        .agg([
            pl.len().alias("peer_bank_count"),
            pl.col("ASSET").sum().alias("peer_total_assets"),
            pl.col("ROA").mean().alias("peer_mean_roa"),
            pl.col("eq_to_asset").mean().alias("peer_mean_eq_to_asset"),
        ])
        .sort(["REPDTE", "STNAME"])
    )

    return selective_lf, peer_lf


def export_polars_query_plans(output_path: Union[str, Path] = REPORTS_DIR / "polars_query_plans.txt") -> None:
    """
    Generate and save unoptimized and optimized AST query plans for both
    the full panel pipeline and the selective query.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel_lf, peer_lf = build_polars_pipeline(is_lazy=True)
    sel_panel_lf, sel_peer_lf = build_selective_polars_query(min_year=2008)

    content = []
    content.append("=" * 80)
    content.append("POLARS QUERY PLAN ANALYSIS: PREDICATE & PROJECTION PUSHDOWN & CSE")
    content.append("Course: BFT65B - Big Data Analytics in Finance")
    content.append("=" * 80)
    content.append("\n" + "#" * 80)
    content.append("1. FULL PANEL PIPELINE - UNOPTIMIZED LOGICAL PLAN")
    content.append("#" * 80)
    content.append(panel_lf.explain(optimized=False))

    content.append("\n" + "#" * 80)
    content.append("2. FULL PANEL PIPELINE - OPTIMIZED PHYSICAL PLAN")
    content.append("#" * 80)
    content.append(panel_lf.explain(optimized=True))

    content.append("\n" + "#" * 80)
    content.append("3. SELECTIVE QUERY (year >= 2008, 6 columns) - UNOPTIMIZED LOGICAL PLAN")
    content.append("#" * 80)
    content.append(sel_peer_lf.explain(optimized=False))

    content.append("\n" + "#" * 80)
    content.append("4. SELECTIVE QUERY (year >= 2008, 6 columns) - OPTIMIZED PHYSICAL PLAN (PUSHDOWNS ACTIVE)")
    content.append("#" * 80)
    content.append(sel_peer_lf.explain(optimized=True))

    full_text = "\n".join(content)
    output_path.write_text(full_text, encoding="utf-8")
    print(f"Exported Polars query plans to: {output_path}")


if __name__ == "__main__":
    export_polars_query_plans()
