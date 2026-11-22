"""
Production Ingestion Script for FDIC BankFind Suite API.
Course: BFT65B - Big Data Analytics in Finance
Subagent 1: Data Pipeline Engineer

Downloads:
1. /financials   : quarterly Call Report panel, saved as Hive-partitioned files:
                   data/raw/financials/REPDTE={q}.parquet
                   (resumable: quarters already downloaded and complete are skipped)
2. /failures     : list of failed banks with COST to deposit insurance fund:
                   data/raw/failures.parquet
3. /institutions : structural & exit information (CHANGEC1, ACTIVE, etc.):
                   data/raw/institutions.parquet
4. Manifests     : data/raw/manifest_financials.json & data/raw/manifest_tables.json
                   recording row counts, API totals, completeness, bytes, and timestamps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import (
    API_BASE_URL,
    PAGE_SIZE,
    START_DATE,
    END_DATE,
    DATA_RAW,
    FIN_FIELDS,
    FAIL_FIELDS,
    INST_FIELDS,
)

sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------------
# Explicit PyArrow Schemas
# ----------------------------------------------------------------------------
FINANCIALS_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64()),
    pa.field("REPDTE", pa.string()),
    pa.field("STNAME", pa.string()),
    *[
        pa.field(col, pa.float64())
        for col in FIN_FIELDS
        if col not in ("CERT", "REPDTE", "STNAME")
    ],
])

FAILURES_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64()),
    pa.field("NAME", pa.string()),
    pa.field("CITYST", pa.string()),
    pa.field("FAILDATE", pa.string()),
    pa.field("FAILYR", pa.int64()),
    pa.field("COST", pa.float64()),
    pa.field("QBFASSET", pa.float64()),
    pa.field("QBFDEP", pa.float64()),
    pa.field("RESTYPE", pa.string()),
])

INSTITUTIONS_SCHEMA = pa.schema([
    pa.field("CERT", pa.int64()),
    pa.field("NAME", pa.string()),
    pa.field("STNAME", pa.string()),
    pa.field("BKCLASS", pa.string()),
    pa.field("ACTIVE", pa.int64()),
    pa.field("ESTYMD", pa.string()),
    pa.field("ENDEFYMD", pa.string()),
    pa.field("CHANGEC1", pa.float64()),
    pa.field("INSDATE", pa.string()),
])

# ----------------------------------------------------------------------------
# HTTP Session Helper
# ----------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BFT65B-EarlyWarning-DataEngineer/1.0"})


def get_json(endpoint: str, params: dict[str, Any], max_retries: int = 6, pause: float = 0.25) -> dict[str, Any]:
    """GET {API_BASE_URL}/{endpoint} and return parsed JSON with exponential backoff."""
    url = f"{API_BASE_URL}/{endpoint}"
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=180)
            if r.status_code == 200:
                time.sleep(pause)
                return r.json()
            print(f"    [WARN] HTTP {r.status_code} (attempt {attempt}/{max_retries}) - retrying...")
        except requests.RequestException as exc:
            print(f"    [WARN] {type(exc).__name__}: {exc} (attempt {attempt}/{max_retries}) - retrying...")
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts with params: {params}")


def fetch_all(
    endpoint: str,
    fields: list[str],
    filters: str | None = None,
    sort_by: str = "CERT",
) -> Tuple[pd.DataFrame, int]:
    """Page through FDIC endpoint with limit/offset. Returns (DataFrame, total reported by API)."""
    rows: list[dict[str, Any]] = []
    offset = 0
    total = None
    while True:
        params = {
            "fields": ",".join(fields),
            "limit": PAGE_SIZE,
            "offset": offset,
            "sort_by": sort_by,
            "sort_order": "ASC",
            "format": "json",
        }
        if filters:
            params["filters"] = filters
        js = get_json(endpoint, params)
        total = js["meta"]["total"]
        page = [rec["data"] for rec in js.get("data", [])]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= total:
            break

    df = pd.DataFrame(rows)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df, total


# ----------------------------------------------------------------------------
# Quarter Helpers
# ----------------------------------------------------------------------------
def quarter_ends(start: date, end: date) -> list[str]:
    """Return all quarter-end dates (YYYYMMDD) between start and end, inclusive."""
    ends = []
    for y in range(start.year, end.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = date(y, m, d)
            if start <= q <= end:
                ends.append(q.strftime("%Y%m%d"))
    return ends


def latest_repdte() -> str:
    """Query FDIC API for the latest available Call Report quarter-end date."""
    js = get_json(
        "financials",
        {"fields": "REPDTE", "sort_by": "REPDTE", "sort_order": "DESC", "limit": 1, "format": "json"},
    )
    return js["data"][0]["data"]["REPDTE"]


# ----------------------------------------------------------------------------
# Ingestion: Financials
# ----------------------------------------------------------------------------
def enforce_financials_schema(df: pd.DataFrame) -> pa.Table:
    """Enforce explicit types on Call Report dataframe and return PyArrow Table."""
    df_clean = df.copy()
    for col in FIN_FIELDS:
        if col not in df_clean.columns:
            df_clean[col] = None

    # Cast keys and strings
    df_clean["CERT"] = pd.to_numeric(df_clean["CERT"], errors="coerce").astype("int64")
    df_clean["REPDTE"] = df_clean["REPDTE"].astype("string")
    df_clean["STNAME"] = df_clean["STNAME"].astype("string")

    # Cast all 30 financial fields to float64
    for col in FIN_FIELDS:
        if col not in ("CERT", "REPDTE", "STNAME"):
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").astype("float64")

    # Ensure column ordering strictly matches schema
    df_clean = df_clean[FIN_FIELDS]
    return pa.Table.from_pandas(df_clean, schema=FINANCIALS_SCHEMA, preserve_index=False)


def download_financials(
    out_dir: Path,
    quarters: list[str],
    compression: str = "snappy",
) -> None:
    """Download Call Report quarterly partitions with explicit schema and audit manifest."""
    fin_dir = out_dir / "financials"
    fin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_financials.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    print(f"--- Ingesting Financials: {len(quarters)} quarter(s) into {fin_dir} ---")
    for i, q in enumerate(quarters, 1):
        target = fin_dir / f"REPDTE={q}.parquet"
        if target.exists() and manifest.get(q, {}).get("complete", False):
            print(f"[{i:3d}/{len(quarters):3d}] Quarter {q}: already downloaded & complete, skipping.")
            continue

        t0 = time.time()
        df, api_total = fetch_all("financials", FIN_FIELDS, filters=f"REPDTE:{q}")
        if df.empty:
            print(f"[{i:3d}/{len(quarters):3d}] Quarter {q}: no records returned (unpublished?).")
            continue

        n_rows = len(df)
        n_duplicates = int(df.duplicated(subset=["CERT", "REPDTE"]).sum())
        is_complete = (n_rows == api_total) and (n_duplicates == 0)

        # Enforce explicit schema and write
        table = enforce_financials_schema(df)
        pq.write_table(table, target, compression=compression)
        file_size_bytes = target.stat().st_size

        manifest[q] = {
            "rows": n_rows,
            "api_total": api_total,
            "complete": bool(is_complete),
            "n_duplicates": n_duplicates,
            "file_size_bytes": file_size_bytes,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        status_str = "OK" if is_complete else f"INCOMPLETE (dups={n_duplicates})"
        print(
            f"[{i:3d}/{len(quarters):3d}] Quarter {q}: {n_rows:>6,} rows "
            f"(API total {api_total:>6,}) [{status_str}] - "
            f"{file_size_bytes / 1024:>7.1f} KB in {time.time() - t0:5.2f}s"
        )


# ----------------------------------------------------------------------------
# Ingestion: Small Reference Tables (Failures & Institutions)
# ----------------------------------------------------------------------------
def enforce_failures_schema(df: pd.DataFrame) -> pa.Table:
    """Enforce explicit types on failures dataframe and return PyArrow Table."""
    df_clean = df.copy()
    for col in FAIL_FIELDS:
        if col not in df_clean.columns:
            df_clean[col] = None

    df_clean["CERT"] = pd.to_numeric(df_clean["CERT"], errors="coerce").astype("Int64")
    df_clean["FAILYR"] = pd.to_numeric(df_clean["FAILYR"], errors="coerce").astype("Int64")
    df_clean["COST"] = pd.to_numeric(df_clean["COST"], errors="coerce").astype("float64")
    df_clean["QBFASSET"] = pd.to_numeric(df_clean["QBFASSET"], errors="coerce").astype("float64")
    df_clean["QBFDEP"] = pd.to_numeric(df_clean["QBFDEP"], errors="coerce").astype("float64")

    for col in ["NAME", "CITYST", "FAILDATE", "RESTYPE"]:
        df_clean[col] = df_clean[col].astype("string")

    df_clean = df_clean[FAIL_FIELDS]
    return pa.Table.from_pandas(df_clean, schema=FAILURES_SCHEMA, preserve_index=False)


def enforce_institutions_schema(df: pd.DataFrame) -> pa.Table:
    """Enforce explicit types on institutions dataframe and return PyArrow Table."""
    df_clean = df.copy()
    for col in INST_FIELDS:
        if col not in df_clean.columns:
            df_clean[col] = None

    df_clean["CERT"] = pd.to_numeric(df_clean["CERT"], errors="coerce").astype("int64")
    df_clean["ACTIVE"] = pd.to_numeric(df_clean["ACTIVE"], errors="coerce").astype("int64")
    df_clean["CHANGEC1"] = pd.to_numeric(df_clean["CHANGEC1"], errors="coerce").astype("float64")

    for col in ["NAME", "STNAME", "BKCLASS", "ESTYMD", "ENDEFYMD", "INSDATE"]:
        df_clean[col] = df_clean[col].astype("string")

    df_clean = df_clean[INST_FIELDS]
    return pa.Table.from_pandas(df_clean, schema=INSTITUTIONS_SCHEMA, preserve_index=False)


def download_small_tables(out_dir: Path, compression: str = "snappy", force: bool = False) -> None:
    """Download failures and institutions tables with explicit schema and audit manifest."""
    manifest_path = out_dir / "manifest_tables.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    print(f"\n--- Ingesting Reference Tables into {out_dir} ---")
    tables_meta = [
        ("failures", FAIL_FIELDS, "FAILDATE", enforce_failures_schema),
        ("institutions", INST_FIELDS, "CERT", enforce_institutions_schema),
    ]

    for name, fields, sort_by, schema_fn in tables_meta:
        target = out_dir / f"{name}.parquet"
        if not force and target.exists() and manifest.get(name, {}).get("complete", False):
            print(f"{name:<14}: already downloaded & complete, skipping.")
            continue

        t0 = time.time()
        df, api_total = fetch_all(name, fields, sort_by=sort_by)
        n_rows = len(df)
        is_complete = (n_rows == api_total)

        table = schema_fn(df)
        pq.write_table(table, target, compression=compression)
        file_size_bytes = target.stat().st_size

        manifest[name] = {
            "rows": n_rows,
            "api_total": api_total,
            "complete": bool(is_complete),
            "file_size_bytes": file_size_bytes,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(
            f"{name:<14}: {n_rows:>6,} rows (API total {api_total:>6,}) "
            f"[{'OK' if is_complete else 'INCOMPLETE'}] - "
            f"{file_size_bytes / 1024:>7.1f} KB in {time.time() - t0:5.2f}s"
        )


# ----------------------------------------------------------------------------
# Completeness & Integrity Audit
# ----------------------------------------------------------------------------
def verify_completeness(out_dir: Path) -> bool:
    """Verify raw data directory completeness, schema integrity, and duplicate checks."""
    fin_manifest_path = out_dir / "manifest_financials.json"
    tab_manifest_path = out_dir / "manifest_tables.json"

    if not fin_manifest_path.exists():
        print("[ERROR] Financials manifest not found.")
        return False

    fin_manifest = json.loads(fin_manifest_path.read_text(encoding="utf-8"))
    bad_quarters = [q for q, m in fin_manifest.items() if not m.get("complete", False)]

    # Load panel keys across partitions
    fin_dir = out_dir / "financials"
    parquet_files = list(fin_dir.glob("REPDTE=*.parquet"))
    if not parquet_files:
        print("[ERROR] No financial parquet files found.")
        return False

    panel_keys = pd.read_parquet(fin_dir, columns=["CERT", "REPDTE"])
    n_panel_rows = len(panel_keys)
    n_distinct_banks = int(panel_keys["CERT"].nunique())
    n_duplicates = int(panel_keys.duplicated(subset=["CERT", "REPDTE"]).sum())

    tab_manifest = json.loads(tab_manifest_path.read_text(encoding="utf-8")) if tab_manifest_path.exists() else {}
    failures_file = out_dir / "failures.parquet"
    inst_file = out_dir / "institutions.parquet"

    n_failures = len(pd.read_parquet(failures_file)) if failures_file.exists() else 0
    n_institutions = len(pd.read_parquet(inst_file)) if inst_file.exists() else 0

    print("\n================ DATA INGESTION AUDIT ================")
    print(f"Quarters downloaded        : {len(fin_manifest):>8,}")
    print(f"Total panel rows           : {n_panel_rows:>8,}")
    print(f"Distinct banks (CERT)      : {n_distinct_banks:>8,}")
    print(f"Duplicate (CERT, REPDTE)   : {n_duplicates:>8,}")
    print(f"Incomplete quarters        : {bad_quarters if bad_quarters else 'None (100% complete)'}")
    print(f"Total failures recorded    : {n_failures:>8,}")
    print(f"Total institutions records : {n_institutions:>8,}")
    print("======================================================")

    is_ok = (n_duplicates == 0) and (len(bad_quarters) == 0) and (n_panel_rows > 0)
    return is_ok


# ----------------------------------------------------------------------------
# Main CLI Entrypoint
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FDIC BankFind Suite API Ingestion with Explicit PyArrow Schemas"
    )
    parser.add_argument("--start", default=START_DATE, help="Start quarter date (YYYY-MM-DD)")
    parser.add_argument("--end", default=END_DATE, help="End quarter date (YYYY-MM-DD or 'latest')")
    parser.add_argument("--out", default=str(DATA_RAW), help="Output raw data directory")
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "zstd", "gzip", "none", "uncompressed"],
        help="Parquet compression codec (default: snappy)",
    )
    parser.add_argument(
        "--skip-reference-tables",
        action="store_true",
        help="Skip downloading failures and institutions tables",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    codec = None if args.compression in ("none", "uncompressed") else args.compression

    end_str = args.end
    if end_str == "latest":
        latest_date_str = latest_repdte()
        end_str = f"{latest_date_str[:4]}-{latest_date_str[4:6]}-{latest_date_str[6:8]}"

    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(end_str)
    quarters = quarter_ends(start_d, end_d)

    print("==========================================================")
    print("  FDIC BankFind Ingestion Pipeline (Production Bronze)    ")
    print("==========================================================")
    print(f"Target Directory : {out_dir.resolve()}")
    print(f"Date Range       : {args.start} -> {end_str} ({len(quarters)} quarter(s))")
    print(f"Compression      : {args.compression}")
    print("==========================================================\n")

    download_financials(out_dir, quarters, compression=codec or "none")
    if not args.skip_reference_tables:
        download_small_tables(out_dir, compression=codec or "none")

    success = verify_completeness(out_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
