"""
Global Configuration for FDIC Bank Failure Early Warning Big Data Pipeline.
Course: BFT65B - Big Data Analytics in Finance
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_RAW_FIN = DATA_RAW / "financials"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_FEATURES = DATA_DIR / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for p in [DATA_RAW_FIN, DATA_PROCESSED, DATA_FEATURES, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# API Settings
API_BASE_URL = "https://api.fdic.gov/banks"
PAGE_SIZE = 10_000

# Course Conventions for Dates & Split Strategy (Option A)
START_DATE = "1984-03-31"
END_DATE = "2026-06-30"

# Train/Dev, Embargo, and Test Split Definitions
DEV_START_DATE = "1984-03-31"
DEV_END_DATE = "2006-12-31"      # Development / Training period (~23 years)
EMBARGO_START_DATE = "2007-01-01"
EMBARGO_END_DATE = "2007-12-31"  # 4-quarter gap (Embargo window, only 3 failures in 2007)
TEST_START_DATE = "2008-01-01"
TEST_END_DATE = "2014-12-31"     # Out-of-time Test set (>500 failures during GFC)
LABEL_CUTOFF_DATE = "2025-06-30" # Max allowed horizon by exam prompt

# Field Definitions
FIN_FIELDS = [
    "CERT", "REPDTE", "STNAME",
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",          # balance sheet, liquidity
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",        # loans, reserves
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",     # asset quality
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",      # earnings
    "ROA", "ROE", "NIMY",                                           # earnings ratios
    "RBC1AAJ", "RBCRWAJ",                                           # capital ratios
]

FAIL_FIELDS = ["CERT", "NAME", "CITYST", "FAILDATE", "FAILYR", "COST", "QBFASSET", "QBFDEP", "RESTYPE"]
INST_FIELDS = ["CERT", "NAME", "STNAME", "BKCLASS", "ACTIVE", "ESTYMD", "ENDEFYMD", "CHANGEC1", "INSDATE"]
