# Scalable Early-Warning System for Bank Distress & Failure
### An End-to-End Leak-Free Big-Data Pipeline on 42 Years of U.S. FDIC Call Reports (1984Q1–2026Q2)

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Polars 2.1.0](https://img.shields.io/badge/polars-2.1.0-orange.svg)](https://pola.rs/)
[![PySpark 4.3.0](https://img.shields.io/badge/pyspark-4.3.0-red.svg)](https://spark.apache.org/)
[![LightGBM 4.7.0](https://img.shields.io/badge/lightgbm-4.7.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Audit Invariants](https://img.shields.io/badge/verification-47%2F47%20PASS-success.svg)](reports/final_audit_certificate.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview & Empirical Highlights

This repository contains the complete, reproducible source code and academic report for the **BFT65B Big Data Analytics in Finance** capstone research project at the **National Economics University (NEU)**, Hanoi, Vietnam.

- **Author:** Nguyễn Thiện Nhân (Student ID: `11231004`)
- **Public Repository:** `https://github.com/ntnhana1nhk55-glitch/bft65b-fdic-early-warning-`
- **Data Acquisition Method & Date:** Automated resumable FDIC BankFind Suite REST API ingestion (`src/data/acquire.py`), downloaded and verified on **November 7, 2026**.

The system builds an enterprise-grade Big Data early-warning pipeline for banking distress and resolution across the entire 42-year digital history of U.S. Federal Deposit Insurance Corporation (FDIC) Call Reports ($1984\text{Q}1–2026\text{Q}2$), encompassing **1,682,616 bank-quarter filings**, **27,835 unique institutions**, and **4,117 regulatory resolutions**.

```
+──────────────────────────────────────────────────────────────────────────────────────────────────+
|                                  SYSTEM ARCHITECTURE AT A GLANCE                                 |
|                                                                                                  |
|   [1. Ingestion & Storage]        [2. Panel & Quality Audit]       [3. Scalable Engine]          |
|   • 170 Quarters (1.68M rows)     • 0 Duplicates, 1,185 Post-Fail  • Polars Lazy: 1,140 ms       |
|   • 33-col PyArrow schema         • YTD Sawtooth De-cumulation     • Pushdown: 43.4 ms (40.3x)   |
|   • Parquet ZSTD: 190.7 MB        • Sentinel-0.0 Decoding          • PySpark Broadcast: 3.42x    |
|   • 459.0x vs Pandas CSV          • Point-in-Time Forward Target   • 4-Way Parity (5.68e-14)     |
|                                                                                                  |
|   [4. CAMELS Feature Pipeline]    [5. Modelling & CV]              [6. Supervisory Economics]    |
|   • 40 CAMELS Features            • 4-Fold Purged Expanding CV     • LightGBM Test PR-AUC: 0.645 |
|   • State-Quarter Peer Dev        • 1-Year Embargo (2007)          • Top 2% Captures 80.2% Fails |
|   • Scikit-Learn Pipeline         • Out-of-Time GFC (2008-2014)    • Optimal tau* = 0.0401       |
|   • Invariance: 1.39e-17          • Hyperparam Regularization      • Net Savings: $4,074.08 M    |
+──────────────────────────────────────────────────────────────────────────────────────────────────+
```

### Key Empirical Results
- **Storage Engineering (Part 1):** Parquet Snappy (**241.88 MB**) and Parquet Zstandard (**190.73 MB**, $2.22\times$ compression) slash storage from raw CSV (**424.12 MB**). Column projection and predicate pushdown accelerate selective supervisory queries on Parquet Snappy to **31.9 ms**—delivering a **$459.0\times$ speedup** over Pandas CSV full reads (**14,652.7 ms**), a **$22.6\times$ speedup** over Polars CSV full scans (**721.9 ms**), and a **$16.4\times$ speedup** over Polars CSV selective queries (**522.5 ms**).
- **Quality & Causality (Part 2):** Verifies **zero duplicate `(CERT, REPDTE)` pairs**, removes **1,185 residual post-terminal receivership filings** across 270 closed banks while preserving **4,425 post-assistance operating quarters** and **14,605 negative equity ($EQ < 0$) technical insolvencies**, eliminates Call Report YTD accumulation sawteeth via contiguous within-bank de-cumulation, decodes **sentinel `0.0` values** for pre-1990 Basel and post-2020 CBLR regimes, and constructs a causal point-in-time forward target $y \in (t, t+4\text{Q}]$ with right-censoring quarantine post June 30, 2025.
- **Engine Performance & Distributed Joins (Part 3):** Polars Lazy compiles execution plans in **1,140 ms** ($8.03\times$ faster than vectorized Pandas at **9,154 ms**) and achieves a **$40.29\times$ speedup** on selective pushdown workloads (**43.4 ms** vs **1,749.7 ms**). PySpark BroadcastHashJoin achieves a **$3.42\times$ speedup** (**1,180.8 ms** vs **4,035.1 ms** SortMergeJoin) by eliminating the $O(N)$ shuffle exchange. Four-way cross-engine calculations verify mathematical parity across all 1,681,431 Silver panel rows ($\max|\Delta| \le 5.68 \times 10^{-14}$).
- **Leak-Free Preprocessing (Part 4):** Standardized Scikit-Learn `Pipeline` fitted strictly inside training folds. An adversarial Silver $\to$ Gold $\to$ Preprocessor perturbation proof confirms zero backward leakage ($\max|\Delta| = 1.39 \times 10^{-17}$ in `reports/final_audit_certificate.json`).
- **Predictive Performance (Part 5):** Champion **LightGBM CAMELS** achieves **0.6454 Test PR-AUC** on the Out-of-Time Global Financial Crisis ($1984–2006$ DEV [2,658 failed banks, 10,225 failure quarters] $\to$ 2007 Embargo $\to$ $2008–2014$ TEST [526 failed banks, 2,052 failure quarters], 212,645 rows), outperforming single-ratio Texas Ratio supervisory heuristics (**0.4662**) by **$+38.4\%$**. At an inspection budget of **Top 2%**, the model captures **80.2% of failing institutions** and **78.1% of failing banking assets**.
- **Supervisory Loss Economics (Part 5 & 6):** Asymmetric loss threshold optimization ($\tau^* = 0.0401$) yields **\$4,074.08 Million in net regulatory cost savings**—an improvement of **+\$2,453.54 Million** over standard naive $\tau = 0.50$.

---

## 2. Repository Directory Structure

```text
bft65b-fdic-early-warning-/
├── data/                                 # Data storage (Raw, Processed Silver, Features Gold)
│   ├── raw/                              # Raw FDIC API downloads (170 quarterly partitions)
│   │   ├── financials/                   # Partitioned Call Reports (REPDTE=YYYYMMDD.parquet)
│   │   ├── institutions.parquet          # FDIC structural demographics (27,835 banks)
│   │   ├── failures.parquet              # FDIC failure & assistance events (4,117 records)
│   │   └── manifest_financials.json      # SHA-256 cryptographic verification manifest
│   ├── processed/                        # Silver layer: Deduplicated longitudinal panel
│   │   └── bank_quarter_panel.parquet    # Cleaned Silver panel (1,681,431 rows x 46 cols)
│   └── features/                         # Gold layer: Leak-free feature table
│       └── gold_feature_table.parquet    # Feature-engineered Gold panel (1,663,902 rows x 57 cols)
├── reports/                              # Generated research artifacts, audits, and figures
│   ├── figures/                          # 6 Publication-grade 300 DPI figures
│   │   ├── fig1_panel_coverage_and_failures.png
│   │   ├── fig2_storage_benchmark.png
│   │   ├── fig3_data_quality_and_ytd_sawtooth.png
│   │   ├── fig4_engine_benchmark_and_scaling.png
│   │   ├── fig5_camels_auc_and_importance.png
│   │   └── fig6_model_evaluation_and_cost_threshold.png
│   ├── FINAL_REPORT_BFT65B.md            # Complete English Academic Research Report
│   ├── PROJECT_LOG.md                    # Detailed multi-phase engineering journal
│   ├── independent_external_audit.md     # Independent zero-trust adversarial audit report
│   ├── final_audit_certificate.json      # Machine-readable audit verification (47/47 PASS)
│   ├── storage_benchmark_full.csv        # 5-layout storage footprint & latency benchmarks
│   ├── data_quality_report.json          # Automated data quality audit statistics
│   ├── part3_engine_benchmark.csv        # Pandas vs Polars Eager/Lazy benchmark results
│   ├── part3_amdahl_scaling.csv          # Multi-threaded scaling measurements & Amdahl fit
│   ├── part3_scaleup_stress.csv          # 1x to 10x (16.8M rows) scale-up stress test
│   ├── part3_spark_join_benchmark.json   # PySpark Catalyst Broadcast vs SortMerge join test
│   ├── part4_feature_catalog.csv         # 40-feature CAMELS dictionary & univariate AUC
│   ├── part5_model_comparison.csv        # Out-of-Time GFC test metrics across 6 models
│   ├── part5_cv_fold_metrics.csv         # 4-Fold Purged Expanding-Window CV metrics
│   ├── part5_feature_importance.csv      # LightGBM gain share & Logistic standardized beta
│   └── part5_threshold_optimization.csv  # Asymmetric cost-ratio sensitivity (10:1 to 250:1)
├── src/                                  # Modular, tested production Python source code
│   ├── configs/                          # Global paths, constants, and seeds
│   │   └── config.py
│   ├── data/                             # Data acquisition & cleaning modules (Parts 1 & 2)
│   │   ├── acquire.py                    # FDIC API client with resumable backoff
│   │   ├── benchmark_storage.py          # 5-layout storage benchmarking suite
│   │   ├── quality_audit.py              # Automated data quality auditing engine
│   │   └── build_panel.py                # Silver point-in-time longitudinal panel builder
│   ├── processing/                       # Engine performance & query planning (Part 3)
│   │   ├── pandas_baseline.py            # Vectorized Pandas processing baseline
│   │   ├── polars_pipeline.py            # Polars LazyFrame query-optimized pipeline
│   │   ├── pyspark_pipeline.py           # PySpark Catalyst & AQE distributed pipeline
│   │   └── benchmark_engines.py          # Latency, Amdahl scaling & stress runner
│   ├── features/                         # Feature engineering & preprocessing (Part 4)
│   │   ├── build_features.py             # 40-feature CAMELS causal feature generator
│   │   └── preprocessing_pipeline.py     # Leak-free sklearn Pipeline & purged CV splitter
│   ├── models/                           # Model training, CV & evaluation (Part 5)
│   │   ├── train_evaluate.py             # Model training, OOF tuning & cost optimizer
│   │   └── spark_mllib_demo.py           # Distributed PySpark MLlib GBT & Logistic demo
│   ├── visualization/                    # Publication-grade figure generator (Part 6)
│   │   └── generate_figures.py           # 6 Publication-grade 300 DPI figures generator
│   └── audit/                            # Deterministic invariant verification suite
│       └── verify_end_to_end.py          # 47-check automated end-to-end audit gate
├── requirements.txt                      # Pinned exact package versions for reproducibility
└── README.md                             # This repository guide
```

---

## 3. Hardware & Pinned Software Environment

This pipeline is 100% reproducible on a single multi-core desktop or developer laptop without requiring paid cloud infrastructure.

### Pinned Dependencies (`requirements.txt`)
```text
# Core Data Manipulation & Big Data
pandas==2.3.3
numpy==2.4.2
polars==2.1.0
pyarrow==23.0.1
fastparquet==2026.5.0

# HTTP & API Acquisition
requests==2.33.1

# Machine Learning & Scientific Computing
scipy==1.17.1
scikit-learn==1.8.0
lightgbm==4.7.0

# System & Memory Profiling
psutil==7.2.2

# Visualization & Reporting
matplotlib==3.10.8
seaborn==0.13.2
openpyxl==3.1.5
python-docx==1.2.0

# Distributed Computing
pyspark==4.3.0
```

### Installation
```bash
# Clone the repository
git clone https://github.com/ntnhana1nhk55-glitch/bft65b-fdic-early-warning-.git
cd bft65b-fdic-early-warning-

# Create and activate a clean virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install exact pinned requirements
pip install -r requirements.txt
```

---

## 4. Step-by-Step End-to-End Reproduction Workflow

To execute the entire pipeline from scratch, run the following sequential commands from the project root using `python -X utf8`:

```bash
# ==============================================================================
# PART 1: DATA ACQUISITION & STORAGE ARCHITECTURE BENCHMARK
# ==============================================================================
# Ingest all 170 quarters (1984Q1–2026Q2) from FDIC BankFind API with retry/backoff
python -X utf8 src/data/acquire.py

# Benchmark 5 physical storage layouts (CSV vs CSV.gz vs Parquet Snappy/ZSTD)
python -X utf8 src/data/benchmark_storage.py

# ==============================================================================
# PART 2: AUTOMATED DATA QUALITY AUDIT & SILVER PANEL CONSTRUCTION
# ==============================================================================
# Run automated schema validation, YTD sawtooth audit, and anomaly catalog
python -X utf8 src/data/quality_audit.py

# Construct clean Silver longitudinal panel (1,681,431 rows x 46 cols)
python -X utf8 src/data/build_panel.py

# ==============================================================================
# PART 3: SCALABLE PROCESSING, QUERY PLANS & DISTRIBUTED BENCHMARKS
# ==============================================================================
# Execute 4-engine benchmark (Pandas vs Polars Eager/Lazy vs PySpark), Amdahl scaling & stress
python -X utf8 src/processing/benchmark_engines.py

# ==============================================================================
# PART 4: LEAK-FREE CAMELS FEATURE PIPELINE & SPLITTING
# ==============================================================================
# Generate 40-feature CAMELS Gold panel (1,663,902 rows x 57 cols)
python -X utf8 src/features/build_features.py

# ==============================================================================
# PART 5: MODELLING, PURGED EXPANDING CV & COST-DRIVEN EVALUATION
# ==============================================================================
# Train baseline, econometric, and tree models; optimize threshold tau*; evaluate GFC
python -X utf8 src/models/train_evaluate.py

# Execute distributed PySpark MLlib GBT & Logistic Regression demo
python -X utf8 src/models/spark_mllib_demo.py

# ==============================================================================
# PART 6: VISUALIZATION & FIGURE GENERATION
# ==============================================================================
# Generate all 6 publication-grade figures (300 DPI PNGs in reports/figures/)
python -X utf8 src/visualization/generate_figures.py

# ==============================================================================
# END-TO-END VERIFICATION & ANTI-LEAKAGE AUDIT SUITE
# ==============================================================================
# Execute zero-trust 47-invariant verification suite
python -X utf8 src/audit/verify_end_to_end.py
```

---

## 5. Summary of Anti-Leakage Architecture (`47/47 PASS`)

Per Section 4 of the BFT65B examination rubric, **any use of future information caps Parts 4 and 5 at 30% of available points**. Our pipeline implements a zero-trust anti-leakage architecture verified by 47 automated tests:

```text
==============================================================================
  BFT65B AUTOMATED VERIFICATION SUITE RESULTS (reports/final_audit_certificate.json)
==============================================================================
  [PART 1] Acquisition & Storage:   10 / 10 Checks PASS
  [PART 2] Panel Quality & Dates:   10 / 10 Checks PASS
  [PART 3] Engine Parity & Scaling:  8 /  8 Checks PASS
  [PART 4] CAMELS Features & Gate:  10 / 10 Checks PASS
  [PART 5] Modelling & Costs:        9 /  9 Checks PASS
==============================================================================
  FINAL AUDIT VERDICT: PASS (ZERO DATA LEAKAGE VERIFIED)
==============================================================================
```

### Seven Anti-Leakage Invariants Enforced in Code
1. **Forward Half-Open Interval:** Target is strictly $(\text{repdte\_date}, \text{repdte\_date} + 4\text{Q}]$. Same-day closures (`fail_date == repdte_date`) are excluded from $t$.
2. **Right-Censoring Quarantine:** Quarters 2025Q3–2026Q2 (17,529 rows) are masked as `target_fail_4q = null` and excluded from model matrices.
3. **Chronological Splitting & Embargo:** 2007Q1–2007Q4 (34,754 rows) is quarantined as an embargo buffer between DEV (1984–2006) and TEST (2008–2014 GFC).
4. **Purged Expanding-Window CV:** All 4 cross-validation folds enforce a 1-year (4-quarter) purge buffer between training endpoints and validation starts.
5. **Causal Feature Functions:** All YTD flow de-cumulations (`shift(1)`), 4Q lags (`shift(4)`), and rolling metrics are strictly backward-looking ($\text{lag} \ge 0$) and guarded by contiguous index checks (`q_idx - q_idx.shift(1) == 1`).
6. **Contemporaneous State Peer Normalization:** Peer benchmarks are computed strictly within contemporaneous `(REPDTE, STNAME)` cells.
7. **Empirical Silver $\to$ Gold $\to$ Preprocessor Perturbation Invariance:** Corrupting all post-2006 Silver data by $\times 10^6 + 999,999$ across the entire panel produces $\max|\Delta| = 1.39 \times 10^{-17}$ on the preprocessed DEV matrix (verified in `reports/final_audit_certificate.json`).

---

## 6. Academic References & AI Transparency

### Academic Literature
- Cassidy, Gerard (1989). *The Texas Ratio: Predicting Bank Failure in Distressed Real Estate Markets*, Commercial Banking Research, Austin, TX.
- Cole, Rebel A., and Jeffery W. Gunther (1995). *Separating the Likelihood and Timing of Bank Failure*, Journal of Banking & Finance, 19(6), 1073–1089.
- Lopez, Jose A. (1999). *Using CAMELS Ratings to Monitor Bank Conditions*, FRBSF Economic Letter, Federal Reserve Bank of San Francisco.
- Wheelock, David C., and Paul W. Wilson (2000). *Why Do Banks Disappear? The Determinants of U.S. Bank Failures and Acquisitions*, Review of Economics and Statistics, 82(1), 127–138.
- Federal Deposit Insurance Corporation (2017). *Crisis and Response: An FDIC History, 2008–2013*, FDIC Division of Resolutions and Receiverships, Washington, D.C.
- State Bank of Vietnam (2021). *Circular No. 11/2021/TT-NHNN on Asset Classification and Provisioning for Credit Institutions*, Hanoi, Vietnam.
- State Bank of Vietnam (2016). *Circular No. 41/2016/TT-NHNN Regulating Capital Adequacy Ratio for Commercial Banks (Basel II Standardized)*, Hanoi, Vietnam.
- National Assembly of Vietnam (2024). *Law on Credit Institutions No. 32/2024/QH15*, Hanoi, Vietnam.
- Course Reference Starter Script: `fdic_download_starter.py` provided by Faculty of Banking and Finance, National Economics University (NEU), Hanoi, Vietnam.

### AI Assistance Disclosure
In accordance with Section 4 of the BFT65B Examination Specification, Generative AI (Claude / Gemini via agentic workflow) was utilized for code scaffolding, automated test assertion design (`src/audit/verify_end_to_end.py`), and documentation styling. All econometric formulas, causal target definitions, CAMELS accounting rules, and analytical conclusions were conceptualized, independently audited, and verified by the author. Every number in this repository matches verified on-disk artifacts.

---
**Course:** BFT65B Big Data Analytics in Finance | **National Economics University (NEU)** | November 2026
