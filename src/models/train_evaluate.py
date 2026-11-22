"""
Part 5: Modelling, Purged Expanding-Window Time-Series Cross-Validation,
Cost-Driven Threshold Optimization, and Single-Shot Test Evaluation.

Course: BFT65B - Big Data Analytics in Finance
Subagent 3: Financial Quant & Machine Learning Engineer

Core Architecture:
1. Strict Anti-Leakage Chronological Partitioning:
   - DEV (1984Q1–2006Q4): 1,193,702 rows, 10,225 failure events.
   - EMBARGO (2007Q1–2007Q4): 34,754 rows (purged 1-year gap).
   - TEST (2008Q1–2014Q4): 212,645 rows, 2,052 failure events.
   - POST-TEST (2015Q1–2025Q2): 222,801 rows, 129 failure events.
2. 4-Fold Purged Expanding-Window Time-Series Cross-Validation strictly inside DEV:
   - Fold 1: Train <= 1986, Purge 1987 (4Q), Val 1988–1989 (S&L Peak).
   - Fold 2: Train <= 1989, Purge 1990 (4Q), Val 1991–1992 (FDICIA/Basel I).
   - Fold 3: Train <= 1992, Purge 1993 (4Q), Val 1994–1999 (Consolidation).
   - Fold 4: Train <= 1999, Purge 2000 (4Q), Val 2001–2006 (Pre-GFC Basel).
3. Models Evaluated on Natural Empirical Class Priors (~1% base rate):
   - Supervisory_Rule_Baseline (Texas Ratio Logistic Regression)
   - Logistic_Regression_CAMELS (C=0.1)
   - HistGradientBoosting_CAMELS (max_iter=200, lr=0.05)
   - LightGBM_CAMELS (Champion, n_estimators=300, lr=0.03, alpha=1.0, lambda=5.0)
   - LightGBM_Default (Tuning baseline on CV)
4. Comprehensive Metrics:
   - PR-AUC (Primary), ROC-AUC, Brier score, Log loss
   - Supervisory Inspection Capacity (Top 1%, 2%, 5%, 10%): Recall@k, Dollar-Recall@k, Precision@k
5. Cost-Driven Threshold Optimization (tau*):
   - Empirical FDIC DIF resolution loss C_FN ($2,532k/quarter) vs C_FP ($100k/inspection)
   - Theoretical Bayes threshold vs Empirical grid-search tau* on DEV OOF
   - Frozen tau* evaluated ONCE on TEST: Confusion Matrix, Dollar-Recall, Net Cost Savings ($M).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.configs.config import (
    DATA_FEATURES,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    KEY_COLUMNS,
    NUMERIC_FEATURES,
)
from src.features.preprocessing_pipeline import (
    build_preprocessing_pipeline,
)


# ---------------------------------------------------------------------------
# 1. Evaluation Metrics Helper
# ---------------------------------------------------------------------------
def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_series: np.ndarray | None = None,
    k_percentages: Tuple[int, ...] = (1, 2, 5, 10),
) -> Dict[str, float]:
    """
    Compute comprehensive classification and supervisory inspection capacity metrics:
    - PR-AUC, ROC-AUC, Brier score, Log loss
    - Top-k% Recall, Top-k% Dollar Recall (FDIC DIF loss), Top-k% Precision
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    n_samples = len(y_true)
    n_positives = int(y_true.sum())

    # Standard Probability Metrics
    pr_auc = float(average_precision_score(y_true, y_prob)) if n_positives > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_prob)) if (0 < n_positives < n_samples) else 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob_clipped))

    metrics: Dict[str, float] = {
        "samples": float(n_samples),
        "positives": float(n_positives),
        "prevalence_pct": round(float(n_positives * 100.0 / n_samples), 4) if n_samples > 0 else 0.0,
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "brier_score": round(brier, 6),
        "log_loss": round(ll, 6),
    }

    # Supervisory Capacity Metrics (Top-k%)
    # Sort indices by descending predicted probability
    desc_order = np.argsort(-y_prob)
    total_dollar_loss = float(cost_series.sum()) if cost_series is not None else 0.0

    for k in k_percentages:
        k_count = max(1, int(np.ceil(n_samples * (k / 100.0))))
        top_k_indices = desc_order[:k_count]
        top_k_y = y_true[top_k_indices]

        captured_fails = int(top_k_y.sum())
        recall_at_k = float(captured_fails / n_positives) if n_positives > 0 else 0.0
        precision_at_k = float(captured_fails / k_count)

        metrics[f"recall_at_{k}pct"] = round(recall_at_k, 6)
        metrics[f"precision_at_{k}pct"] = round(precision_at_k, 6)

        if cost_series is not None and total_dollar_loss > 0:
            top_k_cost = float(cost_series[top_k_indices].sum())
            dollar_recall_at_k = float(top_k_cost / total_dollar_loss)
            metrics[f"dollar_recall_at_{k}pct"] = round(dollar_recall_at_k, 6)
        else:
            metrics[f"dollar_recall_at_{k}pct"] = round(recall_at_k, 6)

    return metrics


# ---------------------------------------------------------------------------
# 2. Model Factory
# ---------------------------------------------------------------------------
def get_model_pipeline(model_name: str) -> Pipeline:
    """
    Instantiate model pipeline with leak-free ColumnTransformer preprocessor:
    All models fit on natural class priors (no class_weight="balanced").
    """
    if model_name == "Supervisory_Rule_Baseline":
        # Single-Variable Gerard Cassidy Texas Ratio Logistic Regression
        return Pipeline([
            ("preprocessor", build_preprocessing_pipeline(
                numeric_features=["texas_ratio_proxy"],
                categorical_features=[]
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=300,
                random_state=RANDOM_SEED,
                solver="lbfgs"
            )),
        ])

    elif model_name == "Logistic_Regression_CAMELS":
        # Full 40 features regularized linear baseline
        return Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("clf", LogisticRegression(
                C=0.1,
                max_iter=500,
                random_state=RANDOM_SEED,
                solver="lbfgs"
            )),
        ])

    elif model_name == "HistGradientBoosting_CAMELS":
        # Scikit-learn native GBDT on full 40 features
        return Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=100,
                l2_regularization=1.0,
                random_state=RANDOM_SEED
            )),
        ])

    elif model_name == "LightGBM_CAMELS":
        # Champion regularized gradient boosted trees on full 40 features
        return Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("clf", lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )),
        ])

    elif model_name == "LightGBM_Default":
        # Tuning comparison baseline (default learning rate & no regularization)
        return Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("clf", lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                min_child_samples=20,
                reg_alpha=0.0,
                reg_lambda=0.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            )),
        ])

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# ---------------------------------------------------------------------------
# 3. Main Part 5 Execution Workflow
# ---------------------------------------------------------------------------
def run_part5_modelling_and_evaluation() -> Dict[str, Any]:
    print("=" * 75)
    print("  PART 5: MODELLING, PURGED EXPANDING-WINDOW CV & COST-DRIVEN THRESHOLD  ")
    print("=" * 75)
    t0_global = time.time()

    # ------------------------------------------------------------------------
    # Step A: Load Gold Feature Table and Partitions
    # ------------------------------------------------------------------------
    gold_path = DATA_FEATURES / "gold_feature_table.parquet"
    print(f"\n[Step A] Loading Gold Feature Table from: {gold_path}...")
    df_gold = pl.read_parquet(str(gold_path))
    total_rows = len(df_gold)
    print(f"Loaded Gold Table: {total_rows:,} rows × {len(df_gold.columns)} columns.")

    # Split into chronological regimes
    df_dev = df_gold.filter(pl.col("split_regime") == "dev")
    df_embargo = df_gold.filter(pl.col("split_regime") == "embargo")
    df_test = df_gold.filter(pl.col("split_regime") == "test")
    df_post_test = df_gold.filter(pl.col("split_regime") == "post_test")

    print(f"  DEV fold       : {len(df_dev):>9,} rows | {df_dev['target_fail_4q'].sum():>5} fails ({df_dev['target_fail_4q'].sum()*100.0/len(df_dev):.3f}%) [1984Q1–2006Q4]")
    print(f"  EMBARGO fold   : {len(df_embargo):>9,} rows | {df_embargo['target_fail_4q'].sum():>5} fails ({df_embargo['target_fail_4q'].sum()*100.0/len(df_embargo):.3f}%) [2007Q1–2007Q4] (PURGED)")
    print(f"  TEST fold      : {len(df_test):>9,} rows | {df_test['target_fail_4q'].sum():>5} fails ({df_test['target_fail_4q'].sum()*100.0/len(df_test):.3f}%) [2008Q1–2014Q4] (GFC Single-Shot)")
    print(f"  POST-TEST fold : {len(df_post_test):>9,} rows | {df_post_test['target_fail_4q'].sum():>5} fails ({df_post_test['target_fail_4q'].sum()*100.0/len(df_post_test):.3f}%) [2015Q1–2025Q2]")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # ------------------------------------------------------------------------
    # Step B: 4-Fold Purged Expanding-Window CV Inside DEV
    # ------------------------------------------------------------------------
    print("\n[Step B] Running 4-Fold Purged Expanding-Window Cross-Validation on DEV...")
    cv_fold_definitions = [
        {
            "fold": 1,
            "name": "Fold 1 (S&L Peak)",
            "train_cond": pl.col("year") <= 1986,
            "purge_cond": pl.col("year") == 1987,
            "val_cond": (pl.col("year") >= 1988) & (pl.col("year") <= 1989),
            "val_regime": "S&L Peak Horizon",
        },
        {
            "fold": 2,
            "name": "Fold 2 (FDICIA/Basel I Transition)",
            "train_cond": pl.col("year") <= 1989,
            "purge_cond": pl.col("year") == 1990,
            "val_cond": (pl.col("year") >= 1991) & (pl.col("year") <= 1992),
            "val_regime": "Post-Basel I Regulatory Horizon",
        },
        {
            "fold": 3,
            "name": "Fold 3 (1990s Consolidation)",
            "train_cond": pl.col("year") <= 1992,
            "purge_cond": pl.col("year") == 1993,
            "val_cond": (pl.col("year") >= 1994) & (pl.col("year") <= 1999),
            "val_regime": "Great Moderation Era",
        },
        {
            "fold": 4,
            "name": "Fold 4 (Pre-GFC Modern Basel)",
            "train_cond": pl.col("year") <= 1999,
            "purge_cond": pl.col("year") == 2000,
            "val_cond": (pl.col("year") >= 2001) & (pl.col("year") <= 2006),
            "val_regime": "Pre-Crisis Credit Expansion",
        },
    ]

    models_to_cv = [
        "Supervisory_Rule_Baseline",
        "Logistic_Regression_CAMELS",
        "HistGradientBoosting_CAMELS",
        "LightGBM_CAMELS",
        "LightGBM_Default",  # Tuning check
    ]

    cv_fold_records = []
    # Dictionary to store out-of-fold validation predictions across folds
    oof_predictions: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in models_to_cv}

    for fold_def in cv_fold_definitions:
        f_idx = fold_def["fold"]
        f_name = fold_def["name"]

        train_sub = df_dev.filter(fold_def["train_cond"])
        purge_sub = df_dev.filter(fold_def["purge_cond"])
        val_sub = df_dev.filter(fold_def["val_cond"])

        n_tr, fails_tr = len(train_sub), int(train_sub["target_fail_4q"].sum())
        n_pg = len(purge_sub)
        n_vl, fails_vl = len(val_sub), int(val_sub["target_fail_4q"].sum())

        print(f"\n--- {f_name} ---")
        print(f"  Train: {n_tr:>7,} rows ({fails_tr:>5} fails, {fails_tr*100.0/n_tr:.3f}%) | "
              f"Purge Gap: {n_pg:>6,} rows (4Q) | "
              f"Val: {n_vl:>7,} rows ({fails_vl:>5} fails, {fails_vl*100.0/n_vl:.3f}%)")

        X_train_df = train_sub.select(feature_cols).to_pandas()
        y_train = train_sub["target_fail_4q"].to_numpy()

        X_val_df = val_sub.select(feature_cols).to_pandas()
        y_val = val_sub["target_fail_4q"].to_numpy()
        val_costs = val_sub["_meta_fail_cost"].to_numpy()

        for model_name in models_to_cv:
            t_m0 = time.time()
            pipe = get_model_pipeline(model_name)
            # Fit strictly on train_sub
            pipe.fit(X_train_df, y_train)
            fit_time = time.time() - t_m0

            # Predict probabilities on val_sub
            y_prob_val = pipe.predict_proba(X_val_df)[:, 1]
            oof_predictions[model_name][f"fold_{f_idx}"] = y_prob_val

            # Compute metrics
            m_metrics = compute_evaluation_metrics(y_val, y_prob_val, cost_series=val_costs)
            m_metrics.update({
                "fold": f_idx,
                "fold_name": f_name,
                "val_regime": fold_def["val_regime"],
                "model": model_name,
                "fit_time_sec": round(fit_time, 2),
            })
            cv_fold_records.append(m_metrics)

            print(f"    {model_name:<28} | PR-AUC: {m_metrics['pr_auc']:.4f} | "
                  f"ROC-AUC: {m_metrics['roc_auc']:.4f} | Recall@2%: {m_metrics['recall_at_2pct']*100:.1f}% | "
                  f"$Recall@2%: {m_metrics['dollar_recall_at_2pct']*100:.1f}% | Fit: {fit_time:.1f}s")

    # Save CV Fold Metrics
    cv_metrics_df = pd.DataFrame(cv_fold_records)
    cv_metrics_path = REPORTS_DIR / "part5_cv_fold_metrics.csv"
    cv_metrics_df.to_csv(cv_metrics_path, index=False, encoding="utf-8")
    cv_metrics_json_path = REPORTS_DIR / "part5_cv_fold_metrics.json"
    cv_metrics_json_path.write_text(json.dumps(cv_fold_records, indent=2), encoding="utf-8")
    print(f"\nSaved Purged CV Fold Metrics to: {cv_metrics_path} and {cv_metrics_json_path}")

    # Compute Pooled Out-of-Fold Metrics on DEV
    print("\n--- Summary of Pooled Out-of-Fold Validation Performance on DEV ---")
    pooled_oof_records = []

    # Get validation indices/labels across folds
    y_val_all = []
    cost_val_all = []
    y_val_post1990 = []
    cost_val_post1990 = []

    for fold_def in cv_fold_definitions:
        val_sub = df_dev.filter(fold_def["val_cond"])
        y_val_part = val_sub["target_fail_4q"].to_numpy()
        cost_val_part = val_sub["_meta_fail_cost"].to_numpy()

        y_val_all.append(y_val_part)
        cost_val_all.append(cost_val_part)
        if fold_def["fold"] >= 2:
            y_val_post1990.append(y_val_part)
            cost_val_post1990.append(cost_val_part)

    y_val_all_concat = np.concatenate(y_val_all)
    cost_val_all_concat = np.concatenate(cost_val_all)
    y_val_post1990_concat = np.concatenate(y_val_post1990)
    cost_val_post1990_concat = np.concatenate(cost_val_post1990)

    for model_name in models_to_cv:
        p_all = np.concatenate([oof_predictions[model_name][f"fold_{i}"] for i in range(1, 5)])
        p_post1990 = np.concatenate([oof_predictions[model_name][f"fold_{i}"] for i in range(2, 5)])

        metrics_all = compute_evaluation_metrics(y_val_all_concat, p_all, cost_val_all_concat)
        metrics_post1990 = compute_evaluation_metrics(y_val_post1990_concat, p_post1990, cost_val_post1990_concat)

        pooled_oof_records.append({
            "model": model_name,
            "scope": "All_4_Folds_1988_2006",
            "samples": metrics_all["samples"],
            "fails": metrics_all["positives"],
            "pr_auc": metrics_all["pr_auc"],
            "roc_auc": metrics_all["roc_auc"],
            "brier_score": metrics_all["brier_score"],
            "recall_at_1pct": metrics_all["recall_at_1pct"],
            "recall_at_2pct": metrics_all["recall_at_2pct"],
            "recall_at_5pct": metrics_all["recall_at_5pct"],
            "dollar_recall_at_2pct": metrics_all["dollar_recall_at_2pct"],
            "precision_at_2pct": metrics_all["precision_at_2pct"],
        })

        pooled_oof_records.append({
            "model": model_name,
            "scope": "Modern_Basel_Folds2_4_1991_2006",
            "samples": metrics_post1990["samples"],
            "fails": metrics_post1990["positives"],
            "pr_auc": metrics_post1990["pr_auc"],
            "roc_auc": metrics_post1990["roc_auc"],
            "brier_score": metrics_post1990["brier_score"],
            "recall_at_1pct": metrics_post1990["recall_at_1pct"],
            "recall_at_2pct": metrics_post1990["recall_at_2pct"],
            "recall_at_5pct": metrics_post1990["recall_at_5pct"],
            "dollar_recall_at_2pct": metrics_post1990["dollar_recall_at_2pct"],
            "precision_at_2pct": metrics_post1990["precision_at_2pct"],
        })

        print(f"  {model_name:<28} | Folds 1-4 PR-AUC: {metrics_all['pr_auc']:.4f} | ROC-AUC: {metrics_all['roc_auc']:.4f} | $Recall@2%: {metrics_all['dollar_recall_at_2pct']*100:.1f}%")
        print(f"  {'':<28} | Folds 2-4 PR-AUC: {metrics_post1990['pr_auc']:.4f} | ROC-AUC: {metrics_post1990['roc_auc']:.4f} | $Recall@2%: {metrics_post1990['dollar_recall_at_2pct']*100:.1f}%")

    # Hyperparameter Verification: Compare LightGBM_CAMELS vs LightGBM_Default
    lgb_tuned_prauc = [r for r in pooled_oof_records if r["model"] == "LightGBM_CAMELS" and r["scope"] == "All_4_Folds_1988_2006"][0]["pr_auc"]
    lgb_default_prauc = [r for r in pooled_oof_records if r["model"] == "LightGBM_Default" and r["scope"] == "All_4_Folds_1988_2006"][0]["pr_auc"]
    print(f"\n>> Hyperparameter Tuning Validation (DEV CV): LightGBM_CAMELS PR-AUC = {lgb_tuned_prauc:.4f} vs Default = {lgb_default_prauc:.4f} (Delta = +{lgb_tuned_prauc - lgb_default_prauc:.4f})")

    # ------------------------------------------------------------------------
    # Step C: Single-Shot Full Training on DEV and Evaluation on TEST & POST-TEST
    # ------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  [Step C] SINGLE-SHOT TEST EVALUATION (Fitted on Full DEV: 1,193,702 rows)")
    print("=" * 75)

    main_models = [
        "Supervisory_Rule_Baseline",
        "Logistic_Regression_CAMELS",
        "HistGradientBoosting_CAMELS",
        "LightGBM_CAMELS",
    ]

    X_dev_full = df_dev.select(feature_cols).to_pandas()
    y_dev_full = df_dev["target_fail_4q"].to_numpy()
    dev_costs_full = df_dev["_meta_fail_cost"].to_numpy()

    X_test_full = df_test.select(feature_cols).to_pandas()
    y_test_full = df_test["target_fail_4q"].to_numpy()
    test_costs_full = df_test["_meta_fail_cost"].to_numpy()

    X_post_test_full = df_post_test.select(feature_cols).to_pandas()
    y_post_test_full = df_post_test["target_fail_4q"].to_numpy()
    post_test_costs_full = df_post_test["_meta_fail_cost"].to_numpy()

    model_comparison_records = []
    fitted_main_pipelines: Dict[str, Pipeline] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    for model_name in main_models:
        print(f"\nFitting {model_name} on Full DEV...")
        t_fit = time.time()
        pipe = get_model_pipeline(model_name)
        pipe.fit(X_dev_full, y_dev_full)
        fit_duration = time.time() - t_fit
        fitted_main_pipelines[model_name] = pipe
        print(f"  Fitted in {fit_duration:.2f} seconds.")

        # Predict on DEV (In-Sample reference)
        p_dev = pipe.predict_proba(X_dev_full)[:, 1]
        m_dev = compute_evaluation_metrics(y_dev_full, p_dev, dev_costs_full)
        m_dev.update({"model": model_name, "partition": "DEV (1984–2006, In-Sample)", "fit_time_sec": round(fit_duration, 2)})
        model_comparison_records.append(m_dev)

        # Single-Shot Predict on TEST (Out-of-Time Primary)
        p_test = pipe.predict_proba(X_test_full)[:, 1]
        test_predictions[model_name] = p_test
        m_test = compute_evaluation_metrics(y_test_full, p_test, test_costs_full)
        m_test.update({"model": model_name, "partition": "TEST (2008–2014 GFC, Out-of-Time)", "fit_time_sec": round(fit_duration, 2)})
        model_comparison_records.append(m_test)

        # Diagnostic Predict on POST-TEST
        p_post = pipe.predict_proba(X_post_test_full)[:, 1]
        m_post = compute_evaluation_metrics(y_post_test_full, p_post, post_test_costs_full)
        m_post.update({"model": model_name, "partition": "POST_TEST (2015–2025, Diagnostic)", "fit_time_sec": round(fit_duration, 2)})
        model_comparison_records.append(m_post)

        print(f"  TEST PR-AUC: {m_test['pr_auc']:.4f} | ROC-AUC: {m_test['roc_auc']:.4f} | Brier: {m_test['brier_score']:.5f} | "
              f"Recall@2%: {m_test['recall_at_2pct']*100:.1f}% | $Recall@2%: {m_test['dollar_recall_at_2pct']*100:.1f}% | Prec@2%: {m_test['precision_at_2pct']*100:.1f}%")

    model_comp_df = pd.DataFrame(model_comparison_records)
    model_comp_path = REPORTS_DIR / "part5_model_comparison.csv"
    model_comp_df.to_csv(model_comp_path, index=False, encoding="utf-8")
    print(f"\nSaved Model Comparison Table to: {model_comp_path}")

    # ------------------------------------------------------------------------
    # Step D: Feature Importance Extraction
    # ------------------------------------------------------------------------
    print("\n[Step D] Extracting Feature Importances on DEV...")
    # Extract feature names from fitted preprocessor
    fitted_preproc = fitted_main_pipelines["LightGBM_CAMELS"].named_steps["preprocessor"]
    ohe_feature_names = fitted_preproc.get_feature_names_out()

    # LightGBM Importances
    lgb_clf = fitted_main_pipelines["LightGBM_CAMELS"].named_steps["clf"]
    split_importances = lgb_clf.feature_importances_
    gain_importances = lgb_clf.booster_.feature_importance(importance_type="gain")
    total_gain = float(gain_importances.sum())
    gain_shares = gain_importances / total_gain if total_gain > 0 else np.zeros_like(gain_importances)

    # Logistic Regression Standardized Coefficients
    lr_clf = fitted_main_pipelines["Logistic_Regression_CAMELS"].named_steps["clf"]
    lr_coefs = lr_clf.coef_[0]

    fi_df = pd.DataFrame({
        "feature": ohe_feature_names,
        "lgb_split_importance": split_importances,
        "lgb_gain_importance": np.round(gain_importances, 2),
        "lgb_gain_share_pct": np.round(gain_shares * 100.0, 4),
        "lr_standardized_coef": np.round(lr_coefs, 6),
    }).sort_values("lgb_gain_importance", ascending=False)

    fi_path = REPORTS_DIR / "part5_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False, encoding="utf-8")
    print(f"Saved Feature Importance Table to: {fi_path}")

    print("\n--- Top 10 Most Important Features in Champion LightGBM (by Gain Share) ---")
    for idx, row in fi_df.head(10).iterrows():
        print(f"  {row['feature']:<30} | Gain Share: {row['lgb_gain_share_pct']:>6.2f}% | "
              f"Splits: {int(row['lgb_split_importance']):>4} | LR Coef: {row['lr_standardized_coef']:>9.4f}")

    # ------------------------------------------------------------------------
    # Step E: Cost-Driven Threshold Optimization (tau*)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  [Step E] COST-DRIVEN THRESHOLD OPTIMIZATION (tau*) STRICTLY ON DEV")
    print("=" * 75)

    # 1. Estimate Empirical FDIC DIF Resolution Loss C_FN on DEV
    dev_fail_costs = df_dev.filter((pl.col("target_fail_4q") == 1) & (pl.col("_meta_fail_cost") > 0))["_meta_fail_cost"]
    annual_median_dif_loss = float(dev_fail_costs.median())
    annual_mean_dif_loss = float(dev_fail_costs.mean())

    quarterly_median_dif_loss = annual_median_dif_loss / 4.0
    quarterly_mean_dif_loss = annual_mean_dif_loss / 4.0

    c_fp_benchmark = 100.0  # $100k per supervisory examination false alarm
    c_fn_primary = quarterly_median_dif_loss  # ~$2,532k to ~$2,600k per failure quarter
    c_fn_annual_median = annual_median_dif_loss  # ~$10,127k to ~$10,399k

    print(f"Empirical FDIC DIF Resolution Loss in DEV (1984–2006):")
    print(f"  Median Annual Bank Failure Loss : ${annual_median_dif_loss:,.2f}k (${annual_median_dif_loss/1000:.2f} Million)")
    print(f"  Mean Annual Bank Failure Loss   : ${annual_mean_dif_loss:,.2f}k (${annual_mean_dif_loss/1000:.2f} Million)")
    print(f"  Primary C_FN (Quarterly Median) : ${c_fn_primary:,.2f}k")
    print(f"  Supervisory C_FP (Audit Cost)   : ${c_fp_benchmark:,.2f}k")
    print(f"  Primary Cost Ratio (C_FN / C_FP): {c_fn_primary / c_fp_benchmark:.2f} : 1")

    # 2. Grid-Search tau* strictly on DEV Out-of-Fold Validation Predictions
    # Using modern Basel folds 2-4 pooled OOF predictions for Champion LightGBM
    p_dev_oof = np.concatenate([oof_predictions["LightGBM_CAMELS"][f"fold_{i}"] for i in range(2, 5)])
    y_dev_oof = y_val_post1990_concat

    tau_grid = np.linspace(0.002, 0.950, 200)

    # Pre-compute Confusion Matrix counts across tau grid
    # Vectorized computation for speed
    n_dev_oof = len(y_dev_oof)
    n_dev_pos = int(y_dev_oof.sum())
    n_dev_neg = n_dev_oof - n_dev_pos

    grid_cost_records = []
    best_cost = float("inf")
    best_tau = 0.5

    for tau in tau_grid:
        tau_r = round(float(tau), 4)
        y_pred = (p_dev_oof >= tau_r).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_dev_oof == 1)))
        fp = int(np.sum((y_pred == 1) & (y_dev_oof == 0)))
        fn = n_dev_pos - tp
        tn = n_dev_neg - fp

        total_cost = fp * c_fp_benchmark + fn * c_fn_primary
        if total_cost < best_cost:
            best_cost = total_cost
            best_tau = tau_r

        grid_cost_records.append({
            "tau": tau_r,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
            "total_cost_k": round(total_cost, 2),
        })

    # Theoretical Bayes Threshold
    tau_bayes_primary = c_fp_benchmark / (c_fp_benchmark + c_fn_primary)
    tau_bayes_annual = c_fp_benchmark / (c_fp_benchmark + c_fn_annual_median)

    print(f"\n--- Cost Optimization Curve on DEV Out-of-Fold (Folds 2–4: 617,504 banks) ---")
    print(f"  Theoretical Bayes Threshold (tau_Bayes*) : {tau_bayes_primary:.4f}")
    print(f"  Empirical Cost-Minimizing Threshold (tau*): {best_tau:.4f} (Min Cost: ${best_cost:,.2f}k)")

    # 3. Sensitivity Analysis across Cost Ratios C_FN / C_FP in {10, 25, 50, 100, 250}
    cost_ratios = [10.0, 25.0, 50.0, 100.0, 250.0]
    sensitivity_records = []

    for ratio in cost_ratios:
        c_fn_test = c_fp_benchmark * ratio
        tau_bayes_ratio = c_fp_benchmark / (c_fp_benchmark + c_fn_test)

        # Find empirical tau* for this ratio on DEV OOF
        opt_cost = float("inf")
        opt_tau = 0.5
        for row in grid_cost_records:
            c = row["fp"] * c_fp_benchmark + row["fn"] * c_fn_test
            if c < opt_cost:
                opt_cost = c
                opt_tau = row["tau"]

        # Evaluate on TEST using this opt_tau
        p_test_champ = test_predictions["LightGBM_CAMELS"]
        y_test_pred = (p_test_champ >= opt_tau).astype(int)
        cm = confusion_matrix(y_test_full, y_test_pred)
        tn_t, fp_t, fn_t, tp_t = cm.ravel()

        p_t = precision_score(y_test_full, y_test_pred, zero_division=0)
        r_t = recall_score(y_test_full, y_test_pred, zero_division=0)
        f1_t = f1_score(y_test_full, y_test_pred, zero_division=0)
        f2_t = fbeta_score(y_test_full, y_test_pred, beta=2.0, zero_division=0)

        # Dollar recall on TEST
        caught_cost = float(test_costs_full[(y_test_pred == 1) & (y_test_full == 1)].sum())
        total_test_cost = float(test_costs_full[y_test_full == 1].sum())
        dollar_recall_t = caught_cost / total_test_cost if total_test_cost > 0 else 0.0

        # Cost savings on TEST
        cost_no_ews = len(test_costs_full[y_test_full == 1]) * c_fn_test
        cost_with_ews = fp_t * c_fp_benchmark + fn_t * c_fn_test
        net_savings_m = (cost_no_ews - cost_with_ews) / 1000.0

        sensitivity_records.append({
            "cost_ratio": ratio,
            "c_fp_k": c_fp_benchmark,
            "c_fn_k": c_fn_test,
            "tau_bayes": round(tau_bayes_ratio, 4),
            "opt_tau_dev": round(opt_tau, 4),
            "test_tp": int(tp_t),
            "test_fp": int(fp_t),
            "test_fn": int(fn_t),
            "test_tn": int(tn_t),
            "test_precision": round(p_t, 4),
            "test_recall": round(r_t, 4),
            "test_dollar_recall": round(dollar_recall_t, 4),
            "test_f1": round(f1_t, 4),
            "test_f2": round(f2_t, 4),
            "test_net_savings_million": round(net_savings_m, 2),
        })

    sens_df = pd.DataFrame(sensitivity_records)
    sens_path = REPORTS_DIR / "part5_threshold_optimization.csv"
    sens_df.to_csv(sens_path, index=False, encoding="utf-8")
    print(f"Saved Threshold Optimization & Sensitivity Table to: {sens_path}")

    print("\n--- Sensitivity Across Asymmetric Cost Ratios (Evaluated on TEST 2008–2014) ---")
    for s in sensitivity_records:
        print(f"  Ratio {int(s['cost_ratio']):>3}:1 | tau*: {s['opt_tau_dev']:.4f} (Bayes: {s['tau_bayes']:.4f}) | "
              f"Recall: {s['test_recall']*100:>5.1f}% | $Recall: {s['test_dollar_recall']*100:>5.1f}% | "
              f"Precision: {s['test_precision']*100:>5.1f}% | Net Savings: ${s['test_net_savings_million']:>8.2f}M")

    # 4. Primary Decision: Evaluate Champion LightGBM at best_tau on TEST (Single-Shot)
    p_test_champ = test_predictions["LightGBM_CAMELS"]
    y_test_pred_champ = (p_test_champ >= best_tau).astype(int)
    cm_champ = confusion_matrix(y_test_full, y_test_pred_champ)
    tn_c, fp_c, fn_c, tp_c = cm_champ.ravel()

    test_prec = float(precision_score(y_test_full, y_test_pred_champ, zero_division=0))
    test_rec = float(recall_score(y_test_full, y_test_pred_champ, zero_division=0))
    test_f1 = float(f1_score(y_test_full, y_test_pred_champ, zero_division=0))
    test_f2 = float(fbeta_score(y_test_full, y_test_pred_champ, beta=2.0, zero_division=0))

    test_caught_cost = float(test_costs_full[(y_test_pred_champ == 1) & (y_test_full == 1)].sum())
    test_total_cost = float(test_costs_full[y_test_full == 1].sum())
    test_dollar_recall = float(test_caught_cost / test_total_cost)

    test_cost_no_ews = int(y_test_full.sum()) * c_fn_primary
    test_cost_with_ews = fp_c * c_fp_benchmark + fn_c * c_fn_primary
    test_net_savings_m = (test_cost_no_ews - test_cost_with_ews) / 1000.0

    # Also compare against Naive Default Threshold (tau = 0.50)
    y_test_pred_naive = (p_test_champ >= 0.50).astype(int)
    tn_n, fp_n, fn_n, tp_n = confusion_matrix(y_test_full, y_test_pred_naive).ravel()
    naive_rec = float(recall_score(y_test_full, y_test_pred_naive, zero_division=0))
    naive_prec = float(precision_score(y_test_full, y_test_pred_naive, zero_division=0))
    naive_caught_cost = float(test_costs_full[(y_test_pred_naive == 1) & (y_test_full == 1)].sum())
    naive_dollar_rec = float(naive_caught_cost / test_total_cost)
    test_cost_naive_ews = fp_n * c_fp_benchmark + fn_n * c_fn_primary
    savings_vs_naive_m = (test_cost_naive_ews - test_cost_with_ews) / 1000.0

    print("\n" + "=" * 75)
    print("  CHAMPION MODEL TEST VERDICT (LightGBM_CAMELS with DEV-learned tau*)  ")
    print("=" * 75)
    print(f"  Learned Threshold tau*       : {best_tau:.4f} (vs Naive tau=0.5000)")
    print(f"  Confusion Matrix (TEST)      : TP={tp_c:,} | FP={fp_c:,} | FN={fn_c:,} | TN={tn_c:,}")
    print(f"  Detection Recall             : {test_rec * 100:.2f}% ({tp_c:,} / {int(y_test_full.sum()):,} GFC failures vs {naive_rec * 100:.2f}% at tau=0.50)")
    print(f"  Economic Dollar-Recall       : {test_dollar_recall * 100:.2f}% (${test_caught_cost / 1e6:.2f}B / ${test_total_cost / 1e6:.2f}B vs {naive_dollar_rec * 100:.2f}% at tau=0.50)")
    print(f"  Precision                    : {test_prec * 100:.2f}% (vs {naive_prec * 100:.2f}% at tau=0.50)")
    print(f"  F1 Score / F2 Score          : {test_f1:.4f} / {test_f2:.4f}")
    print(f"  Expected Net Cost Savings    : ${test_net_savings_m:,.2f} Million (vs No EWS) | +${savings_vs_naive_m:,.2f} Million (vs Naive tau=0.50)")

    # ------------------------------------------------------------------------
    # Step F: Save Comprehensive Summary JSON
    # ------------------------------------------------------------------------
    summary_data = {
        "status": "SUCCESS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_execution_seconds": round(time.time() - t0_global, 2),
        "split_counts": {
            "dev_rows": len(df_dev),
            "dev_fails": int(df_dev["target_fail_4q"].sum()),
            "embargo_rows": len(df_embargo),
            "test_rows": len(df_test),
            "test_fails": int(df_test["target_fail_4q"].sum()),
            "post_test_rows": len(df_post_test),
            "post_test_fails": int(df_post_test["target_fail_4q"].sum()),
        },
        "purged_cv_expanding_window": {
            "description": "4-Fold Purged Expanding-Window Time-Series Cross-Validation inside DEV",
            "purge_gap": "4 Quarters (1 Year) strictly between every Train and Val fold",
            "pooled_oof_performance": pooled_oof_records,
            "hyperparameter_check": {
                "lightgbm_tuned_prauc": lgb_tuned_prauc,
                "lightgbm_default_prauc": lgb_default_prauc,
                "tuning_gain": round(lgb_tuned_prauc - lgb_default_prauc, 4),
            },
        },
        "single_shot_test_evaluation": {
            "champion_model": "LightGBM_CAMELS",
            "all_models_test_metrics": [r for r in model_comparison_records if "TEST" in r["partition"]],
        },
        "cost_driven_threshold_optimization": {
            "supervisory_c_fp_k": c_fp_benchmark,
            "primary_c_fn_quarterly_median_k": round(c_fn_primary, 2),
            "annual_c_fn_median_k": round(annual_median_dif_loss, 2),
            "primary_cost_ratio": round(c_fn_primary / c_fp_benchmark, 2),
            "theoretical_tau_bayes": round(tau_bayes_primary, 4),
            "dev_learned_optimal_tau": round(best_tau, 4),
            "test_verdict_at_optimal_tau": {
                "tp": int(tp_c),
                "fp": int(fp_c),
                "fn": int(fn_c),
                "tn": int(tn_c),
                "precision": round(test_prec, 4),
                "recall": round(test_rec, 4),
                "dollar_recall": round(test_dollar_recall, 4),
                "f1": round(test_f1, 4),
                "f2": round(test_f2, 4),
                "net_savings_million": round(test_net_savings_m, 2),
                "savings_vs_naive_tau050_million": round(savings_vs_naive_m, 2),
                "naive_tau050_recall": round(naive_rec, 4),
                "naive_tau050_dollar_recall": round(naive_dollar_rec, 4),
                "naive_tau050_tp": int(tp_n),
                "naive_tau050_fp": int(fp_n),
                "naive_tau050_fn": int(fn_n),
            },
            "sensitivity_table": sensitivity_records,
        },
    }

    summary_json_path = REPORTS_DIR / "part5_model_comparison.json"
    summary_json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSaved Part 5 Master Summary JSON to: {summary_json_path}")
    print("=" * 75)

    return summary_data


def main():
    run_part5_modelling_and_evaluation()


if __name__ == "__main__":
    main()
