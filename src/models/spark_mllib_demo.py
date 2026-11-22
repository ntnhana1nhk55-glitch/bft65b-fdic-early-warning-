"""
Distributed PySpark MLlib Pipeline Demo for FDIC Bank Failure Early Warning.
Course: BFT65B - Big Data Analytics in Finance
Subagent 3: Financial Quant & Machine Learning Engineer

Key Features:
1. Leak-Free Distributed Preprocessing Pipeline fitted strictly on DEV (1,193,702 rows):
   - StringIndexer + OneHotEncoder on BKCLASS
   - Distributed Median Imputer on all 39 CAMELS numeric features
   - VectorAssembler + StandardScaler
2. Distributed Estimators:
   - Spark MLlib LogisticRegression (L2 regularized)
   - Spark MLlib GBTClassifier (Gradient Boosted Decision Trees)
3. Single-Shot Out-of-Time Evaluation on TEST (212,645 rows, 2008–2014 GFC):
   - PR-AUC (areaUnderPR) and ROC-AUC (areaUnderROC) via BinaryClassificationEvaluator
   - Cross-verification with scikit-learn metrics
4. Export: reports/part5_spark_mllib_results.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure Windows Hadoop Home Fallback
if sys.platform == "win32" and "HADOOP_HOME" not in os.environ:
    hadoop_dummy = PROJECT_ROOT / "data" / "benchmark_tmp" / "hadoop"
    hadoop_dummy.mkdir(parents=True, exist_ok=True)
    (hadoop_dummy / "bin").mkdir(exist_ok=True)
    os.environ["HADOOP_HOME"] = str(hadoop_dummy)

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sklearn.metrics import average_precision_score, roc_auc_score

from src.configs.config import (
    DATA_FEATURES,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def run_spark_mllib_pipeline(
    gold_path: Path | str = DATA_FEATURES / "gold_feature_table.parquet",
    output_json_path: Path | str = REPORTS_DIR / "part5_spark_mllib_results.json",
) -> Dict[str, Any]:
    gold_path = Path(gold_path)
    output_json_path = Path(output_json_path)
    t0_global = time.time()

    print("=" * 75)
    print("      PART 5: DISTRIBUTED PYSPARK MLLIB PIPELINE DEMO               ")
    print("=" * 75)

    # 1. Initialize Local Spark Session
    print("\n[Step 1] Initializing PySpark Session (local[4], 4g driver memory)...")
    spark = (
        SparkSession.builder.appName("FDIC_Bank_Failure_MLlib_Demo")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark Version: {spark.version} | Driver Memory: 4g | Shuffle Partitions: 16")

    # 2. Read Gold Table and Select Feature Columns
    print(f"\n[Step 2] Reading Gold Feature Table from: {gold_path}...")
    selected_cols = ["split_regime", "target_fail_4q", "BKCLASS"] + NUMERIC_FEATURES
    df_spark = spark.read.parquet(str(gold_path)).select(selected_cols)

    # Impute empty string and nulls in BKCLASS to 'MISSING' (prevents empty attribute names in StringIndexer)
    df_spark = df_spark.withColumn(
        "BKCLASS",
        F.when(F.col("BKCLASS").isNull() | (F.trim(F.col("BKCLASS")) == ""), F.lit("MISSING")).otherwise(F.trim(F.col("BKCLASS")))
    )

    # Chronological partition filters
    train_df = df_spark.filter(F.col("split_regime") == "dev").cache()
    test_df = df_spark.filter(F.col("split_regime") == "test").cache()

    n_train = train_df.count()
    fails_train = train_df.filter(F.col("target_fail_4q") == 1).count()
    n_test = test_df.count()
    fails_test = test_df.filter(F.col("target_fail_4q") == 1).count()

    print(f"  Training Split (DEV) : {n_train:,} rows | {fails_train:,} fails ({fails_train*100.0/n_train:.3f}%) [1984Q1–2006Q4]")
    print(f"  Testing Split (TEST) : {n_test:,} rows | {fails_test:,} fails ({fails_test*100.0/n_test:.3f}%) [2008Q1–2014Q4]")

    # 3. Construct Distributed Feature Engineering Stages
    print("\n[Step 3] Constructing PySpark ML Feature Transformation Stages...")
    indexer = StringIndexer(inputCol="BKCLASS", outputCol="BKCLASS_idx", handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=["BKCLASS_idx"], outputCols=["BKCLASS_ohe"])

    # Imputer with median strategy across all 39 CAMELS numeric features
    imputed_num_cols = [f"{c}_imp" for c in NUMERIC_FEATURES]
    imputer = Imputer(strategy="median", inputCols=NUMERIC_FEATURES, outputCols=imputed_num_cols)

    # VectorAssembler combining imputed numeric + one-hot encoded charter class
    assembler_inputs = imputed_num_cols + ["BKCLASS_ohe"]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="keep")

    # StandardScaler with mean centering and unit variance scaling
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)

    fe_stages = [indexer, encoder, imputer, assembler, scaler]

    # Evaluators
    eval_roc = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="target_fail_4q",
        metricName="areaUnderROC"
    )
    eval_pr = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="target_fail_4q",
        metricName="areaUnderPR"
    )

    models_results = {}

    # ------------------------------------------------------------------------
    # Model 1: Spark MLlib LogisticRegression
    # ------------------------------------------------------------------------
    print("\n[Step 4.A] Training Spark MLlib Logistic Regression (maxIter=50, regParam=0.01)...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="target_fail_4q",
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.0,
    )
    lr_pipe = Pipeline(stages=fe_stages + [lr])

    t_lr_fit0 = time.time()
    lr_model = lr_pipe.fit(train_df)
    lr_fit_time = time.time() - t_lr_fit0
    print(f"  Fitted Logistic Regression Pipeline in: {lr_fit_time:.2f} seconds.")

    t_lr_pred0 = time.time()
    lr_preds = lr_model.transform(test_df).cache()
    lr_test_roc = float(eval_roc.evaluate(lr_preds))
    lr_test_pr = float(eval_pr.evaluate(lr_preds))
    lr_pred_time = time.time() - t_lr_pred0

    print(f"  Spark Evaluator -> Test PR-AUC: {lr_test_pr:.4f} | Test ROC-AUC: {lr_test_roc:.4f}")

    # Cross-verify with scikit-learn on collected predictions
    lr_collected = lr_preds.select("target_fail_4q", "probability").toPandas()
    y_true_test = lr_collected["target_fail_4q"].to_numpy().astype(int)
    # probability vector: index 1 is probability of class 1
    lr_probs = np.array([row[1] for row in lr_collected["probability"]])

    sklearn_lr_pr = float(average_precision_score(y_true_test, lr_probs))
    sklearn_lr_roc = float(roc_auc_score(y_true_test, lr_probs))
    print(f"  Sklearn Verify  -> Test PR-AUC: {sklearn_lr_pr:.4f} | Test ROC-AUC: {sklearn_lr_roc:.4f}")

    models_results["Spark_MLlib_LogisticRegression"] = {
        "fit_time_seconds": round(lr_fit_time, 2),
        "prediction_time_seconds": round(lr_pred_time, 2),
        "test_roc_auc_spark": round(lr_test_roc, 6),
        "test_pr_auc_spark": round(lr_test_pr, 6),
        "test_roc_auc_sklearn": round(sklearn_lr_roc, 6),
        "test_pr_auc_sklearn": round(sklearn_lr_pr, 6),
    }

    # ------------------------------------------------------------------------
    # Model 2: Spark MLlib GBTClassifier
    # ------------------------------------------------------------------------
    print("\n[Step 4.B] Training Spark MLlib GBTClassifier (maxIter=25, maxDepth=5)...")
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="target_fail_4q",
        maxIter=25,
        maxDepth=5,
        seed=RANDOM_SEED,
    )
    gbt_pipe = Pipeline(stages=fe_stages + [gbt])

    t_gbt_fit0 = time.time()
    gbt_model = gbt_pipe.fit(train_df)
    gbt_fit_time = time.time() - t_gbt_fit0
    print(f"  Fitted GBTClassifier Pipeline in: {gbt_fit_time:.2f} seconds.")

    t_gbt_pred0 = time.time()
    gbt_preds = gbt_model.transform(test_df).cache()
    gbt_test_roc = float(eval_roc.evaluate(gbt_preds))
    gbt_test_pr = float(eval_pr.evaluate(gbt_preds))
    gbt_pred_time = time.time() - t_gbt_pred0

    print(f"  Spark Evaluator -> Test PR-AUC: {gbt_test_pr:.4f} | Test ROC-AUC: {gbt_test_roc:.4f}")

    # Cross-verify with scikit-learn
    gbt_collected = gbt_preds.select("target_fail_4q", "probability").toPandas()
    gbt_probs = np.array([row[1] for row in gbt_collected["probability"]])

    sklearn_gbt_pr = float(average_precision_score(y_true_test, gbt_probs))
    sklearn_gbt_roc = float(roc_auc_score(y_true_test, gbt_probs))
    print(f"  Sklearn Verify  -> Test PR-AUC: {sklearn_gbt_pr:.4f} | Test ROC-AUC: {sklearn_gbt_roc:.4f}")

    models_results["Spark_MLlib_GBTClassifier"] = {
        "fit_time_seconds": round(gbt_fit_time, 2),
        "prediction_time_seconds": round(gbt_pred_time, 2),
        "test_roc_auc_spark": round(gbt_test_roc, 6),
        "test_pr_auc_spark": round(gbt_test_pr, 6),
        "test_roc_auc_sklearn": round(sklearn_gbt_roc, 6),
        "test_pr_auc_sklearn": round(sklearn_gbt_pr, 6),
    }

    # 5. Clean up Spark and temporary directory
    train_df.unpersist()
    test_df.unpersist()
    lr_preds.unpersist()
    gbt_preds.unpersist()
    spark.stop()
    shutil.rmtree(PROJECT_ROOT / "data" / "benchmark_tmp", ignore_errors=True)
    print("\nPySpark Session closed and temporary files cleaned up successfully.")

    # 6. Save JSON Summary
    output_data = {
        "status": "SUCCESS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_seconds": round(time.time() - t0_global, 2),
        "spark_version": "4.3.0",
        "dataset_summary": {
            "gold_path": str(gold_path),
            "train_dev_rows": n_train,
            "train_dev_fails": fails_train,
            "test_rows": n_test,
            "test_fails": fails_test,
        },
        "pipeline_architecture": [
            "StringIndexer(inputCol='BKCLASS', outputCol='BKCLASS_idx')",
            "OneHotEncoder(inputCols=['BKCLASS_idx'], outputCols=['BKCLASS_ohe'])",
            f"Imputer(strategy='median', inputCols=NUMERIC_FEATURES [39 cols])",
            "VectorAssembler(inputCols=[imputed_39_cols + BKCLASS_ohe], outputCol='raw_features')",
            "StandardScaler(inputCol='raw_features', outputCol='features', withMean=True, withStd=True)",
            "Estimator: Fitted STRICTLY on DEV (1984Q1–2006Q4)",
        ],
        "models": models_results,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"\nSaved Spark MLlib Demo Results to: {output_json_path}")
    print("=" * 75)

    return output_data


def main():
    run_spark_mllib_pipeline()


if __name__ == "__main__":
    main()
