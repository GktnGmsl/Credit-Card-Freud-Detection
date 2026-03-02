"""
Credit Card Fraud Detection — MLflow Experiment Tracking
=========================================================
Logs all model training runs to MLflow, including:
- Parameters (hyperparameters)
- Metrics (AUC, precision, recall, F1)
- Artifacts (confusion matrices, models)
Registers the best model to MLflow Model Registry as Production.

Usage:
    python mlflow_tracking.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow  # type: ignore
import mlflow.sklearn  # type: ignore
import mlflow.xgboost  # type: ignore

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, average_precision_score, f1_score, log_loss
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

EXPERIMENT_NAME = "Credit_Card_Fraud_Detection"
TARGET = "Class"

# Add project root for imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training import (
    load_data,
    compute_roc_pr,
    find_best_threshold,
    plot_confusion_matrices,
    train_logistic_regression_with_loss,
    train_random_forest_with_loss,
    train_xgboost_with_loss,
    run_optuna_optimization,
    train_optimized_model,
)


# ──────────────────────────────────────────────────────────────────────
# Helper: log a single model run
# ──────────────────────────────────────────────────────────────────────

def log_model_run(model, model_name, params, X_val, y_val, X_test, y_test,
                  is_optimized=False):
    """Log a model run to MLflow with parameters, metrics, and artifacts."""

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    report_test = classification_report(y_test, y_test_pred, output_dict=True)
    report_val = classification_report(y_val, y_val_pred, output_dict=True)

    best_threshold, best_f1, _, _ = find_best_threshold(y_test, y_test_proba)

    run_name = f"{model_name}_optimized" if is_optimized else model_name
    tags = {"model_type": model_name, "optimized": str(is_optimized)}

    with mlflow.start_run(run_name=run_name, tags=tags):
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log test metrics
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_proba))
        mlflow.log_metric("test_pr_auc", average_precision_score(y_test, y_test_proba))
        mlflow.log_metric("test_precision_fraud", report_test["1"]["precision"])
        mlflow.log_metric("test_recall_fraud", report_test["1"]["recall"])
        mlflow.log_metric("test_f1_fraud", report_test["1"]["f1-score"])
        mlflow.log_metric("test_log_loss", log_loss(y_test, y_test_proba))
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("test_f1_at_best_threshold", best_f1)

        # Log validation metrics
        mlflow.log_metric("val_accuracy", accuracy_score(y_val, y_val_pred))
        mlflow.log_metric("val_roc_auc", roc_auc_score(y_val, y_val_proba))
        mlflow.log_metric("val_pr_auc", average_precision_score(y_val, y_val_proba))
        mlflow.log_metric("val_precision_fraud", report_val["1"]["precision"])
        mlflow.log_metric("val_recall_fraud", report_val["1"]["recall"])
        mlflow.log_metric("val_f1_fraud", report_val["1"]["f1-score"])

        # Log confusion matrix images if they exist
        cm_file = os.path.join(OUTPUT_DIR,
                               f"cm_{run_name.replace(' ', '_')}.png")
        if os.path.exists(cm_file):
            mlflow.log_artifact(cm_file, "confusion_matrices")

        # Log model
        if "xgb" in model_name.lower() or "xgboost" in model_name.lower():
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"  Logged: {run_name} (run_id: {run_id[:8]}...)")
        print(f"    ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}  "
              f"PR-AUC: {average_precision_score(y_test, y_test_proba):.4f}  "
              f"F1(Fraud): {report_test['1']['f1-score']:.4f}")

        return run_id


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # ── Setup MLflow ──────────────────────────────────────────────────
    tracking_uri = f"file:///{MLRUNS_DIR.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"\nMLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"mlruns directory: {MLRUNS_DIR}")

    # ── Load data ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  LOADING DATA")
    print("="*60)
    X_train, y_train, X_val, y_val, X_test, y_test, features = load_data()

    # ── Train models ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING MODELS")
    print("="*60)

    print("\n[1/3] Training Logistic Regression...")
    lr_model, _ = train_logistic_regression_with_loss(
        X_train, y_train, X_val, y_val, X_test, y_test)

    print("[2/3] Training Random Forest...")
    rf_model, _ = train_random_forest_with_loss(
        X_train, y_train, X_val, y_val, X_test, y_test)

    print("[3/3] Training XGBoost...")
    xgb_model, _ = train_xgboost_with_loss(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # ── Log models to MLflow ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  LOGGING TO MLFLOW")
    print("="*60)

    lr_params = {"solver": "saga", "max_iter": 1000, "C": 1.0, "random_state": 42}
    rf_params = {"n_estimators": 200, "oob_score": True, "random_state": 42,
                 "n_jobs": -1}
    xgb_params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                  "eval_metric": "logloss", "random_state": 42}

    log_model_run(lr_model, "Logistic_Regression", lr_params,
                  X_val, y_val, X_test, y_test)
    log_model_run(rf_model, "Random_Forest", rf_params,
                  X_val, y_val, X_test, y_test)
    log_model_run(xgb_model, "XGBoost", xgb_params,
                  X_val, y_val, X_test, y_test)

    # ── Determine best model by PR-AUC ────────────────────────────────
    models_eval = {
        "XGBoost": (xgb_model, xgb_params),
        "Random_Forest": (rf_model, rf_params),
        "Logistic_Regression": (lr_model, lr_params),
    }
    pr_aucs = {}
    for name, (model, _) in models_eval.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        pr_aucs[name] = average_precision_score(y_test, y_proba)

    best_name = max(pr_aucs, key=pr_aucs.get)
    print(f"\n  Best baseline model: {best_name} (PR-AUC = {pr_aucs[best_name]:.4f})")

    # ── Optuna optimization ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  OPTUNA OPTIMIZATION")
    print("="*60)

    # Map internal names
    optuna_name_map = {
        "XGBoost": "XGBoost",
        "Random_Forest": "Random Forest",
        "Logistic_Regression": "Logistic Regression",
    }
    study = run_optuna_optimization(optuna_name_map[best_name],
                                    X_train, y_train, X_val, y_val, n_trials=30)

    optimized_model = train_optimized_model(optuna_name_map[best_name],
                                            study.best_params,
                                            X_train, y_train)

    # Log optimized model
    print("\n  Logging optimized model...")
    opt_run_id = log_model_run(optimized_model, best_name, study.best_params,
                               X_val, y_val, X_test, y_test, is_optimized=True)

    # ── Register best model ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  MODEL REGISTRY")
    print("="*60)

    model_uri = f"runs:/{opt_run_id}/model"
    registry_name = "CreditCardFraudDetector"

    result = mlflow.register_model(model_uri, registry_name)
    print(f"  Registered model: {registry_name}")
    print(f"  Version: {result.version}")

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(registry_name, "production", result.version)
    print(f"  Alias 'production' set for version {result.version}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  DONE")
    print("="*60)
    print(f"\n  MLflow UI: mlflow ui --backend-store-uri \"{tracking_uri}\"")
    print(f"  Or simply: cd \"{BASE_DIR}\" && mlflow ui")
    print(f"  Then open: http://127.0.0.1:5000")
    print(f"\n  mlruns directory: {MLRUNS_DIR}")
    print(f"  Registered model: {registry_name} (production)")


if __name__ == "__main__":
    main()
