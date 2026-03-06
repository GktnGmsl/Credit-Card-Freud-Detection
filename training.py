"""
Credit Card Fraud Detection — Model Training & Comparison
==========================================================
Trains Logistic Regression, Random Forest, and XGBoost models.
Evaluates with confusion matrices, classification reports, ROC-AUC,
PR-AUC, loss curves, and threshold optimization.
Runs Optuna Bayesian optimization on the best PR-AUC model.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, log_loss, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
import joblib
import json
import xgboost as xgb  # type: ignore
import optuna  # type: ignore

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ──────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "Class"


def load_data():
    """Load train (SMOTE), validation, and test sets from parquet files."""
    train = pd.read_parquet(os.path.join(DATA_DIR, "train_smote.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    features = [c for c in train.columns if c != TARGET]

    X_train, y_train = train[features].values, train[TARGET].values
    X_val, y_val = val[features].values, val[TARGET].values
    X_test, y_test = test[features].values, test[TARGET].values

    print(f"Train : {X_train.shape}  | Fraud ratio: {y_train.mean():.4f}")
    print(f"Val   : {X_val.shape}  | Fraud ratio: {y_val.mean():.4f}")
    print(f"Test  : {X_test.shape}  | Fraud ratio: {y_test.mean():.4f}")

    return X_train, y_train, X_val, y_val, X_test, y_test, features


# ──────────────────────────────────────────────────────────────────────
# 2. EVALUATION UTILITIES
# ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(y_val, y_val_pred, y_test, y_test_pred,
                            model_name, save=True):
    """Plot confusion matrices for validation and test sets side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_true, y_pred, title in [
        (axes[0], y_val, y_val_pred, f"{model_name} — Validation Set"),
        (axes[1], y_test, y_test_pred, f"{model_name} — Test Set"),
    ]:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Fraud"],
                    yticklabels=["Normal", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"cm_{model_name.replace(' ', '_')}.png"),
                     dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def print_classification_report_custom(y_true, y_pred, dataset_name, model_name):
    """Print classification report with accuracy, focused on fraud class."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  {model_name} — {dataset_name}")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.6f}")
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Fraud"], digits=6))


def compute_roc_pr(y_true, y_proba):
    """Compute ROC and PR curve data."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc_val = average_precision_score(y_true, y_proba)
    return {
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_val,
        "precision": precision, "recall": recall, "pr_auc": pr_auc_val,
        "roc_thresholds": roc_thresholds, "pr_thresholds": pr_thresholds,
    }


def find_best_threshold(y_true, y_proba, thresholds=None):
    """Find threshold that maximizes F1-score."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_t, zero_division=0))
    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx], thresholds, f1_scores


def plot_f1_vs_threshold(thresholds, f1_scores, best_threshold, best_f1,
                         model_name, save=True):
    """Plot F1-score vs threshold curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_scores, color="steelblue", lw=2)
    ax.axvline(best_threshold, color="red", ls="--", lw=1.5,
               label=f"Best threshold = {best_threshold:.2f} (F1 = {best_f1:.4f})")
    ax.axvline(0.5, color="gray", ls=":", lw=1, label="Default threshold = 0.50")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-Score")
    ax.set_title(f"{model_name} — F1 vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"f1_threshold_{model_name.replace(' ', '_')}.png"),
                     dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────
# 3. LOSS CURVES
# ──────────────────────────────────────────────────────────────────────

def train_logistic_regression_with_loss(X_train, y_train, X_val, y_val):
    """Train LR incrementally and record log_loss at each checkpoint."""
    checkpoints = [5, 10, 25, 50, 100, 200]
    losses = {"train": [], "val": [], "iters": []}

    lr_model = LogisticRegression(
        solver="saga", max_iter=5, warm_start=True, random_state=42, C=1.0
    )

    for cp in checkpoints:
        lr_model.max_iter = cp
        lr_model.fit(X_train, y_train)
        losses["iters"].append(cp)
        for name, X, y in [("train", X_train, y_train),
                           ("val", X_val, y_val)]:
            proba = lr_model.predict_proba(X)
            losses[name].append(log_loss(y, proba))
        print(f"    LR checkpoint iter={cp} done")

    print(f"  Logistic Regression trained (final iter={checkpoints[-1]})")
    return lr_model, losses


def train_random_forest_with_loss(X_train, y_train, X_val, y_val):
    """Train RF incrementally and record log_loss at each n_estimators checkpoint."""
    checkpoints = [10, 25, 50, 75, 100, 150, 200]
    losses = {"train": [], "val": [], "n_estimators": []}

    rf_model = RandomForestClassifier(
        n_estimators=10, oob_score=True, warm_start=True,
        random_state=42, n_jobs=-1
    )

    for cp in checkpoints:
        rf_model.n_estimators = cp
        rf_model.fit(X_train, y_train)
        losses["n_estimators"].append(cp)
        for name, X, y in [("train", X_train, y_train),
                           ("val", X_val, y_val)]:
            proba = rf_model.predict_proba(X)
            losses[name].append(log_loss(y, proba))
        print(f"    RF checkpoint n_estimators={cp} done")

    print(f"  Random Forest trained (n_estimators={checkpoints[-1]})")
    return rf_model, losses


def train_xgboost_with_loss(X_train, y_train, X_val, y_val):
    """Train XGBoost with eval_set to track logloss per round."""
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
        use_label_encoder=False, verbosity=0
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    results = xgb_model.evals_result()
    losses = {
        "train": results["validation_0"]["logloss"],
        "val": results["validation_1"]["logloss"],
        "rounds": list(range(1, len(results["validation_0"]["logloss"]) + 1)),
    }
    print(f"  XGBoost trained (n_estimators=200)")
    return xgb_model, losses


def plot_loss_curves(losses_dict, x_key, x_label, model_name, save=True):
    """Plot train/val/test loss curves."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = losses_dict[x_key]
    ax.plot(x, losses_dict["train"], label="Train", color="steelblue", lw=2)
    ax.plot(x, losses_dict["val"], label="Validation", color="darkorange", lw=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Log Loss")
    ax.set_title(f"{model_name} — Loss Curves (Train / Validation)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"loss_{model_name.replace(' ', '_')}.png"),
                     dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────
# 4. ROC & PR CURVES
# ──────────────────────────────────────────────────────────────────────

def plot_roc_combined(results_dict, save=True):
    """Plot ROC curves for all models on the same figure."""
    colors = {"Logistic Regression": "steelblue",
              "Random Forest": "darkorange",
              "XGBoost": "green"}
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, res in results_dict.items():
        ax.plot(res["fpr"], res["tpr"], lw=2, color=colors.get(name, None),
                label=f"{name} (AUC = {res['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — All Models")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "roc_combined.png"),
                     dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_roc_individual(results_dict, save=True):
    """Plot ROC curves for each model individually."""
    colors = {"Logistic Regression": "steelblue",
              "Random Forest": "darkorange",
              "XGBoost": "green"}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (name, res) in zip(axes, results_dict.items()):
        ax.plot(res["fpr"], res["tpr"], lw=2, color=colors.get(name, None),
                label=f"AUC = {res['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {name}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "roc_individual.png"),
                     dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_pr_combined(results_dict, save=True):
    """Plot Precision-Recall curves for all models on the same figure."""
    colors = {"Logistic Regression": "steelblue",
              "Random Forest": "darkorange",
              "XGBoost": "green"}
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, res in results_dict.items():
        ax.plot(res["recall"], res["precision"], lw=2, color=colors.get(name, None),
                label=f"{name} (PR-AUC = {res['pr_auc']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — All Models (Test Set)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "pr_combined.png"),
                     dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_pr_individual(results_dict, save=True):
    """Plot PR curves for each model individually."""
    colors = {"Logistic Regression": "steelblue",
              "Random Forest": "darkorange",
              "XGBoost": "green"}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (name, res) in zip(axes, results_dict.items()):
        ax.plot(res["recall"], res["precision"], lw=2, color=colors.get(name, None),
                label=f"PR-AUC = {res['pr_auc']:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall — {name}")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "pr_individual.png"),
                     dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────
# 5. OPTUNA HYPERPARAMETER OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────

def optuna_xgboost_objective(trial, X, y, n_splits=5):
    """Optuna objective for XGBoost: maximize mean PR-AUC via Stratified K-Fold CV."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric": "logloss",
        "random_state": 42,
        "use_label_encoder": False,
        "verbosity": 0,
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pr_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_fold_train, y_fold_train,
                  eval_set=[(X_fold_val, y_fold_val)], verbose=False)
        y_proba = model.predict_proba(X_fold_val)[:, 1]
        pr_aucs.append(average_precision_score(y_fold_val, y_proba))
    return np.mean(pr_aucs)


def optuna_rf_objective(trial, X, y, n_splits=5):
    """Optuna objective for Random Forest: maximize mean PR-AUC via Stratified K-Fold CV."""
    max_depth_choice = trial.suggest_categorical("max_depth", [None, 10, 20, 30, 40, 50])
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": max_depth_choice,
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": 42,
        "n_jobs": -1,
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pr_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        model = RandomForestClassifier(**params)
        model.fit(X_fold_train, y_fold_train)
        y_proba = model.predict_proba(X_fold_val)[:, 1]
        pr_aucs.append(average_precision_score(y_fold_val, y_proba))
    return np.mean(pr_aucs)


def optuna_lr_objective(trial, X, y, n_splits=5):
    """Optuna objective for Logistic Regression: maximize mean PR-AUC via Stratified K-Fold CV."""
    params = {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "solver": trial.suggest_categorical("solver", ["saga", "lbfgs"]),
        "max_iter": trial.suggest_int("max_iter", 500, 2000),
        "random_state": 42,
    }
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pr_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LogisticRegression(**params)
        model.fit(X_fold_train, y_fold_train)
        y_proba = model.predict_proba(X_fold_val)[:, 1]
        pr_aucs.append(average_precision_score(y_fold_val, y_proba))
    return np.mean(pr_aucs)


def run_optuna_optimization(best_model_name, X, y, n_trials=50):
    """Run Optuna Bayesian optimization with Stratified K-Fold CV."""
    objectives = {
        "XGBoost": optuna_xgboost_objective,
        "Random Forest": optuna_rf_objective,
        "Logistic Regression": optuna_lr_objective,
    }
    objective_fn = objectives[best_model_name]

    study = optuna.create_study(direction="maximize",
                                study_name=f"{best_model_name}_optimization")
    study.optimize(
        lambda trial: objective_fn(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
        timeout=3600,  # 60 min max
    )

    print(f"\n{'='*60}")
    print(f"  Optuna Optimization — {best_model_name}")
    print(f"{'='*60}")
    print(f"  Best PR-AUC: {study.best_value:.6f}")
    print(f"  Best Parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study


def train_optimized_model(best_model_name, best_params, X_train, y_train):
    """Train the optimized model with best params from Optuna."""
    if best_model_name == "XGBoost":
        model = xgb.XGBClassifier(
            **best_params,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
        )
    elif best_model_name == "Random Forest":
        model = RandomForestClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = LogisticRegression(
            **best_params,
            random_state=42,
        )
    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────────────────────────────────
# 6. SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────

def build_summary_table(all_results):
    """Build a comparison DataFrame of all models."""
    rows = []
    for name, res in all_results.items():
        rows.append({
            "Model": name,
            "ROC-AUC": res["roc_auc"],
            "PR-AUC": res["pr_auc"],
            "Accuracy": res.get("accuracy", None),
            "Precision (Fraud)": res.get("precision_fraud", None),
            "Recall (Fraud)": res.get("recall_fraud", None),
            "F1 (Fraud)": res.get("f1_fraud", None),
            "Best Threshold": res.get("best_threshold", None),
            "F1 at Best Threshold": res.get("best_f1", None),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def run_full_pipeline():
    """Execute the full training, evaluation, and optimization pipeline."""

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
    lr_model, lr_losses = train_logistic_regression_with_loss(
        X_train, y_train, X_val, y_val)

    print("[2/3] Training Random Forest...")
    rf_model, rf_losses = train_random_forest_with_loss(
        X_train, y_train, X_val, y_val)

    print("[3/3] Training XGBoost...")
    xgb_model, xgb_losses = train_xgboost_with_loss(
        X_train, y_train, X_val, y_val)

    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
    }

    # ── Loss curves ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  LOSS CURVES")
    print("="*60)
    plot_loss_curves(lr_losses, "iters", "Iterations", "Logistic Regression")
    plot_loss_curves(rf_losses, "n_estimators", "Number of Trees", "Random Forest")
    plot_loss_curves(xgb_losses, "rounds", "Boosting Rounds", "XGBoost")

    # ── Per-model evaluation ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  MODEL EVALUATION")
    print("="*60)

    all_results = {}

    for name, model in models.items():
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Confusion matrices
        plot_confusion_matrices(y_val, y_val_pred, y_test, y_test_pred, name)

        # Classification reports
        print_classification_report_custom(y_val, y_val_pred, "Validation", name)
        print_classification_report_custom(y_test, y_test_pred, "Test", name)

        # ROC & PR on test set
        res = compute_roc_pr(y_test, y_test_proba)

        # Extract fraud-class metrics from classification report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        res["accuracy"] = accuracy_score(y_test, y_test_pred)
        res["precision_fraud"] = report["1"]["precision"]
        res["recall_fraud"] = report["1"]["recall"]
        res["f1_fraud"] = report["1"]["f1-score"]

        # Threshold optimization
        best_t, best_f1, thresholds, f1_scores = find_best_threshold(
            y_test, y_test_proba)
        res["best_threshold"] = best_t
        res["best_f1"] = best_f1

        plot_f1_vs_threshold(thresholds, f1_scores, best_t, best_f1, name)
        print(f"  {name} — Best threshold: {best_t:.2f}, F1: {best_f1:.4f}")

        # Evaluation at optimal threshold
        y_test_pred_opt = (y_test_proba >= best_t).astype(int)
        print_classification_report_custom(y_test, y_test_pred_opt,
                                           f"Test @threshold={best_t:.2f}", name)

        all_results[name] = res

    # ── Combined ROC & PR plots ───────────────────────────────────────
    print("\n" + "="*60)
    print("  ROC & PRECISION-RECALL CURVES")
    print("="*60)
    plot_roc_combined(all_results)
    plot_roc_individual(all_results)
    plot_pr_combined(all_results)
    plot_pr_individual(all_results)

    # ── Summary table ─────────────────────────────────────────────────
    summary_df = build_summary_table(all_results)
    print("\n" + "="*60)
    print("  MODEL COMPARISON SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    # ── Identify best model by PR-AUC ─────────────────────────────────
    best_model_name = max(all_results, key=lambda k: all_results[k]["pr_auc"])
    print(f"\n  Best model by PR-AUC: {best_model_name} "
          f"(PR-AUC = {all_results[best_model_name]['pr_auc']:.4f})")

    # ── Optuna optimization ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    # Load original (non-SMOTE) training data for K-Fold CV evaluation
    train_orig_df = pd.read_parquet(os.path.join(DATA_DIR, "train_original.parquet"))
    X_train_orig = train_orig_df.drop(columns=[TARGET])
    y_train_orig = train_orig_df[TARGET]

    study = run_optuna_optimization(best_model_name, X_train_orig, y_train_orig,
                                    n_trials=50)

    # Train optimized model
    optimized_model = train_optimized_model(best_model_name,
                                            study.best_params,
                                            X_train, y_train)

    # Evaluate optimized model
    y_test_proba_opt = optimized_model.predict_proba(X_test)[:, 1]
    y_test_pred_opt = optimized_model.predict(X_test)
    y_val_pred_opt = optimized_model.predict(X_val)

    opt_name = f"{best_model_name} (Optimized)"

    plot_confusion_matrices(y_val, y_val_pred_opt, y_test, y_test_pred_opt, opt_name)
    print_classification_report_custom(y_test, y_test_pred_opt, "Test", opt_name)
    print_classification_report_custom(y_val, y_val_pred_opt, "Validation", opt_name)

    opt_res = compute_roc_pr(y_test, y_test_proba_opt)
    report_opt = classification_report(y_test, y_test_pred_opt, output_dict=True)
    opt_res["accuracy"] = accuracy_score(y_test, y_test_pred_opt)
    opt_res["precision_fraud"] = report_opt["1"]["precision"]
    opt_res["recall_fraud"] = report_opt["1"]["recall"]
    opt_res["f1_fraud"] = report_opt["1"]["f1-score"]

    best_t_opt, best_f1_opt, thresholds_opt, f1_scores_opt = find_best_threshold(
        y_test, y_test_proba_opt)
    opt_res["best_threshold"] = best_t_opt
    opt_res["best_f1"] = best_f1_opt
    plot_f1_vs_threshold(thresholds_opt, f1_scores_opt, best_t_opt, best_f1_opt, opt_name)

    # Classification report at optimal threshold
    y_test_pred_opt_t = (y_test_proba_opt >= best_t_opt).astype(int)
    print_classification_report_custom(y_test, y_test_pred_opt_t,
                                       f"Test @threshold={best_t_opt:.2f}", opt_name)

    all_results[opt_name] = opt_res

    # ── Final combined plots with optimized model ─────────────────────
    plot_roc_combined(all_results)
    plot_pr_combined(all_results)

    # Final summary
    final_summary = build_summary_table(all_results)
    print("\n" + "="*60)
    print("  FINAL MODEL COMPARISON (Including Optimized)")
    print("="*60)
    print(final_summary.to_string(index=False))
    final_summary.to_csv(os.path.join(OUTPUT_DIR, "final_comparison.csv"), index=False)

    print("\n" + "="*60)
    print(f"  Optimized {best_model_name} — Best Hyperparameters")
    print("="*60)
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"    PR-AUC (Optuna best trial): {study.best_value:.6f}")
    print(f"    PR-AUC (Test set):           {opt_res['pr_auc']:.6f}")

    # ── Save models & artifacts ───────────────────────────────────────
    model_output_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(model_output_dir, exist_ok=True)

    for name, mdl in models.items():
        fname = name.lower().replace(" ", "_")
        joblib.dump(mdl, os.path.join(model_output_dir, f"{fname}.joblib"))
    joblib.dump(optimized_model, os.path.join(model_output_dir, "optimized_model.joblib"))

    # Save scaler (fit on original unscaled training data)
    from sklearn.preprocessing import StandardScaler
    cols_to_scale = [
        "Amount", "Time", "Time_in_day", "Amount_log",
        "Time_Amount", "Time_Amount_sq", "Amount_per_Time",
    ]
    train_original = pd.read_parquet(os.path.join(DATA_DIR, "train_original.parquet"))
    scaler_obj = StandardScaler()
    scaler_obj.fit(train_original[cols_to_scale])
    joblib.dump(scaler_obj, os.path.join(model_output_dir, "scaler.joblib"))

    # Save model config (threshold, best model info)
    model_config = {
        "best_threshold": float(best_t_opt),
        "best_model": best_model_name,
        "best_params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                        for k, v in study.best_params.items()},
    }
    with open(os.path.join(model_output_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n  Models, scaler, and config saved to {model_output_dir}")

    return {
        "models": models,
        "optimized_model": optimized_model,
        "optimized_model_name": best_model_name,
        "study": study,
        "all_results": all_results,
        "losses": {
            "Logistic Regression": lr_losses,
            "Random Forest": rf_losses,
            "XGBoost": xgb_losses,
        },
        "data": (X_train, y_train, X_val, y_val, X_test, y_test, features),
    }


if __name__ == "__main__":
    results = run_full_pipeline()
