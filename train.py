"""
train.py
--------
Trains two flood-detection classifiers:
  1. Random Forest  (sklearn)
  2. XGBoost        (xgboost)

Evaluates both, picks the best one, and saves the model + scaler to disk.

Run from the project root:
    python train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from sklearn.model_selection import cross_val_score
import xgboost as xgb

from data_loader import download_dataset, preprocess, FEATURE_COLS, BINARY_TARGET

# ── Config ─────────────────────────────────────────────────────────────────────
MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
SAMPLE_SIZE  = 50_000   # cap for faster local training (set None for full dataset)
RANDOM_STATE = 42


def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Load & preprocess ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Flood Early Warning System — ML Training Pipeline")
    print("=" * 60)
    df_full = download_dataset()

    if SAMPLE_SIZE and len(df_full) > SAMPLE_SIZE:
        df_full = df_full.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
        print(f"[train] Sampled {SAMPLE_SIZE:,} rows for training speed.")

    X_train, X_test, y_train, y_test, scaler, df_clean, feature_cols = preprocess(df_full)
    print(f"[train] Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"[train] Class balance — Flood: {y_train.mean()*100:.1f}%  No-Flood: {(1-y_train.mean())*100:.1f}%")

    # ── Model definitions ──────────────────────────────────────────────────────
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n[train] Training {name}...")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]

        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")
        auc    = roc_auc_score(y_test, y_prob)
        cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()

        results[name] = {
            "model":    model,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "accuracy": acc,
            "f1":       f1,
            "auc":      auc,
            "cv_acc":   cv_acc,
        }

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  5-Fold CV: {cv_acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))

    # ── Pick best model ────────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["auc"])
    best      = results[best_name]
    print(f"\n[train] Best model: {best_name} (AUC = {best['auc']:.4f})")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    joblib.dump(best["model"],  f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(scaler,         f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(feature_cols,   f"{MODELS_DIR}/feature_cols.pkl")

    # Save metrics JSON (read by the Streamlit app)
    metrics = {
        "best_model_name": best_name,
        "accuracy":  round(best["accuracy"], 4),
        "f1_score":  round(best["f1"], 4),
        "roc_auc":   round(best["auc"], 4),
        "cv_accuracy": round(best["cv_acc"], 4),
        "all_models": {
            k: {
                "accuracy": round(v["accuracy"], 4),
                "f1":       round(v["f1"], 4),
                "auc":      round(v["auc"], 4),
                "cv_acc":   round(v["cv_acc"], 4),
            } for k, v in results.items()
        }
    }
    with open(f"{REPORTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Generate report plots ──────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, best["y_pred"], best_name)
    _plot_roc_curves(y_test, results)
    _plot_feature_importance(best["model"], feature_cols, best_name)
    _plot_model_comparison(results)

    print(f"\n[train] Done. Model saved to {MODELS_DIR}/  |  Plots saved to {REPORTS_DIR}/")
    return metrics


def _plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Flood", "Flood"],
                yticklabels=["No Flood", "Flood"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=120)
    plt.close(fig)


def _plot_roc_curves(y_test, results):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1D9E75", "#378ADD", "#E24B4A"]
    for i, (name, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{name}  (AUC = {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/roc_curves.png", dpi=120)
    plt.close(fig)


def _plot_feature_importance(model, feature_cols, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1D9E75" if v > np.median(importances) else "#9FE1CB" for v in importances[idx]]
    ax.barh([feature_cols[i] for i in idx][::-1],
            importances[idx][::-1],
            color=colors[::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/feature_importance.png", dpi=120)
    plt.close(fig)


def _plot_model_comparison(results):
    names   = list(results.keys())
    metrics_to_plot = ["accuracy", "f1", "auc", "cv_acc"]
    labels  = ["Accuracy", "F1 Score", "AUC-ROC", "CV Accuracy"]
    x = np.arange(len(names))
    width = 0.18
    colors = ["#1D9E75", "#378ADD", "#E24B4A", "#BA7517"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (metric, label) in enumerate(zip(metrics_to_plot, labels)):
        vals = [results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=label, color=colors[i])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/model_comparison.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    train_and_evaluate()
