"""
Diabetes Prediction – Final Model Evaluation
=============================================
Loads all trained models, runs the winning ensemble (soft voting:
KNN + RF + XGBoost + MLP), and produces the key visualizations.

Run from the project root:
    python main.py
"""

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent

# =============================================================================
# 1. DATA
# =============================================================================

def read_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "diabetes.csv")


def impute_missing(df_train: pd.DataFrame, df_test: pd.DataFrame, cols: list) -> tuple:
    """Median imputation on train set; same median applied to test set.
    Adds binary missing-indicator flag for each imputed column."""
    for col in cols:
        df_train[col + "_missing_flag"] = (df_train[col] == 0).astype(int)
        df_test[col + "_missing_flag"]  = (df_test[col]  == 0).astype(int)
        median = df_train[col].replace(0, np.nan).median()
        df_train[col] = df_train[col].replace(0, median)
        df_test[col]  = df_test[col].replace(0, median)
    return df_train, df_test


def engineer_poly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise and triple interaction terms selected by permutation importance."""
    df = df.copy()
    df["Glucose x BMI"]               = df["Glucose"]      * df["BMI"]
    df["Age x BMI"]                   = df["Age"]          * df["BMI"]
    df["Pregnancies x Insulin"]        = df["Pregnancies"]  * df["Insulin"]
    df["SkinThickness x Age"]          = df["SkinThickness"]* df["Age"]
    df["Insulin x Age"]                = df["Insulin"]      * df["Age"]
    df["Pregnancies x Insulin x BMI"]  = df["Pregnancies"]  * df["Insulin"] * df["BMI"]
    return df


def engineer_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clinically motivated binary threshold features (used by XGBoost only)."""
    df = df.copy()
    df["Glucose_high"]        = (df["Glucose"]       >= 126).astype(int)
    df["Glucose_prediabetes"] = (df["Glucose"]       >= 100).astype(int)
    df["BMI_obese"]           = (df["BMI"]           >= 30 ).astype(int)
    df["BMI_overweight"]      = (df["BMI"]           >= 25 ).astype(int)
    df["Insulin_high"]        = (df["Insulin"]       >  166).astype(int)
    df["Age_risk"]            = (df["Age"]           >= 45 ).astype(int)
    df["BP_high"]             = (df["BloodPressure"] >= 90 ).astype(int)
    return df


def drop_features(df_train: pd.DataFrame, df_test: pd.DataFrame, cols: list) -> tuple:
    return df_train.drop(columns=cols), df_test.drop(columns=cols)


# =============================================================================
# 2. FEATURE SETS
# =============================================================================

MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

KNN_FEATURES = [
    "Age x BMI", "BMI", "Glucose", "Pregnancies x Insulin x BMI", "Insulin"
]
RF_DROP = [
    "Glucose_missing_flag", "BMI_missing_flag", "SkinThickness_missing_flag",
    "BloodPressure_missing_flag", "Insulin_missing_flag", "Pregnancies",
]
LOG_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "BMI",
    "DiabetesPedigreeFunction", "Age", "Insulin_missing_flag",
    "Glucose x BMI", "Age x BMI", "Pregnancies x Insulin",
    "SkinThickness x Age", "Insulin x Age", "Pregnancies x Insulin x BMI",
]
SVM_FEATURES = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "SkinThickness_missing_flag",
    "Glucose x BMI", "Age x BMI", "Pregnancies x Insulin",
    "SkinThickness x Age", "Insulin x Age", "Pregnancies x Insulin x BMI",
]

MODEL_LABELS = {
    "knn": "KNN",
    "rf":  "Random Forest",
    "log": "Logistic Regression",
    "svm": "SVM",
    "xgb": "XGBoost",
    "mlp": "MLP",
}
MODEL_COLORS = {
    "knn": "#4C72B0",
    "rf":  "#55A868",
    "log": "#C44E52",
    "svm": "#8172B2",
    "xgb": "#CCB974",
    "mlp": "#64B5CD",
}


def build_feature_sets(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
) -> dict:
    """Return per-model (train, test) DataFrames."""
    ft_poly, ftest_poly = impute_missing(
        features_train.copy(), features_test.copy(), MISSING_COLS
    )
    ft_poly    = engineer_poly_features(ft_poly)
    ftest_poly = engineer_poly_features(ftest_poly)

    ft_full    = engineer_threshold_features(ft_poly)
    ftest_full = engineer_threshold_features(ftest_poly)

    ft_rf, ftest_rf = drop_features(ft_poly.copy(), ftest_poly.copy(), RF_DROP)

    return {
        "knn": (ft_poly[KNN_FEATURES],  ftest_poly[KNN_FEATURES]),
        "rf":  (ft_rf,                  ftest_rf),
        "log": (ft_poly[LOG_FEATURES],  ftest_poly[LOG_FEATURES]),
        "svm": (ft_poly[SVM_FEATURES],  ftest_poly[SVM_FEATURES]),
        "xgb": (ft_full,                ftest_full),
        "mlp": (ft_poly,                ftest_poly),
    }


# =============================================================================
# 3. LOAD MODELS
# =============================================================================

def load_models(base_dir: Path) -> dict:
    models = {}
    for name in MODEL_LABELS:
        with open(base_dir / f"final_model_{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    return models


# =============================================================================
# 4. EVALUATE INDIVIDUAL MODELS
# =============================================================================

def evaluate_models(
    models: dict,
    feature_sets: dict,
    target_test: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        _, f_test = feature_sets[name]
        pred  = model.predict(f_test)
        proba = model.predict_proba(f_test)[:, 1]
        rows.append({
            "Model":     MODEL_LABELS[name],
            "F1":        f1_score(target_test, pred),
            "Precision": precision_score(target_test, pred),
            "Recall":    recall_score(target_test, pred),
            "ROC-AUC":   roc_auc_score(target_test, proba),
        })
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


# =============================================================================
# 5. SOFT VOTING ENSEMBLE (KNN + RF + XGBoost + MLP)
# =============================================================================

def soft_voting_ensemble(
    models: dict,
    feature_sets: dict,
    target_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Winning ensemble: unweighted average of predicted probabilities
    from KNN, Random Forest, XGBoost and MLP.
    LogReg and SVM are excluded — they reduced ensemble F1 in testing.
    """
    ensemble_members = ["knn", "rf", "xgb", "mlp"]
    probas = []
    for name in ensemble_members:
        _, f_test = feature_sets[name]
        probas.append(models[name].predict_proba(f_test)[:, 1])

    soft_proba = np.column_stack(probas).mean(axis=1)
    soft_pred  = (soft_proba >= threshold).astype(int)

    return {
        "proba": soft_proba,
        "pred":  soft_pred,
        "f1":        f1_score(target_test, soft_pred),
        "precision": precision_score(target_test, soft_pred),
        "recall":    recall_score(target_test, soft_pred),
        "roc_auc":   roc_auc_score(target_test, soft_proba),
    }


# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================

def plot_metrics_bar(results: pd.DataFrame, save_path: Path) -> None:
    metrics = ["F1", "Precision", "Recall", "ROC-AUC"]
    x     = np.arange(len(results))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results[metric], width, label=metric, zorder=3)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – Test Set Metrics")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_roc_curves(
    models: dict,
    feature_sets: dict,
    target_test: pd.Series,
    ensemble_proba: np.ndarray,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        _, f_test = feature_sets[name]
        proba = model.predict_proba(f_test)[:, 1]
        auc   = roc_auc_score(target_test, proba)
        RocCurveDisplay.from_predictions(
            target_test, proba,
            name=f"{MODEL_LABELS[name]} (AUC={auc:.3f})",
            color=MODEL_COLORS[name],
            ax=ax,
        )
    # Ensemble
    ens_auc = roc_auc_score(target_test, ensemble_proba)
    RocCurveDisplay.from_predictions(
        target_test, ensemble_proba,
        name=f"Soft Voting – KNN/RF/XGB/MLP (AUC={ens_auc:.3f})",
        color="black", linestyle="--", lw=2,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="Random")
    ax.set_title("ROC Curves – All Models + Ensemble")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_threshold_curve(
    ensemble_proba: np.ndarray,
    target_test: pd.Series,
    best_threshold: float,
    save_path: Path,
) -> None:
    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        pred = (ensemble_proba >= t).astype(int)
        rows.append({
            "Threshold": t,
            "Precision": precision_score(target_test, pred, zero_division=0),
            "Recall":    recall_score(target_test, pred, zero_division=0),
            "F1":        f1_score(target_test, pred, zero_division=0),
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Threshold"], df["F1"],        label="F1")
    ax.plot(df["Threshold"], df["Precision"], label="Precision")
    ax.plot(df["Threshold"], df["Recall"],    label="Recall")
    ax.axvline(best_threshold, linestyle="--", color="black",
               label=f"Best threshold = {best_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Soft Voting Ensemble – Precision, Recall and F1 by Decision Threshold")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_confusion_matrices(
    models: dict,
    feature_sets: dict,
    target_test: pd.Series,
    ensemble_pred: np.ndarray,
    save_path: Path,
) -> None:
    entries = list(models.items()) + [("ensemble", None)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, (name, model) in zip(axes, entries):
        if name == "ensemble":
            pred  = ensemble_pred
            title = "Soft Voting\n(KNN + RF + XGB + MLP)"
        else:
            _, f_test = feature_sets[name]
            pred  = model.predict(f_test)
            title = MODEL_LABELS[name]
        cm   = confusion_matrix(target_test, pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
    # hide unused subplot (2×4 = 8 panels, 7 items)
    for ax in axes[len(entries):]:
        ax.set_visible(False)
    fig.suptitle("Confusion Matrices – Test Set", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Diabetes Prediction – Final Evaluation")
    print("=" * 60)

    # --- Data ---
    df       = read_data(BASE_DIR)
    features = df.drop(columns=["Outcome"])
    target   = df["Outcome"]
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    feature_sets = build_feature_sets(features_train, features_test)

    # --- Models ---
    print("\nLoading models ...")
    models = load_models(BASE_DIR)

    # --- Individual model evaluation ---
    print("\nIndividual model results (test set):")
    results = evaluate_models(models, feature_sets, target_test)
    print(results.to_string(index=False, float_format="{:.4f}".format))

    # --- Soft voting ensemble ---
    BEST_THRESHOLD = 0.5   # optimal threshold found during tuning
    ensemble = soft_voting_ensemble(models, feature_sets, target_test, threshold=BEST_THRESHOLD)
    print(f"\nSoft Voting Ensemble (KNN + RF + XGBoost + MLP), threshold={BEST_THRESHOLD}:")
    print(f"  F1:        {ensemble['f1']:.4f}")
    print(f"  Precision: {ensemble['precision']:.4f}")
    print(f"  Recall:    {ensemble['recall']:.4f}")
    print(f"  ROC-AUC:   {ensemble['roc_auc']:.4f}")

    # --- Plots ---
    print("\nGenerating plots ...")
    plot_metrics_bar(results, BASE_DIR / "plot_metrics_bar.png")
    plot_roc_curves(models, feature_sets, target_test,
                    ensemble["proba"], BASE_DIR / "plot_roc_curves.png")
    plot_threshold_curve(ensemble["proba"], target_test,
                         BEST_THRESHOLD, BASE_DIR / "plot_threshold_curve.png")
    plot_confusion_matrices(models, feature_sets, target_test,
                            ensemble["pred"], BASE_DIR / "plot_confusion_matrices.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
