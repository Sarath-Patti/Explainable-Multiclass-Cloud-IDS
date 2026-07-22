"""XGBoost Model Training & Evaluation Module for Explainable-Multiclass-Cloud-IDS.

This module loads preprocessed datasets and artifacts, performs hyperparameter tuning
using RandomizedSearchCV with 3-Fold Stratified Cross-Validation on a representative
stratified tuning subset (250k samples), selects the optimal hyperparameters, and trains
the final XGBoost model on the complete training dataset (1.76M samples).
Evaluates performance on validation and test sets, and outputs metrics, visualization plots,
model artifacts, and a detailed performance report.
"""

import json
import logging
import time
from pathlib import Path
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("XGBoostBaseline")


def load_preprocessed_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    np.ndarray, np.ndarray, np.ndarray,
    object, dict, dict
]:
    """Loads preprocessed datasets and artifacts (supporting both CSV and NPY formats).

    Returns:
        tuple containing:
            - X_train, X_valid, X_test (DataFrames)
            - y_train, y_valid, y_test (1D numpy arrays)
            - label_encoder (LabelEncoder)
            - class_weights (dict)
            - label_mapping (dict)
    """
    logger.info("Loading preprocessed feature splits and target labels...")

    if (PROCESSED_DATA_DIR / "X_train.npy").exists():
        X_train = pd.DataFrame(np.load(PROCESSED_DATA_DIR / "X_train.npy"))
        X_valid = pd.DataFrame(np.load(PROCESSED_DATA_DIR / "X_valid.npy"))
        X_test = pd.DataFrame(np.load(PROCESSED_DATA_DIR / "X_test.npy"))
        y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
        y_valid = np.load(PROCESSED_DATA_DIR / "y_valid.npy")
        y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
    else:
        X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
        X_valid = pd.read_csv(PROCESSED_DATA_DIR / "X_valid.csv")
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
        y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")["Label"].values
        y_valid = pd.read_csv(PROCESSED_DATA_DIR / "y_valid.csv")["Label"].values
        y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")["Label"].values

    logger.info(f"Loaded X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"Loaded X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
    logger.info(f"Loaded X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    logger.info("Loading preprocessing artifacts (label_encoder.pkl, class_weights.pkl, label_mapping.json)...")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    class_weights = joblib.load(MODELS_DIR / "class_weights.pkl")

    label_mapping_path = MODELS_DIR / "label_mapping.json"
    if label_mapping_path.exists():
        with open(label_mapping_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
    else:
        label_mapping = {str(c): i for i, c in enumerate(label_encoder.classes_)}

    return X_train, X_valid, X_test, y_train, y_valid, y_test, label_encoder, class_weights, label_mapping


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_class: int,
    tuning_size: int = 250000
) -> tuple[dict, float, float, int]:
    """Performs hyperparameter tuning using RandomizedSearchCV on a stratified tuning subset.

    Args:
        X_train: Complete training predictor features.
        y_train: Complete training target labels.
        num_class: Total number of target classes.
        tuning_size: Size of the stratified tuning subset.

    Returns:
        tuple containing:
            - best_params: Dict of best hyperparameters.
            - best_cv_score: Best Macro F1 cross-validation score on tuning subset.
            - tuning_time: Tuning execution duration in seconds.
            - tuning_samples: Number of samples in the tuning subset.
    """
    logger.info(f"Creating a stratified tuning subset of {tuning_size:,} samples from X_train...")
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train,
        train_size=tuning_size,
        stratify=y_train,
        random_state=42
    )
    tuning_samples = len(X_tune)
    logger.info(f"Tuning subset created successfully. X_tune shape: {X_tune.shape}")

    base_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        tree_method="hist",
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )

    param_distributions = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.05, 0.1, 0.15],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    xgb_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_distributions,
        n_iter=5,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    logger.info("Executing RandomizedSearchCV on XGBoost tuning subset (3-Fold CV, 5 iterations)...")
    start_tune_time = time.time()
    xgb_search.fit(X_tune, y_tune)
    tuning_time = time.time() - start_tune_time

    best_params = xgb_search.best_params_
    best_cv_score = float(xgb_search.best_score_)

    logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds.")
    logger.info(f"Best Tuning Subset CV Macro F1 Score: {best_cv_score:.6f}")
    logger.info(f"Selected Best Hyperparameters: {best_params}")

    return best_params, best_cv_score, tuning_time, tuning_samples


def train_final_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_class: int,
    best_params: dict
) -> tuple[XGBClassifier, float]:
    """Trains the final XGBoost model on the complete training dataset.

    Args:
        X_train: Complete training features dataset (1.76M samples).
        y_train: Complete training target labels.
        num_class: Total number of target classes.
        best_params: Selected optimal hyperparameters.

    Returns:
        tuple containing:
            - final_model: Fitted XGBClassifier on complete training data.
            - train_time: Duration of full model training in seconds.
    """
    logger.info("Instantiating final XGBoost model with selected best parameters...")
    final_xgb = XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        tree_method="hist",
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )

    logger.info(f"Retraining final XGBoost model on COMPLETE training dataset ({len(X_train):,} samples)...")
    start_train = time.time()
    final_xgb.fit(X_train, y_train)
    train_time = time.time() - start_train

    logger.info(f"Full XGBoost training completed in {train_time:.2f} seconds.")

    return final_xgb, train_time


def evaluate_model(
    model: XGBClassifier,
    X: pd.DataFrame,
    y_true: np.ndarray,
    class_names: list[str]
) -> tuple[dict, float, np.ndarray, np.ndarray]:
    """Evaluates the XGBoost model on a dataset split and calculates detailed metrics.

    Args:
        model: Trained XGBClassifier.
        X: Features dataset split.
        y_true: Ground truth target labels.
        class_names: List of target class names.

    Returns:
        tuple containing:
            - metrics: Dict of evaluation metrics.
            - inference_time: Time taken to run predictions.
            - y_pred: Array of predicted class indices.
            - y_proba: Matrix of predicted class probabilities.
    """
    start_time = time.time()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    inference_time = time.time() - start_time

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weight = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    try:
        roc_auc_mac = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        roc_auc_weight = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    except Exception as e:
        logger.warning(f"Could not calculate overall ROC-AUC: {e}")
        roc_auc_mac = 0.0
        roc_auc_weight = 0.0

    p_class, r_class, f_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class_metrics = {}
    for idx, cname in enumerate(class_names):
        try:
            c_auc = float(roc_auc_score(y_true == idx, y_proba[:, idx]))
        except Exception:
            c_auc = 0.0

        per_class_metrics[cname] = {
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1_score": float(f_class[idx]),
            "roc_auc": c_auc
        }

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_mac,
        "f1_weighted": f1_weight,
        "roc_auc_macro": roc_auc_mac,
        "roc_auc_weighted": roc_auc_weight,
        "inference_time_seconds": float(inference_time),
        "per_class": per_class_metrics
    }

    return metrics, inference_time, y_pred, y_proba


def generate_plots(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_proba: np.ndarray,
    class_names: list[str],
    plots_dir: Path
) -> None:
    """Generates and saves publication-quality evaluation plots.

    Args:
        model: Trained XGBoost model.
        X_test: Test features DataFrame.
        y_test: Ground truth test labels.
        y_test_pred: Predicted test labels.
        y_test_proba: Predicted test probabilities.
        class_names: List of class labels.
        plots_dir: Path to directory to save plots.
    """
    logger.info(f"Generating evaluation plots in {plots_dir}...")
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Sample Count"}
    )
    plt.title("XGBoost Baseline - Test Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Class Label", fontsize=12)
    plt.ylabel("True Class Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / "xgb_confusion_matrix.png", dpi=300)
    plt.close()

    # 2. Feature Importances (Top 20)
    importances = model.feature_importances_
    features = X_test.columns
    feat_imp_df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    top_20_df = feat_imp_df.head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="importance",
        y="feature",
        data=top_20_df,
        hue="feature",
        palette="viridis",
        legend=False
    )
    plt.title("XGBoost Baseline - Top 20 Feature Importances", fontsize=14, pad=15)
    plt.xlabel("Feature Importance Weight", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "xgb_feature_importance.png", dpi=300)
    plt.close()

    # 3. One-vs-Rest ROC Curves
    plt.figure(figsize=(10, 8))
    for idx, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test == idx, y_test_proba[:, idx])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label=f"{cname} (AUC = {roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("XGBoost Baseline - One-vs-Rest ROC Curves", fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "xgb_roc_curves.png", dpi=300)
    plt.close()

    # 4. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for idx, cname in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_test == idx, y_test_proba[:, idx])
        pr_auc_val = auc(rec, prec)
        plt.plot(rec, prec, lw=1.5, label=f"{cname} (PR-AUC = {pr_auc_val:.3f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("XGBoost Baseline - Precision-Recall Curves", fontsize=14, pad=15)
    plt.legend(loc="lower left", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "xgb_precision_recall.png", dpi=300)
    plt.close()

    logger.info("All evaluation plots generated successfully.")


def save_artifacts_and_metrics(
    model: XGBClassifier,
    best_params: dict,
    best_cv_score: float,
    tuning_time: float,
    train_time: float,
    tuning_samples: int,
    total_train_samples: int,
    val_metrics: dict,
    test_metrics: dict,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    class_names: list[str]
) -> None:
    """Saves model binaries, metrics JSONs, and CSV reports.

    Args:
        model: Trained XGBoost model.
        best_params: Best hyperparameters dictionary.
        best_cv_score: Best CV macro F1 score.
        tuning_time: Tuning duration in seconds.
        train_time: Final model training duration in seconds.
        tuning_samples: Sample size of tuning subset.
        total_train_samples: Sample size of full training dataset.
        val_metrics: Validation metrics dictionary.
        test_metrics: Test metrics dictionary.
        y_test: True test target values.
        y_test_pred: Predicted test target values.
        class_names: List of target class names.
    """
    logger.info("Saving trained model and output metrics...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save serialized model
    joblib.dump(model, MODELS_DIR / "xgboost_model.pkl")

    # 2. Save best parameters & search info JSON
    best_params_payload = {
        "best_params": best_params,
        "best_cv_macro_f1": best_cv_score,
        "n_trees": int(model.n_estimators),
        "tuning_subset_samples": tuning_samples,
        "full_training_samples": total_train_samples,
        "cv_folds": 3,
        "random_search_iterations": 5,
        "tuning_time_seconds": float(tuning_time),
        "final_training_time_seconds": float(train_time),
        "val_inference_time_seconds": float(val_metrics["inference_time_seconds"]),
        "test_inference_time_seconds": float(test_metrics["inference_time_seconds"])
    }
    with open(METRICS_DIR / "xgb_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_payload, f, indent=4)

    # 3. Save comprehensive metrics JSON
    metrics_payload = {
        "validation": val_metrics,
        "test": test_metrics
    }
    with open(METRICS_DIR / "xgboost_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    # 4. Save classification report CSV
    clf_report_dict = classification_report(
        y_test, y_test_pred, target_names=class_names, output_dict=True
    )
    clf_report_df = pd.DataFrame(clf_report_dict).T
    clf_report_df.to_csv(METRICS_DIR / "xgb_classification_report.csv", index=True)

    # 5. Save confusion matrix CSV
    cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(METRICS_DIR / "xgb_confusion_matrix.csv", index=True)

    logger.info("Model binaries and metrics saved successfully.")


def generate_report(
    model: XGBClassifier,
    best_params: dict,
    best_cv_score: float,
    tuning_time: float,
    train_time: float,
    tuning_samples: int,
    total_train_samples: int,
    val_metrics: dict,
    test_metrics: dict,
    feature_names: list[str]
) -> str:
    """Generates the text report summarizing model performance.

    Args:
        model: Trained XGBoost model.
        best_params: Best hyperparameters dictionary.
        best_cv_score: Best CV macro F1 score.
        tuning_time: Hyperparameter tuning duration in seconds.
        train_time: Full model training duration in seconds.
        tuning_samples: Number of samples in tuning subset.
        total_train_samples: Number of samples in full training dataset.
        val_metrics: Validation metrics dictionary.
        test_metrics: Test metrics dictionary.
        feature_names: List of feature names.

    Returns:
        str: Formatted plain text report.
    """
    logger.info("Compiling XGBoost baseline report...")

    # Feature importances top 20
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    top_20 = feat_imp_df.head(20)

    lines = [
        "==================================================",
        "XGBOOST BASELINE MODEL PERFORMANCE REPORT",
        "==================================================",
        "",
        "1. MODEL CONFIGURATION & HYPERPARAMETER TUNING",
        "----------------------------------------------",
        f"Algorithm: XGBClassifier (Gradient Boosted Trees)",
        f"Objective: multi:softprob",
        f"Evaluation Metric: mlogloss",
        f"Tree Method: hist",
        f"Tuning Subset Size: {tuning_samples:,} samples",
        f"Cross-Validation Configuration: 3-Fold StratifiedKFold",
        f"RandomizedSearch Iterations: 5",
        f"Full Training Dataset Size: {total_train_samples:,} samples",
        f"Number of Trees (n_estimators): {model.n_estimators}",
        f"Best CV Macro F1 Score (Tuning Subset): {best_cv_score:.6f}",
        f"Hyperparameter Tuning Time: {tuning_time:.2f} seconds",
        f"Final Full-Dataset Training Time: {train_time:.2f} seconds",
        f"Validation Inference Time: {val_metrics['inference_time_seconds']:.4f} seconds",
        f"Test Inference Time: {test_metrics['inference_time_seconds']:.4f} seconds",
        "",
        "Best Hyperparameters:",
    ]

    for k, v in sorted(best_params.items()):
        lines.append(f"  - {k}: {v}")

    lines.extend([
        "",
        "2. VALIDATION SET EVALUATION METRICS",
        "------------------------------------",
        f"Accuracy: {val_metrics['accuracy']:.6f}",
        f"Macro Precision: {val_metrics['precision_macro']:.6f}",
        f"Weighted Precision: {val_metrics['precision_weighted']:.6f}",
        f"Macro Recall: {val_metrics['recall_macro']:.6f}",
        f"Weighted Recall: {val_metrics['recall_weighted']:.6f}",
        f"Macro F1 Score: {val_metrics['f1_macro']:.6f}",
        f"Weighted F1 Score: {val_metrics['f1_weighted']:.6f}",
        f"Macro One-vs-Rest ROC-AUC: {val_metrics['roc_auc_macro']:.6f}",
        f"Weighted One-vs-Rest ROC-AUC: {val_metrics['roc_auc_weighted']:.6f}",
        "",
        "3. TEST SET EVALUATION METRICS",
        "------------------------------",
        f"Accuracy: {test_metrics['accuracy']:.6f}",
        f"Macro Precision: {test_metrics['precision_macro']:.6f}",
        f"Weighted Precision: {test_metrics['precision_weighted']:.6f}",
        f"Macro Recall: {test_metrics['recall_macro']:.6f}",
        f"Weighted Recall: {test_metrics['recall_weighted']:.6f}",
        f"Macro F1 Score: {test_metrics['f1_macro']:.6f}",
        f"Weighted F1 Score: {test_metrics['f1_weighted']:.6f}",
        f"Macro One-vs-Rest ROC-AUC: {test_metrics['roc_auc_macro']:.6f}",
        f"Weighted One-vs-Rest ROC-AUC: {test_metrics['roc_auc_weighted']:.6f}",
        "",
        "4. PER-CLASS TEST PERFORMANCE BREAKDOWN",
        "--------------------------------------",
    ])

    headers = f"{'Class Label':<22}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'ROC-AUC':<12}"
    lines.append(headers)
    lines.append("-" * 70)

    for cname, pdict in test_metrics["per_class"].items():
        lines.append(
            f"{cname:<22}{pdict['precision']:<12.4f}{pdict['recall']:<12.4f}"
            f"{pdict['f1_score']:<12.4f}{pdict['roc_auc']:<12.4f}"
        )

    lines.extend([
        "",
        "5. TOP 20 MOST IMPORTANT FEATURES",
        "---------------------------------",
    ])

    for rank, (_, row) in enumerate(top_20.iterrows(), 1):
        lines.append(f"  {rank:2d}. {row['feature']:<35}: {row['importance']:.6f}")

    lines.append("\n==============================================")
    return "\n".join(lines)


def main() -> None:
    """Main execution function to orchestrate the XGBoost baseline pipeline."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "xgboost.log"

    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Starting XGBoost Baseline Training & Evaluation pipeline...")

    try:
        # 1. Load data and artifacts
        (
            X_train, X_valid, X_test,
            y_train, y_valid, y_test,
            label_encoder, class_weights, label_mapping
        ) = load_preprocessed_data()

        class_names = list(label_encoder.classes_)
        num_class = len(class_names)

        # 2. Hyperparameter tuning on 250k stratified subset
        best_params, best_cv_score, tuning_time, tuning_samples = tune_hyperparameters(
            X_train, y_train, num_class, tuning_size=250000
        )

        # 3. Retrain final XGBoost model on complete training dataset (1.76M samples)
        final_xgb, train_time = train_final_model(
            X_train, y_train, num_class, best_params
        )

        # 4. Evaluate on validation and test sets
        logger.info("Evaluating XGBoost model on validation split...")
        val_metrics, val_inf_time, _, _ = evaluate_model(final_xgb, X_valid, y_valid, class_names)

        logger.info("Evaluating XGBoost model on test split...")
        test_metrics, test_inf_time, y_test_pred, y_test_proba = evaluate_model(
            final_xgb, X_test, y_test, class_names
        )

        # 5. Generate plots
        generate_plots(
            final_xgb, X_test, y_test, y_test_pred, y_test_proba, class_names, PLOTS_DIR
        )

        # 6. Save artifacts and metrics
        save_artifacts_and_metrics(
            final_xgb, best_params, best_cv_score, tuning_time, train_time,
            tuning_samples, len(X_train), val_metrics, test_metrics, y_test, y_test_pred, class_names
        )

        # 7. Generate and save text report
        report_text = generate_report(
            final_xgb, best_params, best_cv_score, tuning_time, train_time,
            tuning_samples, len(X_train), val_metrics, test_metrics, list(X_train.columns)
        )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "xgboost_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("XGBoost Baseline pipeline completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during XGBoost baseline pipeline execution:")
        raise e


if __name__ == "__main__":
    main()
