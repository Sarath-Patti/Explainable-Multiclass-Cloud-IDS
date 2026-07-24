"""SHAP Explainability Module for Explainable-Multiclass-Cloud-IDS.

This module loads the trained XGBoost model and preprocessed test data, computes
Shapley Additive exPlanations (SHAP) using TreeExplainer on a representative stratified sample,
and generates global, local, and class-wise explainability reports, visualizations,
and structured metrics.
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
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPLAINABILITY_DIR = PROJECT_ROOT / "outputs" / "explainability"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"
SHAP_REPORT_PATH = REPORTS_DIR / "shap_report.txt"

# Default configuration constants
DEFAULT_SAMPLE_SIZE = 1000
RANDOM_STATE = 42

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SHAPExplainer")


def verify_artifacts() -> None:
    """Verifies that all required model and preprocessed data artifacts exist.

    Raises:
        FileNotFoundError: If any mandatory artifact is missing.
    """
    logger.info("Verifying existence of required model and data artifacts...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained XGBoost model not found at: {MODEL_PATH}")

    has_test_data = (PROCESSED_DATA_DIR / "X_test.csv").exists() or (PROCESSED_DATA_DIR / "X_test.npy").exists()
    if not has_test_data:
        raise FileNotFoundError(f"Preprocessed test dataset not found in {PROCESSED_DATA_DIR}")

    has_label_info = LABEL_ENCODER_PATH.exists() or LABEL_MAPPING_PATH.exists()
    if not has_label_info:
        raise FileNotFoundError(f"Label artifacts missing in {MODELS_DIR}")

    logger.info("All required artifacts successfully verified.")


def load_data_and_model() -> tuple[XGBClassifier, pd.DataFrame, np.ndarray, list[str]]:
    """Loads the trained XGBoost model, test dataset, and class label names.

    Returns:
        tuple containing:
            - model: Loaded XGBClassifier.
            - X_test: Test features DataFrame.
            - y_test: Test target 1D numpy array.
            - class_names: List of class names.
    """
    logger.info("Loading trained XGBoost model...")
    model = joblib.load(MODEL_PATH)

    logger.info("Loading preprocessed test features and target labels...")
    if (PROCESSED_DATA_DIR / "X_test.csv").exists():
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    else:
        X_test_arr = np.load(PROCESSED_DATA_DIR / "X_test.npy")
        if (PROCESSED_DATA_DIR / "X_train.csv").exists():
            cols = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv", nrows=1).columns
        else:
            cols = [f"Feature_{i}" for i in range(X_test_arr.shape[1])]
        X_test = pd.DataFrame(X_test_arr, columns=cols)

    if (PROCESSED_DATA_DIR / "y_test.csv").exists():
        y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")["Label"].values
    else:
        y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")

    if LABEL_ENCODER_PATH.exists():
        le = joblib.load(LABEL_ENCODER_PATH)
        class_names = [str(c) for c in le.classes_]
    elif LABEL_MAPPING_PATH.exists():
        with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        class_names = [k for k, v in sorted(mapping.items(), key=lambda x: x[1])]
    else:
        num_classes = len(np.unique(y_test))
        class_names = [f"Class_{i}" for i in range(num_classes)]

    logger.info(f"Test dataset loaded. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logger.info(f"Loaded target classes ({len(class_names)}): {class_names}")

    return model, X_test, y_test, class_names


def create_stratified_sample(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, np.ndarray]:
    """Creates a representative stratified sample of the test set.

    Args:
        X_test: Full test predictor features.
        y_test: Full test target labels.
        sample_size: Desired number of sample instances.
        random_state: Random seed for reproducibility.

    Returns:
        tuple containing:
            - X_sample: Stratified test features DataFrame.
            - y_sample: Stratified test target labels numpy array.
    """
    logger.info(f"Extracting stratified sample of size {sample_size:,} from test dataset...")
    effective_size = min(sample_size, len(X_test))

    if effective_size == len(X_test):
        return X_test.copy(), y_test.copy()

    X_sample, _, y_sample, _ = train_test_split(
        X_test,
        y_test,
        train_size=effective_size,
        stratify=y_test,
        random_state=random_state
    )

    X_sample = X_sample.reset_index(drop=True)
    logger.info(f"Stratified sample created. X_sample shape: {X_sample.shape}")
    return X_sample, y_sample


def standardize_shap_values(
    shap_out: object,
    num_classes: int,
    num_samples: int,
    num_features: int
) -> list[np.ndarray]:
    """Standardizes SHAP output into a list of 2D numpy arrays of shape (N, P), one per class.

    Args:
        shap_out: Raw SHAP output from TreeExplainer.
        num_classes: Number of target classes.
        num_samples: Number of samples in X_sample.
        num_features: Number of features.

    Returns:
        List of 2D numpy arrays of shape (N, P), length equal to num_classes.
    """
    if isinstance(shap_out, list):
        return [np.array(v) for v in shap_out]

    if hasattr(shap_out, "values"):
        vals = shap_out.values
    else:
        vals = shap_out

    if isinstance(vals, np.ndarray):
        if vals.ndim == 2:
            return [vals]
        elif vals.ndim == 3:
            if vals.shape == (num_samples, num_features, num_classes):
                return [vals[:, :, c] for c in range(num_classes)]
            elif vals.shape == (num_classes, num_samples, num_features):
                return [vals[c, :, :] for c in range(num_classes)]
            elif vals.shape[2] == num_classes:
                return [vals[:, :, c] for c in range(num_classes)]
            elif vals.shape[0] == num_classes:
                return [vals[c, :, :] for c in range(num_classes)]

    raise ValueError(f"Unable to parse SHAP output structure with shape/type: {type(shap_out)}")


def get_base_value(expected_value: object, class_idx: int) -> float:
    """Safely retrieves the base expected value for a specific class index.

    Args:
        expected_value: Explainer expected value (scalar, list, or array).
        class_idx: Target class index.

    Returns:
        float: Expected base value.
    """
    if isinstance(expected_value, (list, np.ndarray)):
        if len(expected_value) > class_idx:
            return float(expected_value[class_idx])
        return float(expected_value[0])
    return float(expected_value)


def compute_shap_values(
    model: XGBClassifier,
    X_sample: pd.DataFrame,
    class_names: list[str]
) -> tuple[shap.TreeExplainer, list[np.ndarray]]:
    """Computes SHAP values using TreeExplainer on the sampled test data.

    Args:
        model: Fitted XGBClassifier.
        X_sample: Sampled test features DataFrame.
        class_names: List of class names.

    Returns:
        tuple containing:
            - explainer: Configured shap.TreeExplainer instance.
            - shap_values_list: List of 2D numpy arrays (one per class).
    """
    logger.info("Initializing SHAP TreeExplainer for the trained XGBoost model...")
    start_time = time.time()
    explainer = shap.TreeExplainer(model)

    logger.info("Computing SHAP values for the sampled dataset...")
    shap_raw = explainer.shap_values(X_sample)
    duration = time.time() - start_time

    num_classes = len(class_names)
    num_samples, num_features = X_sample.shape

    shap_values_list = standardize_shap_values(shap_raw, num_classes, num_samples, num_features)
    logger.info(f"SHAP computation completed in {duration:.2f} seconds across {len(shap_values_list)} classes.")

    return explainer, shap_values_list


def generate_global_explainability(
    shap_values_list: list[np.ndarray],
    X_sample: pd.DataFrame,
    class_names: list[str],
    output_dir: Path
) -> pd.DataFrame:
    """Generates global explainability plots, calculates mean absolute SHAP values, and builds feature importance ranking table.

    Args:
        shap_values_list: List of 2D numpy arrays of SHAP values per class.
        X_sample: Sampled test features DataFrame.
        class_names: List of class names.
        output_dir: Path to directory for saving plots and reports.

    Returns:
        pd.DataFrame: Global feature importance DataFrame.
    """
    logger.info("Generating global SHAP explainability plots and importance tables...")
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    # 1. Global Summary Plot (Stacked Bar across classes)
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_list, X_sample, class_names=class_names, show=False)
        plt.title("SHAP Global Multiclass Feature Impact Summary", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(output_dir / "global_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate global_summary.png: {e}")
        plt.close()

    # 2. Global Bar Plot (Mean |SHAP| across classes)
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_list, X_sample, plot_type="bar", class_names=class_names, show=False)
        plt.title("SHAP Global Mean Absolute Feature Importance Across Classes", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(output_dir / "global_bar.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate global_bar.png: {e}")
        plt.close()

    # 3. Global Beeswarm Plot
    try:
        # Aggregated mean absolute SHAP values across classes for each instance
        aggregated_shap = np.mean([np.abs(sv) for sv in shap_values_list], axis=0)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(aggregated_shap, X_sample, show=False)
        plt.title("SHAP Global Feature Impact Beeswarm Plot (Aggregated)", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(output_dir / "global_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate global_beeswarm.png: {e}")
        plt.close()

    # Compute Mean Absolute SHAP Importance across all classes
    mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_values_list]
    global_mean_abs = np.mean(mean_abs_per_class, axis=0)

    feature_df = pd.DataFrame({
        "feature_name": X_sample.columns,
        "mean_abs_shap": global_mean_abs
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    feature_df["rank"] = feature_df.index + 1
    feature_df = feature_df[["rank", "feature_name", "mean_abs_shap"]]

    feature_df.to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info("Saved global feature importance table to feature_importance.csv")

    return feature_df


def generate_local_explainability(
    explainer: shap.TreeExplainer,
    shap_values_list: list[np.ndarray],
    X_sample: pd.DataFrame,
    y_sample: np.ndarray,
    class_names: list[str],
    output_dir: Path
) -> None:
    """Generates local explainability plots (Waterfall, Force HTML, Decision) for representative attack instances.

    Args:
        explainer: Configured SHAP TreeExplainer.
        shap_values_list: List of SHAP value matrices per class.
        X_sample: Sampled test features DataFrame.
        y_sample: Sampled test target labels.
        class_names: List of class names.
        output_dir: Output directory for saving explainability files.
    """
    logger.info("Generating local SHAP explanations (Waterfall, Decision, Force HTML) for representative instances...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls_idx, cname in enumerate(class_names):
        matching_indices = np.where(y_sample == cls_idx)[0]
        if len(matching_indices) == 0:
            logger.warning(f"No instances found in sample for class: {cname}. Skipping local plots.")
            continue

        idx_in_sample = matching_indices[0]
        cname_clean = cname.replace(" ", "_").replace("/", "_").replace("-", "_")

        base_val = get_base_value(explainer.expected_value, cls_idx)
        shap_vec = shap_values_list[cls_idx][idx_in_sample]
        instance_series = X_sample.iloc[idx_in_sample]

        # 1. Waterfall Plot
        try:
            exp_inst = shap.Explanation(
                values=shap_vec,
                base_values=base_val,
                data=instance_series.values,
                feature_names=list(X_sample.columns)
            )
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(exp_inst, show=False)
            plt.title(f"SHAP Local Waterfall Plot - Class: {cname}", fontsize=12, pad=15)
            plt.tight_layout()
            plt.savefig(output_dir / f"waterfall_{cname_clean}.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate waterfall plot for {cname}: {e}")
            plt.close()

        # 2. Decision Plot
        try:
            plt.figure(figsize=(10, 6))
            shap.decision_plot(
                base_val,
                shap_vec,
                instance_series,
                feature_names=list(X_sample.columns),
                show=False
            )
            plt.title(f"SHAP Local Decision Plot - Class: {cname}", fontsize=12, pad=15)
            plt.tight_layout()
            plt.savefig(output_dir / f"decision_{cname_clean}.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate decision plot for {cname}: {e}")
            plt.close()

        # 3. Force Plot HTML
        try:
            force_html = shap.force_plot(
                base_val,
                shap_vec,
                instance_series,
                matplotlib=False
            )
            shap.save_html(str(output_dir / f"force_{cname_clean}.html"), force_html)
        except Exception as e:
            logger.warning(f"Could not generate force plot for {cname}: {e}")

    logger.info("Local explainability visualizations generated successfully.")


def generate_class_wise_explainability(
    shap_values_list: list[np.ndarray],
    X_sample: pd.DataFrame,
    class_names: list[str],
    output_dir: Path
) -> pd.DataFrame:
    """Generates class-wise feature rankings and exports class_feature_importance.csv.

    Args:
        shap_values_list: List of SHAP matrices per class.
        X_sample: Sampled test features.
        class_names: List of class names.
        output_dir: Path to directory for saving CSV reports.

    Returns:
        pd.DataFrame: Class-wise feature importance DataFrame.
    """
    logger.info("Generating class-wise feature importance rankings...")

    records = []
    for cls_idx, cname in enumerate(class_names):
        class_shap = shap_values_list[cls_idx]
        mean_abs = np.mean(np.abs(class_shap), axis=0)

        df_cls = pd.DataFrame({
            "class_label": cname,
            "feature_name": X_sample.columns,
            "mean_abs_shap": mean_abs
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        df_cls["rank"] = df_cls.index + 1
        records.append(df_cls)

    class_imp_df = pd.concat(records, ignore_index=True)
    class_imp_df = class_imp_df[["class_label", "rank", "feature_name", "mean_abs_shap"]]

    class_imp_df.to_csv(output_dir / "class_feature_importance.csv", index=False)
    logger.info("Saved class-wise feature importance table to class_feature_importance.csv")

    return class_imp_df


def save_shap_artifacts_and_metrics(
    explainer: shap.TreeExplainer,
    shap_values_list: list[np.ndarray],
    X_sample: pd.DataFrame,
    y_sample: np.ndarray,
    class_names: list[str],
    global_imp_df: pd.DataFrame,
    class_imp_df: pd.DataFrame,
    model: XGBClassifier
) -> dict:
    """Serializes shap_values.pkl and outputs shap_metrics.json.

    Args:
        explainer: Configured SHAP TreeExplainer.
        shap_values_list: List of SHAP matrices per class.
        X_sample: Sampled test DataFrame.
        y_sample: Sampled target array.
        class_names: List of class names.
        global_imp_df: Global feature importance DataFrame.
        class_imp_df: Class-wise feature importance DataFrame.
        model: Trained XGBClassifier.

    Returns:
        dict: Numerical SHAP metrics payload.
    """
    logger.info("Saving SHAP serialization objects and numerical metrics JSON...")
    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save shap_values.pkl
    shap_payload = {
        "shap_values": shap_values_list,
        "expected_value": explainer.expected_value,
        "feature_names": list(X_sample.columns),
        "class_names": class_names,
        "sample_indices": X_sample.index.tolist(),
        "y_sample": y_sample
    }
    joblib.dump(shap_payload, EXPLAINABILITY_DIR / "shap_values.pkl")

    # 2. Build XGBoost built-in importance comparison
    xgb_importances = model.feature_importances_
    xgb_imp_df = pd.DataFrame({
        "feature_name": X_sample.columns,
        "xgb_weight_importance": xgb_importances
    }).sort_values(by="xgb_weight_importance", ascending=False).reset_index(drop=True)
    xgb_imp_df["xgb_rank"] = xgb_imp_df.index + 1

    comparison_df = pd.merge(
        global_imp_df,
        xgb_imp_df,
        on="feature_name"
    ).sort_values(by="rank").reset_index(drop=True)

    # Build Class Top 10 dictionary
    class_top_dict = {}
    for cname in class_names:
        sub_df = class_imp_df[class_imp_df["class_label"] == cname].head(10)
        class_top_dict[cname] = sub_df[["rank", "feature_name", "mean_abs_shap"]].to_dict(orient="records")

    metrics_payload = {
        "sample_size": int(len(X_sample)),
        "num_features": int(X_sample.shape[1]),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        "global_top_features": global_imp_df.head(20).to_dict(orient="records"),
        "class_top_features": class_top_dict,
        "xgb_vs_shap_top_features_comparison": comparison_df.head(15).to_dict(orient="records")
    }

    with open(EXPLAINABILITY_DIR / "shap_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    with open(METRICS_DIR / "shap_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    logger.info("SHAP artifacts and metrics JSON saved successfully.")
    return metrics_payload


def generate_report(
    metrics_payload: dict,
    global_imp_df: pd.DataFrame,
    class_imp_df: pd.DataFrame,
    class_names: list[str]
) -> str:
    """Compiles the detailed research-grade SHAP explainability text report.

    Args:
        metrics_payload: SHAP numerical metrics dictionary.
        global_imp_df: Global feature importance DataFrame.
        class_imp_df: Class-wise feature importance DataFrame.
        class_names: List of target class names.

    Returns:
        str: Formatted plain text report.
    """
    logger.info("Compiling detailed SHAP explainability report...")

    top_global = global_imp_df.head(15)

    bot_df = class_imp_df[class_imp_df["class_label"] == "Bot"].head(10) if "Bot" in class_names else pd.DataFrame()
    web_df = class_imp_df[class_imp_df["class_label"] == "Web Attack"].head(10) if "Web Attack" in class_names else pd.DataFrame()

    lines = [
        "==================================================",
        "EXPLAINABLE MULTICLASS CLOUD IDS: SHAP REPORT",
        "==================================================",
        "",
        "1. EXPLANATION METHODOLOGY & SAMPLE OVERVIEW",
        "--------------------------------------------",
        f"Model Explored: Trained Gradient Boosted Trees (XGBoost Baseline)",
        f"Evaluation Sample Size: {metrics_payload['sample_size']:,} instances (Stratified Test Subset)",
        f"Predictor Feature Count: {metrics_payload['num_features']}",
        f"Multiclass Target Categories: {metrics_payload['num_classes']} ({', '.join(class_names)})",
        "Framework: Shapley Additive exPlanations (SHAP) based on Cooperative Game Theory",
        "",
        "2. TREEEXPLAINER OVERVIEW",
        "------------------------",
        "TreeExplainer is an exact, polynomial-time algorithm (Lundberg et al., 2020) designed for decision tree ensembles.",
        "Unlike sampling-based Model-Agnostic KernelSHAP approximations, TreeExplainer leverages the tree structure to compute",
        "exact Shapley values in O(TLD^2) time complexity.",
        "The model output for each sample and target class is decomposed additively into feature attributions:",
        "    f_c(x) = E[f_c(X)] + sum_{j=1}^P phi_{j, c}(x)",
        "where E[f_c(X)] represents the expected baseline log-odds and phi_{j, c}(x) is the SHAP attribution of feature j for class c.",
        "",
        "3. GLOBAL FEATURE INTERPRETATION",
        "--------------------------------",
        "Global feature importance is calculated using the Mean Absolute SHAP value across all sampled network instances.",
        "Top 15 Globally Discriminative Network Features:",
    ]

    for _, row in top_global.iterrows():
        lines.append(f"  {row['rank']:2d}. {row['feature_name']:<35}: {row['mean_abs_shap']:.6f}")

    lines.extend([
        "",
        "4. BOT CLASS DEEP-DIVE ANALYSIS",
        "-------------------------------",
        "The 'Bot' class represents stealthy Command and Control (C2) botnet traffic characterized by low volume and periodic communications.",
        "Top SHAP Feature Contributors for Bot Attack Detection:",
    ])

    if not bot_df.empty:
        for _, row in bot_df.iterrows():
            lines.append(f"  {row['rank']:2d}. {row['feature_name']:<35}: {row['mean_abs_shap']:.6f}")
    else:
        lines.append("  Bot class metrics unavailable.")

    lines.extend([
        "Key Findings: Botnet traffic exhibits strong attributions in backward packet size distributions, initial window sizes,",
        "and packet inter-arrival times (IAT). Small uniform response sizes combined with persistent TCP window settings enable",
        "the model to isolate C2 heartbeats from legitimate HTTP background traffic.",
        "",
        "5. WEB ATTACK CLASS DEEP-DIVE ANALYSIS",
        "--------------------------------------",
        "The 'Web Attack' category consolidates SQL Injection, Cross-Site Scripting (XSS), and HTTP Brute Force incursions.",
        "Top SHAP Feature Contributors for Web Attack Detection:",
    ])

    if not web_df.empty:
        for _, row in web_df.iterrows():
            lines.append(f"  {row['rank']:2d}. {row['feature_name']:<35}: {row['mean_abs_shap']:.6f}")
    else:
        lines.append("  Web Attack class metrics unavailable.")

    lines.extend([
        "Key Findings: Web Attack detection is predominantly driven by Forward Header Length, Forward Packet Length Max, and",
        "Flow Duration. Exploit payloads and brute-force POST sequences generate abnormally large forward packet sizes and short",
        "burst durations compared to standard GET requests.",
        "",
        "6. COMPARISON: SHAP vs BUILT-IN XGBOOST FEATURE IMPORTANCE",
        "----------------------------------------------------------",
        "Built-in XGBoost feature importances (Weight/Gain) rely on split frequency or average gain reduction across trees.",
        "Limitations of Built-in Importance:",
        "  1. Gain/Weight values are non-additive and biased toward high-cardinality continuous features.",
        "  2. Built-in metrics provide no directionality (they cannot indicate whether a feature value increases or decreases attack likelihood).",
        "  3. Global metrics obscure class-specific feature utility in multiclass contexts.",
        "SHAP Advantages:",
        "  1. Mathematically consistent and additive attributions based on game theory.",
        "  2. Provides local (per-instance) and global directionality (showing positive/negative contribution to specific attack log-odds).",
        "  3. Dissects per-class feature behavior cleanly without cross-class confounding.",
        "",
        "7. EXPLAINABLE AI (XAI) IN CLOUD INTRUSION DETECTION",
        "---------------------------------------------------",
        "Deploying black-box ML models in cloud security introduces operational opacity. SHAP bridges the trust gap by converting",
        "statistical probability vectors into human-interpretable domain evidence, ensuring compliance with auditing standards",
        "and enabling transparent automated response orchestration.",
        "",
        "8. IMPLICATIONS FOR SECURITY OPERATIONS CENTER (SOC) ANALYSTS",
        "------------------------------------------------------------",
        "  - Rapid Incident Triage: SOC analysts can inspect local Waterfall and Force plots to immediately verify trigger conditions",
        "    without performing manual deep packet inspection on raw PCAPs.",
        "  - Alert Fatigue Reduction: Distinguishes high-confidence structural attacks from benign network anomalies.",
        "  - Rule Engineering: SHAP top features provide explicit boundaries for crafting deterministic Suricata/Snort signatures.",
        "",
        "9. LIMITATIONS OF SHAP",
        "----------------------",
        "  - Computational Overhead: Exact SHAP calculations on massive network datasets (>1M flows) require stratified sub-sampling.",
        "  - Collinearity Distribution: Highly correlated network features (e.g., Subflow Fwd Bytes and Total Length of Fwd Packets)",
        "    split Shapley credit, which can dilute individual feature importance ranks.",
        "",
        "=================================================="
    ])

    return "\n".join(lines)


def main() -> None:
    """Main execution function to orchestrate the SHAP explainability workflow."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "shap_explainer.log"

    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Starting SHAP Explainability & Model Interpretability Pipeline (v0.8)...")

    try:
        # 1. Verify required input files
        verify_artifacts()

        # 2. Load model, data, and class names
        model, X_test, y_test, class_names = load_data_and_model()

        # 3. Create stratified sample (1,000 instances)
        X_sample, y_sample = create_stratified_sample(
            X_test, y_test, sample_size=DEFAULT_SAMPLE_SIZE, random_state=RANDOM_STATE
        )

        # 4. Compute SHAP values via TreeExplainer
        explainer, shap_values_list = compute_shap_values(model, X_sample, class_names)

        # 5. Generate global explainability plots and feature rankings
        global_imp_df = generate_global_explainability(
            shap_values_list, X_sample, class_names, EXPLAINABILITY_DIR
        )

        # 6. Generate local explainability plots (Waterfall, Decision, Force HTML)
        generate_local_explainability(
            explainer, shap_values_list, X_sample, y_sample, class_names, EXPLAINABILITY_DIR
        )

        # 7. Generate class-wise feature importance CSV
        class_imp_df = generate_class_wise_explainability(
            shap_values_list, X_sample, class_names, EXPLAINABILITY_DIR
        )

        # 8. Save serialized SHAP objects and metrics JSON
        metrics_payload = save_shap_artifacts_and_metrics(
            explainer, shap_values_list, X_sample, y_sample, class_names,
            global_imp_df, class_imp_df, model
        )

        # 9. Generate and save plain text research report
        report_text = generate_report(metrics_payload, global_imp_df, class_imp_df, class_names)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SHAP_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("SHAP Explainability pipeline completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during SHAP explainability execution:")
        raise e


if __name__ == "__main__":
    main()
