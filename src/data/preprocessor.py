"""Dataset Preprocessing & Prepartion Module for Explainable-Multiclass-Cloud-IDS.

This module separates features and target labels, encodes target classes, splits the
dataset into stratified train, validation, and test subsets, computes class weights
for tree-based model learning, and outputs serialization artifacts and validation reports.
"""

import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

PREPROCESSING_REPORT_PATH = REPORTS_DIR / "preprocessing_report.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Preprocessor")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Loads the cleaned dataset from a CSV file.

    Args:
        file_path: Path to the clean CSV file.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {file_path}")
    logger.info(f"Loading cleaned dataset from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset loaded. Shape: {df.shape}")
    return df


def separate_features_and_target(df: pd.DataFrame, target_col: str = "Label") -> tuple[pd.DataFrame, pd.Series]:
    """Separates the dataset into predictor features (X) and target labels (y).

    Args:
        df: The dataset DataFrame.
        target_col: Name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: X and y.
    """
    logger.info(f"Separating features and target column: {target_col}...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder, dict]:
    """Encodes target labels using LabelEncoder.

    Args:
        y: Target label Series.

    Returns:
        tuple[np.ndarray, LabelEncoder, dict]: Encoded labels, fitted LabelEncoder, and mapping.
    """
    logger.info("Encoding target labels using LabelEncoder...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Generate label mapping
    mapping = {str(label): int(idx) for idx, label in enumerate(le.classes_)}
    logger.info(f"Encoded class mapping: {mapping}")
    return y_encoded, le, mapping


def split_dataset(
    X: pd.DataFrame,
    y: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits the dataset into stratified train, validation, and test splits (70/15/15).

    Args:
        X: Predictor features.
        y: Encoded target labels.

    Returns:
        tuple containing:
            - X_train, X_valid, X_test
            - y_train, y_valid, y_test
    """
    logger.info("Splitting dataset into stratified train, validation, and test splits...")

    # Stage 1: split into train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Stage 2: split temp into validation (15% of total) and test (15% of total)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    logger.info("Split completed:")
    logger.info(f"  Train: X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"  Valid: X_valid={X_valid.shape}, y_valid={y_valid.shape}")
    logger.info(f"  Test:  X_test={X_test.shape}, y_test={y_test.shape}")

    # Convert y splits to pandas Series to keep things consistent
    y_train = pd.Series(y_train, name="Label", index=X_train.index)
    y_valid = pd.Series(y_valid, name="Label", index=X_valid.index)
    y_test = pd.Series(y_test, name="Label", index=X_test.index)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def compute_class_weights(y_train: pd.Series) -> dict:
    """Computes balanced class weights from the training labels.

    Args:
        y_train: Encoded training target labels.

    Returns:
        dict: Mapping of class index to class weight.
    """
    logger.info("Computing balanced class weights from training labels...")
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.values
    )

    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    logger.info(f"Computed class weights: {class_weights}")
    return class_weights


def validate_splits(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    num_classes: int
) -> dict:
    """Validates the split datasets.

    Args:
        X_train: Training features.
        X_valid: Validation features.
        X_test: Test features.
        y_train: Training labels.
        y_valid: Validation labels.
        y_test: Test labels.
        num_classes: Expected number of classes.

    Returns:
        dict: Validation results.
    """
    logger.info("Validating splits...")
    results = {}

    # 1. Feature counts
    feats_train = X_train.shape[1]
    feats_valid = X_valid.shape[1]
    feats_test = X_test.shape[1]
    identical_features = (feats_train == feats_valid == feats_test)
    results["identical_features"] = identical_features

    # 2. No missing values
    missing_train = int(X_train.isna().sum().sum() + y_train.isna().sum())
    missing_valid = int(X_valid.isna().sum().sum() + y_valid.isna().sum())
    missing_test = int(X_test.isna().sum().sum() + y_test.isna().sum())
    no_missing = (missing_train == 0 and missing_valid == 0 and missing_test == 0)
    results["no_missing"] = no_missing

    # 3. No infinite values
    def count_infs(df: pd.DataFrame) -> int:
        num_cols = df.select_dtypes(include=[np.number])
        if num_cols.empty:
            return 0
        return int(np.isinf(num_cols).values.sum())

    inf_train = count_infs(X_train)
    inf_valid = count_infs(X_valid)
    inf_test = count_infs(X_test)
    no_infinite = (inf_train == 0 and inf_valid == 0 and inf_test == 0)
    results["no_infinite"] = no_infinite

    # 4. All classes present in each split
    classes_train = set(y_train.unique())
    classes_valid = set(y_valid.unique())
    classes_test = set(y_test.unique())
    all_classes_present = (
        len(classes_train) == num_classes and
        len(classes_valid) == num_classes and
        len(classes_test) == num_classes
    )
    results["all_classes_present"] = all_classes_present

    # 5. Stratification preserved
    dist_train = y_train.value_counts(normalize=True).sort_index().to_dict()
    dist_valid = y_valid.value_counts(normalize=True).sort_index().to_dict()
    dist_test = y_test.value_counts(normalize=True).sort_index().to_dict()

    max_diff_valid = max(abs(dist_train.get(c, 0) - dist_valid.get(c, 0)) for c in range(num_classes))
    max_diff_test = max(abs(dist_train.get(c, 0) - dist_test.get(c, 0)) for c in range(num_classes))
    stratification_preserved = (max_diff_valid < 0.01 and max_diff_test < 0.01)
    results["stratification_preserved"] = stratification_preserved
    results["class_distributions"] = {
        "train": dist_train,
        "valid": dist_valid,
        "test": dist_test
    }

    logger.info("Validation Results:")
    logger.info(f"  Identical feature counts: {identical_features} ({feats_train})")
    logger.info(f"  No missing values: {no_missing}")
    logger.info(f"  No infinite values: {no_infinite}")
    logger.info(f"  All classes present: {all_classes_present}")
    logger.info(f"  Stratification preserved: {stratification_preserved} (max_diff_valid={max_diff_valid:.6f}, max_diff_test={max_diff_test:.6f})")

    return results


def save_processed_data(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Saves split datasets to CSV files.

    Args:
        X_train, X_valid, X_test: Feature splits.
        y_train, y_valid, y_test: Target splits.
        output_dir: Path to directory to save files.
    """
    logger.info(f"Saving processed splits to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_valid.to_csv(output_dir / "X_valid.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_valid.to_csv(output_dir / "y_valid.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    logger.info("Processed datasets saved successfully.")


def save_artifacts(
    le: LabelEncoder,
    class_weights: dict,
    label_mapping: dict,
    models_dir: Path
) -> None:
    """Saves preprocessing artifacts (label encoder, class weights, label mapping).

    Args:
        le: Fitted LabelEncoder.
        class_weights: Dictionary of class weights.
        label_mapping: Label to index mapping dict.
        models_dir: Path to models directory.
    """
    logger.info(f"Saving preprocessing artifacts to {models_dir}...")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save label encoder
    joblib.dump(le, models_dir / "label_encoder.pkl")

    # Save class weights
    joblib.dump(class_weights, models_dir / "class_weights.pkl")

    # Save label mapping
    with open(models_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=4)

    logger.info("Artifacts saved successfully.")


def generate_report(
    original_size: int,
    split_sizes: dict,
    feature_count: int,
    class_distributions: dict,
    label_mapping: dict,
    class_weights: dict,
    validation_results: dict
) -> str:
    """Generates the text preprocessing report.

    Args:
        original_size: Number of samples in clean dataset.
        split_sizes: Sizes of train, validation, and test splits.
        feature_count: Number of predictor features.
        class_distributions: Class distributions per split.
        label_mapping: Class label to index mapping.
        class_weights: Computed class weights.
        validation_results: Split validation results.

    Returns:
        str: Formatted report text.
    """
    logger.info("Generating preprocessing report...")

    idx_to_label = {v: k for k, v in label_mapping.items()}

    lines = [
        "==================================================",
        "CICIDS2017 DATA PREPROCESSING & PREPARATION REPORT",
        "==================================================",
        "",
        "1. DATASET SPLIT DETAILS",
        "------------------------",
        f"Original Dataset Size: {original_size:,} samples",
        f"Predictor Features: {feature_count}",
        f"Train Split Size: {split_sizes['train']:,} samples ({split_sizes['train_pct']:.2f}%)",
        f"Validation Split Size: {split_sizes['valid']:,} samples ({split_sizes['valid_pct']:.2f}%)",
        f"Test Split Size: {split_sizes['test']:,} samples ({split_sizes['test_pct']:.2f}%)",
        "",
        "2. ENCODED CLASS MAPPING",
        "------------------------",
    ]

    for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        lines.append(f"  - {idx}: {label}")

    lines.extend([
        "",
        "3. COMPUTED CLASS WEIGHTS (balanced)",
        "------------------------------------",
    ])

    for idx, weight in sorted(class_weights.items()):
        class_name = idx_to_label[idx]
        lines.append(f"  - Class {idx} ({class_name}): {weight:.6f}")

    lines.extend([
        "",
        "4. CLASS DISTRIBUTION PER SPLIT (Proportions)",
        "---------------------------------------------",
    ])

    headers = f"{'Class ID':<10}{'Class Label':<22}{'Train %':<12}{'Valid %':<12}{'Test %':<12}"
    lines.append(headers)
    lines.append("-" * 68)

    for idx in sorted(idx_to_label.keys()):
        class_name = idx_to_label[idx]
        tr_pct = class_distributions["train"].get(idx, 0.0) * 100
        val_pct = class_distributions["valid"].get(idx, 0.0) * 100
        te_pct = class_distributions["test"].get(idx, 0.0) * 100
        lines.append(
            f"{idx:<10}{class_name:<22}{tr_pct:<12.4f}{val_pct:<12.4f}{te_pct:<12.4f}"
        )

    lines.extend([
        "",
        "5. PREPROCESSING VALIDATION RESULTS",
        "-----------------------------------",
        f"  - Identical feature counts: {validation_results['identical_features']}",
        f"  - No missing values: {validation_results['no_missing']}",
        f"  - No infinite values: {validation_results['no_infinite']}",
        f"  - All classes present in splits: {validation_results['all_classes_present']}",
        f"  - Stratification check (< 1% variance): {validation_results['stratification_preserved']}",
        "",
        "Validation Status: SUCCESS" if all([
            validation_results['identical_features'],
            validation_results['no_missing'],
            validation_results['no_infinite'],
            validation_results['all_classes_present'],
            validation_results['stratification_preserved']
        ]) else "Validation Status: FAILED",
        "=================================================="
    ])

    return "\n".join(lines)


def main() -> None:
    """Main execution function to orchestrate the preprocessing workflow."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "preprocessor.log"

    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Starting CICIDS2017 dataset Preprocessing & Preparation pipeline...")

    try:
        df = load_dataset(CLEANED_DATA_PATH)
        original_size = len(df)

        X, y = separate_features_and_target(df)
        feature_count = X.shape[1]

        y_encoded, le, label_mapping = encode_labels(y)
        num_classes = len(le.classes_)

        X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(X, y_encoded)

        class_weights = compute_class_weights(y_train)

        validation_results = validate_splits(
            X_train, X_valid, X_test,
            y_train, y_valid, y_test,
            num_classes
        )

        split_sizes = {
            "train": len(X_train),
            "train_pct": (len(X_train) / original_size) * 100,
            "valid": len(X_valid),
            "valid_pct": (len(X_valid) / original_size) * 100,
            "test": len(X_test),
            "test_pct": (len(X_test) / original_size) * 100
        }

        report_text = generate_report(
            original_size,
            split_sizes,
            feature_count,
            validation_results["class_distributions"],
            label_mapping,
            class_weights,
            validation_results
        )

        # Save splits and reports
        save_processed_data(X_train, X_valid, X_test, y_train, y_valid, y_test, PROCESSED_DATA_DIR)
        save_artifacts(le, class_weights, label_mapping, MODELS_DIR)

        # Save report text
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(PREPROCESSING_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("Preprocessing & Preparation pipeline completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during preprocessing:")
        raise e


if __name__ == "__main__":
    main()
