"""Data Cleaning Module for Explainable-Multiclass-Cloud-IDS.

This module processes the merged CICIDS2017 dataset by removing rare classes,
replacing infinite values with NaN, dropping duplicate and missing rows, and
removing specific constant features.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_DATA_PATH = PROJECT_ROOT / "data" / "merged" / "cicids2017_multiclass.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

CLEANING_REPORT_PATH = REPORTS_DIR / "data_cleaning_report.txt"
CLASS_DIST_CLEAN_PATH = REPORTS_DIR / "class_distribution_clean.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataCleaner")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Loads the merged dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame containing the dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Merged dataset not found at: {file_path}")
    logger.info(f"Loading merged dataset from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset loaded. Shape: {df.shape}")
    return df


def remove_rare_classes(df: pd.DataFrame, classes_to_remove: list[str]) -> tuple[pd.DataFrame, dict]:
    """Removes rare classes from the dataset and records stats.

    Args:
        df: The dataset DataFrame.
        classes_to_remove: List of class labels to drop.

    Returns:
        A tuple containing the filtered DataFrame and a dictionary of removed counts.
    """
    logger.info(f"Removing rare classes: {classes_to_remove}...")
    df = df.copy()
    removed_stats = {}

    for cls in classes_to_remove:
        count = int((df["Label"] == cls).sum())
        removed_stats[cls] = count
        logger.info(f"Class '{cls}': {count} samples will be removed.")

    filtered_df = df[~df["Label"].isin(classes_to_remove)].copy()
    logger.info(f"Shape after removing rare classes: {filtered_df.shape}")
    return filtered_df, removed_stats


def replace_infinities(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replaces positive and negative infinities with NaN.

    Args:
        df: The dataset DataFrame.

    Returns:
        A tuple of the modified DataFrame and the total count of infinity values replaced.
    """
    logger.info("Replacing positive and negative infinities with NaN...")
    df = df.copy()

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Count infs
    inf_count = int(np.isinf(df[numeric_cols]).values.sum())
    logger.info(f"Found {inf_count} infinite values to replace.")

    # Replace infs
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df, inf_count


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Removes duplicate rows from the dataset.

    Args:
        df: The dataset DataFrame.

    Returns:
        A tuple containing the deduplicated DataFrame and a dictionary of duplicate statistics.
    """
    logger.info("Removing duplicate rows...")
    dup_count_before = int(df.duplicated().sum())

    deduped_df = df.drop_duplicates(keep="first").copy()

    dup_count_after = int(deduped_df.duplicated().sum())  # Should be 0
    removed_count = dup_count_before - dup_count_after

    logger.info(f"Duplicates before: {dup_count_before}, after: {dup_count_after}. Removed: {removed_count}")

    stats = {
        "duplicates_before": dup_count_before,
        "duplicates_removed": removed_count,
    }
    return deduped_df, stats


def remove_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Removes rows with missing (NaN) values.

    Args:
        df: The dataset DataFrame.

    Returns:
        A tuple containing the cleaned DataFrame and a dictionary of missing values stats.
    """
    logger.info("Removing rows containing missing values...")
    missing_rows_before = int(df.isna().any(axis=1).sum())

    cleaned_df = df.dropna().copy()

    missing_rows_after = int(cleaned_df.isna().any(axis=1).sum())  # Should be 0
    removed_count = missing_rows_before - missing_rows_after

    logger.info(f"Rows with NaNs before: {missing_rows_before}, after: {missing_rows_after}. Removed: {removed_count}")

    stats = {
        "missing_rows_before": missing_rows_before,
        "missing_rows_removed": removed_count,
    }
    return cleaned_df, stats


def remove_constant_features(df: pd.DataFrame, features_to_remove: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Removes specified constant features from the DataFrame.

    Args:
        df: The dataset DataFrame.
        features_to_remove: List of column names to drop.

    Returns:
        A tuple containing the DataFrame with features removed and a list of actually removed features.
    """
    logger.info(f"Dropping constant features: {features_to_remove}...")
    df = df.copy()
    existing_to_remove = [col for col in features_to_remove if col in df.columns]

    df.drop(columns=existing_to_remove, inplace=True)
    logger.info(f"Dropped features: {existing_to_remove}. Remaining features: {df.shape[1]}")
    return df, existing_to_remove


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validates the properties of the cleaned dataset.

    Args:
        df: The cleaned dataset DataFrame.

    Returns:
        A dictionary containing validation metrics.
    """
    logger.info("Validating cleaned dataset...")
    shape = df.shape
    num_samples = shape[0]
    num_features = shape[1]

    unique_classes = sorted(list(df["Label"].unique()))
    num_classes = len(unique_classes)

    remaining_duplicates = int(df.duplicated().sum())
    remaining_missing = int(df.isna().sum().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    remaining_infinite = int(np.isinf(df[numeric_cols]).values.sum()) if not numeric_cols.empty else 0

    return {
        "shape": shape,
        "num_samples": num_samples,
        "num_features": num_features,
        "classes": unique_classes,
        "num_classes": num_classes,
        "remaining_duplicates": remaining_duplicates,
        "remaining_missing": remaining_missing,
        "remaining_infinite": remaining_infinite,
    }


def generate_cleaning_report(
    audit_trail: dict,
    validation_metrics: dict
) -> str:
    """Generates a text report showing the complete audit trail of the cleaning process.

    Args:
        audit_trail: Dictionary of intermediate counts and stats from each step.
        validation_metrics: Dictionary of final validation metrics.

    Returns:
        Formatted cleaning report string.
    """
    logger.info("Generating data cleaning report...")

    lines = [
        "==================================================",
        "CICIDS2017 DATA CLEANING AUDIT TRAIL REPORT",
        "==================================================",
        "",
        "Original Dataset",
        "↓",
        f"Original Samples: {audit_trail['original_samples']:,}",
        "↓",
        f"Removed Heartbleed: {audit_trail['removed_heartbleed']:,} samples",
        "↓",
        f"Removed Infiltration: {audit_trail['removed_infiltration']:,} samples",
        "↓",
        f"Replaced Infinite Values: {audit_trail['replaced_infinities']:,} values replaced with NaN",
        "↓",
        f"Removed Duplicate Rows: {audit_trail['removed_duplicates']:,} rows dropped",
        "↓",
        f"Removed Missing Rows: {audit_trail['removed_missing']:,} rows containing NaN dropped",
        "↓",
        f"Dropped Constant Features: {len(audit_trail['dropped_features'])} features dropped ({', '.join(audit_trail['dropped_features'])})",
        "↓",
        f"Final Dataset Shape: {validation_metrics['shape']}",
        "↓",
        f"Remaining Classes: {validation_metrics['num_classes']} ({', '.join(validation_metrics['classes'])})",
        "↓",
        f"Remaining Features: {validation_metrics['num_features']}",
        "↓",
        f"Remaining Missing Values: {validation_metrics['remaining_missing']}",
        "↓",
        f"Remaining Infinite Values: {validation_metrics['remaining_infinite']}",
        "↓",
        "Cleaning Completed",
        "",
        "=================================================="
    ]

    return "\n".join(lines)


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Saves the cleaned DataFrame to a CSV file.

    Args:
        df: The cleaned DataFrame.
        output_path: Path to the output CSV file.
    """
    logger.info(f"Saving cleaned dataset to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Cleaned dataset saved successfully.")


def save_reports(
    report_text: str,
    class_dist: pd.DataFrame,
    report_path: Path,
    class_dist_path: Path
) -> None:
    """Saves the cleaning audit report and class distribution CSV.

    Args:
        report_text: Formatted audit trail text.
        class_dist: DataFrame of cleaned class distribution.
        report_path: Path to save the audit report.
        class_dist_path: Path to save the class distribution CSV.
    """
    logger.info("Saving cleaning reports...")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    class_dist_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    class_dist.to_csv(class_dist_path, index=False)
    logger.info("Reports saved successfully.")


def main() -> None:
    """Main execution function to run the data cleaning pipeline."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "data_cleaner.log"

    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Starting CICIDS2017 dataset Data Cleaning pipeline...")

    # Define rare classes and constant features to drop
    classes_to_drop = ["Heartbleed", "Infiltration"]
    constant_features_to_drop = [
        "Bwd PSH Flags",
        "Bwd URG Flags",
        "Fwd Avg Bytes/Bulk",
        "Fwd Avg Packets/Bulk",
        "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk",
        "Bwd Avg Packets/Bulk",
        "Bwd Avg Bulk Rate"
    ]

    try:
        df = load_dataset(MERGED_DATA_PATH)
        audit_trail = {"original_samples": len(df)}

        # Step 1: Remove rare classes
        df, removed_classes_stats = remove_rare_classes(df, classes_to_drop)
        audit_trail["removed_heartbleed"] = removed_classes_stats.get("Heartbleed", 0)
        audit_trail["removed_infiltration"] = removed_classes_stats.get("Infiltration", 0)

        # Step 2: Replace infinities with NaN
        df, replaced_infs_count = replace_infinities(df)
        audit_trail["replaced_infinities"] = replaced_infs_count

        # Step 3: Remove duplicate rows
        df, duplicate_stats = remove_duplicates(df)
        audit_trail["removed_duplicates"] = duplicate_stats["duplicates_removed"]

        # Step 4: Remove missing rows
        df, missing_stats = remove_missing_values(df)
        audit_trail["removed_missing"] = missing_stats["missing_rows_removed"]

        # Step 5: Remove constant features
        df, dropped_features = remove_constant_features(df, constant_features_to_drop)
        audit_trail["dropped_features"] = dropped_features

        # Validate
        validation_metrics = validate_dataset(df)

        # Generate cleaned class distribution
        counts = df["Label"].value_counts(dropna=False)
        total_samples = len(df)
        class_dist_records = []
        for label, count in counts.items():
            pct = (count / total_samples) * 100 if total_samples > 0 else 0.0
            class_dist_records.append({
                "class_label": str(label),
                "sample_count": int(count),
                "percentage": pct
            })
        class_dist_df = pd.DataFrame(class_dist_records)

        # Generate report text
        report_text = generate_cleaning_report(audit_trail, validation_metrics)

        # Save outputs
        save_dataset(df, CLEANED_DATA_PATH)
        save_reports(report_text, class_dist_df, CLEANING_REPORT_PATH, CLASS_DIST_CLEAN_PATH)

        logger.info("Data Cleaning pipeline completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during dataset cleaning:")
        raise e


if __name__ == "__main__":
    main()
