"""Dataset Profiling & Validation Module for Explainable-Multiclass-Cloud-IDS.

This module profiles and validates the merged dataset (before any preprocessing)
to evaluate feature quality, target distribution, numerical properties, and
identify potential data issues. Reports are saved in text and CSV formats.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_DATA_PATH = PROJECT_ROOT / "data" / "merged" / "cicids2017_multiclass.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

PROFILE_REPORT_PATH = REPORTS_DIR / "dataset_profile.txt"
CLASS_DIST_PATH = REPORTS_DIR / "class_distribution.csv"
FEATURE_SUMMARY_PATH = REPORTS_DIR / "feature_summary.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DatasetProfiler")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Loads the merged dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Pandas DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Merged dataset file not found at: {file_path}")
    logger.info(f"Loading merged dataset from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validates the basic properties of the dataset.

    Args:
        df: The dataset DataFrame.

    Returns:
        A dictionary containing basic dataset metrics.
    """
    logger.info("Validating basic dataset properties...")
    shape = df.shape
    num_samples = shape[0]
    num_features = shape[1]
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    memory_usage = df.memory_usage(deep=True).sum()

    return {
        "shape": shape,
        "num_samples": num_samples,
        "num_features": num_features,
        "columns": columns,
        "dtypes": dtypes,
        "memory_usage": memory_usage,
    }


def profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes the quality of each feature in the dataset.

    Computes missing/infinite counts, uniqueness, constant/near-constant status,
    and identifies data types.

    Args:
        df: The dataset DataFrame.

    Returns:
        A pandas DataFrame summary of all features.
    """
    logger.info("Profiling features...")
    records = []
    num_samples = len(df)

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)

        # Missing values
        missing_count = int(series.isna().sum())
        missing_pct = (missing_count / num_samples) * 100 if num_samples > 0 else 0.0

        # Infinite values
        inf_count = 0
        if pd.api.types.is_numeric_dtype(series):
            inf_count = int(np.isinf(series).sum())
        inf_pct = (inf_count / num_samples) * 100 if num_samples > 0 else 0.0

        # Uniqueness
        unique_count = int(series.nunique(dropna=False))

        # Constant check
        is_const = (unique_count == 1)

        # Near constant check (most frequent value accounts for >= 99%)
        is_near_const = False
        if not is_const and num_samples > 0:
            val_counts = series.value_counts(dropna=False)
            if not val_counts.empty:
                most_freq_ratio = val_counts.iloc[0] / num_samples
                is_near_const = (most_freq_ratio >= 0.99)

        is_empty = (missing_count == num_samples)
        is_object = (series.dtype == "object" and col != "Label")
        is_high_card = (is_object and unique_count > 10)

        records.append({
            "feature_name": col,
            "dtype": dtype,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "infinite_count": inf_count,
            "infinite_pct": inf_pct,
            "unique_count": unique_count,
            "is_constant": is_const,
            "is_near_constant": is_near_const,
            "is_empty": is_empty,
            "is_object_type": is_object,
            "is_high_cardinality": is_high_card,
        })

    summary_df = pd.DataFrame(records)
    return summary_df


def profile_labels(df: pd.DataFrame, label_column: str = "Label") -> pd.DataFrame:
    """Profiles the label column to analyze class distribution.

    Identifies unique classes, counts, percentages, and marks severely
    imbalanced classes (< 0.5% of total dataset).

    Args:
        df: The dataset DataFrame.
        label_column: The name of the label column.

    Returns:
        A pandas DataFrame summary of target labels.
    """
    logger.info("Profiling target labels...")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    counts = df[label_column].value_counts(dropna=False)
    total_samples = len(df)

    records = []
    for label, count in counts.items():
        pct = (count / total_samples) * 100 if total_samples > 0 else 0.0
        # Severe imbalance threshold: less than 0.5% of total samples
        is_imbalanced = (pct < 0.5)
        records.append({
            "class_label": str(label),
            "sample_count": int(count),
            "percentage": pct,
            "is_severely_imbalanced": is_imbalanced,
        })

    summary_df = pd.DataFrame(records)
    return summary_df


def generate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generates descriptive statistics for numeric features.

    Calculates mean, median, std, min, max, and percentiles.

    Args:
        df: The dataset DataFrame.

    Returns:
        A pandas DataFrame of descriptive statistics.
    """
    logger.info("Generating descriptive statistics for numerical features...")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logger.warning("No numeric columns found in the dataset.")
        return pd.DataFrame()

    stats_df = numeric_df.describe(percentiles=[0.25, 0.50, 0.75]).T
    stats_df = stats_df[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats_df.rename(columns={"50%": "median"}, inplace=True)
    stats_df.index.name = "feature_name"
    stats_df.reset_index(inplace=True)
    return stats_df


def generate_reports(
    basic_metrics: dict,
    feature_summary: pd.DataFrame,
    class_dist: pd.DataFrame,
    stats_df: pd.DataFrame,
    duplicate_count: int
) -> str:
    """Compiles individual profiles and summaries into a comprehensive text report.

    Args:
        basic_metrics: Dictionary of basic dataset validation metrics.
        feature_summary: DataFrame summarizing feature quality.
        class_dist: DataFrame profiling target labels.
        stats_df: DataFrame containing numerical statistics.
        duplicate_count: Total count of duplicate rows.

    Returns:
        A formatted string containing the complete dataset profile report.
    """
    logger.info("Compiling comprehensive profile report...")

    # Format memory usage
    mem_bytes = basic_metrics["memory_usage"]
    if mem_bytes >= 1024**3:
        mem_str = f"{mem_bytes / (1024**3):.2f} GB"
    else:
        mem_str = f"{mem_bytes / (1024**2):.2f} MB"

    # Extract lists of issues from feature_summary
    missing_features = feature_summary[feature_summary["missing_count"] > 0]
    infinite_features = feature_summary[feature_summary["infinite_count"] > 0]
    empty_features = feature_summary[feature_summary["is_empty"] == True]
    constant_features = feature_summary[feature_summary["is_constant"] == True]
    near_constant_features = feature_summary[feature_summary["is_near_constant"] == True]
    object_features = feature_summary[feature_summary["is_object_type"] == True]
    high_cardinality_features = feature_summary[feature_summary["is_high_cardinality"] == True]

    lines = [
        "==================================================",
        "CICIDS2017 DATASET PROFILE & VALIDATION REPORT",
        "==================================================",
        "",
        "1. DATASET INFORMATION",
        "----------------------",
        f"Dataset Shape: {basic_metrics['shape']}",
        f"Number of Samples: {basic_metrics['num_samples']:,}",
        f"Number of Features: {basic_metrics['num_features']}",
        f"Memory Usage: {mem_str}",
        "",
        "Column Names & Data Types:",
    ]

    for col, dtype in basic_metrics["dtypes"].items():
        lines.append(f"  - {col}: {dtype}")

    lines.extend([
        "",
        "2. TARGET LABELS & CLASS DISTRIBUTION",
        "-------------------------------------",
        f"Number of Classes: {len(class_dist)}",
        "Class Distribution:",
    ])

    for _, row in class_dist.iterrows():
        imbalance_str = " (SEVERELY IMBALANCED)" if row["is_severely_imbalanced"] else ""
        lines.append(
            f"  - {row['class_label']}: {row['sample_count']:,} ({row['percentage']:.4f}%){imbalance_str}"
        )

    num_samples = basic_metrics['num_samples']
    dup_pct = (duplicate_count / num_samples * 100) if num_samples > 0 else 0.0

    lines.extend([
        "",
        "3. DETECTED DATA QUALITY ISSUES",
        "-------------------------------",
        f"Duplicate Row Count: {duplicate_count:,} ({dup_pct:.2f}% of samples)",
        "",
        f"Empty Columns (100% missing) [{len(empty_features)}]:",
    ])
    for _, row in empty_features.iterrows():
        lines.append(f"  - {row['feature_name']}")

    lines.append(f"\nMissing Values per Feature (non-zero) [{len(missing_features)}]:")
    for _, row in missing_features.iterrows():
        lines.append(f"  - {row['feature_name']}: {row['missing_count']:,} ({row['missing_pct']:.4f}%)")

    lines.append(f"\nInfinite Values per Feature (non-zero) [{len(infinite_features)}]:")
    for _, row in infinite_features.iterrows():
        lines.append(f"  - {row['feature_name']}: {row['infinite_count']:,} ({row['infinite_pct']:.4f}%)")

    lines.append(f"\nConstant Features (1 unique value) [{len(constant_features)}]:")
    for _, row in constant_features.iterrows():
        lines.append(f"  - {row['feature_name']}")

    lines.append(f"\nNear-Constant Features (>= 99% same value) [{len(near_constant_features)}]:")
    for _, row in near_constant_features.iterrows():
        lines.append(f"  - {row['feature_name']}")

    lines.append(f"\nObject-Type Columns (excluding Label) [{len(object_features)}]:")
    for _, row in object_features.iterrows():
        lines.append(f"  - {row['feature_name']} (unique count: {row['unique_count']})")

    lines.append(f"\nHigh Cardinality Categorical Features (>10 unique values) [{len(high_cardinality_features)}]:")
    for _, row in high_cardinality_features.iterrows():
        lines.append(f"  - {row['feature_name']} (unique count: {row['unique_count']})")

    lines.extend([
        "",
        "4. NUMERICAL FEATURE DESCRIPTIVE STATISTICS",
        "-------------------------------------------",
    ])

    if not stats_df.empty:
        headers = ["Feature Name", "Mean", "Median", "Std Dev", "Min", "Max", "25%", "75%"]
        col_widths = [32, 12, 12, 12, 12, 12, 12, 12]

        header_line = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * sum(col_widths))

        for _, row in stats_df.iterrows():
            row_str = (
                f"{row['feature_name'][:30]:<32}"
                f"{row['mean']:<12.4f}"
                f"{row['median']:<12.4f}"
                f"{row['std']:<12.4f}"
                f"{row['min']:<12.4f}"
                f"{row['max']:<12.4f}"
                f"{row['25%']:<12.4f}"
                f"{row['75%']:<12.4f}"
            )
            lines.append(row_str)
    else:
        lines.append("No numerical features available for statistics.")

    lines.append("\n==============================================")
    return "\n".join(lines)


def save_reports(
    profile_text: str,
    class_dist: pd.DataFrame,
    feature_summary: pd.DataFrame,
    profile_path: Path,
    class_dist_path: Path,
    feature_summary_path: Path
) -> None:
    """Saves all generated reports to their designated output directories.

    Args:
        profile_text: The compiled text profile.
        class_dist: DataFrame of class distributions.
        feature_summary: DataFrame of feature summaries.
        profile_path: File path to save the text profile.
        class_dist_path: File path to save class distribution CSV.
        feature_summary_path: File path to save feature summary CSV.
    """
    logger.info("Saving validation and profiling reports...")

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    class_dist_path.parent.mkdir(parents=True, exist_ok=True)
    feature_summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(profile_path, "w", encoding="utf-8") as f:
        f.write(profile_text)

    class_dist.to_csv(class_dist_path, index=False)
    feature_summary.to_csv(feature_summary_path, index=False)

    logger.info("All reports saved successfully.")


def main() -> None:
    """Main execution function to run the profiling and validation workflow."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "dataset_profiler.log"

    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Starting CICIDS2017 dataset Profiling & Validation pipeline...")

    try:
        df = load_dataset(MERGED_DATA_PATH)
        basic_metrics = validate_dataset(df)
        feature_summary = profile_features(df)
        class_dist = profile_labels(df)
        stats_df = generate_statistics(df)

        duplicate_count = int(df.duplicated().sum())

        profile_text = generate_reports(
            basic_metrics,
            feature_summary,
            class_dist,
            stats_df,
            duplicate_count
        )

        save_reports(
            profile_text,
            class_dist,
            feature_summary,
            PROFILE_REPORT_PATH,
            CLASS_DIST_PATH,
            FEATURE_SUMMARY_PATH
        )

        logger.info("Profiling & Validation pipeline completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during dataset profiling:")
        raise e


if __name__ == "__main__":
    main()
