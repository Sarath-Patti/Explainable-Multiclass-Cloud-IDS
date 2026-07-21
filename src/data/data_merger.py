"""Dataset Ingestion & Standardization Module for Explainable-Multiclass-Cloud-IDS.

This module discovers, loads, validates, standardizes, and merges individual CSV
files from the CICIDS2017 dataset in the raw data directory. It produces a single,
standardized, multiclass dataset and generates a summary report.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Define paths and project roots
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MERGED_DATA_DIR = PROJECT_ROOT / "data" / "merged"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

MERGED_DATA_PATH = MERGED_DATA_DIR / "cicids2017_multiclass.csv"
SUMMARY_REPORT_PATH = REPORTS_DIR / "dataset_summary.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DatasetMerger")


def discover_csv_files(raw_dir: Path) -> list[Path]:
    """Discovers all CSV files in the raw data directory.

    Args:
        raw_dir: Path to the directory containing raw CSV files.

    Returns:
        List of Path objects representing the CSV files.

    Raises:
        FileNotFoundError: If raw_dir does not exist.
        ValueError: If no CSV files are found in raw_dir.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {raw_dir}")

    logger.info(f"Discovered {len(csv_files)} CSV files in {raw_dir}")
    return sorted(csv_files)


def validate_dataset(df: pd.DataFrame, file_path: Path) -> None:
    """Validates that the loaded dataset conforms to requirements.

    Specifically checks for the presence of the "Label" column.

    Args:
        df: Pandas DataFrame to validate.
        file_path: Path to the file from which the DataFrame was loaded.

    Raises:
        ValueError: If the required 'Label' column is missing.
    """
    if "Label" not in df.columns:
        raise ValueError(f"Dataset {file_path.name} is missing the required 'Label' column.")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes columns in the DataFrame.

    Trims leading and trailing whitespace and collapses duplicate spaces.
    Otherwise, preserves the original feature names.

    Args:
        df: Pandas DataFrame whose columns need standardization.

    Returns:
        DataFrame with standardized column names.
    """
    df = df.copy()
    df.columns = [" ".join(col.strip().split()) for col in df.columns]
    return df


def load_datasets(file_paths: list[Path]) -> list[pd.DataFrame]:
    """Loads all CSV datasets from the given file paths.

    Attempts to load files with UTF-8 encoding first, falling back to Latin-1
    in case of decoding anomalies. Standardizes the columns and validates
    each DataFrame before returning.

    Args:
        file_paths: List of Path objects to CSV files.

    Returns:
        List of pandas DataFrames with standardized columns.
    """
    dfs = []
    for path in file_paths:
        logger.info(f"Loading dataset: {path.name}")
        try:
            # Try reading with utf-8
            df = pd.read_csv(path, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not load {path.name} with utf-8 ({e}). Falling back to latin-1.")
            df = pd.read_csv(path, encoding="latin-1")

        # Standardize columns right away to allow proper validation
        df = standardize_columns(df)
        validate_dataset(df, path)
        dfs.append(df)

    return dfs


def standardize_labels(df: pd.DataFrame, label_column: str = "Label") -> pd.DataFrame:
    """Standardizes values in the Label column.

    Normalizes spellings, trims whitespace, handles encoding variants,
    and groups all Web Attack subclasses (Brute Force, XSS, SQL Injection)
    into a single category.

    Args:
        df: Pandas DataFrame containing the label column.
        label_column: Name of the label column.

    Returns:
        DataFrame with standardized labels.
    """
    df = df.copy()

    # Mapping for common inconsistent spellings (case-insensitive keys)
    label_mapping = {
        "benign": "BENIGN",
        "bot": "Bot",
        "botnet": "Bot",
        "ddos": "DDoS",
        "dos goldeneye": "DoS GoldenEye",
        "dos hulk": "DoS Hulk",
        "dos slowhttptest": "DoS Slowhttptest",
        "dos slowloris": "DoS slowloris",
        "ftp-patator": "FTP-Patator",
        "ssh-patator": "SSH-Patator",
        "portscan": "PortScan",
        "infiltration": "Infiltration",
        "heartbleed": "Heartbleed"
    }

    def normalize_label(val) -> str:
        if not isinstance(val, str):
            return str(val).strip()

        # Trim leading/trailing whitespace and collapse duplicate spaces
        val_clean = " ".join(val.strip().split())
        val_lower = val_clean.lower()

        # Merge all Web Attack variants (handles encoding variants like )
        if "web attack" in val_lower:
            return "Web Attack"

        # Map known categories case-insensitively
        if val_lower in label_mapping:
            return label_mapping[val_lower]

        return val_clean

    df[label_column] = df[label_column].apply(normalize_label)
    return df


def merge_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merges a list of DataFrames into a single DataFrame.

    Preserves all records and features using an outer join.

    Args:
        dfs: List of DataFrames to merge.

    Returns:
        A single merged DataFrame.
    """
    logger.info("Merging datasets...")
    # Use outer join to preserve every feature and row
    merged_df = pd.concat(dfs, ignore_index=True, join="outer")
    logger.info(f"Merged dataset shape: {merged_df.shape}")
    return merged_df


def generate_summary(
    dfs: list[pd.DataFrame],
    merged_df: pd.DataFrame,
    file_paths: list[Path]
) -> dict:
    """Generates a summary of the merged dataset metrics.

    Args:
        dfs: List of individual loaded DataFrames.
        merged_df: The final merged DataFrame.
        file_paths: List of paths to the loaded CSV files.

    Returns:
        A dictionary containing dataset summary statistics.
    """
    logger.info("Generating dataset summary statistics...")

    num_files = len(file_paths)
    dataset_names = [path.name for path in file_paths]
    total_samples = len(merged_df)
    num_features = len(merged_df.columns)
    feature_names = sorted(list(merged_df.columns))

    unique_labels = sorted(list(merged_df["Label"].unique()))
    class_distribution = merged_df["Label"].value_counts().to_dict()

    memory_usage_bytes = merged_df.memory_usage(deep=True).sum()

    duplicate_count = int(merged_df.duplicated().sum())
    missing_count = int(merged_df.isna().sum().sum())

    numeric_df = merged_df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric_df).values.sum()) if not numeric_df.empty else 0

    return {
        "num_files": num_files,
        "dataset_names": dataset_names,
        "total_samples": total_samples,
        "num_features": num_features,
        "feature_names": feature_names,
        "unique_labels": unique_labels,
        "class_distribution": class_distribution,
        "memory_usage_bytes": memory_usage_bytes,
        "duplicate_count": duplicate_count,
        "missing_count": missing_count,
        "inf_count": inf_count,
    }


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Saves the merged DataFrame to a CSV file.

    Args:
        df: The DataFrame to save.
        output_path: The file path to save the CSV to.
    """
    logger.info(f"Saving merged dataset to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved successfully.")


def save_summary(summary_data: dict, output_path: Path) -> None:
    """Saves the summary statistics to a text report file.

    Args:
        summary_data: Dictionary of summary statistics.
        output_path: The file path to save the text report to.
    """
    logger.info(f"Saving dataset summary to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mem_bytes = summary_data["memory_usage_bytes"]
    if mem_bytes >= 1024**3:
        mem_str = f"{mem_bytes / (1024**3):.2f} GB"
    else:
        mem_str = f"{mem_bytes / (1024**2):.2f} MB"

    lines = [
        "==================================================",
        "CICIDS2017 Dataset Ingestion & Integration Summary",
        "==================================================",
        f"Number of CSV files loaded: {summary_data['num_files']}",
        "",
        "Dataset Names:",
        *[f"  - {name}" for name in summary_data["dataset_names"]],
        "",
        f"Total Samples: {summary_data['total_samples']:,}",
        f"Number of Features: {summary_data['num_features']}",
        "",
        "Feature Names:",
        *[f"  - {col}" for col in summary_data["feature_names"]],
        "",
        "Unique Attack Labels:",
        *[f"  - {label}" for label in summary_data["unique_labels"]],
        "",
        "Class Distribution:",
        *[f"  - {label}: {count:,}" for label, count in sorted(summary_data["class_distribution"].items(), key=lambda x: x[1], reverse=True)],
        "",
        f"Dataset Memory Usage: {mem_str}",
        f"Duplicate Row Count: {summary_data['duplicate_count']:,}",
        f"Missing Value Count: {summary_data['missing_count']:,}",
        f"Infinite Value Count: {summary_data['inf_count']:,}",
        "=================================================="
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Summary report saved successfully.")


def main() -> None:
    """Main execution function to run the ingestion and standardization workflow."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "dataset_merger.log"

    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Starting CICIDS2017 dataset Ingestion & Standardization pipeline...")

    try:
        csv_files = discover_csv_files(RAW_DATA_DIR)
        dfs = load_datasets(csv_files)

        # Standardize labels for each loaded dataframe before merging
        standardized_dfs = [standardize_labels(df) for df in dfs]

        merged_df = merge_datasets(standardized_dfs)

        summary = generate_summary(dfs, merged_df, csv_files)

        save_dataset(merged_df, MERGED_DATA_PATH)
        save_summary(summary, SUMMARY_REPORT_PATH)

        logger.info("Ingestion & Standardization pipeline completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during dataset ingestion:")
        raise e


if __name__ == "__main__":
    main()
