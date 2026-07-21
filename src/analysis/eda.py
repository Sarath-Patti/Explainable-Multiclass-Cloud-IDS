"""Exploratory Data Analysis (EDA) Module for Explainable-Multiclass-Cloud-IDS.

This module performs statistical analysis, computes correlation matrices, detects
outliers and skewness, and generates publication-quality plots from the cleaned dataset.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Set matplotlib backend to Agg to run in headless mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EDA")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Loads the cleaned dataset from the given path.

    Args:
        file_path: Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {file_path}")
    logger.info(f"Loading cleaned dataset from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


def dataset_overview(df: pd.DataFrame, target_col: str = "Label") -> dict:
    """Generates an overview of the dataset's basic characteristics.

    Args:
        df: The dataset DataFrame.
        target_col: Name of the target column.

    Returns:
        dict: Basic overview metrics.
    """
    logger.info("Generating dataset overview...")
    num_samples = len(df)
    predictor_cols = [col for col in df.columns if col != target_col]
    num_predictors = len(predictor_cols)

    unique_classes = df[target_col].unique()
    num_classes = len(unique_classes)

    memory_usage = df.memory_usage(deep=True).sum()

    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_counts = {str(k): int(v) for k, v in dtype_counts.items()}

    return {
        "num_samples": num_samples,
        "num_predictors": num_predictors,
        "target_col": target_col,
        "num_classes": num_classes,
        "memory_usage": memory_usage,
        "dtype_counts": dtype_counts,
        "dtypes_all": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def class_distribution_analysis(df: pd.DataFrame, target_col: str = "Label") -> tuple[pd.DataFrame, dict]:
    """Analyzes target class distributions and identifies majority/minority classes.

    Args:
        df: The dataset DataFrame.
        target_col: The target label column.

    Returns:
        tuple[pd.DataFrame, dict]: Distribution DataFrame and summary dictionary.
    """
    logger.info("Analyzing target class distribution...")
    counts = df[target_col].value_counts()
    total = len(df)

    records = []
    majority_class = counts.index[0]

    for label, count in counts.items():
        pct = (count / total) * 100 if total > 0 else 0.0
        role = "Majority" if label == majority_class else "Minority"
        records.append({
            "class_label": str(label),
            "sample_count": int(count),
            "percentage": pct,
            "role": role
        })

    dist_df = pd.DataFrame(records)

    summary = {
        "majority_class": str(majority_class),
        "majority_count": int(counts.loc[majority_class]),
        "majority_pct": (counts.loc[majority_class] / total) * 100 if total > 0 else 0.0,
        "minority_classes": [str(x) for x in counts.index[1:]]
    }

    return dist_df, summary


def numerical_feature_summary(df: pd.DataFrame, target_col: str = "Label") -> pd.DataFrame:
    """Generates summary statistics for numerical features.

    Args:
        df: The dataset DataFrame.
        target_col: Name of the target column to exclude.

    Returns:
        pd.DataFrame: Summary statistics.
    """
    logger.info("Generating numerical feature summary...")
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])

    if numeric_df.empty:
        return pd.DataFrame()

    stats = numeric_df.describe(percentiles=[0.25, 0.50, 0.75]).T
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats.rename(columns={"50%": "median"}, inplace=True)
    stats.index.name = "feature_name"
    stats.reset_index(inplace=True)
    return stats


def correlation_analysis(df: pd.DataFrame, target_col: str = "Label") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes the Pearson correlation matrix and identifies highly correlated feature pairs.

    Args:
        df: The dataset DataFrame.
        target_col: Target column to exclude.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Correlation matrix and highly correlated pairs summary.
    """
    logger.info("Computing Pearson correlation matrix...")
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])

    corr_matrix = numeric_df.corr(method="pearson")

    # Find upper triangle correlations only (to avoid duplicates like A-B and B-A)
    pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.90:
                pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": val,
                    "abs_correlation": abs(val)
                })

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df.sort_values(by="abs_correlation", ascending=False, inplace=True)
        pairs_df.reset_index(drop=True, inplace=True)
    else:
        pairs_df = pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation"])

    return corr_matrix, pairs_df


def feature_distribution_analysis(df: pd.DataFrame, target_col: str = "Label") -> dict:
    """Analyzes and summarizes feature distributions.

    Args:
        df: The dataset DataFrame.
        target_col: Name of the target column.

    Returns:
        dict: Summary of observed distributions.
    """
    logger.info("Analyzing feature distributions...")
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])

    # Select representative features for visual analysis and reporting
    rep_features = ["Flow Duration", "Total Fwd Packets", "Fwd Packet Length Max", "Bwd Packet Length Mean"]
    rep_features = [col for col in rep_features if col in numeric_df.columns]
    if len(rep_features) < 4:
        rep_features = list(numeric_df.columns[:4])

    rep_summaries = {}
    for col in rep_features:
        series = numeric_df[col]
        rep_summaries[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "unique_values": int(series.nunique())
        }

    return {
        "representative_features": rep_features,
        "representative_summaries": rep_summaries
    }


def outlier_analysis(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    """Detects outliers using the IQR method for selected features.

    Args:
        df: The dataset DataFrame.
        selected_features: List of column names to analyze.

    Returns:
        pd.DataFrame: Outlier percentage and counts per feature.
    """
    logger.info("Performing outlier analysis using IQR method...")
    records = []
    num_samples = len(df)

    for col in selected_features:
        if col not in df.columns:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_series = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers_series)
        outlier_pct = (outlier_count / num_samples) * 100 if num_samples > 0 else 0.0

        records.append({
            "feature_name": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct
        })

    outliers_df = pd.DataFrame(records)
    return outliers_df


def skewness_analysis(df: pd.DataFrame, target_col: str = "Label") -> pd.DataFrame:
    """Computes skewness for all numerical features and classifies them.

    Args:
        df: The dataset DataFrame.
        target_col: Name of the target column.

    Returns:
        pd.DataFrame: Skewness values and classification per feature.
    """
    logger.info("Computing feature skewness...")
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])

    records = []
    for col in numeric_df.columns:
        series = numeric_df[col]
        if series.std() == 0:
            skew = 0.0
        else:
            skew = float(series.skew(skipna=True))

        abs_skew = abs(skew)

        if abs_skew <= 0.5:
            classification = "Symmetric"
        elif abs_skew <= 1.0:
            classification = "Moderately Skewed"
        else:
            classification = "Highly Skewed"

        records.append({
            "feature_name": col,
            "skewness": skew,
            "abs_skewness": abs_skew,
            "classification": classification
        })

    skew_df = pd.DataFrame(records)
    if not skew_df.empty:
        skew_df.sort_values(by="abs_skewness", ascending=False, inplace=True)
        skew_df.reset_index(drop=True, inplace=True)
    return skew_df


def generate_visualizations(
    df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    class_dist: pd.DataFrame,
    rep_features: list[str],
    plots_dir: Path
) -> None:
    """Generates and saves publication-quality visualizations.

    Args:
        df: The dataset DataFrame.
        corr_matrix: The correlation matrix.
        class_dist: Class distribution DataFrame.
        rep_features: Representative features to plot.
        plots_dir: Directory to save plots.
    """
    logger.info("Generating publication-quality visualizations...")
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")

    # 1. Class Distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="sample_count",
        y="class_label",
        data=class_dist,
        hue="class_label",
        palette="viridis",
        legend=False
    )
    plt.title("CICIDS2017 Cleaned Class Distribution (Log Scale)", fontsize=14, pad=15)
    plt.xlabel("Sample Count (Log Scale)", fontsize=12)
    plt.ylabel("Class Label", fontsize=12)
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(plots_dir / "class_distribution.png", dpi=300)
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )
    plt.title("Pearson Correlation Coefficient Matrix Heatmap", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", dpi=300)
    plt.close()

    # 3. Feature Histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, col in enumerate(rep_features):
        if idx >= len(axes):
            break
        sns.histplot(
            df[col],
            bins=50,
            kde=True,
            ax=axes[idx],
            color="skyblue"
        )
        axes[idx].set_title(f"Distribution of {col}", fontsize=12)
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel("Frequency", fontsize=10)
        if df[col].max() / (df[col].min() + 1e-5) > 1000:
            axes[idx].set_yscale("log")
            axes[idx].set_ylabel("Frequency (Log Scale)", fontsize=10)
    plt.suptitle("Histograms of Representative Features", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_histograms.png", dpi=300)
    plt.close()

    # 4. Feature Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, col in enumerate(rep_features):
        if idx >= len(axes):
            break
        sns.boxplot(
            y=df[col],
            ax=axes[idx],
            color="lightcoral"
        )
        axes[idx].set_title(f"Boxplot of {col}", fontsize=12)
        axes[idx].set_ylabel(col, fontsize=10)
        if df[col].max() / (df[col].min() + 1e-5) > 1000:
            axes[idx].set_yscale("log")
            axes[idx].set_ylabel(f"{col} (Log Scale)", fontsize=10)
    plt.suptitle("Boxplots of Representative Features (Outlier Detection)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_boxplots.png", dpi=300)
    plt.close()


def generate_report(
    overview: dict,
    class_summary: dict,
    class_dist: pd.DataFrame,
    high_corr: pd.DataFrame,
    skewness: pd.DataFrame,
    outliers: pd.DataFrame,
    rep_features: list[str]
) -> str:
    """Generates the comprehensive plain text EDA summary report.

    Args:
        overview: Basic dataset overview dictionary.
        class_summary: Summary of class distributions.
        class_dist: Class distribution DataFrame.
        high_corr: Highly correlated feature pairs.
        skewness: Skewness classification DataFrame.
        outliers: Outliers percentage per representative feature.
        rep_features: Selected representative features.

    Returns:
        str: Formatted report.
    """
    logger.info("Compiling EDA summary report...")

    mem_bytes = overview["memory_usage"]
    if mem_bytes >= 1024**3:
        mem_str = f"{mem_bytes / (1024**3):.2f} GB"
    else:
        mem_str = f"{mem_bytes / (1024**2):.2f} MB"

    skew_counts = skewness["classification"].value_counts().to_dict()

    lines = [
        "==================================================",
        "CICIDS2017 EXPLORATORY DATA ANALYSIS (EDA) REPORT",
        "==================================================",
        "",
        "1. DATASET OVERVIEW",
        "-------------------",
        f"Total Samples: {overview['num_samples']:,}",
        f"Predictor Features: {overview['num_predictors']}",
        f"Target Column: {overview['target_col']}",
        f"Number of Classes: {overview['num_classes']}",
        f"Dataset Memory Usage: {mem_str}",
        "Data Types Count:",
    ]

    for dtype, count in overview["dtype_counts"].items():
        lines.append(f"  - {dtype}: {count}")

    lines.extend([
        "",
        "2. CLASS IMBALANCE ANALYSIS",
        "---------------------------",
        f"Majority Class: {class_summary['majority_class']} ({class_summary['majority_count']:,} samples, {class_summary['majority_pct']:.4f}%)",
        "Minority Classes:",
    ])

    for _, row in class_dist.iterrows():
        if row["class_label"] != class_summary["majority_class"]:
            lines.append(f"  - {row['class_label']}: {row['sample_count']:,} samples ({row['percentage']:.4f}%)")

    lines.extend([
        "",
        "3. HIGHLY CORRELATED FEATURES (> 0.90 absolute Pearson correlation)",
        "--------------------------------------------------------------------",
        f"Total highly correlated pairs found: {len(high_corr)}",
    ])

    if not high_corr.empty:
        for idx, row in high_corr.head(20).iterrows():
            lines.append(
                f"  - {row['feature_1']} <-> {row['feature_2']}: {row['correlation']:.4f}"
            )
        if len(high_corr) > 20:
            lines.append(f"  ... and {len(high_corr) - 20} more highly correlated pairs.")
    else:
        lines.append("  No highly correlated feature pairs found.")

    lines.extend([
        "",
        "4. DISTRIBUTION & SKEWNESS OBSERVATIONS",
        "---------------------------------------",
        f"Symmetric Features (|skew| <= 0.5): {skew_counts.get('Symmetric', 0)}",
        f"Moderately Skewed Features (0.5 < |skew| <= 1.0): {skew_counts.get('Moderately Skewed', 0)}",
        f"Highly Skewed Features (|skew| > 1.0): {skew_counts.get('Highly Skewed', 0)}",
        "",
        "Top 10 Most Skewed Features:",
    ])

    for _, row in skewness.head(10).iterrows():
        lines.append(f"  - {row['feature_name']}: {row['skewness']:.4f} ({row['classification']})")

    lines.extend([
        "",
        "5. OUTLIER OBSERVATIONS (IQR Method)",
        "------------------------------------",
        "Outlier statistics for representative features:",
    ])

    for _, row in outliers.iterrows():
        lines.append(
            f"  - {row['feature_name']}: {row['outlier_count']:,} outliers ({row['outlier_percentage']:.2f}% of samples)"
        )

    lines.extend([
        "",
        "6. RECOMMENDATIONS FOR PREPROCESSING",
        "------------------------------------",
        "Based on the EDA observations, the following preprocessing steps are highly recommended:",
        "1. Outlier Robust Scaling: Due to extreme range differences and high skewness, standard min-max scaling",
        "   may squeeze normal data. A RobustScaler or PowerTransformer (Box-Cox or Yeo-Johnson) is recommended.",
        "2. Skewness Reduction: Apply log1p or power transformation on highly skewed predictor features.",
        "3. Multicollinearity Handling: Address highly correlated features (absolute correlation > 0.90)",
        "   either via dimensionality reduction (PCA) or by dropping redundant features during feature selection.",
        "4. Addressing Class Imbalance: The dataset exhibits severe class imbalance (BENIGN is 80%+ while some classes",
        "   are < 0.1%). Consider subclass grouping, weighted loss functions, or ensemble techniques during model training.",
        "",
        "=================================================="
    ])

    return "\n".join(lines)


def save_outputs(
    eda_summary: str,
    stats_df: pd.DataFrame,
    high_corr: pd.DataFrame,
    skewness: pd.DataFrame,
    reports_dir: Path
) -> None:
    """Saves all generated CSV reports and the plain text EDA report.

    Args:
        eda_summary: Compiled summary text.
        stats_df: Numerical feature statistics DataFrame.
        high_corr: Highly correlated pairs DataFrame.
        skewness: Skewness classification DataFrame.
        reports_dir: Directory to save the outputs.
    """
    logger.info("Saving EDA summary and CSV reports...")
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "eda_summary.txt", "w", encoding="utf-8") as f:
        f.write(eda_summary)

    stats_df.to_csv(reports_dir / "feature_statistics.csv", index=False)
    high_corr.to_csv(reports_dir / "correlation_summary.csv", index=False)
    skewness.to_csv(reports_dir / "skewness_summary.csv", index=False)
    logger.info("Reports saved successfully.")


def main() -> None:
    """Main execution function to orchestrate the EDA workflow."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "eda.log"

    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Starting CICIDS2017 dataset Exploratory Data Analysis (EDA)...")

    try:
        df = load_dataset(CLEANED_DATA_PATH)

        overview = dataset_overview(df)
        class_dist, class_summary = class_distribution_analysis(df)
        stats_df = numerical_feature_summary(df)
        corr_matrix, high_corr = correlation_analysis(df)

        dist_summary = feature_distribution_analysis(df)
        rep_features = dist_summary["representative_features"]

        outliers_df = outlier_analysis(df, rep_features)
        skewness_df = skewness_analysis(df)

        # Save plots
        generate_visualizations(df, corr_matrix, class_dist, rep_features, PLOTS_DIR)

        # Compile report
        report_text = generate_report(
            overview,
            class_summary,
            class_dist,
            high_corr,
            skewness_df,
            outliers_df,
            rep_features
        )

        # Save reports
        save_outputs(report_text, stats_df, high_corr, skewness_df, REPORTS_DIR)

        logger.info("Exploratory Data Analysis completed successfully.")

    except Exception as e:
        logger.exception("An error occurred during EDA execution:")
        raise e


if __name__ == "__main__":
    main()
