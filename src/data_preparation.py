"""
Olist Dataset Preparation Pipeline for Recommendation System.
"""

from pathlib import Path
import importlib
import logging

import pandas as pd

try:
    logger = importlib.import_module("loguru").logger
except Exception:  # pragma: no cover - fallback for environments without loguru
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "README.md"


def resolve_data_dir() -> Path:
    """Resolve the folder that contains raw CSV files.

    Prefers data/raw for legacy layouts and falls back to data/ when files
    are stored directly under the data folder.
    """
    raw_dir = PROJECT_ROOT / "data" / "raw"
    data_dir = PROJECT_ROOT / "data"

    if raw_dir.exists():
        return raw_dir

    logger.warning(
        f"Directory {raw_dir} not found. Falling back to {data_dir}."
    )
    return data_dir


def load_raw_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load relevant CSVs from the raw data directory.

    Args:
        data_dir: Path to the directory containing raw CSV files.

    Returns:
        Dictionary of DataFrames mapped by lowercased dataset name.
    """
    logger.info(f"Loading raw data from {data_dir}")

    # Load required datasets along with payments to extract payment_type
    datasets = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "products": "olist_products_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "translation": "product_category_name_translation.csv",
        "payments": "olist_order_payments_dataset.csv",
    }

    data = {}
    for name, filename in datasets.items():
        filepath = data_dir / filename
        try:
            df = pd.read_csv(filepath)
            data[name] = df
            logger.info(f"Loaded '{name}': shape {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            raise

    return data


def merge_olist_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge the raw DataFrames into a single interactions DataFrame.

    Args:
        data: Dictionary of loaded DataFrames.

    Returns:
        Merged DataFrame.
    """
    logger.info("Starting merges...")

    # orders + items (INNER)
    merged = pd.merge(data["orders"], data["items"], on="order_id", how="inner")
    logger.info(f"Shape after orders + items (INNER): {merged.shape}")

    # merged + customers (INNER)
    merged = pd.merge(merged, data["customers"], on="customer_id", how="inner")
    logger.info(f"Shape after + customers (INNER): {merged.shape}")

    # merged + products (INNER)
    merged = pd.merge(merged, data["products"], on="product_id", how="inner")
    logger.info(f"Shape after + products (INNER): {merged.shape}")

    # merged + reviews (LEFT)
    merged = pd.merge(merged, data["reviews"], on="order_id", how="left")
    logger.info(f"Shape after + reviews (LEFT): {merged.shape}")

    # merged + translation (LEFT)
    merged = pd.merge(
        merged, data["translation"], on="product_category_name", how="left"
    )
    logger.info(f"Shape after + translation (LEFT): {merged.shape}")

    # merged + payments (LEFT)
    # Group payments by order_id to avoid exploding the DataFrame if there are multiple payment types
    payments_dedup = data["payments"].groupby("order_id").first().reset_index()
    merged = pd.merge(merged, payments_dedup, on="order_id", how="left")
    logger.info(f"Shape after + payments (LEFT): {merged.shape}")

    return merged


def filter_delivered_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter only delivered orders.

    Args:
        df: Merged DataFrame.

    Returns:
        Filtered DataFrame with only delivered orders.
    """
    initial_shape = df.shape[0]
    df = df[df["order_status"] == "delivered"].copy()
    final_shape = df.shape[0]

    removed = initial_shape - final_shape
    logger.info(
        f"Filtered delivered orders. Removed {removed} records. New shape: {df.shape}"
    )
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean columns and engineer new features.

    Args:
        df: Filtered DataFrame.

    Returns:
        Cleaned and engineered DataFrame.
    """
    logger.info("Cleaning data and engineering features...")

    # Log nulls before
    nulls_before = df.isnull().sum().sum()
    logger.info(f"Total nulls before cleaning: {nulls_before}")

    # Create has_review (1 if review_score is not null, 0 otherwise)
    df["has_review"] = df["review_score"].notnull().astype(int)

    # Fill product_category_name_english with 'unknown'
    if "product_category_name_english" in df.columns:
        df["product_category_name_english"] = df[
            "product_category_name_english"
        ].fillna("unknown")
    else:
        logger.warning(
            "'product_category_name_english' not found. Adding as 'unknown'."
        )
        df["product_category_name_english"] = "unknown"

    # Convert order_purchase_timestamp to datetime
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    # Log nulls after
    nulls_after = df.isnull().sum().sum()
    logger.info(f"Total nulls after cleaning: {nulls_after}")

    return df


def aggregate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate interactions to user-item level.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Aggregated user-item interactions DataFrame.
    """
    logger.info("Aggregating interactions...")
    initial_shape = df.shape[0]

    # Ensure payment_type exists (in case it got dropped or wasn't loaded)
    if "payment_type" not in df.columns:
        logger.warning("'payment_type' column missing. Filling with 'unknown'.")
        df["payment_type"] = "unknown"

    agg_funcs = {
        "review_score": "mean",
        "price": "first",
        "freight_value": "mean",
        "payment_type": "first",
        "order_purchase_timestamp": "max",
        "has_review": "max",
    }

    grouped = df.groupby(
        ["customer_unique_id", "product_id", "product_category_name_english"]
    )

    # Calculate purchase count (size of group)
    purchase_counts = grouped.size().reset_index(name="purchase_count")

    # Apply aggregations
    aggregated = grouped.agg(agg_funcs).reset_index()

    # Merge purchase count
    final_df = pd.merge(
        aggregated,
        purchase_counts,
        on=["customer_unique_id", "product_id", "product_category_name_english"],
        how="inner",
    )

    final_shape = final_df.shape[0]
    logger.info(
        f"Aggregated {initial_shape} records into {final_shape} records. Duplicates aggregated: {initial_shape - final_shape}"
    )

    return final_df


def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the processed DataFrame as a parquet file.

    Args:
        df: Processed DataFrame.
        output_path: Path to save the parquet file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved processed data to {output_path} (Size: {size_mb:.2f} MB)")


def generate_validation_report(df: pd.DataFrame) -> dict:
    """
    Generate validation metrics for the processed data.

    Args:
        df: Processed DataFrame.

    Returns:
        Dictionary containing validation metrics.
    """
    logger.info("Generating validation report...")

    total_interactions = len(df)
    unique_users = df["customer_unique_id"].nunique()
    unique_products = df["product_id"].nunique()
    unique_categories = df["product_category_name_english"].nunique()

    sparsity = (
        1.0 - (total_interactions / (unique_users * unique_products))
        if unique_users > 0 and unique_products > 0
        else 0
    )
    missing_review_count = df["review_score"].isnull().sum()
    mean_purchase_count_per_pair = df["purchase_count"].mean()

    date_range = (
        df["order_purchase_timestamp"].min(),
        df["order_purchase_timestamp"].max(),
    )

    report = {
        "total_interactions": total_interactions,
        "unique_users": unique_users,
        "unique_products": unique_products,
        "unique_categories": unique_categories,
        "sparsity": sparsity,
        "missing_review_count": int(missing_review_count),
        "mean_purchase_count_per_pair": float(mean_purchase_count_per_pair),
        "date_range": (str(date_range[0]), str(date_range[1])),
    }

    for k, v in report.items():
        logger.info(f"Validation Metric - {k}: {v}")

    return report


def save_metadata_readme(report: dict, metadata_path: Path) -> None:
    """Save a lightweight markdown summary for the prepared dataset."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# Processed Interactions Dataset",
            "",
            "Generated by `src/data_preparation.py`.",
            "",
            "## Validation Summary",
            f"- total_interactions: {report['total_interactions']}",
            f"- unique_users: {report['unique_users']}",
            f"- unique_products: {report['unique_products']}",
            f"- unique_categories: {report['unique_categories']}",
            f"- sparsity: {report['sparsity']}",
            f"- missing_review_count: {report['missing_review_count']}",
            (
                "- date_range: "
                f"{report['date_range'][0]} to {report['date_range'][1]}"
            ),
        ]
    )
    metadata_path.write_text(content + "\n", encoding="utf-8")
    logger.info(f"Saved metadata report to {metadata_path}")


def main():
    logger.info("Starting Olist Data Preparation Pipeline...")

    try:
        data_dir = resolve_data_dir()

        # 1. Load Data
        data = load_raw_data(data_dir)

        # 2. Merge Data
        merged_df = merge_olist_data(data)

        # 3. Filter
        filtered_df = filter_delivered_orders(merged_df)

        # 4. Clean & Feature Engineer
        cleaned_df = clean_and_engineer_features(filtered_df)

        # 5. Aggregate
        interactions_df = aggregate_interactions(cleaned_df)

        # 6. Save
        save_processed_data(interactions_df, OUTPUT_PATH)

        # 7. Validate
        report = generate_validation_report(interactions_df)

        # 8. Metadata
        save_metadata_readme(report, METADATA_PATH)

        logger.info("Pipeline completed successfully.")

    except Exception:
        logger.exception("Pipeline failed:")
        raise


if __name__ == "__main__":
    main()
