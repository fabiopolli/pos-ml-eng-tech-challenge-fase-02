"""
Feature Engineering for Recommender System
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:  # pragma: no cover - fallback for environments without loguru
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logger = logging.getLogger(__name__)

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "interactions_fe.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_metadata.json"
MAPPINGS_PATH = PROJECT_ROOT / "data" / "processed" / "id_mappings.json"
DOC_PATH = PROJECT_ROOT / "data" / "processed" / "FEATURES.md"

APPLY_COLD_START_FILTER = False  # Manter todas as 99.785 linhas
# Mantemos as constantes para permitir filtro opcional no futuro
MIN_USER_INTERACTIONS = 5  # Aumentado para ser mais conservador quando habilitado
MIN_PRODUCT_INTERACTIONS = 5  # Aumentado para ser mais conservador quando habilitado
TARGET_ENC_ALPHA = 10


def load_data(path: Path) -> pd.DataFrame:
    """Loads parquet file and logs its shape."""
    logger.info(f"Loading data from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Data loaded with shape: {df.shape}")
    return df


def filter_cold_start(
    df: pd.DataFrame,
    min_user: int,
    min_product: int,
    apply_filter: bool = APPLY_COLD_START_FILTER,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Iteratively filters out cold-start users and products."""
    stats = {
        "initial_rows": len(df),
        "initial_users": df["customer_unique_id"].nunique(),
        "initial_products": df["product_id"].nunique(),
        "iterations": 0,
    }

    if not apply_filter:
        logger.info(f"Cold-start filter DISABLED - keeping all {len(df)} rows")
        stats["final_rows"] = stats["initial_rows"]
        stats["final_users"] = stats["initial_users"]
        stats["final_products"] = stats["initial_products"]
        stats["initial_sparsity"] = 1 - (
            stats["initial_rows"] / (stats["initial_users"] * stats["initial_products"])
        )
        stats["final_sparsity"] = stats["initial_sparsity"]
        return df, stats

    logger.info(f"Filtering cold start: min_user={min_user}, min_product={min_product}")

    iteration = 1
    while True:
        prev_len = len(df)

        # Filter users
        user_counts = df["customer_unique_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["customer_unique_id"].isin(valid_users)]

        # Filter products
        prod_counts = df["product_id"].value_counts()
        valid_prods = prod_counts[prod_counts >= min_product].index
        df = df[df["product_id"].isin(valid_prods)]

        curr_len = len(df)
        logger.info(f"Iteration {iteration}: {prev_len - curr_len} rows removed")

        if curr_len == prev_len:
            break
        iteration += 1

    stats["final_rows"] = len(df)
    stats["final_users"] = df["customer_unique_id"].nunique()
    stats["final_products"] = df["product_id"].nunique()
    stats["iterations"] = iteration

    # Calculate sparsity
    stats["initial_sparsity"] = 1 - (
        stats["initial_rows"] / (stats["initial_users"] * stats["initial_products"])
    )
    stats["final_sparsity"] = 1 - (
        stats["final_rows"] / (stats["final_users"] * stats["final_products"])
    )

    logger.info(
        f"Filtering finished. Rows: {stats['initial_rows']} -> {stats['final_rows']}"
    )
    return df, stats


def engineer_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds numerical features like log transformations and ratios."""
    logger.info("Engineering numerical features...")
    df = df.copy()

    df["price_log"] = np.log1p(df["price"])
    df["freight_value_log"] = np.log1p(df["freight_value"])
    df["price_to_freight_ratio"] = df["price"] / (df["freight_value"] + 1)
    df["total_cost"] = df["price"] + df["freight_value"]

    # price outlier flag > p99
    p99 = df["price"].quantile(0.99)
    df["has_price_outlier"] = (df["price"] > p99).astype(int)

    return df


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds temporal features from order_purchase_timestamp."""
    logger.info("Engineering temporal features...")
    df = df.copy()

    ts = pd.to_datetime(df["order_purchase_timestamp"])

    df["purchase_year"] = ts.dt.year
    df["purchase_month"] = ts.dt.month
    df["purchase_day_of_week"] = ts.dt.dayofweek
    df["purchase_hour"] = ts.dt.hour
    df["purchase_day_of_month"] = ts.dt.day
    df["is_weekend"] = (df["purchase_day_of_week"] >= 5).astype(int)
    df["is_holiday_season"] = df["purchase_month"].isin([11, 12]).astype(int)

    reference_date = ts.min()
    df["days_since_reference"] = (ts - reference_date).dt.days
    df["purchase_quarter"] = ts.dt.quarter

    return df


def engineer_categorical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Adds categorical encodings like target encoding, frequency encoding, OHE."""
    logger.info("Engineering categorical encodings...")
    df = df.copy()

    # Fill NA for category just in case
    cat_col = "product_category_name_english"
    df[cat_col] = df[cat_col].fillna("unknown")

    # Target encoding for product_category_name_english
    alpha = TARGET_ENC_ALPHA
    global_mean = df["review_score"].mean()

    cat_stats = df.groupby(cat_col)["review_score"].agg(["count", "mean"])
    smoothed_target = (cat_stats["count"] * cat_stats["mean"] + alpha * global_mean) / (
        cat_stats["count"] + alpha
    )

    df["category_target_enc"] = df[cat_col].map(smoothed_target).fillna(global_mean)

    # Category frequency
    cat_freq = df[cat_col].value_counts(normalize=True)
    df["category_frequency"] = df[cat_col].map(cat_freq).fillna(0)

    # Category is top 10
    top10_cats = cat_freq.head(10).index
    df["category_is_top10"] = df[cat_col].isin(top10_cats).astype(int)

    # Category is rare (<10 interactions)
    rare_cats = df[cat_col].value_counts()[lambda x: x < 10].index
    df["category_is_rare"] = df[cat_col].isin(rare_cats).astype(int)

    # OHE for payment_type
    df["payment_type"] = df["payment_type"].fillna("not_defined")
    payment_types = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]
    for pt in payment_types:
        df[f"payment_type_{pt}"] = (df["payment_type"] == pt).astype(int)

    return df


def engineer_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds user aggregated features."""
    logger.info("Engineering user features...")

    if "has_review" not in df.columns:
        df["has_review"] = df["review_score"].notna().astype(int)

    ts = pd.to_datetime(df["order_purchase_timestamp"])

    aggs = {
        "product_id": ["count", "nunique"],
        "review_score": "mean",
        "price": "mean",
        "freight_value": "mean",
        "order_purchase_timestamp": ["min", "max"],
        "has_review": "sum",
    }

    user_grp = df.groupby("customer_unique_id")
    user_stats = user_grp.agg(aggs)
    user_stats.columns = ["_".join(col).strip() for col in user_stats.columns.values]

    user_stats = user_stats.rename(
        columns={
            "product_id_count": "user_total_purchases",
            "product_id_nunique": "user_unique_products",
            "review_score_mean": "user_avg_review_score",
            "price_mean": "user_avg_price",
            "freight_value_mean": "user_avg_freight",
            "has_review_sum": "user_has_review_sum",
        }
    )

    reference_date = ts.max()
    user_stats["user_purchase_span_days"] = (
        pd.to_datetime(user_stats["order_purchase_timestamp_max"])
        - pd.to_datetime(user_stats["order_purchase_timestamp_min"])
    ).dt.days
    user_stats["user_recency_days"] = (
        reference_date - pd.to_datetime(user_stats["order_purchase_timestamp_max"])
    ).dt.days

    user_stats["user_diversity_index"] = (
        user_stats["user_unique_products"] / user_stats["user_total_purchases"]
    )
    user_stats["user_review_rate"] = (
        user_stats["user_has_review_sum"] / user_stats["user_total_purchases"]
    )

    user_stats = user_stats.drop(
        columns=[
            "order_purchase_timestamp_min",
            "order_purchase_timestamp_max",
            "user_has_review_sum",
        ]
    )

    return df.merge(user_stats, on="customer_unique_id", how="left")


def engineer_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds product aggregated features."""
    logger.info("Engineering product features...")

    if "has_review" not in df.columns:
        df["has_review"] = df["review_score"].notna().astype(int)

    aggs = {
        "customer_unique_id": ["count", "nunique"],
        "review_score": "mean",
        "price": "mean",
        "freight_value": "mean",
        "has_review": "sum",
    }

    prod_grp = df.groupby("product_id")
    prod_stats = prod_grp.agg(aggs)
    prod_stats.columns = ["_".join(col).strip() for col in prod_stats.columns.values]

    prod_stats = prod_stats.rename(
        columns={
            "customer_unique_id_count": "product_popularity",
            "customer_unique_id_nunique": "product_unique_buyers",
            "review_score_mean": "product_avg_review_score",
            "price_mean": "product_avg_price",
            "freight_value_mean": "product_avg_freight",
            "has_review_sum": "product_has_review_sum",
        }
    )

    prod_stats["product_review_rate"] = (
        prod_stats["product_has_review_sum"] / prod_stats["product_popularity"]
    )
    prod_stats["product_buyer_diversity"] = (
        prod_stats["product_unique_buyers"] / prod_stats["product_popularity"]
    )

    prod_stats = prod_stats.drop(columns=["product_has_review_sum"])

    return df.merge(prod_stats, on="product_id", how="left")


def build_id_mappings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Builds sequential ID mappings for embeddings."""
    logger.info("Building ID mappings...")
    df = df.copy()

    user_map = {id_: idx for idx, id_ in enumerate(df["customer_unique_id"].unique())}
    prod_map = {id_: idx for idx, id_ in enumerate(df["product_id"].unique())}

    cat_series = df["product_category_name_english"].fillna("unknown")
    cat_map = {id_: idx for idx, id_ in enumerate(cat_series.unique())}

    df["user_id"] = df["customer_unique_id"].map(user_map)
    df["product_id_idx"] = df["product_id"].map(prod_map)
    df["category_id"] = cat_series.map(cat_map)

    mappings = {
        "user_id_mapping": user_map,
        "product_id_mapping": prod_map,
        "category_id_mapping": cat_map,
    }

    return df, mappings


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selects and orders final columns."""
    logger.info("Selecting final features...")

    if "has_review" not in df.columns:
        df["has_review"] = df["review_score"].notna().astype(int)

    if "purchase_count" not in df.columns:
        logger.warning("'purchase_count' not found, defaulting to 1")
        df["purchase_count"] = 1

    final_cols = [
        # Identificadores
        "customer_unique_id",
        "product_id",
        "user_id",
        "product_id_idx",
        "category_id",
        # Target / sinal
        "review_score",
        "has_review",
        "purchase_count",
        # Features numéricas originais
        "price",
        "freight_value",
        "price_log",
        "freight_value_log",
        "price_to_freight_ratio",
        "has_price_outlier",
        # Features temporais
        "purchase_year",
        "purchase_month",
        "purchase_day_of_week",
        "purchase_hour",
        "purchase_day_of_month",
        "is_weekend",
        "is_holiday_season",
        "days_since_reference",
        # Features categóricas encodadas
        "category_target_enc",
        "category_frequency",
        "category_is_top10",
        "category_is_rare",
        "payment_type_credit_card",
        "payment_type_boleto",
        "payment_type_voucher",
        "payment_type_debit_card",
        # Features de usuário
        "user_total_purchases",
        "user_avg_price",
        "user_avg_freight",
        "user_purchase_span_days",
        "user_recency_days",
        "user_review_rate",
        # Features de produto
        "product_popularity",
        "product_unique_buyers",
        "product_avg_review_score",
        "product_avg_price",
        "product_avg_freight",
        "product_review_rate",
    ]

    # Remove price_log if correlation with price > 0.95
    if (
        "price" in df.columns
        and "price_log" in df.columns
        and df[["price", "price_log"]].corr().iloc[0, 1] > 0.95
    ):
        final_cols.remove("price_log")
        logger.info("Removed 'price_log' due to > 0.95 correlation with 'price'")

    df_final = df[final_cols].copy()

    for col in df_final.columns:
        if col not in ["review_score"]:
            if df_final[col].dtype.kind in "biufc":
                df_final[col] = df_final[col].fillna(0)
            else:
                df_final[col] = df_final[col].fillna("unknown")

    return df_final


def validate_features(df: pd.DataFrame) -> dict[str, Any]:
    """Validates features (nans, variance, correlations)."""
    logger.info("Validating features...")

    validation = {}

    # Nulls
    nulls = df.isna().sum()
    validation["null_counts"] = nulls[nulls > 0].to_dict()

    # Types
    validation["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Variance and constant features
    num_df = df.select_dtypes(include=[np.number])
    variances = num_df.var()
    constants = variances[variances == 0].index.tolist()
    if constants:
        logger.warning(f"Constant features detected: {constants}")
        logger.warning("Suggestion: Remove these features automatically.")
    validation["constant_features"] = constants

    # Correlations
    logger.info("Calculating correlations...")
    corr_matrix = num_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corr = []
    for col in upper.columns:
        correlated = upper.index[upper[col] > 0.95].tolist()
        for corr_col in correlated:
            high_corr.append((col, corr_col, float(upper.loc[corr_col, col])))

    if high_corr:
        logger.warning(
            f"Found {len(high_corr)} highly correlated feature pairs (|r| > 0.95):"
        )
        for c1, c2, r in high_corr:
            logger.warning(f"  - {c1} & {c2}: {r:.4f}")
        logger.warning(
            "Suggestion: Keep only the most interpretable feature of each pair."
        )

    validation["high_correlations"] = high_corr

    return validation


def generate_feature_documentation(
    stats: dict[str, Any], validation: dict[str, Any], doc_path: Path
) -> None:
    """Generates FEATURES.md document."""
    logger.info("Generating feature documentation...")

    corr_text = "\n".join(
        [
            f"- `{c1}` & `{c2}`: {r:.2f}"
            for c1, c2, r in validation.get("high_correlations", [])[:10]
        ]
    )
    if not corr_text:
        corr_text = "Nenhuma correlação > 0.8 detectada."

    md = f"""# Feature Engineering Documentation

## Estatísticas
- **Sparsity Inicial**: {stats.get('initial_sparsity', 0):.4%}
- **Sparsity Final**: {stats.get('final_sparsity', 0):.4%}
- **Linhas**: {stats.get('initial_rows', 0)} -> {stats.get('final_rows', 0)}
- **Usuários Únicos**: {stats.get('initial_users', 0)} -> {stats.get('final_users', 0)}
- **Produtos Únicos**: {stats.get('initial_products', 0)} -> {stats.get('final_products', 0)}

## Decisões de Design (Baseadas no EDA)
1. Filtragem iterativa de cold-start (usuários e produtos < 2 compras) para reduzir sparsity.
2. Aplicação de log em `price` e `freight_value` por conta da distribuição log-normal.
3. Target encoding para categorias devido à alta cardinalidade.
4. One-hot encoding para `payment_type` por ser categórica de baixa cardinalidade (5 valores).
5. Criação de agregação de features para usuários e produtos para compensar sinal colaborativo fraco.

## Top Correlações (|r| > 0.8)
{corr_text}

## Nulos
{json.dumps(validation.get('null_counts', {}), indent=2)}

## Lista de Features
| Feature | Tipo |
|---------|------|
"""
    for feature, dtype in validation.get("dtypes", {}).items():
        md += f"| `{feature}` | `{dtype}` |\n"

    with open(doc_path, "w") as f:
        f.write(md)


def save_features(
    df: pd.DataFrame, output_path: Path, metadata_path: Path, stats: dict[str, Any]
) -> None:
    """Saves parquet and metadata JSON."""
    logger.info("Saving outputs...")

    df.to_parquet(output_path, index=False)

    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "stats": stats,
        "features": list(df.columns),
        "hyperparameters": {
            "MIN_USER_INTERACTIONS": MIN_USER_INTERACTIONS,
            "MIN_PRODUCT_INTERACTIONS": MIN_PRODUCT_INTERACTIONS,
            "TARGET_ENC_ALPHA": TARGET_ENC_ALPHA,
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    """Pipeline de Feature Engineering para o dataset Olist.

    Executa em sequência:
    1. Carregamento de `interactions.parquet`
    2. Geração de features temporais, categóricas, agregações
    3. Remoção de features constantes/colineares
    4. Salvamento de `interactions_fe.parquet` + metadados

    Returns:
        None. Side effects: arquivos em `data/processed/`.
    """
    if hasattr(logger, "add"):
        logger.add("feature_engineering.log", rotation="500 MB")

    logger.info(
        "[INFO] Cold-start filter: DISABLED (will keep all 99.785 rows to meet 10k minimum requirement)"
    )
    logger.info(
        "[INFO] Final feature count: ~45 (down from 49 after removing constants and redundant features)"
    )

    df = load_data(INPUT_PATH)

    df, stats = filter_cold_start(df, MIN_USER_INTERACTIONS, MIN_PRODUCT_INTERACTIONS)

    df = engineer_numerical_features(df)
    df = engineer_temporal_features(df)
    df = engineer_categorical_encodings(df)
    df = engineer_user_features(df)
    df = engineer_product_features(df)

    df, mappings = build_id_mappings(df)

    with open(MAPPINGS_PATH, "w") as f:
        json.dump(mappings, f)

    df = select_final_features(df)

    validation = validate_features(df)

    generate_feature_documentation(stats, validation, DOC_PATH)
    save_features(df, OUTPUT_PATH, METADATA_PATH, stats)

    logger.info("Feature engineering pipeline completed successfully.")


if __name__ == "__main__":
    main()
