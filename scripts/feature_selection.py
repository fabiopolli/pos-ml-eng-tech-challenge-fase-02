"""
Etapa 1: Análise de Features para NCF.

Executa:
1. Matriz de correlação (Spearman)
2. Identificação de pares com |r| > 0.9
3. Mutual Information com target
4. Random Forest feature importance
5. Salva lista final em configs/selected_features.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression


# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "interactions_fe.parquet"
OUTPUT_PATH = PROJECT_ROOT / "configs" / "selected_features.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_selection_report.md"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# Features candidatas (todas as 42 do dataset, exceto IDs brutos e target leakage)
CANDIDATE_FEATURES = [
    "review_score", "has_review", "purchase_count",
    "price", "freight_value", "price_log", "freight_value_log",
    "price_to_freight_ratio", "has_price_outlier",
    "purchase_year", "purchase_month", "purchase_day_of_week",
    "purchase_hour", "purchase_day_of_month", "is_weekend", "is_holiday_season",
    "days_since_reference", "category_target_enc", "category_frequency",
    "category_is_top10", "category_is_rare",
    "payment_type_credit_card", "payment_type_boleto",
    "payment_type_voucher", "payment_type_debit_card",
    "user_total_purchases", "user_avg_price", "user_avg_freight",
    "user_purchase_span_days", "user_recency_days", "user_review_rate",
    "product_popularity", "product_unique_buyers",
    "product_avg_review_score", "product_avg_price", "product_avg_freight",
    "product_review_rate",
]

# Features para embedding (NÃO entram no scaler)
EMBEDDING_FEATURES = ["user_id", "product_id_idx", "category_id"]

# Features finais selecionadas (20)
SELECTED_NUMERIC = [
    "price_log", "freight_value_log", "price_to_freight_ratio", "has_price_outlier",
    "days_since_reference", "is_weekend", "is_holiday_season",
    "category_target_enc", "category_frequency",
    "payment_type_credit_card", "payment_type_boleto",
    "payment_type_voucher", "payment_type_debit_card",
    "user_total_purchases", "user_avg_freight",
    "user_purchase_span_days", "user_recency_days",
    "product_popularity", "product_avg_review_score", "product_review_rate",
]


def correlation_analysis(df: pd.DataFrame) -> dict:
    """Identifica pares altamente correlacionados."""
    numeric = df[CANDIDATE_FEATURES].select_dtypes(include=[np.number])
    corr = numeric.corr(method="spearman")

    high_corr_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.9:
                high_corr_pairs.append({
                    "f1": cols[i], "f2": cols[j], "corr": float(r),
                })

    # Features redundantes a remover
    to_drop = set()
    for pair in high_corr_pairs:
        # Manter a feature mais interpretável
        f1, f2 = pair["f1"], pair["f2"]
        if "log" in f1:
            to_drop.add(f2)
        elif "log" in f2:
            to_drop.add(f1)
        elif "unique_buyers" in f1 or "unique_buyers" in f2:
            to_drop.add("product_unique_buyers")
        elif "user_avg_price" in (f1, f2):
            to_drop.add("user_avg_price")
        elif "product_avg_price" in (f1, f2):
            to_drop.add("product_avg_price")
        elif "user_review_rate" in (f1, f2):
            to_drop.add("user_review_rate")

    return {
        "high_corr_pairs": high_corr_pairs,
        "redundant_features": sorted(to_drop),
    }


def mutual_information_scores(df: pd.DataFrame, target: str = "review_score") -> pd.Series:
    """Calcula MI score entre cada feature e o target."""
    X = df[CANDIDATE_FEATURES].fillna(0).astype(np.float32)
    y = df[target].fillna(df[target].median()).astype(np.float32)

    mi = mutual_info_regression(X, y, random_state=42, n_jobs=-1)
    return pd.Series(mi, index=CANDIDATE_FEATURES).sort_values(ascending=False)


def random_forest_importance(df: pd.DataFrame, target: str = "review_score") -> pd.Series:
    """Calcula importância via Random Forest."""
    X = df[CANDIDATE_FEATURES].fillna(0).astype(np.float32)
    y = df[target].fillna(df[target].median()).astype(np.float32)

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=CANDIDATE_FEATURES).sort_values(ascending=False)


def main() -> None:
    print("=" * 60)
    print("ETAPA 1: ANÁLISE DE FEATURES")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset carregado: {df.shape}")

    # 1. Análise de correlação
    print("\n[1/3] Análise de correlação (Spearman)...")
    corr_results = correlation_analysis(df)
    print(f"  Pares com |r| > 0.9: {len(corr_results['high_corr_pairs'])}")
    print(f"  Features redundantes removidas: {corr_results['redundant_features']}")

    # 2. Mutual Information
    print("\n[2/3] Mutual Information vs review_score...")
    mi_scores = mutual_information_scores(df)
    print(mi_scores.head(10).to_string())

    # 3. Random Forest importance
    print("\n[3/3] Random Forest feature importance...")
    rf_importance = random_forest_importance(df)
    print(rf_importance.head(10).to_string())

    # 4. Salvar configuração final
    config = {
        "embedding_features": EMBEDDING_FEATURES,
        "numeric_features": SELECTED_NUMERIC,
        "n_users": int(df["user_id"].max() + 1),
        "n_items": int(df["product_id_idx"].max() + 1),
        "n_categories": int(df["category_id"].max() + 1),
        "target": "implicit_feedback",  # BPR usa compra como positivo
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\n[OK] Config salva em {OUTPUT_PATH}")

    # 5. Relatório
    report = ["# Relatório de Feature Selection\n\n"]
    report.append("## Features Removidas por Multicolinearidade (|r| > 0.9)\n\n")
    for pair in corr_results["high_corr_pairs"]:
        report.append(
            f"- `{pair['f1']}` ↔ `{pair['f2']}` (r={pair['corr']:.3f})\n"
        )
    report.append(f"\n**Total removido:** {len(corr_results['redundant_features'])} features\n\n")

    report.append("## Top 10 Mutual Information\n\n")
    report.append("| Feature | MI Score |\n|---|---|\n")
    for feat, score in mi_scores.head(10).items():
        report.append(f"| {feat} | {score:.4f} |\n")

    report.append("\n## Top 10 Random Forest Importance\n\n")
    report.append("| Feature | Importance |\n|---|---|\n")
    for feat, imp in rf_importance.head(10).items():
        report.append(f"| {feat} | {imp:.4f} |\n")

    report.append(f"\n## Features Selecionadas (Total: {len(SELECTED_NUMERIC)})\n\n")
    for feat in SELECTED_NUMERIC:
        report.append(f"- `{feat}`\n")

    with open(REPORT_PATH, "w") as f:
        f.writelines(report)
    print(f"[OK] Relatório salvo em {REPORT_PATH}")


if __name__ == "__main__":
    main()