"""Métricas de avaliação @K para sistemas de recomendação."""
from __future__ import annotations

from typing import Dict, Set

import numpy as np
import pandas as pd
import torch


# Aux columns (mesmas do config)
_AUX_COLS = [
    "price_log", "freight_value_log", "price_to_freight_ratio", "has_price_outlier",
    "days_since_reference", "is_weekend", "is_holiday_season",
    "category_target_enc", "category_frequency",
    "payment_type_credit_card", "payment_type_boleto",
    "payment_type_voucher", "payment_type_debit_card",
    "user_total_purchases", "user_avg_freight",
    "user_purchase_span_days", "user_recency_days",
    "product_popularity", "product_avg_review_score", "product_review_rate",
]


def _build_item_lookup(df: pd.DataFrame, aux_cols: list[str]) -> Dict[int, np.ndarray]:
    """Lookup por product_id_idx -> (categoria, vetor aux).

    Para itens com múltiplas linhas, pega a primeira ocorrência.
    """
    lookup: Dict[int, np.ndarray] = {}
    cat_lookup: Dict[int, int] = {}
    df_unique = df.drop_duplicates(subset=["product_id_idx"], keep="first")
    for _, row in df_unique.iterrows():
        item_id = int(row["product_id_idx"])
        cat_lookup[item_id] = int(row["category_id"])
        lookup[item_id] = row[aux_cols].values.astype(np.float32)
    return cat_lookup, lookup


def calculate_metrics_at_k(
    recommended_list: np.ndarray,
    true_items_set: set,
    k: int = 10,
) -> Dict[str, float]:
    """Calcula métricas de ranking @K para um único usuário.

    Args:
        recommended_list: Array ordenado de IDs de itens recomendados.
        true_items_set: Set de IDs de itens realmente relevantes.
        k: Cutoff K.

    Returns:
        Dict com HitRate@K, Recall@K, Precision@K, NDCG@K, MAP@K.
    """
    rec_k = recommended_list[:k]
    hits = np.array([1 if item in true_items_set else 0 for item in rec_k])
    num_hits = int(hits.sum())

    hit_rate = 1.0 if num_hits > 0 else 0.0
    precision = num_hits / k if k > 0 else 0.0
    recall = num_hits / len(true_items_set) if len(true_items_set) > 0 else 0.0

    # NDCG
    positions = np.arange(len(hits))
    discounts = 1.0 / np.log2(positions + 2)
    dcg = float((hits * discounts).sum())

    n_relevant = min(len(true_items_set), k)
    idcg = float((1.0 / np.log2(np.arange(n_relevant) + 2)).sum())
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # MAP (Average Precision)
    if num_hits == 0:
        ap = 0.0
    else:
        cum_hits = np.cumsum(hits)
        precisions_at_hits = cum_hits[hits == 1] / (positions[hits == 1] + 1)
        ap = float(precisions_at_hits.mean())
    map_k = ap

    return {
        "HitRate@K": hit_rate,
        "Recall@K": recall,
        "Precision@K": precision,
        "NDCG@K": ndcg,
        "MAP@K": map_k,
    }


def evaluate_model(
    model,
    eval_df: pd.DataFrame,
    eval_aux_normalized: np.ndarray,
    user_items_map: Dict[int, Set[int]],
    all_item_ids: np.ndarray,
    k: int = 10,
    n_neg_eval: int = 99,
    device: str = "cpu",
    filter_cold_start: bool = True,
) -> Dict[str, float]:
    """Avalia o modelo em pares (positivos do eval_df, n_neg_eval negativos aleatórios).

    IMPORTANTE: `eval_aux_normalized` deve estar na MESMA ORDEM de `eval_df`,
    com features já normalizadas pelo scaler fitado no treino.

    Args:
        model: Modelo NCF treinado.
        eval_df: DataFrame com user_id, product_id_idx, category_id.
        eval_aux_normalized: Array (n_eval, n_features) já normalizado.
        user_items_map: Dict user_id -> set de itens já vistos (treino+val).
        all_item_ids: Array com todos os IDs de itens.
        k: Cutoff K.
        n_neg_eval: Número de negativos por positivo.
        device: Device para inferência.
        filter_cold_start: Se True, ignora usuários sem histórico.

    Returns:
        Dict com métricas médias.
    """
    model.eval()

    # Lookups: (item_id -> idx em eval_df)
    item_to_row_idx: Dict[int, int] = {}
    item_to_cat: Dict[int, int] = {}
    for i, row in eval_df.reset_index(drop=True).iterrows():
        item_id = int(row["product_id_idx"])
        if item_id not in item_to_row_idx:
            item_to_row_idx[item_id] = i
            item_to_cat[item_id] = int(row["category_id"])

    n_features = eval_aux_normalized.shape[1]
    rng = np.random.default_rng(42)

    all_metrics: Dict[str, list[float]] = {
        "HitRate@K": [],
        "Recall@K": [],
        "Precision@K": [],
        "NDCG@K": [],
        "MAP@K": [],
    }

    eval_users = eval_df["user_id"].unique()
    test_users_eval = []
    if filter_cold_start:
        for u in eval_users:
            if int(u) in user_items_map and len(user_items_map[int(u)]) > 0:
                test_users_eval.append(u)
            else:
                pass
        n_cold_start = len(eval_users) - len(test_users_eval)
        if n_cold_start > 0:
            print(f"  [eval] {n_cold_start}/{len(eval_users)} usuários cold-start ignorados")
    else:
        test_users_eval = list(eval_users)

    with torch.no_grad():
        for uid in test_users_eval:
            user_data = eval_df[eval_df["user_id"] == uid]
            true_items = set(int(i) for i in user_data["product_id_idx"].tolist())
            if not true_items:
                continue

            seen = user_items_map.get(int(uid), set())

            # Candidatos: positivos + negativos
            candidates: list[int] = []
            for pos_item in true_items:
                candidates.append(int(pos_item))
                neg_count = 0
                attempts = 0
                while neg_count < n_neg_eval and attempts < n_neg_eval * 20:
                    neg = int(rng.choice(all_item_ids))
                    if neg not in seen and neg not in true_items:
                        candidates.append(neg)
                        neg_count += 1
                    attempts += 1

            if not candidates:
                continue

            # Tensores
            n_cand = len(candidates)
            user_tensor = torch.full((n_cand,), int(uid), dtype=torch.long, device=device)
            item_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

            # Categorias
            cats = [item_to_cat.get(c, 0) for c in candidates]
            cat_tensor = torch.tensor(cats, dtype=torch.long, device=device)

            # Aux features: usar do item real se conhecido, senão zeros
            aux_rows = []
            for c in candidates:
                if c in item_to_row_idx:
                    aux_rows.append(eval_aux_normalized[item_to_row_idx[c]])
                else:
                    aux_rows.append(np.zeros(n_features, dtype=np.float32))
            aux_tensor = torch.tensor(np.stack(aux_rows), dtype=torch.float32, device=device)

            scores = model(user_tensor, item_tensor, cat_tensor, aux_tensor).cpu().numpy()
            top_k_idx = np.argsort(scores)[-k:][::-1]
            recommended = np.array([candidates[i] for i in top_k_idx])

            metrics = calculate_metrics_at_k(recommended, true_items, k=k)
            for mname, val in metrics.items():
                all_metrics[mname].append(val)

    if not all_metrics["Recall@K"]:
        return {m: 0.0 for m in all_metrics}

    return {m: float(np.mean(v)) for m, v in all_metrics.items() if v}