"""Dataset PyTorch para feedback implícito com negative sampling on-the-fly."""
from __future__ import annotations

from typing import Dict, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ImplicitFeedbackDataset(Dataset):
    """Dataset de pares (user, item_positivo, item_negativo, features).

    O negative sampling é feito on-the-fly a cada época para garantir
    variabilidade e evitar overfitting a negativos fixos.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        all_item_ids: np.ndarray,
        user_items_map: Dict[int, Set[int]],
        numeric_features: np.ndarray,
        category_ids: np.ndarray,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_negatives: int = 1,
    ) -> None:
        """Inicializa o dataset.

        Args:
            interactions_df: DataFrame com colunas user_id, product_id_idx.
            all_item_ids: Array com todos os IDs de produtos possíveis.
            user_items_map: Dict mapeando user_id -> set de items positivos.
            numeric_features: Array (n_interactions, n_features) normalizado.
            category_ids: Array (n_interactions,) com categoria de cada item.
            user_ids: Array (n_interactions,) com user_id indexado.
            item_ids: Array (n_interactions,) com product_id_idx.
            n_negatives: Número de negativos por positivo.
        """
        self.n_interactions = len(interactions_df)
        self.all_item_ids = all_item_ids
        self.user_items_map = user_items_map
        self.numeric_features = torch.from_numpy(numeric_features).float()
        self.category_ids = torch.from_numpy(category_ids).long()
        self.user_ids = torch.from_numpy(user_ids).long()
        self.item_ids = torch.from_numpy(item_ids).long()
        self.n_negatives = n_negatives

    def __len__(self) -> int:
        return self.n_interactions

    def __getitem__(self, idx: int):
        user = self.user_ids[idx]
        pos_item = self.item_ids[idx]
        cat_pos = self.category_ids[idx]
        aux = self.numeric_features[idx]

        # Negative sampling on-the-fly
        seen = self.user_items_map.get(user.item(), set())
        neg_items = []
        for _ in range(self.n_negatives):
            neg_item = int(np.random.choice(self.all_item_ids))
            while neg_item in seen:
                neg_item = int(np.random.choice(self.all_item_ids))
            neg_items.append(neg_item)

        return {
            "user": user,
            "pos_item": pos_item,
            "pos_category": cat_pos,
            "neg_item": torch.tensor(neg_items, dtype=torch.long),
            "aux_features": aux,
        }


def build_user_items_map(df: pd.DataFrame) -> Dict[int, Set[int]]:
    """Constrói mapeamento user -> set de itens positivos."""
    user_items = df.groupby("user_id")["product_id_idx"].apply(set).to_dict()
    return user_items