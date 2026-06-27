"""Neural Collaborative Filtering com side-information."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class NCFHybrid(nn.Module):
    """NCF Híbrido: Embeddings (user/item/category) + Aux Features → MLP → Score.

    Arquitetura:
        [User ID]   → Embed(emb_dim)
        [Item ID]   → Embed(emb_dim)
        [Cat ID]    → Embed(cat_emb_dim)
        [Aux Feat]  → (já normalizado)
                          ↓
                    Concatenate → MLP → Score
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_categories: int,
        n_aux_features: int,
        emb_dim: int = 32,
        cat_emb_dim: int = 8,
        hidden: List[int] = None,
        dropout: float = 0.3,
    ) -> None:
        """Inicializa o modelo.

        Args:
            n_users: Número total de usuários.
            n_items: Número total de itens.
            n_categories: Número total de categorias.
            n_aux_features: Dimensão das features auxiliares.
            emb_dim: Dimensão dos embeddings de user/item.
            cat_emb_dim: Dimensão do embedding de categoria.
            hidden: Lista de dimensões das camadas ocultas.
            dropout: Taxa de dropout.
        """
        super().__init__()

        if hidden is None:
            hidden = [128, 64, 32]

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.cat_emb = nn.Embedding(n_categories, cat_emb_dim)

        input_dim = (emb_dim * 2) + cat_emb_dim + n_aux_features

        layers: list[nn.Module] = []
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicialização Xavier para MLP e normal para embeddings."""
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.cat_emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        category: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            user: Tensor (batch,) com user IDs.
            item: Tensor (batch,) com item IDs.
            category: Tensor (batch,) com category IDs.
            aux_features: Tensor (batch, n_aux_features) com features normalizadas.

        Returns:
            Tensor (batch,) com scores logit.
        """
        u_vec = self.user_emb(user)
        i_vec = self.item_emb(item)
        c_vec = self.cat_emb(category)
        x = torch.cat([u_vec, i_vec, c_vec, aux_features], dim=-1)
        return self.mlp(x).squeeze(-1)