"""Loss functions para recomendação."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking Loss.

    Maximiza a diferença de score entre item positivo e negativo.

    L = -log(sigmoid(pos_score - neg_score))

    Args:
        pos_scores: Scores preditos para itens positivos (batch,).
        neg_scores: Scores preditos para itens negativos (batch,).

    Returns:
        Loss escalar (média do batch).
    """
    return -F.logsigmoid(pos_scores - neg_scores).mean()