"""Pré-processamento de features numéricas com StandardScaler."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> StandardScaler:
    """Ajusta StandardScaler apenas no conjunto de treino.

    Args:
        train_df: DataFrame de treino.
        feature_cols: Lista de colunas numéricas para normalizar.

    Returns:
        Scaler ajustado.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].fillna(0).astype(np.float32))
    return scaler


def transform_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> np.ndarray:
    """Aplica transformação do scaler nas features.

    Args:
        df: DataFrame a transformar.
        scaler: Scaler previamente ajustado.
        feature_cols: Colunas a normalizar.

    Returns:
        Array numpy (n_samples, n_features) normalizado.
    """
    return scaler.transform(df[feature_cols].fillna(0).astype(np.float32))


def save_scaler_stats(
    scaler: StandardScaler,
    feature_cols: list[str],
    output_path: Path,
) -> None:
    """Persiste estatísticas do scaler para auditoria."""
    stats = {
        "feature_cols": feature_cols,
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)