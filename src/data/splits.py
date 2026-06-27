"""Split temporal de dados para sistemas de recomendação.

Garante que train < val < test cronologicamente, evitando data leakage.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    time_col: str = "days_since_reference",
    train_size: float = 0.70,
    val_size: float = 0.15,
    descending: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide dataset em train/val/test baseado em ordenação temporal.

    Por padrão, assume que `days_since_reference` é proxy temporal:
    valores menores = compras mais recentes. Usamos ascending para que
    os dados mais recentes fiquem no test set (cenário realista).

    Args:
        df: DataFrame com coluna temporal.
        time_col: Nome da coluna proxy de tempo.
        train_size: Proporção do treino (default 0.70).
        val_size: Proporção da validação (default 0.15). Teste recebe o resto.
        descending: Se True, ordena decrescente (mais recente primeiro).

    Returns:
        Tupla (train_df, val_df, test_df).

    Raises:
        AssertionError: Se houver vazamento temporal.
    """
    df_sorted = df.sort_values(by=time_col, ascending=not descending).reset_index(drop=True)

    n_total = len(df_sorted)
    train_end = int(n_total * train_size)
    val_end = int(n_total * (train_size + val_size))

    train_data = df_sorted.iloc[:train_end].copy()
    val_data = df_sorted.iloc[train_end:val_end].copy()
    test_data = df_sorted.iloc[val_end:].copy()

    # Validação rigorosa de leakage temporal
    train_max = train_data[time_col].max()
    val_min, val_max = val_data[time_col].min(), val_data[time_col].max()
    test_min = test_data[time_col].min()

    assert train_max <= val_min, (
        f"Vazamento: treino ({train_max}) avança sobre validação ({val_min})"
    )
    assert val_max <= test_min, (
        f"Vazamento: validação ({val_max}) avança sobre teste ({test_min})"
    )

    print(f"Treino: {len(train_data):,} registros (até {train_max})")
    print(f"Validação: {len(val_data):,} registros ({val_min} → {val_max})")
    print(f"Teste: {len(test_data):,} registros (a partir de {test_min})")

    return train_data, val_data, test_data