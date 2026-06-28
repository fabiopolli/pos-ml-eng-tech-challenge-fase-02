"""Strategy Pattern para estratégias de split/preprocessamento de dados.

Permite trocar a estratégia de particionamento em tempo de execução
(temporal vs random) sem modificar o código cliente.

Padrão: **Strategy** (Gang of Four).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class SplitStrategy(ABC):
    """Interface abstrata para estratégias de split."""

    @abstractmethod
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide o DataFrame em (train, val, test).

        Args:
            df: DataFrame completo.

        Returns:
            Tupla com três DataFrames.
        """
        raise NotImplementedError


class TemporalSplitStrategy(SplitStrategy):
    """Split temporal (ordenação por coluna de tempo, sem shuffling)."""

    def __init__(
        self,
        train_size: float = 0.70,
        val_size: float = 0.15,
        time_col: str = "days_since_reference",
        descending: bool = False,
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.time_col = time_col
        self.descending = descending

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide preservando ordem temporal (anti-leakage)."""
        from src.data.splits import temporal_split
        return temporal_split(
            df,
            time_col=self.time_col,
            train_size=self.train_size,
            val_size=self.val_size,
            descending=self.descending,
        )


class RandomSplitStrategy(SplitStrategy):
    """Split aleatório (para baselines comparativos e sanity checks)."""

    def __init__(self, train_size: float = 0.70, val_size: float = 0.15, seed: int = 42):
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Divide aleatoriamente (NÃO recomendado para avaliação de prod)."""
        train, temp = train_test_split(
            df, train_size=self.train_size, random_state=self.seed
        )
        val_size_adjusted = self.val_size / (1 - self.train_size)
        val, test = train_test_split(
            temp, train_size=val_size_adjusted, random_state=self.seed
        )
        return train, val, test


class SplitContext:
    """Context do Strategy: encapsula a estratégia escolhida."""

    def __init__(self, strategy: SplitStrategy):
        self._strategy = strategy

    @property
    def strategy(self) -> SplitStrategy:
        """Retorna a estratégia atualmente configurada."""
        return self._strategy

    def set_strategy(self, strategy: SplitStrategy) -> None:
        """Troca a estratégia em tempo de execução.

        Args:
            strategy: Nova implementação de SplitStrategy a ser usada.
        """
        self._strategy = strategy

    def execute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Delega a chamada à estratégia atual."""
        return self._strategy.split(df)