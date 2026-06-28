"""Configurações centralizadas do projeto via pydantic-settings.

Este módulo provê uma única fonte de verdade para paths, hiperparâmetros e
configurações de tracking/avaliação. Todas as variáveis podem ser sobrescritas
via arquivo .env ou variáveis de ambiente.

Example:
    >>> from src.config import settings
    >>> settings.ncf_emb_dim
    32
    >>> settings.mlflow_uri
    'sqlite:///./artifacts/mlflow.db'
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Configurações globais do projeto Olist Recommender System."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Paths ----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data" / "processed"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    configs_dir: Path = PROJECT_ROOT / "configs"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # ---- Datasets ----
    interactions_path: Path = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
    interactions_fe_path: Path = PROJECT_ROOT / "data" / "processed" / "interactions_fe.parquet"
    id_mappings_path: Path = PROJECT_ROOT / "data" / "processed" / "id_mappings.json"
    feature_metadata_path: Path = PROJECT_ROOT / "data" / "processed" / "feature_metadata.json"

    # ---- MLflow ----
    mlflow_uri: str = "sqlite:///./artifacts/mlflow.db"
    experiment_name: str = "Olist_Recommender"

    # ---- Reprodutibilidade ----
    seed: int = 42

    # ---- Device ----
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    # ---- Split temporal ----
    train_size: float = 0.70
    val_size: float = 0.15
    time_col: str = "days_since_reference"

    # ---- Hiperparâmetros NCF ----
    ncf_emb_dim: int = 32
    ncf_hidden: list[int] = Field(default_factory=lambda: [64, 32])
    ncf_dropout: float = 0.5
    ncf_lr: float = 5e-4
    ncf_weight_decay: float = 5e-4
    ncf_batch_size: int = 2048
    n_epochs: int = 15
    n_negatives: int = 8

    # ---- Avaliação top-K ----
    k_eval: int = 10
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    n_neg_eval: int = 99

    # ---- Baselines ----
    svd_n_components: int = 50
    item_cf_min_support: int = 2

    def get_device(self) -> str:
        """Resolve device efetivo (considera torch.cuda.is_available())."""
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def ensure_dirs(self) -> None:
        """Cria diretórios essenciais se não existirem."""
        for d in [self.artifacts_dir, self.reports_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
