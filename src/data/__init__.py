"""Módulo de dados: splits, pré-processamento e datasets."""
from src.data.dataset import ImplicitFeedbackDataset, build_user_items_map
from src.data.preprocessing import fit_scaler, save_scaler_stats, transform_features
from src.data.splits import temporal_split

__all__ = [
    "ImplicitFeedbackDataset",
    "build_user_items_map",
    "fit_scaler",
    "save_scaler_stats",
    "temporal_split",
    "transform_features",
]