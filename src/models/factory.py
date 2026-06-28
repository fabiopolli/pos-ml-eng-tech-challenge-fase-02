"""Factory Method para instanciação de modelos de recomendação.

Permite criar instâncias de modelos a partir de uma string identificadora,
facilitando a configuração declarativa de experimentos via YAML/CLI.

Padrão: **Factory Method** (Gang of Four).
"""
from __future__ import annotations

from typing import Any, Callable

from src.train import (
    ItemItemCF,
    PopularityBaseline,
    TopRatedBaseline,
    TruncatedSVDBaseline,
)


# Registry nomeado -> classe
_REGISTRY: dict[str, type] = {
    "popularity": PopularityBaseline,
    "top_rated": TopRatedBaseline,
    "item_cf": ItemItemCF,
    "svd": TruncatedSVDBaseline,
}


def model_factory(name: str, **kwargs: Any):
    """Instancia um modelo de recomendação pelo nome.

    Args:
        name: Identificador do modelo (chave em _REGISTRY).
        **kwargs: Parâmetros forwarded para o construtor da classe.

    Returns:
        Instância do modelo (não treinada).

    Raises:
        ValueError: Se o nome não estiver no registry.

    Examples:
        >>> model = model_factory("popularity", k=10)
        >>> model = model_factory("svd", k=20, n_components=100)
        >>> model.fit(train_df)
    """
    name_lower = name.lower()
    if name_lower not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Modelo '{name}' desconhecido. Opções disponíveis: {available}"
        )
    cls = _REGISTRY[name_lower]
    return cls(**kwargs)


def register_model(name: str, cls: type) -> None:
    """Registra uma nova classe de modelo no factory.

    Args:
        name: Identificador único do modelo.
        cls: Classe que implementa a interface recomendada (fit/recommend).
    """
    _REGISTRY[name.lower()] = cls


def list_models() -> list[str]:
    """Retorna a lista de modelos disponíveis."""
    return sorted(_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Verifica se um modelo está registrado."""
    return name.lower() in _REGISTRY