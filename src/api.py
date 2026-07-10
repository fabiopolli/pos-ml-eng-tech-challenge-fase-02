"""API FastAPI do Sistema de Recomendação Olist.

Stub do contrato HTTP para deploy. O carregamento real do modelo é feito no
startup; o endpoint `/recommend/{user_id}` retorna uma resposta mockada até o
modelo Production ser plugado.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from loguru import logger

app = FastAPI(
    title="Olist Recommendation API",
    description="API para o Sistema de Recomendação baseado em NCF (PyTorch)",
    version="1.0.0",
)

# Variável global para armazenar o modelo carregado em memória
modelo_ncf = None


@app.on_event("startup")
def load_model() -> None:
    """Carrega o modelo PyTorch quando o container sobe."""
    global modelo_ncf
    logger.info("Inicializando a API e carregando o modelo PyTorch...")

    # Caminho onde o modelo deve estar salvo
    model_path = "models/best_ncf_model.pt"

    try:
        if os.path.exists(model_path):
            # Quando o modelo Production for plugado, instanciar:
            # modelo_ncf = NCFHybrid(...)
            # modelo_ncf.load_state_dict(torch.load(model_path))
            # modelo_ncf.eval()
            logger.success("Modelo carregado com sucesso!")
        else:
            logger.warning(
                f"Modelo não encontrado em {model_path}. "
                "Rodando em modo de fallback (Mock)."
            )
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {e}")


@app.get("/")
def health_check() -> dict:
    """Rota raiz para verificação de saúde do container."""
    return {"status": "ok", "message": "Olist NCF Recommendation API está rodando!"}


@app.get("/recommend/{user_id}")
def get_recommendations(user_id: str, top_k: int = 5) -> dict:
    """Retorna as top K recomendações para um usuário específico."""
    return {
        "user_id": user_id,
        "recommendations": [
            {"product_id": "aca2eb7d00ea1a7b8ebd4e68314663af", "score": 0.95},
            {"product_id": "99a4788cb24856965c36a24e339b6058", "score": 0.88},
            {"product_id": "422879e10f46682990de24d770e7f83d", "score": 0.82},
        ][:top_k],
    }
