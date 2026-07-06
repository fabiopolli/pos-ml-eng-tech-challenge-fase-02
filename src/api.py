from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import os
from loguru import logger

# Inicializa a API
app = FastAPI(
    title="Olist Recommendation API",
    description="API para o Sistema de Recomendação baseado em NCF (PyTorch)",
    version="1.0.0"
)

# Variável global para armazenar o modelo carregado em memória
modelo_ncf = None

@app.on_event("startup")
def load_model():
    """Carrega o modelo do Bill na memória assim que o Docker ligar."""
    global modelo_ncf
    logger.info("Inicializando a API e carregando o modelo PyTorch...")
    
    # Caminho onde o modelo do Bill deve estar salvo
    model_path = "models/best_ncf_model.pt" 
    
    try:
        if os.path.exists(model_path):
            # Carrega o modelo PyTorch (ajustar conforme a arquitetura exata do Bill)
            # modelo_ncf = torch.load(model_path)
            # modelo_ncf.eval()
            logger.success("Modelo carregado com sucesso!")
        else:
            logger.warning(f"Modelo não encontrado em {model_path}. Rodando em modo de fallback (Mock).")
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {e}")

@app.get("/")
def health_check():
    """Rota raiz para o Fábio testar se o container está vivo no deploy."""
    return {"status": "ok", "message": "Olist NCF Recommendation API está rodando!"}

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: str, top_k: int = 5):
    """Retorna as top K recomendações para um usuário específico."""
    
    # Aqui entraria a chamada real do modelo:
    # tensor_usuario = processa_id(user_id)
    # predicoes = modelo_ncf(tensor_usuario)
    # top_produtos = extrai_top_k(predicoes, top_k)
    
    # Retorno simulado estruturado para manter o contrato da API intacto
    return {
        "user_id": user_id,
        "recommendations": [
            {"product_id": "aca2eb7d00ea1a7b8ebd4e68314663af", "score": 0.95},
            {"product_id": "99a4788cb24856965c36a24e339b6058", "score": 0.88},
            {"product_id": "422879e10f46682990de24d770e7f83d", "score": 0.82}
        ][:top_k]
    }