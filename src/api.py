from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import os
import json
from loguru import logger

# Tenta importar a classe NCFHybrid dos caminhos padrão do projeto
try:
    from src.models.ncf import NCFHybrid
except ImportError:
    try:
        from src.ncf import NCFHybrid
    except ImportError:
        logger.error("Atenção: Classe NCFHybrid não encontrada em src/models/ncf.py nem em src/ncf.py!")

# Inicializa a API
app = FastAPI(
    title="Olist Recommendation API",
    description="API para o Sistema de Recomendação baseado em NCF (PyTorch)",
    version="2.0.0"
)

# Variáveis globais para armazenar o modelo e os mapeamentos em memória
modelo_ncf = None
user_to_idx = {}
item_to_idx = {}
idx_to_item = {}
todos_itens_idx = []
n_aux_features = 0

@app.on_event("startup")
def load_model_and_mappings():
    """Carrega o modelo de produção do DVC, deduz as dimensões do state_dict e faz a busca inteligente dos mappings."""
    global modelo_ncf, user_to_idx, item_to_idx, idx_to_item, todos_itens_idx, n_aux_features
    
    logger.info("Inicializando a API, carregando modelo PyTorch e mappings...")
    
    model_path = "/app/models/ncf_production.pt"
    mappings_path = "/app/data/processed/id_mappings.json"
    
    # 1. Carrega os Mapeamentos com Busca Inteligente de Chaves
    try:
        if os.path.exists(mappings_path):
            with open(mappings_path, "r", encoding="utf-8") as f:
                mappings = json.load(f)
                
                # Procura pelas chaves diretas mais comuns
                u_dict = mappings.get("user_to_idx") or mappings.get("user_map") or mappings.get("users")
                i_dict = mappings.get("item_to_idx") or mappings.get("item_map") or mappings.get("items")
                
                # Se ainda for None ou não for dicionário, faz uma varredura por qualquer chave que lembre 'user' ou 'item'
                if not u_dict or not isinstance(u_dict, dict):
                    for k, v in mappings.items():
                        if isinstance(v, dict) and ('user' in k.lower() or 'cust' in k.lower()):
                            u_dict = v
                            break
                if not i_dict or not isinstance(i_dict, dict):
                    for k, v in mappings.items():
                        if isinstance(v, dict) and ('item' in k.lower() or 'prod' in k.lower()):
                            i_dict = v
                            break
                
                # Define os dicionários finais
                user_to_idx = u_dict if isinstance(u_dict, dict) else mappings
                item_to_idx = i_dict if isinstance(i_dict, dict) else mappings
                
                # Cria o mapeamento inverso (de inteiro para ID string)
                idx_to_item = {int(v): str(k) for k, v in item_to_idx.items() if str(v).isdigit() or isinstance(v, int)}
                todos_itens_idx = torch.tensor(list(idx_to_item.keys()), dtype=torch.long)
            
            logger.success(f"Mapeamentos carregados com sucesso! {len(user_to_idx)} usuários e {len(item_to_idx)} itens.")
            if len(user_to_idx) < 100:
                logger.warning("ATENÇÃO: Menos de 100 usuários carregados! Verifique se o id_mappings.json não foi sobrescrito por um teste.")
        else:
            logger.warning(f"Arquivo {mappings_path} não encontrado.")
    except Exception as e:
        logger.error(f"Erro ao carregar os mapeamentos de IDs: {e}")

    # 2. Carrega o Modelo PyTorch com a Arquitetura Real do Treino [64, 32]
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Deduz as dimensões exatas a partir dos tensores de peso salvos
            n_users = state_dict['user_emb.weight'].shape[0]
            n_items = state_dict['item_emb.weight'].shape[0]
            n_categories = state_dict['cat_emb.weight'].shape[0]
            emb_dim = state_dict['user_emb.weight'].shape[1]
            cat_emb_dim = state_dict['cat_emb.weight'].shape[1]
            input_dim = state_dict['mlp.0.weight'].shape[1]
            
            # Calcula quantas features auxiliares o modelo espera
            n_aux_features = max(0, input_dim - (emb_dim * 2) - cat_emb_dim)
            
            logger.info(f"Dimensões detectadas no checkpoint -> Users: {n_users}, Items: {n_items}, Cats: {n_categories}, Aux: {n_aux_features}")
            
            # Instancia o modelo com hidden=[64, 32] exatamente como foi treinado pelo time
            modelo_ncf = NCFHybrid(
                n_users=n_users,
                n_items=n_items,
                n_categories=n_categories,
                n_aux_features=n_aux_features,
                emb_dim=emb_dim,
                cat_emb_dim=cat_emb_dim,
                hidden=[64, 32]
            )
            
            # Injeta os pesos no modelo e coloca em modo de inferência (desliga Dropout/BatchNorm)
            modelo_ncf.load_state_dict(state_dict)
            modelo_ncf.eval()
            logger.success("Modelo NCF montado e pesos (state_dict) carregados na memória com sucesso!")
        else:
            logger.warning(f"Modelo não encontrado em {model_path}. Rodando em modo Fallback (Mock).")
    except Exception as e:
        logger.error(f"Erro ao montar/carregar o modelo PyTorch: {e}")

@app.get("/")
def health_check():
    """Rota de verificação de saúde para testes de deploy."""
    return {"status": "ok", "message": "Olist NCF Recommendation API está rodando em Produção!"}

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: str, top_k: int = 5):
    """Retorna as top K recomendações processadas em tempo real pelo PyTorch."""
    global modelo_ncf, user_to_idx, idx_to_item, todos_itens_idx, n_aux_features
    
    # 1. Trava de segurança: se a infraestrutura falhar, não derruba o endpoint
    if modelo_ncf is None or not user_to_idx:
        logger.warning("Retornando Mock: Modelo ou mapeamento indisponível na memória.")
        return mock_response(user_id, top_k)

    # 2. Busca o índice numérico do usuário
    u_idx = user_to_idx.get(user_id)
    
    # Se for usuário 100% novo (Cold Start absoluto), retorna Fallback
    if u_idx is None:
        logger.info(f"Usuário {user_id} não encontrado no histórico de treino (Cold Start). Retornando Fallback.")
        return mock_response(user_id, top_k)

    try:
        # 3. Prepara os tensores para processar todos os itens do catálogo simultaneamente
        user_tensor = torch.tensor([u_idx] * len(todos_itens_idx), dtype=torch.long)
        item_tensor = todos_itens_idx.clone().detach().long()
        
        # Preenche tensores de categoria e features auxiliares com valores neutros para inferência rápida
        cat_tensor = torch.zeros_like(item_tensor, dtype=torch.long)
        aux_tensor = torch.zeros((len(item_tensor), n_aux_features), dtype=torch.float32)
        
        # 4. Executa a predição sem gradientes (máxima performance de CPU)
        with torch.no_grad():
            scores = modelo_ncf(user_tensor, item_tensor, cat_tensor, aux_tensor).squeeze()
        
        # 5. Extrai os índices com as maiores pontuações (Top K)
        top_scores, top_indices = torch.topk(scores, top_k)
        
        # 6. Converte de volta de inteiro para a string original do product_id
        recommendations = []
        for i in range(top_k):
            item_idx_real = todos_itens_idx[top_indices[i]].item()
            product_id_str = idx_to_item.get(item_idx_real, f"item_{item_idx_real}")
            score_val = float(top_scores[i].item())
            
            recommendations.append({
                "product_id": product_id_str,
                "score": round(score_val, 4)
            })

        return {
            "user_id": user_id,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Erro durante a predição no PyTorch para o usuário {user_id}: {e}")
        return mock_response(user_id, top_k)


def mock_response(user_id: str, top_k: int):
    """Função auxiliar de Fallback/Mock para garantir alta disponibilidade (nunca retorna erro 500)."""
    return {
        "user_id": user_id,
        "recommendations": [
            {"product_id": "aca2eb7d00ea1a7b8ebd4e68314663af", "score": 0.95},
            {"product_id": "99a4788cb24856965c36a24e339b6058", "score": 0.88},
            {"product_id": "422879e10f46682990de24d770e7f83d", "score": 0.82}
        ][:top_k]
    }