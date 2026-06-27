"""
Pipeline de Treinamento de Modelos Baseline - Sistema de Recomendação Olist

Este script implementa:
1. Split temporal (70% treino / 15% validação / 15% teste)
2. 3 modelos baseline: Popularidade, Top-Rated, Item-Item CF
3. Métricas reais de ranking: MAP@K, NDCG@K, Recall@K, Precision@K, Hit Rate@K
4. Tracking de experimentos no MLflow
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Configuração do MLflow
MLFLOW_AVAILABLE = False
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Olist_Recommendation_Baselines")
    MLFLOW_AVAILABLE = True
    logger.info("MLflow tracking enabled")
except Exception as e:
    logger.warning(f"MLflow não disponível: {e}. Continuando sem tracking.")

# Paths
DATA_PATH = Path("data/processed/interactions.parquet")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constantes
K_VALUES = [5, 10, 20]  # Diferentes valores de K para avaliação


# =============================================================================
# 1. SPLIT TEMPORAL
# =============================================================================

def temporal_split(
    df: pd.DataFrame,
    time_col: str = "order_purchase_timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Realiza split temporal do dataset.
    
    Args:
        df: DataFrame com os dados
        time_col: Nome da coluna de timestamp
        train_ratio: Proporção para treino
        val_ratio: Proporção para validação
    
    Returns:
        Tuple de (train, val, test) DataFrames
    """
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    logger.info(f"Temporal Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    logger.info(f"  Treino: {train_df[time_col].min()} -> {train_df[time_col].max()}")
    logger.info(f"  Validação: {val_df[time_col].min()} -> {val_df[time_col].max()}")
    logger.info(f"  Teste: {test_df[time_col].min()} -> {test_df[time_col].max()}")
    
    return train_df, val_df, test_df


def get_user_interactions(df: pd.DataFrame, user_col: str = "customer_unique_id") -> Dict:
    """Retorna dicionário {user_id: set of items} para cada split."""
    return df.groupby(user_col)["product_id"].apply(set).to_dict()


# =============================================================================
# 2. MODELOS BASELINE
# =============================================================================

class PopularityBaseline:
    """Baseline 1: Recomenda os itens mais populares globalmente."""
    
    def __init__(self, k: int = 10):
        self.k = k
        self.popular_items: List = []
    
    def fit(self, df: pd.DataFrame, item_col: str = "product_id") -> "PopularityBaseline":
        """Treina usando o conjunto de treino."""
        counts = df[item_col].value_counts()
        self.popular_items = counts.head(self.k).index.tolist()
        logger.info(f"PopularityBaseline: Top {len(self.popular_items)} items selected")
        return self
    
    def recommend(self, user_id: str, user_items: set, n: int = None) -> List:
        """Retorna mesma lista para todos os usuários."""
        n = n or self.k
        return self.popular_items[:n]


class TopRatedBaseline:
    """Baseline 2: Recomenda itens com melhor nota média (com mínimo de reviews)."""
    
    def __init__(self, k: int = 10, min_reviews: int = 5):
        self.k = k
        self.min_reviews = min_reviews
        self.top_items: List = []
    
    def fit(self, df: pd.DataFrame, item_col: str = "product_id") -> "TopRatedBaseline":
        """Treina usando o conjunto de treino."""
        ratings = (
            df.groupby(item_col)
            .agg(avg_rating=("review_score", "mean"), count=("review_score", "count"))
            .reset_index()
        )
        filtered = ratings[ratings["count"] >= self.min_reviews]
        self.top_items = (
            filtered.sort_values(by="avg_rating", ascending=False)
            .head(self.k)[item_col]
            .tolist()
        )
        logger.info(f"TopRatedBaseline: {len(self.top_items)} items (min_reviews={self.min_reviews})")
        return self
    
    def recommend(self, user_id: str, user_items: set, n: int = None) -> List:
        """Retorna mesma lista para todos os usuários."""
        n = n or self.k
        return self.top_items[:n]


class ItemItemCF:
    """Baseline 3: Filtragem Colaborativa Item-Item via similaridade de cosseno."""
    
    def __init__(self, k: int = 10, n_neighbors: int = 50):
        self.k = k
        self.n_neighbors = n_neighbors
        self.similarity_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.all_items = []
    
    def fit(self, df: pd.DataFrame, user_col: str = "customer_unique_id", 
            item_col: str = "product_id") -> "ItemItemCF":
        """Treina calculando matriz de similaridade item-item."""
        # Criar mapeamentos
        self.all_items = df[item_col].unique().tolist()
        self.item_to_idx = {item: idx for idx, item in enumerate(self.all_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Matriz usuário-item
        user_idx = df[user_col].astype("category").cat.codes
        item_idx = df[item_col].map(self.item_to_idx)
        
        matrix = csr_matrix(
            (np.ones(len(df)), (user_idx, item_idx)),
            shape=(user_idx.max() + 1, len(self.all_items))
        )
        
        # Similaridade de cosseno entre itens
        self.similarity_matrix = cosine_similarity(matrix.T, dense_output=True)
        logger.info(f"ItemItemCF: similarity matrix shape {self.similarity_matrix.shape}")
        return self
    
    def recommend(self, user_id: str, user_items: set, n: int = None) -> List:
        """Gera recomendações baseadas em itens que o usuário já interagiu."""
        n = n or self.k
        
        if not user_items:
            # Cold-start: retorna popularidade (vazio por padrão)
            return []
        
        # Agregar scores de similaridade dos itens do usuário
        scores = np.zeros(len(self.all_items))
        for item in user_items:
            if item in self.item_to_idx:
                item_idx = self.item_to_idx[item]
                scores += self.similarity_matrix[item_idx]
        
        # Ordenar por score e filtrar itens já conhecidos
        ranked_indices = np.argsort(scores)[::-1]
        recommendations = []
        for idx in ranked_indices:
            item = self.idx_to_item[idx]
            if item not in user_items:
                recommendations.append(item)
                if len(recommendations) >= n:
                    break
        
        return recommendations


# =============================================================================
# 3. MÉTRICAS DE RANKING
# =============================================================================

def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at K."""
    relevance = np.asarray(relevance)[:k]
    if relevance.size:
        return np.sum((2**relevance - 1) / np.log2(np.arange(2, relevance.size + 2)))
    return 0.0


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Normalized DCG at K."""
    dcg = dcg_at_k(relevance, k)
    ideal_relevance = np.sort(relevance)[::-1][:k]
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevant_items: set, recommended_items: List, k: int) -> float:
    """Precision@K: fração de itens relevantes nas K recomendações."""
    recommended = recommended_items[:k]
    if not recommended:
        return 0.0
    return len(set(recommended) & relevant_items) / k


def recall_at_k(relevant_items: set, recommended_items: List, k: int) -> float:
    """Recall@K: fração de itens relevantes recuperados."""
    recommended = recommended_items[:k]
    if not relevant_items:
        return 0.0
    return len(set(recommended) & relevant_items) / len(relevant_items)


def hit_rate_at_k(relevant_items: set, recommended_items: List, k: int) -> float:
    """Hit Rate@K: 1 se há pelo menos 1 hit, 0 caso contrário."""
    recommended = set(recommended_items[:k])
    return 1.0 if recommended & relevant_items else 0.0


def average_precision_at_k(relevant_items: set, recommended_items: List, k: int) -> float:
    """Average Precision@K (para MAP@K)."""
    if not relevant_items:
        return 0.0
    
    score = 0.0
    num_hits = 0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            num_hits += 1
            score += num_hits / (i + 1)
    
    return score / min(len(relevant_items), k)


def evaluate_model(
    model,
    train_interactions: Dict,
    test_df: pd.DataFrame,
    user_col: str = "customer_unique_id",
    item_col: str = "product_id",
    k_values: List[int] = K_VALUES,
) -> Dict[str, Dict[str, float]]:
    """
    Avalia um modelo em múltiplos valores de K.
    
    Returns:
        Dict[metric_name, Dict[k_value, score]]
    """
    # Ground truth: itens no teste para cada usuário
    test_interactions = test_df.groupby(user_col)[item_col].apply(set).to_dict()
    
    # Filtrar usuários que estão no treino E no teste
    users_to_eval = set(train_interactions.keys()) & set(test_interactions.keys())
    
    results = {f"MAP@{k}": [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})
    results.update({f"Precision@{k}": [] for k in k_values})
    results.update({f"Recall@{k}": [] for k in k_values})
    results.update({f"HitRate@{k}": [] for k in k_values})
    
    for user_id in users_to_eval:
        user_train_items = train_interactions[user_id]
        user_test_items = test_interactions[user_id]
        
        # Recomendar K*2 itens para ter margem (filtrar conhecidos)
        recs = model.recommend(user_id, user_train_items, n=max(k_values) * 2)
        
        # Calcular relevância (1 se item está no test set)
        relevance = np.array([1.0 if item in user_test_items else 0.0 for item in recs])
        
        for k in k_values:
            results[f"MAP@{k}"].append(average_precision_at_k(user_test_items, recs, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(relevance, k))
            results[f"Precision@{k}"].append(precision_at_k(user_test_items, recs, k))
            results[f"Recall@{k}"].append(recall_at_k(user_test_items, recs, k))
            results[f"HitRate@{k}"].append(hit_rate_at_k(user_test_items, recs, k))
    
    # Médias
    return {metric: np.mean(scores) for metric, scores in results.items()}


# =============================================================================
# 4. PIPELINE PRINCIPAL
# =============================================================================

def run_experiment(
    run_name: str,
    model_class,
    model_params: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """Executa um experimento com log no MLflow (se disponível)."""
    
    def _execute():
        # Treinar modelo
        logger.info(f"Treinando {model_class.__name__}...")
        model = model_class(**model_params)
        model.fit(train_df)
        
        # Interações do treino para filtrar recomendações
        train_interactions = get_user_interactions(train_df)
        
        # Avaliar
        logger.info(f"Avaliando {model_class.__name__}...")
        metrics = evaluate_model(model, train_interactions, test_df)
        
        # Log métricas
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Salvar artefatos
        if hasattr(model, 'popular_items'):
            items = model.popular_items
        elif hasattr(model, 'top_items'):
            items = model.top_items
        else:
            items = []
        
        recommendations_df = pd.DataFrame({f"top_{model_params.get('k', 10)}": items})
        output_file = OUTPUT_DIR / f"recommendations_{model_class.__name__}_{run_name}.csv"
        recommendations_df.to_csv(output_file, index=False)
        
        return metrics, model
    
    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name=run_name):
                # Log parâmetros
                mlflow.log_param("model_type", model_class.__name__)
                for key, value in model_params.items():
                    mlflow.log_param(key, value)
                mlflow.log_param("train_size", len(train_df))
                mlflow.log_param("test_size", len(test_df))
                
                metrics, model = _execute()
                
                # Log métricas no MLflow
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                
                logger.success(f"Run '{run_name}' concluída (MLflow)!")
                return metrics
        except Exception as e:
            logger.warning(f"MLflow error: {e}. Executando sem tracking.")
    
    metrics, _ = _execute()
    logger.success(f"Run '{run_name}' concluída!")
    return metrics


def main():
    """Pipeline principal de treinamento."""
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE DE TREINAMENTO - BASELINES")
    logger.info("=" * 60)
    
    # Carregar dados
    logger.info(f"Carregando dados de {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_PATH}. Rode o pipeline de dados primeiro.")
    
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Dados carregados: {len(df)} registros")
    
    # Split temporal
    logger.info("Realizando split temporal...")
    train_df, val_df, test_df = temporal_split(df)
    
    # Salvar splits para debug
    train_df.to_parquet(OUTPUT_DIR / "train_split.parquet", index=False)
    val_df.to_parquet(OUTPUT_DIR / "val_split.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test_split.parquet", index=False)
    
    # Testar diferentes valores de K
    all_results = {}
    
    # Experimento 1: Popularity Baseline
    for k in [5, 10, 20]:
        run_name = f"PopularityBaseline_K{k}"
        metrics = run_experiment(
            run_name,
            PopularityBaseline,
            {"k": k},
            train_df,
            test_df
        )
        all_results[run_name] = metrics
    
    # Experimento 2: Top-Rated Baseline
    for k, min_reviews in [(10, 5), (10, 15), (20, 5)]:
        run_name = f"TopRated_K{k}_MinRev{min_reviews}"
        metrics = run_experiment(
            run_name,
            TopRatedBaseline,
            {"k": k, "min_reviews": min_reviews},
            train_df,
            test_df
        )
        all_results[run_name] = metrics
    
    # Experimento 3: Item-Item CF
    for k in [10, 20]:
        run_name = f"ItemItemCF_K{k}"
        metrics = run_experiment(
            run_name,
            ItemItemCF,
            {"k": k},
            train_df,
            test_df
        )
        all_results[run_name] = metrics
    
    # Resumo final
    logger.info("=" * 60)
    logger.info("RESUMO DOS RESULTADOS")
    logger.info("=" * 60)
    print(f"\n{'Model':<35} {'MAP@10':>8} {'NDCG@10':>8} {'Recall@10':>8} {'HitRate@10':>10}")
    print("-" * 75)
    
    for run_name, metrics in all_results.items():
        print(f"{run_name:<35} {metrics['MAP@10']:>8.4f} {metrics['NDCG@10']:>8.4f} "
              f"{metrics['Recall@10']:>8.4f} {metrics['HitRate@10']:>10.4f}")
    
    logger.success("Pipeline concluído com sucesso!")


if __name__ == "__main__":
    main()
