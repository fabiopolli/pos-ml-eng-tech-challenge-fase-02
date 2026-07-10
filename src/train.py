"""
Pipeline de Treinamento de Modelos Baseline - Sistema de Recomendação Olist

Este script implementa:
1. Split temporal (70% treino / 15% validação / 15% teste)
2. 3 modelos baseline: Popularidade, Top-Rated, Item-Item CF
3. Métricas reais de ranking: MAP@K, NDCG@K, Recall@K, Precision@K, Hit Rate@K
4. Tracking de experimentos no MLflow
"""


import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings
from src.data.splits import temporal_split as _temporal_split_with_asserts

# Configuração do MLflow
MLFLOW_AVAILABLE = False
try:
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.experiment_name)
    MLFLOW_AVAILABLE = True
    logger.info("MLflow tracking enabled")
except Exception as e:
    logger.warning(f"MLflow não disponível: {e}. Continuando sem tracking.")

# Paths
DATA_PATH = settings.interactions_path
OUTPUT_DIR = settings.data_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constantes
K_VALUES = settings.k_values  # Diferentes valores de K para avaliação


# =============================================================================
# 1. SPLIT TEMPORAL
# =============================================================================

def temporal_split(
    df: pd.DataFrame,
    time_col: str = "order_purchase_timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wrapper sobre `src.data.splits.temporal_split` com assertions anti-leakage.

    Mantém a assinatura original (compatibilidade com chamadas em `src/train.py`)
    mas delega a implementação para a versão em `src/data/splits.py`, que inclui
    assertions rigorosas contra data leakage temporal.
    """
    train_df, val_df, test_df = _temporal_split_with_asserts(
        df,
        time_col=time_col,
        train_size=train_ratio,
        val_size=val_ratio,
    )
    logger.info(
        f"Temporal Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )
    logger.info(f"  Treino: {train_df[time_col].min()} -> {train_df[time_col].max()}")
    logger.info(f"  Validação: {val_df[time_col].min()} -> {val_df[time_col].max()}")
    logger.info(f"  Teste: {test_df[time_col].min()} -> {test_df[time_col].max()}")
    return train_df, val_df, test_df


def get_user_interactions(df: pd.DataFrame, user_col: str = "customer_unique_id") -> dict:
    """Retorna dicionário {user_id: set of items} para cada split."""
    return df.groupby(user_col)["product_id"].apply(set).to_dict()


# =============================================================================
# 2. MODELOS BASELINE
# =============================================================================

class PopularityBaseline:
    """Baseline 1: Recomenda os itens mais populares globalmente."""

    def __init__(self, k: int = 10):
        self.k = k
        self.popular_items: list = []

    def fit(self, df: pd.DataFrame, item_col: str = "product_id") -> "PopularityBaseline":
        """Treina usando o conjunto de treino."""
        counts = df[item_col].value_counts()
        self.popular_items = counts.head(self.k).index.tolist()
        logger.info(f"PopularityBaseline: Top {len(self.popular_items)} items selected")
        return self

    def recommend(self, user_id: str, user_items: set, n: int | None = None) -> list:
        """Retorna mesma lista para todos os usuários."""
        n = n or self.k
        return self.popular_items[:n]


class TopRatedBaseline:
    """Baseline 2: Recomenda itens com melhor nota média (com mínimo de reviews)."""

    def __init__(self, k: int = 10, min_reviews: int = 5):
        self.k = k
        self.min_reviews = min_reviews
        self.top_items: list = []

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

    def recommend(self, user_id: str, user_items: set, n: int | None = None) -> list:
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

    def recommend(self, user_id: str, user_items: set, n: int | None = None) -> list:
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


class TruncatedSVDBaseline:
    """Baseline 4: Fatoração de matriz via TruncatedSVD em matriz CSR user-item.

    Usa TruncatedSVD (LSA/LSI) sobre a matriz de interações user x item para
    obter embeddings latentes. Recomendações são feitas por dot product entre
    o vetor latente do usuário e o embedding do item, filtrando itens já vistos."""

    def __init__(
        self,
        k: int = 10,
        n_components: int | None = None,
        random_state: int | None = None,
    ):
        self.k = k
        self.n_components = n_components if n_components is not None else settings.svd_n_components
        self.random_state = random_state if random_state is not None else settings.seed
        self.svd: TruncatedSVD | None = None
        self.user_factors: np.ndarray | None = None
        self.user_idx_map: dict = {}
        self.item_to_idx: dict = {}
        self.idx_to_item: dict = {}
        self.all_item_scores: np.ndarray | None = None

    def fit(
        self,
        df: pd.DataFrame,
        user_col: str = "customer_unique_id",
        item_col: str = "product_id",
    ) -> "TruncatedSVDBaseline":
        """Ajusta o modelo TruncatedSVD na matriz CSR user-item."""
        # Mapeamentos
        users = df[user_col].unique()
        items = df[item_col].unique()
        self.user_idx_map = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_item = dict(enumerate(items))

        n_users = len(users)
        n_items = len(items)
        actual_components = min(self.n_components, n_items - 1)

        row = df[user_col].map(self.user_idx_map).values
        col = df[item_col].map(self.item_to_idx).values
        data = np.ones(len(df), dtype=np.float32)

        matrix = csr_matrix((data, (row, col)), shape=(n_users, n_items))

        # Ajuste do SVD
        self.svd = TruncatedSVD(
            n_components=actual_components, random_state=self.random_state
        )
        self.user_factors = self.svd.fit_transform(matrix)
        # Reconstruir scores precomputados: user_factors @ Vt -> shape (n_users, n_items)
        self.all_item_scores = self.user_factors @ self.svd.components_
        logger.info(
            f"TruncatedSVDBaseline: n_users={n_users}, n_items={n_items}, "
            f"components={actual_components}, explained_var={self.svd.explained_variance_ratio_.sum():.4f}"
        )
        return self

    def recommend(self, user_id: str, user_items: set, n: int | None = None) -> list:
        """Retorna top-N itens para o usuário, excluindo os já vistos."""
        n = n or self.k
        if user_id not in self.user_idx_map:
            return []
        u_idx = self.user_idx_map[user_id]
        scores = self.all_item_scores[u_idx].copy()
        # Máscara eficiente via item_to_idx
        for item in user_items:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:n]
        return [self.idx_to_item[int(i)] for i in top_indices if scores[int(i)] > -np.inf]


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


def precision_at_k(relevant_items: set, recommended_items: list, k: int) -> float:
    """Precision@K: fração de itens relevantes nas K recomendações."""
    recommended = recommended_items[:k]
    if not recommended:
        return 0.0
    return len(set(recommended) & relevant_items) / k


def recall_at_k(relevant_items: set, recommended_items: list, k: int) -> float:
    """Recall@K: fração de itens relevantes recuperados."""
    recommended = recommended_items[:k]
    if not relevant_items:
        return 0.0
    return len(set(recommended) & relevant_items) / len(relevant_items)


def hit_rate_at_k(relevant_items: set, recommended_items: list, k: int) -> float:
    """Hit Rate@K: 1 se há pelo menos 1 hit, 0 caso contrário."""
    recommended = set(recommended_items[:k])
    return 1.0 if recommended & relevant_items else 0.0


def average_precision_at_k(relevant_items: set, recommended_items: list, k: int) -> float:
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


def _init_results_dict(k_values: list[int]) -> dict[str, list[float]]:
    """Inicializa dicionário de métricas vazio para todos os Ks."""
    results = {}
    for metric in ("MAP", "NDCG", "Precision", "Recall", "HitRate"):
        for k in k_values:
            results[f"{metric}@{k}"] = []
    return results


def _score_user_for_all_k(
    model,
    user_id,
    user_train_items: set,
    user_test_items: set,
    k_values: list[int],
    max_k: int,
) -> dict[str, list[float]]:
    """Calcula todas as métricas para um único usuário em todos os Ks."""
    recs = model.recommend(user_id, user_train_items, n=max_k * 2)
    relevance = np.array([1.0 if item in user_test_items else 0.0 for item in recs])
    user_scores = {}
    for k in k_values:
        user_scores[f"MAP@{k}"] = average_precision_at_k(user_test_items, recs, k)
        user_scores[f"NDCG@{k}"] = ndcg_at_k(relevance, k)
        user_scores[f"Precision@{k}"] = precision_at_k(user_test_items, recs, k)
        user_scores[f"Recall@{k}"] = recall_at_k(user_test_items, recs, k)
        user_scores[f"HitRate@{k}"] = hit_rate_at_k(user_test_items, recs, k)
    return user_scores


def evaluate_model(
    model,
    train_interactions: dict,
    test_df: pd.DataFrame,
    user_col: str = "customer_unique_id",
    item_col: str = "product_id",
    k_values: list[int] = K_VALUES,
) -> dict[str, dict[str, float]]:
    """
    Avalia um modelo em múltiplos valores de K.

    Returns:
        Dict[metric_name, Dict[k_value, score]]
    """
    test_interactions = test_df.groupby(user_col)[item_col].apply(set).to_dict()
    users_to_eval = set(train_interactions.keys()) & set(test_interactions.keys())
    max_k = max(k_values)
    results = _init_results_dict(k_values)

    for user_id in users_to_eval:
        user_scores = _score_user_for_all_k(
            model, user_id, train_interactions[user_id], test_interactions[user_id],
            k_values, max_k,
        )
        for metric, score in user_scores.items():
            results[metric].append(score)

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
        baselines_dir = settings.artifacts_dir / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        output_file = baselines_dir / f"recommendations_{model_class.__name__}_{run_name}.csv"
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

                metrics, _ = _execute()

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

    # Experimento 4: TruncatedSVD Baseline
    for k in [10, 20]:
        run_name = f"TruncatedSVD_K{k}"
        metrics = run_experiment(
            run_name,
            TruncatedSVDBaseline,
            {"k": k, "n_components": settings.svd_n_components},
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
