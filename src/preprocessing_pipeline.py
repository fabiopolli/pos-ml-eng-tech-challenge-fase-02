"""
Pipeline de Pré-processamento para o Dataset Olist E-Commerce.

Segue os princípios SOLID e implementa o padrão de projeto Strategy para o pré-processamento.
Inclui anotações de tipo completas (type hints) e docstrings no formato Google.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class OlistDataLoader:
    """Carrega e mescla os conjuntos de dados da Olist."""

    def __init__(self, dataset_path: str):
        """Inicializa o carregador com o caminho do dataset.

        Args:
            dataset_path: Caminho para o diretório contendo os arquivos CSV da Olist.
        """
        self.dataset_path: str = dataset_path

    def load_all_data(self) -> pd.DataFrame:
        """Carrega e une todos os arquivos CSV da Olist.

        Returns:
            DataFrame unificado contendo os dados de clientes, pedidos e avaliações.
        """
        customers: pd.DataFrame = pd.read_csv(f"{self.dataset_path}/olist_customers_dataset.csv")
        order_items: pd.DataFrame = pd.read_csv(f"{self.dataset_path}/olist_order_items_dataset.csv")
        order_payments: pd.DataFrame = pd.read_csv(f"{self.dataset_path}/olist_order_payments_dataset.csv")
        order_reviews: pd.DataFrame = pd.read_csv(f"{self.dataset_path}/olist_order_reviews_dataset.csv")
        orders: pd.DataFrame = pd.read_csv(f"{self.dataset_path}/olist_orders_dataset.csv")
        products: pd.DataFrame = pd.read_csv(f"{self.dataset_path}/olist_products_dataset.csv")
        sellers: pd.DataFrame = pd.read_csv(f"{self.dataset_path}/olist_sellers_dataset.csv")
        category_translation: pd.DataFrame = pd.read_csv(f"{self.dataset_path}/product_category_name_translation.csv")

        # União dos conjuntos de dados
        df: pd.DataFrame = orders.merge(order_items, on='order_id', how='left')
        df = df.merge(order_payments, on='order_id', how='outer', validate='m:m')
        df = df.merge(order_reviews, on='order_id', how='outer')
        df = df.merge(products, on='product_id', how='outer')
        df = df.merge(customers, on='customer_id', how='outer')
        df = df.merge(sellers, on='seller_id', how='outer')
        df = df.merge(category_translation, on='product_category_name', how='left')

        return df


class PreprocessingStrategy(ABC):
    """Classe base abstrata que define a Estratégia de Pré-processamento."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'PreprocessingStrategy':
        """Ajusta os parâmetros da estratégia com os dados de treino.

        Args:
            df: DataFrame de entrada.
            target_col: Nome da coluna alvo.

        Returns:
            A própria instância ajustada.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica as transformações no DataFrame de entrada.

        Args:
            df: DataFrame de entrada.

        Returns:
            DataFrame transformado.
        """
        pass


class OlistRecommendationPrepStrategy(PreprocessingStrategy):
    """Estratégia voltada para sistemas de recomendação baseados em Filtragem Colaborativa.

    Mapeia IDs de alta cardinalidade de usuários e itens para índices numéricos contínuos,
    preparando os dados para camadas de embedding ou modelos baseados em matrizes esparsas.
    """

    def __init__(self) -> None:
        """Inicializa a estratégia de recomendação."""
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.features_to_keep: List[str] = ['customer_unique_id', 'product_id', 'review_score']

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'OlistRecommendationPrepStrategy':
        """Gera os mapeamentos exclusivos para usuários e produtos.

        Args:
            df: DataFrame de entrada.
            target_col: Ignorado.

        Returns:
            A própria instância.
        """
        unique_users: List[str] = df['customer_unique_id'].dropna().unique().tolist()
        unique_items: List[str] = df['product_id'].dropna().unique().tolist()

        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Substitui IDs originais por índices numéricos contínuos.

        Args:
            df: DataFrame de entrada.

        Returns:
            DataFrame contendo apenas user_idx e item_idx.
        """
        transformed_df = df.copy()

        transformed_df['user_idx'] = transformed_df['customer_unique_id'].map(self.user_mapping)
        unseen_user_idx = len(self.user_mapping)
        transformed_df['user_idx'] = transformed_df['user_idx'].fillna(unseen_user_idx).astype(int)

        transformed_df['item_idx'] = transformed_df['product_id'].map(self.item_mapping)
        unseen_item_idx = len(self.item_mapping)
        transformed_df['item_idx'] = transformed_df['item_idx'].fillna(unseen_item_idx).astype(int)

        return transformed_df[['user_idx', 'item_idx']]


class OlistTabularPrepStrategy(PreprocessingStrategy):
    """Estratégia de pré-processamento clássica para modelos preditivos de classificação/regressão."""

    def __init__(self) -> None:
        """Inicializa escaladores, listas de atributos e configurações de limpeza."""
        self.scaler: StandardScaler = StandardScaler()
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.frequency_maps: Dict[str, Dict[str, float]] = {}
        self.features_to_drop: List[str] = [
            'order_id', 'seller_id', 'customer_id', 'order_item_id',
            'product_id', 'review_id', 'customer_unique_id',
            'seller_zip_code_prefix', 'product_category_name'
        ]

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata valores faltantes para variáveis numéricas e categóricas."""
        df = df.copy()
        if 'price' in df.columns:
            df['price'] = df['price'].fillna(-1.0)
        if 'freight_value' in df.columns:
            df['freight_value'] = df['freight_value'].fillna(0.0)

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'review_score' and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('unknown')
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Realiza engenharia de atributos temporais, regionais e físicos."""
        df = df.copy()
        timestamp_cols = [
            'order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date', 'review_creation_date',
            'review_answer_timestamp', 'shipping_limit_date'
        ]

        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Cálculo de tempos de entrega
        if 'order_purchase_timestamp' in df.columns and 'order_delivered_customer_date' in df.columns:
            df['delivery_time_days'] = (
                df['order_delivered_customer_date'] - df['order_purchase_timestamp']
            ).dt.total_seconds() / (24 * 3600)
            df['delivery_time_days'] = df['delivery_time_days'].clip(lower=0)

        if 'order_purchase_timestamp' in df.columns and 'order_estimated_delivery_date' in df.columns:
            df['estimated_delivery_time_days'] = (
                df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
            ).dt.total_seconds() / (24 * 3600)
            df['estimated_delivery_time_days'] = df['estimated_delivery_time_days'].clip(lower=0)

        if 'delivery_time_days' in df.columns and 'estimated_delivery_time_days' in df.columns:
            df['delivery_delay_days'] = df['delivery_time_days'] - df['estimated_delivery_time_days']

        if 'order_purchase_timestamp' in df.columns and 'order_approved_at' in df.columns:
            df['processing_time_days'] = (
                df['order_approved_at'] - df['order_purchase_timestamp']
            ).dt.total_seconds() / (24 * 3600)
            df['processing_time_days'] = df['processing_time_days'].clip(lower=0)

        # Mapeamento para regiões brasileiras
        regions = {
            'Southeast': ['SP', 'RJ', 'ES', 'MG'],
            'Northeast': ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA'],
            'North': ['AM', 'RR', 'AP', 'PA', 'TO', 'RO', 'AC'],
            'Midwest': ['MT', 'GO', 'MS', 'DF'],
            'South': ['SC', 'RS', 'PR']
        }

        def map_state_to_region(state: Union[str, float]) -> str:
            if not isinstance(state, str):
                return 'Other'
            for region, states in regions.items():
                if state in states:
                    return region
            return 'Other'

        if 'customer_state' in df.columns:
            df['customer_region'] = df['customer_state'].apply(map_state_to_region)
        if 'seller_state' in df.columns:
            df['seller_region'] = df['seller_state'].apply(map_state_to_region)

        if 'price' in df.columns:
            df['price_log'] = np.log(df['price'] + 1.5)

        if 'order_item_id' in df.columns:
            df['order_size'] = '1_item'
            df.loc[df['order_item_id'].isin([2, 3, 4, 5, 6]), 'order_size'] = '2-6_items'
            df.loc[df['order_item_id'].isin([7, 8, 9, 10]), 'order_size'] = '7-10_items'
            df.loc[df['order_item_id'] > 10, 'order_size'] = '10+_items'

        if all(col in df.columns for col in ['product_length_cm', 'product_height_cm', 'product_width_cm']):
            df['product_volume_cm3'] = (
                df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
            )

        if 'product_volume_cm3' in df.columns and 'product_weight_g' in df.columns:
            df['product_density_g_cm3'] = df['product_weight_g'] / (df['product_volume_cm3'] + 1e-8)

        return df

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'OlistTabularPrepStrategy':
        """Identifica tipos de colunas e calcula métricas para normalização e codificação.

        Args:
            df: DataFrame de treinamento.
            target_col: Nome opcional da coluna alvo que deve ser desconsiderada.

        Returns:
            Instância ajustada da estratégia.
        """
        df_clean = self._handle_missing_values(df)
        df_engineered = self._feature_engineering(df_clean)

        exclude_cols = self.features_to_drop + ([] if target_col is None else [target_col])
        feature_cols = [col for col in df_engineered.columns if col not in exclude_cols]

        self.numerical_cols = df_engineered[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df_engineered[feature_cols].select_dtypes(include=['object']).columns.tolist()

        # Codificação por frequência para variáveis categóricas
        for col in self.categorical_cols:
            self.frequency_maps[col] = df_engineered[col].value_counts().to_dict()

        if self.numerical_cols:
            self.scaler.fit(df_engineered[self.numerical_cols].fillna(0.0))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a normalização numérica e codificação categórica nas colunas.

        Args:
            df: DataFrame de entrada.

        Returns:
            DataFrame totalmente pré-processado pronto para algoritmos de aprendizado.
        """
        df_clean = self._handle_missing_values(df)
        df_engineered = self._feature_engineering(df_clean)

        for col in self.categorical_cols:
            if col in df_engineered.columns:
                freq_map = self.frequency_maps.get(col, {})
                df_engineered[f'{col}_freq'] = df_engineered[col].map(freq_map).fillna(0.0)

        if self.numerical_cols:
            df_engineered[self.numerical_cols] = self.scaler.transform(df_engineered[self.numerical_cols].fillna(0.0))

        # Remoção de colunas de texto brutas e temporais redundantes
        cols_to_drop = self.features_to_drop + self.categorical_cols
        timestamp_cols = [
            'order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date', 'review_creation_date',
            'review_answer_timestamp', 'shipping_limit_date'
        ]
        cols_to_drop.extend([t for t in timestamp_cols if t in df_engineered.columns])

        return df_engineered.drop(columns=cols_to_drop, errors='ignore')


class OlistPreprocessor(BaseEstimator, TransformerMixin):
    """Classe unificada de pré-processamento que delega as ações para uma estratégia específica."""

    def __init__(self, strategy: PreprocessingStrategy) -> None:
        """Inicializa com a estratégia de pré-processamento configurada.

        Args:
            strategy: Instância concreta de PreprocessingStrategy.
        """
        self.strategy: PreprocessingStrategy = strategy

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OlistPreprocessor':
        """Ajusta a estratégia envelopada.

        Args:
            X: Atributos de entrada para treino.
            y: Alvo de treino (opcional).

        Returns:
            A própria classe unificada.
        """
        self.strategy.fit(X, target_col='review_score')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma as colunas de entrada delegando para a estratégia ativa.

        Args:
            X: Atributos de entrada.

        Returns:
            DataFrame final transformado.
        """
        return self.strategy.transform(X)


def load_and_preprocess_data(
    dataset_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    recommender_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], OlistPreprocessor]:
    """Orquestra o carregamento, limpeza, engenharia e divisão do dataset Olist.

    Args:
        dataset_path: Diretório contendo os arquivos CSV.
        test_size: Proporção da base de teste.
        random_state: Semente de aleatoriedade para reprodutibilidade.
        recommender_mode: Define se a saída será otimizada para modelos de recomendação.

    Returns:
        Um conjunto contendo (X_train, X_test, y_train, y_test, nomes_atributos, preprocessor).
    """
    loader = OlistDataLoader(dataset_path)
    df = loader.load_all_data()

    if 'review_score' not in df.columns:
        raise ValueError("Coluna 'review_score' não encontrada no conjunto de dados")

    df = df.dropna(subset=['review_score'])

    X = df.drop('review_score', axis=1)
    y = df['review_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if recommender_mode:
        strategy = OlistRecommendationPrepStrategy()
    else:
        strategy = OlistTabularPrepStrategy()

    preprocessor = OlistPreprocessor(strategy)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = X_train_processed.columns.tolist()

    return X_train_processed, X_test_processed, y_train, y_test, feature_names, preprocessor
