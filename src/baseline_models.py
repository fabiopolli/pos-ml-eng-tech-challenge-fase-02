"""
Modelos de Baseline para Previsão de Notas de Avaliações (Review Score) do E-Commerce Olist.

Segue os princípios SOLID e implementa o padrão de projeto Factory para criação de modelos.
Inclui anotações de tipo completas (type hints) e docstrings no formato Google.
"""

from typing import Dict, Tuple, List, Optional, Any, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')


class ModelFactory:
    """Fábrica (Factory Pattern) para instanciar classificadores de baseline do Scikit-Learn."""

    @staticmethod
    def create_model(model_type: str, **kwargs: Any) -> BaseEstimator:
        """Cria e retorna uma instância do modelo especificado.

        Args:
            model_type: Tipo do modelo (ex: 'logistic_regression', 'random_forest').
            **kwargs: Parâmetros adicionais passados para o construtor do modelo.

        Returns:
            Um estimador do Scikit-Learn correspondente.

        Raises:
            ValueError: Se o tipo de modelo fornecido for inválido.
        """
        model_type_lower = model_type.lower().replace(" ", "_")

        if model_type_lower == "logistic_regression":
            params = {"max_iter": 1000, "random_state": 42, "multi_class": "ovr"}
            params.update(kwargs)
            return LogisticRegression(**params)

        elif model_type_lower == "decision_tree":
            params = {"max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42}
            params.update(kwargs)
            return DecisionTreeClassifier(**params)

        elif model_type_lower == "random_forest":
            params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1}
            params.update(kwargs)
            return RandomForestClassifier(**params)

        elif model_type_lower == "gradient_boosting":
            params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42}
            params.update(kwargs)
            return GradientBoostingClassifier(**params)

        else:
            raise ValueError(
                f"Tipo de modelo não suportado: {model_type}. "
                f"Suportados: 'logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting'"
            )


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None
) -> Tuple[BaseEstimator, Optional[np.ndarray]]:
    """Treina um modelo de Regressão Logística obtido pela fábrica.

    Args:
        X_train: Atributos de treino.
        y_train: Alvos de treino.
        X_test: Atributos de teste opcionais.
        y_test: Alvos de teste opcionais.

    Returns:
        Tupla com o modelo ajustado e as predições de teste (se dados de teste fornecidos).
    """
    model = ModelFactory.create_model("logistic_regression")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None
) -> Tuple[BaseEstimator, Optional[np.ndarray]]:
    """Treina um modelo de Árvore de Decisão obtido pela fábrica.

    Args:
        X_train: Atributos de treino.
        y_train: Alvos de treino.
        X_test: Atributos de teste opcionais.
        y_test: Alvos de teste opcionais.

    Returns:
        Tupla com o modelo ajustado e as predições de teste (se dados de teste fornecidos).
    """
    model = ModelFactory.create_model("decision_tree")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None
) -> Tuple[BaseEstimator, Optional[np.ndarray]]:
    """Treina um modelo de Floresta Aleatória obtido pela fábrica.

    Args:
        X_train: Atributos de treino.
        y_train: Alvos de treino.
        X_test: Atributos de teste opcionais.
        y_test: Alvos de teste opcionais.

    Returns:
        Tupla com o modelo ajustado e as predições de teste (se dados de teste fornecidos).
    """
    model = ModelFactory.create_model("random_forest")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None
) -> Tuple[BaseEstimator, Optional[np.ndarray]]:
    """Treina um modelo de Gradient Boosting obtido pela fábrica.

    Args:
        X_train: Atributos de treino.
        y_train: Alvos de treino.
        X_test: Atributos de teste opcionais.
        y_test: Alvos de teste opcionais.

    Returns:
        Tupla com o modelo ajustado e as predições de teste (se dados de teste fornecidos).
    """
    model = ModelFactory.create_model("gradient_boosting")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) if X_test is not None else None
    return model, y_pred


def evaluate_model(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Calcula métricas de classificação multiclasse comparando predições e rótulos reais.

    Args:
        y_true: Rótulos reais.
        y_pred: Rótulos preditos.
        model_name: Nome do modelo para o relatório.

    Returns:
        Dicionário com acurácia, precisão, recall, f1-score (ponderados e macro), matriz de confusão e relatório de classificação.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    precision_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_m = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'precision_macro': precision_m,
        'recall_macro': recall_m,
        'f1_macro': f1_m,
        'classification_report': classification_report(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> Dict[str, Any]:
    """Realiza validação cruzada nos dados fornecidos e retorna as estatísticas.

    Args:
        model: Estimador do Scikit-Learn.
        X: DataFrame de atributos.
        y: Série com rótulos.
        cv: Quantidade de dobras (folds).

    Returns:
        Dicionário com pontuações brutas, média, desvio padrão, mínimo e máximo.
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return {
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_min': cv_scores.min(),
        'cv_max': cv_scores.max()
    }


def hyperparameter_tuning_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Otimiza hiperparâmetros de uma Floresta Aleatória por meio de busca em grade (GridSearchCV).

    Args:
        X_train: Atributos de treino.
        y_train: Alvos de treino.

    Returns:
        Tupla contendo o melhor modelo e o dicionário de hiperparâmetros ótimos.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = ModelFactory.create_model("random_forest")
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def plot_feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 15
) -> Optional[plt.Figure]:
    """Gera um gráfico de importância dos atributos para modelos baseados em árvores.

    Args:
        model: Modelo treinado contendo atributo 'feature_importances_'.
        feature_names: Lista contendo os nomes das colunas.
        model_name: Identificador para título do gráfico.
        top_n: Quantidade limite de variáveis a exibir.

    Returns:
        Uma Figura do matplotlib ou None caso o modelo não suporte importância de atributos.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f'Importância dos Atributos - {model_name}')
        ax.bar(range(top_n), importances[indices])
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        return fig
    else:
        print(f"O modelo {model_name} não possui o atributo de importância das variáveis")
        return None


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: Optional[List[str]] = None,
    model_name: str = "Model"
) -> plt.Figure:
    """Renderiza a matriz de confusão como um mapa de calor estilizado.

    Args:
        cm: Matriz de confusão (array numérico).
        class_labels: Rótulos customizados para as classes.
        model_name: Nome do modelo para o cabeçalho.

    Returns:
        Objeto Figura do matplotlib contendo o mapa de calor.
    """
    if class_labels is None:
        class_labels = [str(i) for i in range(1, 6)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels, ax=ax)
    ax.set_title(f'Matriz de Confusão - {model_name}')
    ax.set_ylabel('Classe Real')
    ax.set_xlabel('Classe Predita')
    plt.tight_layout()
    return fig


def train_and_evaluate_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """Treina, valida e avalia sequencialmente todos os classificadores clássicos configurados.

    Args:
        X_train: Atributos pré-processados de treino.
        X_test: Atributos pré-processados de teste.
        y_train: Alvos de treino.
        y_test: Alvos de teste.

    Returns:
        Um dicionário contendo modelos ajustados, predições, métricas e resultados de validação cruzada.
    """
    models_results: Dict[str, Dict[str, Any]] = {}

    model_configs = [
        ("Logistic Regression", train_logistic_regression),
        ("Decision Tree", train_decision_tree),
        ("Random Forest", train_random_forest),
        ("Gradient Boosting", train_gradient_boosting)
    ]

    for model_name, train_func in model_configs:
        print(f"Treinando {model_name}...")
        model, y_pred = train_func(X_train, y_train, X_test, y_test)

        if y_pred is not None:
            metrics = evaluate_model(y_test, y_pred, model_name)
            cv_results = cross_validate_model(model, X_train, y_train, cv=5)

            models_results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'metrics': metrics,
                'cv_results': cv_results
            }

            print(f"  Acurácia: {metrics['accuracy']:.4f}")
            print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
            print(f"  Acurácia CV: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']*2:.4f})\n")

    return models_results


def compare_models(models_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Consolida as principais métricas de todos os modelos avaliados para fins de comparação.

    Args:
        models_results: Retorno da função `train_and_evaluate_all_models`.

    Returns:
        DataFrame consolidado ordenado pelo F1-Score ponderado em ordem decrescente.
    """
    comparison_data = []

    for model_name, results in models_results.items():
        metrics = results['metrics']
        cv_results = results['cv_results']

        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision_Weighted': metrics['precision_weighted'],
            'Recall_Weighted': metrics['recall_weighted'],
            'F1_Weighted': metrics['f1_weighted'],
            'Precision_Macro': metrics['precision_macro'],
            'Recall_Macro': metrics['recall_macro'],
            'F1_Macro': metrics['f1_macro'],
            'CV_Accuracy_Mean': cv_results['cv_mean'],
            'CV_Accuracy_Std': cv_results['cv_std']
        })

    return pd.DataFrame(comparison_data).sort_values('F1_Weighted', ascending=False)
