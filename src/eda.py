from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib configuration
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.bbox"] = "tight"

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "interactions.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "eda_report.md"


def load_data(path: Path) -> pd.DataFrame:
    """
    Carrega o dataset parquet e realiza logs iniciais.

    Args:
        path (Path): Caminho para o arquivo parquet.

    Returns:
        pd.DataFrame: DataFrame carregado.
    """
    logger.info(f"Carregando dados de: {path}")
    df = pd.read_parquet(path)
    mem_usage = df.memory_usage(deep=True).sum() / (1024**2)
    logger.info(f"Shape do dataset: {df.shape}")
    logger.info(f"Uso de memória: {mem_usage:.2f} MB")
    logger.info(f"Primeiras 3 linhas:\n{df.head(3)}")
    return df


def basic_info(df: pd.DataFrame) -> dict:
    """
    Calcula e retorna informações básicas sobre o dataset.

    Args:
        df (pd.DataFrame): DataFrame para analisar.

    Returns:
        dict: Dicionário contendo shape, duplicatas, memória, dtypes e nulos.
    """
    logger.info("Calculando informações básicas do dataset...")
    mem_usage = df.memory_usage(deep=True).sum() / (1024**2)
    missing_count = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()

    info = {
        "shape": df.shape,
        "n_duplicates": int(df.duplicated().sum()),
        "memory_mb": mem_usage,
        "dtypes_count": df.dtypes.astype(str).value_counts().to_dict(),
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    }

    logger.info(f"Shape: {info['shape']}")
    logger.info(f"Linhas duplicadas: {info['n_duplicates']}")
    logger.info(f"Uso de memória: {info['memory_mb']:.2f} MB")
    logger.info(f"Tipos de dados: {info['dtypes_count']}")
    logger.info(f"Valores nulos por coluna: {info['missing_count']}")

    return info


def analyze_target_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Analisa a distribuição da variável target (review_score).

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        figures_dir (Path): Diretório para salvar as figuras.
    """
    logger.info("Analisando distribuição do target (review_score)...")

    stats = df["review_score"].describe()
    logger.info(f"Estatísticas de review_score:\n{stats}")

    counts = df["review_score"].value_counts(dropna=False).sort_index()
    logger.info(f"Contagem por nota:\n{counts}")

    # Sentimento
    positivas = df["review_score"] >= 4
    negativas = df["review_score"] <= 2
    neutras = df["review_score"] == 3
    ausentes = df["review_score"].isna()

    total = len(df)
    pct_pos = positivas.sum() / total * 100
    pct_neg = negativas.sum() / total * 100
    pct_neu = neutras.sum() / total * 100
    pct_aus = ausentes.sum() / total * 100

    logger.info(f"% Positivas (>=4): {pct_pos:.2f}%")
    logger.info(f"% Negativas (<=2): {pct_neg:.2f}%")
    logger.info(f"% Neutras (3): {pct_neu:.2f}%")
    logger.info(f"% Ausentes: {pct_aus:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.countplot(
        data=df.dropna(subset=["review_score"]),
        x="review_score",
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title("Distribuição de Review Score")
    axes[0].set_xlabel("Review Score")
    axes[0].set_ylabel("Contagem")

    labels = ["Positivas", "Neutras", "Negativas", "Ausentes"]
    sizes = [pct_pos, pct_neu, pct_neg, pct_aus]
    colors = sns.color_palette("viridis", 4)
    axes[1].pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[1].set_title("Sentimento das Avaliações")

    plt.tight_layout()
    fig.savefig(figures_dir / "01_review_score_distribution.png")
    plt.close()


def analyze_numerical_features(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Analisa características numéricas (price e freight_value).

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        figures_dir (Path): Diretório para salvar as figuras.
    """
    logger.info("Analisando variáveis numéricas (price, freight_value)...")

    features = ["price", "freight_value"]

    for feat in features:
        if feat in df.columns:
            percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            stats = df[feat].describe(percentiles=percentiles)
            logger.info(f"Estatísticas descritivas para {feat}:\n{stats}")

            q1 = df[feat].quantile(0.25)
            q3 = df[feat].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[feat] < lower_bound) | (df[feat] > upper_bound)]
            pct_outliers = len(outliers) / len(df) * 100
            logger.info(f"% de outliers detectados para {feat}: {pct_outliers:.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    if "price" in df.columns:
        p99_price = df["price"].quantile(0.99)
        sns.histplot(
            df[df["price"] <= p99_price]["price"],
            bins=50,
            ax=axes[0, 0],
            color="skyblue",
        )
        axes[0, 0].set_title("Histograma de Price (Cap em p99)")

        sns.boxplot(x=df["price"], ax=axes[0, 1], color="skyblue")
        axes[0, 1].set_title("Boxplot de Price")

    if "freight_value" in df.columns:
        p99_freight = df["freight_value"].quantile(0.99)
        sns.histplot(
            df[df["freight_value"] <= p99_freight]["freight_value"],
            bins=50,
            ax=axes[1, 0],
            color="lightgreen",
        )
        axes[1, 0].set_title("Histograma de Freight Value (Cap em p99)")

        sns.boxplot(x=df["freight_value"], ax=axes[1, 1], color="lightgreen")
        axes[1, 1].set_title("Boxplot de Freight Value")

    plt.tight_layout()
    fig.savefig(figures_dir / "02_numerical_distributions.png")
    plt.close()


def analyze_categorical_features(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Analisa características categóricas (product_category_name_english e payment_type).

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        figures_dir (Path): Diretório para salvar as figuras.
    """
    logger.info("Analisando variáveis categóricas...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    cat_col = "product_category_name_english"
    if cat_col in df.columns:
        counts = df[cat_col].value_counts()
        total = len(df[cat_col].dropna())

        top15 = counts.head(15)
        top15_pct = top15 / total * 100
        logger.info(f"Top 15 categorias:\n{top15_pct}")

        top5_pct = counts.head(5).sum() / total * 100
        top10_pct = counts.head(10).sum() / total * 100
        logger.info(f"% acumulada Top 5: {top5_pct:.2f}%")
        logger.info(f"% acumulada Top 10: {top10_pct:.2f}%")

        rare_cats = (counts < 10).sum()
        logger.info(f"Número de categorias raras (<10 interações): {rare_cats}")

        sns.barplot(x=top15.values, y=top15.index, ax=axes[0], palette="viridis")
        axes[0].set_title("Top 15 Categorias de Produtos")
        axes[0].set_xlabel("Contagem")

    pay_col = "payment_type"
    if pay_col in df.columns:
        pay_counts = df[pay_col].value_counts()
        pay_pct = pay_counts / len(df[pay_col].dropna()) * 100
        logger.info(f"Distribuição de tipos de pagamento:\n{pay_pct}")

        sns.barplot(x=pay_counts.index, y=pay_counts.values, ax=axes[1], palette="Set2")
        axes[1].set_title("Distribuição de Tipos de Pagamento")
        axes[1].set_ylabel("Contagem")
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    fig.savefig(figures_dir / "03_categorical_distribution.png")
    plt.close()


def analyze_temporal_features(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Analisa características temporais do dataset.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        figures_dir (Path): Diretório para salvar as figuras.
    """
    logger.info("Analisando variáveis temporais...")

    time_col = "order_purchase_timestamp"
    if time_col not in df.columns:
        logger.warning(f"Coluna {time_col} não encontrada.")
        return

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    min_date = df[time_col].min()
    max_date = df[time_col].max()
    range_days = (max_date - min_date).days

    logger.info(
        f"Data mínima: {min_date}, máxima: {max_date}, range: {range_days} dias"
    )

    df["year"] = df[time_col].dt.year
    df["month"] = df[time_col].dt.month
    df["year_month"] = df[time_col].dt.to_period("M").astype(str)
    df["day_of_week"] = df[time_col].dt.day_name()
    df["hour"] = df[time_col].dt.hour

    year_dist = df["year"].value_counts(normalize=True) * 100
    logger.info(f"Distribuição por ano:\n{year_dist}")

    top_months = df["year_month"].value_counts().head(5)
    logger.info(f"Top 5 meses com mais interações:\n{top_months}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. Série temporal por mês
    monthly_counts = df["year_month"].value_counts().sort_index()
    sns.lineplot(
        x=monthly_counts.index, y=monthly_counts.values, ax=axes[0, 0], marker="o"
    )
    axes[0, 0].set_title("Interações por Mês")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Barplot por dia da semana
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    sns.countplot(
        data=df, x="day_of_week", order=order, ax=axes[0, 1], palette="viridis"
    )
    axes[0, 1].set_title("Interações por Dia da Semana")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # 3. Barplot por hora do dia
    sns.countplot(data=df, x="hour", ax=axes[1, 0], color="skyblue")
    axes[1, 0].set_title("Interações por Hora do Dia")

    # 4. Heatmap ano x mês
    heatmap_data = df.groupby(["year", "month"]).size().unstack(fill_value=0)
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d", ax=axes[1, 1])
    axes[1, 1].set_title("Heatmap Ano x Mês")

    plt.tight_layout()
    fig.savefig(figures_dir / "04_temporal_distribution.png")
    plt.close()


def analyze_user_behavior(df: pd.DataFrame, figures_dir: Path) -> dict:
    """
    Analisa o comportamento dos usuários.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        figures_dir (Path): Diretório para salvar as figuras.

    Returns:
        dict: Métricas do comportamento do usuário.
    """
    logger.info("Analisando comportamento do usuário...")

    user_col = "customer_unique_id"
    if user_col not in df.columns:
        logger.warning(f"Coluna {user_col} não encontrada.")
        return {}

    interactions = df[user_col].value_counts()

    percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    stats = interactions.describe(percentiles=percentiles)
    logger.info(f"Estatísticas de interações por usuário:\n{stats}")

    # Cold start distribution
    bins = [0, 1, 2, 3, 4, np.inf]
    labels = ["1", "2", "3", "4", "5+"]
    cold_start = (
        pd.cut(interactions, bins=bins, labels=labels).value_counts(normalize=True)
        * 100
    )
    cold_start = cold_start.sort_index()
    logger.info(f"Distribuição de cold-start (usuários):\n{cold_start}")

    top10_users = interactions.head(10)
    logger.info(f"Top 10 usuários ativos:\n{top10_users}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histograma em log scale
    sns.histplot(
        interactions, bins=50, log_scale=(False, True), ax=axes[0], color="salmon"
    )
    axes[0].set_title("Distribuição de Interações por Usuário (Log Scale)")
    axes[0].set_xlabel("Número de Interações")

    # Cold start pie/bar chart
    sns.barplot(x=cold_start.index, y=cold_start.values, ax=axes[1], palette="viridis")
    axes[1].set_title("Distribuição de Cold Start - Usuários")
    axes[1].set_ylabel("% de Usuários")

    plt.tight_layout()
    fig.savefig(figures_dir / "05_user_behavior.png")
    plt.close()

    return {"user_stats": stats.to_dict(), "user_cold_start": cold_start.to_dict()}


def analyze_product_behavior(df: pd.DataFrame, figures_dir: Path) -> dict:
    """
    Analisa o comportamento dos produtos.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        figures_dir (Path): Diretório para salvar as figuras.

    Returns:
        dict: Métricas do comportamento do produto.
    """
    logger.info("Analisando comportamento do produto...")

    item_col = "product_id"
    if item_col not in df.columns:
        logger.warning(f"Coluna {item_col} não encontrada.")
        return {}

    interactions = df[item_col].value_counts()

    percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    stats = interactions.describe(percentiles=percentiles)
    logger.info(f"Estatísticas de interações por produto:\n{stats}")

    # Cold start distribution
    bins = [0, 1, 5, 10, 50, np.inf]
    labels = ["1", "2-5", "6-10", "11-50", "51+"]
    cold_start = (
        pd.cut(interactions, bins=bins, labels=labels).value_counts(normalize=True)
        * 100
    )
    cold_start = cold_start.sort_index()
    logger.info(f"Distribuição de cold-start (produtos):\n{cold_start}")

    top10_products = interactions.head(10)
    logger.info(f"Top 10 produtos mais comprados:\n{top10_products}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histograma em log scale
    sns.histplot(
        interactions, bins=50, log_scale=(False, True), ax=axes[0], color="orchid"
    )
    axes[0].set_title("Distribuição de Interações por Produto (Log Scale)")
    axes[0].set_xlabel("Número de Interações")

    # Cold start barplot
    sns.barplot(x=cold_start.index, y=cold_start.values, ax=axes[1], palette="Set2")
    axes[1].set_title("Distribuição de Cold Start - Produtos")
    axes[1].set_ylabel("% de Produtos")

    plt.tight_layout()
    fig.savefig(figures_dir / "06_product_behavior.png")
    plt.close()

    return {
        "product_stats": stats.to_dict(),
        "product_cold_start": cold_start.to_dict(),
    }


def analyze_interaction_patterns(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Analisa os padrões de interação (purchase_count, has_review).

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        figures_dir (Path): Diretório para salvar as figuras.
    """
    logger.info("Analisando padrões de interação...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # purchase_count
    if "purchase_count" in df.columns:
        stats = df["purchase_count"].describe()
        logger.info(f"Estatísticas de purchase_count:\n{stats}")

        bins = [0, 1, 2, 3, 4, np.inf]
        labels = ["1", "2", "3", "4", "5+"]
        purchase_dist = (
            pd.cut(df["purchase_count"], bins=bins, labels=labels).value_counts(
                normalize=True
            )
            * 100
        )
        purchase_dist = purchase_dist.sort_index()
        logger.info(f"Distribuição de purchase_count:\n{purchase_dist}")

        repetidas = (df["purchase_count"] > 1).sum() / len(df) * 100
        logger.info(f"% de interações que são compras repetidas: {repetidas:.2f}%")

        # Plot capped at 5
        df_plot = df.copy()
        df_plot["purchase_count_cap"] = df_plot["purchase_count"].clip(upper=5)
        sns.countplot(
            data=df_plot, x="purchase_count_cap", ax=axes[0], palette="viridis"
        )
        axes[0].set_title("Distribuição de Purchase Count (Cap em 5)")
        axes[0].set_xticklabels(["1", "2", "3", "4", "5+"])

    # has_review
    if "has_review" in df.columns:
        review_dist = df["has_review"].value_counts(normalize=True) * 100
        logger.info(f"% com e sem review:\n{review_dist}")

        axes[1].pie(
            review_dist.values,
            labels=review_dist.index,
            autopct="%1.1f%%",
            colors=sns.color_palette("Set2", 2),
            startangle=90,
        )
        axes[1].set_title("Distribuição de Has Review")

    plt.tight_layout()
    fig.savefig(figures_dir / "07_interaction_patterns.png")
    plt.close()


def analyze_bivariate(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Realiza análise bivariada entre as principais variáveis.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        figures_dir (Path): Diretório para salvar as figuras.
    """
    logger.info("Realizando análise bivariada...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Price vs Review Score
    if "price" in df.columns and "review_score" in df.columns:
        df_plot = df.dropna(subset=["price", "review_score"]).copy()
        df_plot["price_log"] = np.log1p(df_plot["price"])

        sns.boxplot(
            data=df_plot, x="review_score", y="price_log", ax=axes[0], palette="viridis"
        )
        axes[0].set_title("Boxplot: Price (Log) vs Review Score")

        corr = df_plot[["price", "review_score"]].corr().iloc[0, 1]
        logger.info(f"Correlação Pearson price vs review_score: {corr:.4f}")

    # Purchase Count vs Has Review
    if "purchase_count" in df.columns and "has_review" in df.columns:
        crosstab = (
            pd.crosstab(
                df["purchase_count"].clip(upper=5), df["has_review"], normalize="index"
            )
            * 100
        )
        logger.info(f"Crosstab Purchase Count (Cap=5) vs Has Review (%):\n{crosstab}")

        crosstab.plot(
            kind="bar", stacked=True, ax=axes[1], color=sns.color_palette("Set2", 2)
        )
        axes[1].set_title("Purchase Count vs Has Review")
        axes[1].set_xlabel("Purchase Count (Capped at 5)")
        axes[1].set_ylabel("%")
        axes[1].set_xticklabels(["1", "2", "3", "4", "5+"], rotation=0)

    plt.tight_layout()
    fig.savefig(figures_dir / "08_bivariate_analysis.png")
    plt.close()


def compute_sparsity(df: pd.DataFrame) -> dict:
    """
    Calcula o sparsity da matriz User-Item.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.

    Returns:
        dict: Métricas de densidade/sparsity.
    """
    logger.info("Calculando sparsity da matriz User-Item...")

    user_col = "customer_unique_id"
    item_col = "product_id"

    if user_col not in df.columns or item_col not in df.columns:
        logger.warning("Colunas de usuário/item não encontradas.")
        return {}

    n_users = df[user_col].nunique()
    n_products = df[item_col].nunique()
    n_interactions = len(df)

    possible_interactions = n_users * n_products
    actual_density = n_interactions / possible_interactions
    sparsity = 1 - actual_density

    logger.info(f"Usuários únicos: {n_users}")
    logger.info(f"Produtos únicos: {n_products}")
    logger.info(f"Interações totais: {n_interactions}")
    logger.info(f"Interações possíveis: {possible_interactions}")
    logger.info(f"Densidade real: {actual_density:.8f}")
    logger.info(f"Sparsity: {sparsity:.8f}")

    # Comparações aproximadas
    # MovieLens 1M: ~4.5% density (95.5% sparsity)
    # Amazon Reviews: < 0.1% density (> 99.9% sparsity)
    logger.info("Comparações de benchmark:")
    logger.info("MovieLens 1M: Sparsity ~95.5%")
    logger.info("Amazon Reviews: Sparsity >99.9%")

    return {
        "n_users": n_users,
        "n_products": n_products,
        "n_interactions": n_interactions,
        "possible_interactions": possible_interactions,
        "actual_density": actual_density,
        "sparsity": sparsity,
    }


def generate_markdown_report(metrics: dict, report_path: Path) -> None:
    """
    Gera o relatório em formato Markdown.

    Args:
        metrics (dict): Dicionário global contendo todas as métricas extraídas.
        report_path (Path): Caminho de saída do relatório.
    """
    logger.info(f"Gerando relatório markdown em {report_path}...")

    basic = metrics.get("basic_info", {})
    user_stats = metrics.get("user_behavior", {}).get("user_stats", {})
    prod_stats = metrics.get("product_behavior", {}).get("product_stats", {})
    sparsity = metrics.get("sparsity", {})

    content = f"""# EDA Report — Olist Interactions Dataset

## 1. Visão Geral
Este documento apresenta a Análise Exploratória de Dados do dataset de interações do Olist, preparado para o desenvolvimento de um Sistema de Recomendação.

## 2. Estatísticas Básicas
- **Total de linhas**: {basic.get('shape', ['-'])[0]}
- **Total de colunas**: {basic.get('shape', ['-'])[1]}
- **Uso de Memória**: {basic.get('memory_mb', 0):.2f} MB
- **Linhas Duplicadas**: {basic.get('n_duplicates', 0)}

## 3. Distribuição do Target (review_score)
A distribuição da variável alvo foi analisada para entender o sentimento geral das compras (positivas >= 4, neutras = 3, negativas <= 2).
![Distribuição Target](figures/01_review_score_distribution.png)

## 4. Features Numéricas (price, freight_value)
Análise do comportamento de preços de produtos e custos de frete, incluindo tratamento visual para outliers (ex: clipping no percentil 99).
![Distribuição Numéricas](figures/02_numerical_distributions.png)

## 5. Features Categóricas (category, payment_type)
O dataset exibe concentrações significativas em top categorias (head) assim como grande variedade na "cauda longa" (long-tail).
![Distribuição Categóricas](figures/03_categorical_distribution.png)

## 6. Análise Temporal
Análise da evolução das interações ao longo dos anos, variação sazonal por meses e comportamentos de dias e horas da compra.
![Distribuição Temporal](figures/04_temporal_distribution.png)

## 7. Comportamento de Usuários
- **Média de interações/usuário**: {user_stats.get('mean', 0):.2f}
- **Mediana**: {user_stats.get('50%', 0):.2f}
- **Max**: {user_stats.get('max', 0):.2f}
![Comportamento Usuário](figures/05_user_behavior.png)

## 8. Comportamento de Produtos
- **Média de interações/produto**: {prod_stats.get('mean', 0):.2f}
- **Mediana**: {prod_stats.get('50%', 0):.2f}
- **Max**: {prod_stats.get('max', 0):.2f}
![Comportamento Produto](figures/06_product_behavior.png)

## 9. Padrões de Interação
Frequência de recompras e presença/ausência de avaliações após o pedido.
![Padrões de Interação](figures/07_interaction_patterns.png)

## 10. Análise Bivariada
Correlações entre nota de avaliação e o preço pago, e a relação de recorrência com propensão de fazer reviews.
![Análise Bivariada](figures/08_bivariate_analysis.png)

## 11. Sparsity da Matriz User-Item
A esparsidade da matriz de recomendação afeta o quão "fria" a base se encontra.
- **Usuários únicos**: {sparsity.get('n_users', 0)}
- **Produtos únicos**: {sparsity.get('n_products', 0)}
- **Total de interações**: {sparsity.get('n_interactions', 0)}
- **Sparsity**: {sparsity.get('sparsity', 0):.6%}
- *Benchmark MovieLens 1M*: ~95.5% sparsity
- *Benchmark Amazon Reviews*: >99.9% sparsity

## 12. Visualizações Geradas
Todas as visualizações encontram-se na pasta `reports/figures/`:
1. `01_review_score_distribution.png`
2. `02_numerical_distributions.png`
3. `03_categorical_distribution.png`
4. `04_temporal_distribution.png`
5. `05_user_behavior.png`
6. `06_product_behavior.png`
7. `07_interaction_patterns.png`
8. `08_bivariate_analysis.png`

## 13. Conclusões e Recomendações
Com base no exposto, recomenda-se especial atenção à esparsidade na modelagem de recomendação, dada a alta concentração de comportamentos do tipo *cold-start* tanto em usuários quanto em produtos.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Relatório markdown gerado com sucesso.")


def main():
    """
    Função principal de orquestração do script.
    """
    logger.info("Iniciando Análise Exploratória de Dados (EDA)")

    # Criar diretórios se não existirem
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Carregar dados
        df = load_data(DATA_PATH)

        metrics = {}

        # Executar análises
        metrics["basic_info"] = basic_info(df)
        analyze_target_distribution(df, FIGURES_DIR)
        analyze_numerical_features(df, FIGURES_DIR)
        analyze_categorical_features(df, FIGURES_DIR)
        analyze_temporal_features(df, FIGURES_DIR)

        metrics["user_behavior"] = analyze_user_behavior(df, FIGURES_DIR)
        metrics["product_behavior"] = analyze_product_behavior(df, FIGURES_DIR)

        analyze_interaction_patterns(df, FIGURES_DIR)
        analyze_bivariate(df, FIGURES_DIR)

        metrics["sparsity"] = compute_sparsity(df)

        # Gerar relatório
        generate_markdown_report(metrics, REPORT_PATH)

        logger.info("EDA finalizada com sucesso!")

    except Exception as e:
        logger.error(f"Erro durante a execução da EDA: {e}")
        raise


if __name__ == "__main__":
    main()
