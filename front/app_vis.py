"""
Dashboard Analítico - Olist Recommender System

Baseado no padrão visual aprovado no app_vis.py do projeto Fase 01.
Stack: Streamlit + Plotly + Seaborn/Matplotlib (dark mode premium).

Abas:
  1. Visão Geral do Dataset (interactions.parquet)
  2. Features Engenheiradas (interactions_fe.parquet)
  3. Análise por Categoria e Usuário
  4. Recomendações e Baselines (se existirem artefatos do train.py)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# --- Path para imports absolutos ---
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --- Caminhos dos artefatos processados ---
RAW_DATA_DIR = BASE_DIR / "data"
INTERACTIONS_PATH = BASE_DIR / "data" / "processed" / "interactions.parquet"
INTERACTIONS_FE_PATH = BASE_DIR / "data" / "processed" / "interactions_fe.parquet"
ID_MAPPINGS_PATH = BASE_DIR / "data" / "processed" / "id_mappings.json"
BASELINE_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "temporary_baseline_recommendations.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FEATURES_DOC_PATH = BASE_DIR / "data" / "processed" / "FEATURES.md"

# --- Configuração da página ---
st.set_page_config(
    page_title="Olist Recommender - Dashboard Analítico",
    layout="wide",
    page_icon="🛒",
)

# --- CSS Dark Mode Premium (mesma identidade visual da Fase 01) ---
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117 !important; }
    [data-testid="stAppViewContainer"] { background-color: #0e1117 !important; }
    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown { color: #e6edf3 !important; }
    div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 800 !important;
        text-shadow: 0px 0px 10px rgba(88, 166, 255, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #161b22;
        border-radius: 8px 8px 0px 0px;
        color: #8b949e;
        border: 1px solid #30363d;
        border-bottom: none;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #ffffff !important;
        border-top: 3px solid #58a6ff !important;
    }
    .stDataFrame { border: 1px solid #30363d; }
    .stAlert { background-color: #161b22; color: #e6edf3; border: 1px solid #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛒 Painel Analítico: Olist Recommender System")
st.markdown(
    "Análise do pipeline de dados, features engenheiradas e avaliação de modelos "
    "de recomendação para o marketplace Olist."
)

# Configurar estilo dark para matplotlib
plt.style.use("dark_background")
sns.set_palette("husl")
plt.rcParams["figure.facecolor"] = "#0e1117"
plt.rcParams["axes.facecolor"] = "#0e1117"
plt.rcParams["savefig.facecolor"] = "#0e1117"


# --- Cache de carregamento ---
@st.cache_data
def load_interactions() -> pd.DataFrame | None:
    if INTERACTIONS_PATH.exists():
        return pd.read_parquet(INTERACTIONS_PATH)
    return None


@st.cache_data
def load_interactions_fe() -> pd.DataFrame | None:
    if INTERACTIONS_FE_PATH.exists():
        return pd.read_parquet(INTERACTIONS_FE_PATH)
    return None


@st.cache_data
def load_baseline_outputs() -> pd.DataFrame | None:
    if BASELINE_OUTPUT_PATH.exists():
        return pd.read_csv(BASELINE_OUTPUT_PATH)
    return None


df = load_interactions()
df_fe = load_interactions_fe()
df_baseline = load_baseline_outputs()


# --- Helpers ---
def plotly_template() -> dict:
    return dict(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#e6edf3",
    )


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    return df[col] if col in df.columns else None


# --- Verificação de artefatos ---
if df is None and df_fe is None:
    st.error(
        "❌ Nenhum artefato do pipeline encontrado. Execute `dvc repro` "
        "para gerar `data/processed/interactions.parquet` e `interactions_fe.parquet`."
    )
    st.stop()

# --- Definição das abas ---
tab_names = ["📊 Visão Geral", "🔧 Feature Engineering"]
if df_baseline is not None:
    tab_names.append("🎯 Recomendações")
tab_names.append("ℹ️ Sobre o Pipeline")

tabs = st.tabs(tab_names)


# ============================================================================
# ABA 1 — VISÃO GERAL DO DATASET
# ============================================================================
with tabs[0]:
    st.header("📊 Visão Geral do Dataset Processado")

    if df is None:
        st.warning("⚠️ `interactions.parquet` não encontrado.")
    else:
        # KPIs principais
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Interações", f"{len(df):,}")
        col2.metric("Usuários Únicos", f"{df['customer_unique_id'].nunique():,}")
        col3.metric("Produtos Únicos", f"{df['product_id'].nunique():,}")
        col4.metric("Categorias", f"{df['product_category_name_english'].nunique() if 'product_category_name_english' in df.columns else 0:,}")

        st.markdown("---")

        # Distribuição do target (review_score)
        st.subheader("Distribuição do `review_score` (Feedback Explícito)")
        col_a, col_b = st.columns(2)

        if "review_score" in df.columns:
            counts = df["review_score"].value_counts().sort_index()
            pct = (df["review_score"].value_counts(normalize=True).sort_index() * 100).round(2)

            with col_a:
                fig = px.bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    color=counts.index.astype(str),
                    labels={"x": "review_score", "y": "Frequência"},
                    title="Volume de Avaliações por Nota",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                fig.update_layout(showlegend=False, **plotly_template())
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig = px.bar(
                    x=pct.index.astype(str),
                    y=pct.values,
                    color=pct.index.astype(str),
                    labels={"x": "review_score", "y": "Percentual (%)"},
                    title="Distribuição Percentual",
                    text=[f"{v:.1f}%" for v in pct.values],
                    color_discrete_sequence=px.colors.sequential.Oranges_r,
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, **plotly_template())
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Distribuições numéricas
        st.subheader("📈 Distribuições Numéricas (price, freight_value)")
        col_c, col_d = st.columns(2)
        for col_name, col_container, title in [
            ("price", col_c, "Preço (R$)"),
            ("freight_value", col_d, "Frete (R$)"),
        ]:
            series = _safe_col(df, col_name)
            if series is not None:
                with col_container:
                    fig = px.histogram(
                        series.dropna(),
                        nbins=60,
                        title=f"Distribuição de {title}",
                        labels={"value": title},
                        color_discrete_sequence=["#58a6ff"],
                    )
                    fig.update_layout(**plotly_template())
                    st.plotly_chart(fig, use_container_width=True)

        # Log transform para visualizar caudas longas
        st.subheader("📐 Análise de Cauda Longa (Escala Logarítmica)")
        if "price" in df.columns:
            log_price = np.log1p(df["price"].dropna())
            fig = px.histogram(
                log_price,
                nbins=60,
                title="Distribuição Log(price+1) — Simetria após transformação",
                labels={"value": "log(price + 1)"},
                color_discrete_sequence=["#3fb950"],
            )
            fig.update_layout(**plotly_template())
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Top categorias
        if "product_category_name_english" in df.columns:
            st.subheader("🛍️ Top 15 Categorias Mais Vendidas")
            top_cats = (
                df["product_category_name_english"]
                .value_counts()
                .head(15)
                .reset_index()
            )
            top_cats.columns = ["Categoria", "Interações"]
            fig = px.bar(
                top_cats.sort_values("Interações"),
                x="Interações",
                y="Categoria",
                orientation="h",
                color="Interações",
                color_continuous_scale="Blues",
                title="Concentração de Vendas por Categoria",
            )
            fig.update_layout(height=500, **plotly_template())
            st.plotly_chart(fig, use_container_width=True)

        # Análise temporal
        st.markdown("---")
        st.subheader("📅 Comportamento Temporal")
        if "order_purchase_timestamp" in df.columns:
            ts = pd.to_datetime(df["order_purchase_timestamp"])
            ts_df = pd.DataFrame({"timestamp": ts, "year_month": ts.dt.to_period("M").astype(str)})
            ts_agg = ts_df.groupby("year_month").size().reset_index(name="volume")
            fig = px.line(
                ts_agg,
                x="year_month",
                y="volume",
                title="Volume de Interações ao Longo do Tempo",
                markers=True,
            )
            fig.update_traces(line_color="#58a6ff", marker_color="#3fb950")
            fig.update_layout(**plotly_template())
            st.plotly_chart(fig, use_container_width=True)

        # Distribuição por hora do dia
        if "order_purchase_timestamp" in df.columns:
            col_e, col_f = st.columns(2)
            ts = pd.to_datetime(df["order_purchase_timestamp"])
            with col_e:
                hour_counts = ts.dt.hour.value_counts().sort_index()
                fig = px.bar(
                    x=hour_counts.index,
                    y=hour_counts.values,
                    labels={"x": "Hora do dia", "y": "Frequência"},
                    title="Distribuição por Hora do Dia",
                    color_discrete_sequence=["#f0883e"],
                )
                fig.update_layout(**plotly_template())
                st.plotly_chart(fig, use_container_width=True)
            with col_f:
                dow_counts = ts.dt.dayofweek.value_counts().sort_index()
                day_names = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
                fig = px.bar(
                    x=[day_names[i] for i in dow_counts.index],
                    y=dow_counts.values,
                    labels={"x": "Dia da semana", "y": "Frequência"},
                    title="Distribuição por Dia da Semana",
                    color_discrete_sequence=["#a371f7"],
                )
                fig.update_layout(**plotly_template())
                st.plotly_chart(fig, use_container_width=True)

        # Análise de sparsity
        st.markdown("---")
        st.subheader("🕳️ Sparsity da Matriz User-Item")
        if "customer_unique_id" in df.columns and "product_id" in df.columns:
            n_users = df["customer_unique_id"].nunique()
            n_items = df["product_id"].nunique()
            n_interactions = len(df)
            total = n_users * n_items
            sparsity = 1 - (n_interactions / total)

            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("Usuários × Produtos", f"{total:,}")
            cs2.metric("Interações Reais", f"{n_interactions:,}")
            cs3.metric("Sparsity", f"{sparsity:.4%}")

            st.info(
                "💡 **Interpretação:** Sparsity > 99.99% indica que a maioria dos pares (user, item) "
                "nunca foram observados. Isso é esperado em e-commerce e justifica o uso de técnicas "
                "de embeddings e filtragem colaborativa."
            )


# ============================================================================
# ABA 2 — FEATURE ENGINEERING
# ============================================================================
with tabs[1]:
    st.header("🔧 Análise das Features Engenheiradas")

    if df_fe is None:
        st.warning("⚠️ `interactions_fe.parquet` não encontrado. Execute `dvc repro --single-stage featurize`.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Linhas", f"{len(df_fe):,}")
        col2.metric("Colunas", f"{df_fe.shape[1]}")
        col3.metric("Categorias mapeadas", f"{df_fe['category_id'].nunique() if 'category_id' in df_fe.columns else 0:,}")

        st.markdown("---")

        # Agrupamento das features
        st.subheader("📋 Mapa de Features por Categoria")
        feature_groups = {
            "Identificadores": ["customer_unique_id", "product_id", "user_id", "product_id_idx", "category_id"],
            "Target/Sinal": ["review_score", "has_review", "purchase_count"],
            "Numéricas": ["price", "freight_value", "price_log", "freight_value_log", "price_to_freight_ratio", "has_price_outlier"],
            "Temporais": ["purchase_year", "purchase_month", "purchase_day_of_week", "purchase_hour", "purchase_day_of_month", "is_weekend", "is_holiday_season", "days_since_reference"],
            "Categóricas Encodadas": ["category_target_enc", "category_frequency", "category_is_top10", "category_is_rare", "payment_type_credit_card", "payment_type_boleto", "payment_type_voucher", "payment_type_debit_card"],
            "Agregações Usuário": ["user_total_purchases", "user_avg_price", "user_avg_freight", "user_purchase_span_days", "user_recency_days", "user_review_rate"],
            "Agregações Produto": ["product_popularity", "product_unique_buyers", "product_avg_review_score", "product_avg_price", "product_avg_freight", "product_review_rate"],
        }

        # Validar quais grupos estão presentes
        group_counts = {k: sum(1 for c in v if c in df_fe.columns) for k, v in feature_groups.items()}
        st.bar_chart(pd.Series(group_counts).sort_values())

        st.markdown("---")

        # Análise comparativa antes/depois
        st.subheader("🔄 Comparativo Antes/Depois da FE")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Antes** (interactions.parquet):")
            if df is not None:
                st.metric("Colunas", df.shape[1])
            else:
                st.metric("Colunas", "N/A")
        with col_b:
            st.markdown("**Depois** (interactions_fe.parquet):")
            st.metric("Colunas", df_fe.shape[1])

        # Distribuição de review_score pós-FE
        st.markdown("---")
        st.subheader("⭐ Distribuição de Notas (Pós-FE)")
        if "review_score" in df_fe.columns:
            rs = df_fe["review_score"].dropna()
            fig = px.histogram(rs, nbins=5, title="Distribuição de review_score", color_discrete_sequence=["#3fb950"])
            fig.update_layout(**plotly_template())
            st.plotly_chart(fig, use_container_width=True)

        # Top 10 categorias por target encoding
        if "category_target_enc" in df_fe.columns and "product_category_name_english" in df_fe.columns:
            st.subheader("🏆 Categorias com Maior Target Encoding (review médio)")
            cat_target = (
                df_fe.groupby("product_category_name_english")["category_target_enc"]
                .first()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            cat_target.columns = ["Categoria", "Target Encoding"]
            fig = px.bar(
                cat_target.sort_values("Target Encoding"),
                x="Target Encoding",
                y="Categoria",
                orientation="h",
                color="Target Encoding",
                color_continuous_scale="Viridis",
                title="Ranking de Categorias (Target Encoding)",
            )
            fig.update_layout(height=500, **plotly_template())
            st.plotly_chart(fig, use_container_width=True)

        # Heatsubset de features numéricas para correlação
        st.markdown("---")
        st.subheader("🔗 Mapa de Correlação (Features Numéricas)")
        num_cols = [
            "price", "freight_value", "price_log", "price_to_freight_ratio",
            "user_total_purchases", "user_recency_days", "user_avg_price",
            "product_popularity", "product_avg_review_score", "category_target_enc",
        ]
        num_cols = [c for c in num_cols if c in df_fe.columns]
        if len(num_cols) >= 3:
            corr = df_fe[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                ax=ax,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title("Correlações entre Features Numéricas", fontsize=14, pad=15)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)


# ============================================================================
# ABA 3 — RECOMENDAÇÕES (CONDICIONAL)
# ============================================================================
baseline_tab_idx = 2 if len(tabs) > 3 else None
if baseline_tab_idx is not None:
    with tabs[baseline_tab_idx]:
        st.header("🎯 Top Recomendações Geradas pelos Baselines")

        if df_baseline is None or df_baseline.empty:
            st.warning(
                "⚠️ Artefato `temporary_baseline_recommendations.csv` não encontrado. "
                "Execute `uv run python src/train.py` para gerar as recomendações dos baselines."
            )
        else:
            # Carrega nome das categorias para enriquecer a tabela
            category_map = {}
            if df is not None and "product_id" in df.columns and "product_category_name_english" in df.columns:
                category_map = (
                    df.drop_duplicates("product_id")
                    .set_index("product_id")["product_category_name_english"]
                    .to_dict()
                )

            col1, col2, col3 = st.columns(3)
            col1.metric("Linhas de Recomendação", f"{len(df_baseline)}")
            col2.metric("Modelos Gerados", f"{df_baseline.shape[1] // 2}")
            col3.metric("Top-N", "10")

            st.markdown("---")
            st.subheader("📋 Tabela de Recomendações Enriquecidas")

            # Enriquecer com categorias se possível
            display_df = df_baseline.copy()
            for col_prefix in ["Popularity", "TopRated"]:
                cat_col = f"{col_prefix}_Category"
                id_col = f"{col_prefix}_ID"
                if id_col in display_df.columns and category_map:
                    display_df[cat_col] = display_df[id_col].map(category_map).fillna("unknown")

            st.dataframe(display_df.head(20), use_container_width=True)

            # Distribuição de categorias recomendadas
            if "Popularity_Category" in display_df.columns:
                st.subheader("📊 Categorias Recomendadas (Baseline Popularidade)")
                cat_counts = display_df["Popularity_Category"].value_counts().head(15)
                fig = px.bar(
                    x=cat_counts.values,
                    y=cat_counts.index,
                    orientation="h",
                    labels={"x": "Frequência", "y": "Categoria"},
                    title="Concentração das Recomendações",
                    color_discrete_sequence=["#58a6ff"],
                )
                fig.update_layout(height=500, **plotly_template())
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# ABA 4 — SOBRE O PIPELINE
# ============================================================================
with tabs[-1]:
    st.header("ℹ️ Sobre o Pipeline de Dados")

    st.markdown(
        """
        ### 🎯 Objetivo
        Construir um sistema de recomendação de produtos para o marketplace Olist,
        baseado no histórico de compras e avaliações dos clientes.

        ### 📦 Pipeline DVC
        ```
        Raw CSVs (9 arquivos)
          ↓
        prepare (src/data_preparation.py)
          ↓ interactions.parquet (10 colunas, ~100k interações)
        featurize (src/feature_engineering.py)
          ↓ interactions_fe.parquet (42 colunas)
        validate (shape == (99785, 42))
          ↓
        train (src/train.py)
          ↓ Baseline popularity + top-rated + item-item CF
          ↓ MLflow tracking
        ```

        ### 🧪 Decisões Arquiteturais
        - **Cold-start desabilitado**: filtro reduziria o dataset de 99.785 para 2.656 linhas.
        - **Target encoding** para categorias com Bayesian smoothing (alpha=10).
        - **Split temporal** (não aleatório) para evitar data leakage.
        - **Embeddings** para `user_id`, `product_id_idx`, `category_id`.

        ### 📊 Métricas Avaliadas
        - NDCG@K (principal)
        - Recall@K
        - MAP@K
        - Hit Rate@K

        ### 🔧 Como Reproduzir
        ```bash
        uv run python src/data_preparation.py
        uv run python src/feature_engineering.py
        uv run python src/train.py
        mlflow ui --host 127.0.0.1 --port 5000
        ```
        """
    )

    # Listar artefatos
    st.markdown("---")
    st.subheader("📁 Artefatos do Projeto")
    if INTERACTIONS_PATH.exists():
        size_kb = INTERACTIONS_PATH.stat().st_size / 1024
        st.success(f"✅ `data/processed/interactions.parquet` — {size_kb:.1f} KB")
    else:
        st.error("❌ `data/processed/interactions.parquet`")

    if INTERACTIONS_FE_PATH.exists():
        size_kb = INTERACTIONS_FE_PATH.stat().st_size / 1024
        st.success(f"✅ `data/processed/interactions_fe.parquet` — {size_kb:.1f} KB")
    else:
        st.error("❌ `data/processed/interactions_fe.parquet`")

    if ID_MAPPINGS_PATH.exists():
        size_kb = ID_MAPPINGS_PATH.stat().st_size / 1024
        st.success(f"✅ `data/processed/id_mappings.json` — {size_kb:.1f} KB")
    else:
        st.warning("⚠️ `data/processed/id_mappings.json`")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.8rem;'>"
    "Painel Analítico de Recomendação v2.0 | Olist Tech Challenge Fase 02</div>",
    unsafe_allow_html=True,
)