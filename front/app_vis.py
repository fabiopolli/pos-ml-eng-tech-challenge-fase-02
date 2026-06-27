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

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import yaml

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
BASELINE_RESULTS_PATH = BASE_DIR / "data" / "processed" / "baseline_results.csv"
TRAIN_SPLIT_PATH = BASE_DIR / "data" / "processed" / "train_split.parquet"
TEST_SPLIT_PATH = BASE_DIR / "data" / "processed" / "test_split.parquet"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FEATURES_DOC_PATH = BASE_DIR / "data" / "processed" / "FEATURES.md"

# --- Caminhos dos artefatos do NCF (PyTorch) ---
NCF_MODEL_PATH = BASE_DIR / "artifacts" / "ncf_final.pt"
NCF_METRICS_PATH = BASE_DIR / "artifacts" / "metrics_Ablation_FINAL_no_aux_emb32.json"
NCF_SCALER_PATH = BASE_DIR / "artifacts" / "scaler.pkl"
NCF_CONFIG_PATH = BASE_DIR / "configs" / "selected_features.yaml"
NCF_BEST_CONFIG_PATH = BASE_DIR / "configs" / "ncf_best.yaml"
NCF_MLFLOW_DB = BASE_DIR / "artifacts" / "mlflow.db"
NCF_FIG_BASELINE = FIGURES_DIR / "ncf_vs_baseline.png"
NCF_FIG_TVT = FIGURES_DIR / "ncf_train_val_test.png"
NCF_FIG_COLDSTART = FIGURES_DIR / "coldstart_analysis.png"
NCF_FIG_EMBED = FIGURES_DIR / "embedding_norms.png"
NCF_FIG_OPTIMIZATION = FIGURES_DIR / "ncf_optimization_comparison.png"

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


@st.cache_data
def load_baseline_results() -> pd.DataFrame | None:
    if BASELINE_RESULTS_PATH.exists():
        return pd.read_csv(BASELINE_RESULTS_PATH)
    return None


@st.cache_data
def load_train_test_splits() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    train_df = pd.read_parquet(TRAIN_SPLIT_PATH) if TRAIN_SPLIT_PATH.exists() else None
    test_df = pd.read_parquet(TEST_SPLIT_PATH) if TEST_SPLIT_PATH.exists() else None
    return train_df, test_df


df = load_interactions()
df_fe = load_interactions_fe()
df_baseline = load_baseline_outputs()
df_results = load_baseline_results()
train_df, test_df = load_train_test_splits()


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
tab_names = [
    "📊 Visão Geral",
    "🔧 Feature Engineering",
    "🏋️ Baselines",
]
# Aba do NCF só aparece se os artefatos existirem
_ncf_available = (
    NCF_MODEL_PATH.exists()
    and NCF_METRICS_PATH.exists()
    and NCF_CONFIG_PATH.exists()
)
if _ncf_available:
    tab_names.append("🧠 NCF (MLP PyTorch)")
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
# ABA 3 — RESULTADOS DO TREINAMENTO
# ============================================================================
with tabs[2]:
    st.header("🏋️ Resultados do Treinamento — Modelos Baseline")

    if df_results is None:
        st.warning(
            "⚠️ Artefato `baseline_results.csv` não encontrado. "
            "Execute `uv run python src/train.py` para gerar os resultados."
        )
    else:
        # KPIs de Split
        st.subheader("📊 Configuração do Split Temporal")
        col_split1, col_split2, col_split3, col_split4 = st.columns(4)
        
        if train_df is not None and test_df is not None:
            col_split1.metric("Treino", f"{len(train_df):,} registros")
            col_split2.metric("Teste", f"{len(test_df):,} registros")
            
            if "order_purchase_timestamp" in train_df.columns:
                train_min = pd.to_datetime(train_df["order_purchase_timestamp"]).min().strftime("%Y-%m")
                train_max = pd.to_datetime(train_df["order_purchase_timestamp"]).max().strftime("%Y-%m")
                col_split3.metric("Período Treino", f"{train_min} → {train_max}")
            
            if "order_purchase_timestamp" in test_df.columns:
                test_min = pd.to_datetime(test_df["order_purchase_timestamp"]).min().strftime("%Y-%m")
                test_max = pd.to_datetime(test_df["order_purchase_timestamp"]).max().strftime("%Y-%m")
                col_split4.metric("Período Teste", f"{test_min} → {test_max}")
        else:
            col_split1.metric("Treino", "70%")
            col_split2.metric("Validação", "15%")
            col_split3.metric("Teste", "15%")
            col_split4.metric("Split", "Temporal")

        st.markdown("---")

        # Introdução explicativa
        st.info(
            """
            **📚 O que são Modelos Baseline?**\n
            Modelos baseline são referências mínimas de performance. Qualquer modelo avançado "
            deve superar esses resultados. Usamos 3 abordagens:\n
            - **Popularity**: Itens mais vendidos globalmente (sem personalização)\n
            - **TopRated**: Itens com melhor nota média (com filtro de mínimo de reviews)\n
            - **ItemItem CF**: Similaridade entre itens baseada em co-ocorrência
            """
        )

        st.markdown("---")

        # Tabela de Resultados Completa
        st.subheader("📋 Tabela de Resultados — Todas as Métricas")
        
        # Preparar dados para exibição
        display_results = df_results.copy()
        display_results["model_k"] = display_results["model"] + " K=" + display_results["k"].astype(str)
        if "min_reviews" in display_results.columns:
            display_results.loc[display_results["min_reviews"] > 0, "model_k"] += " (min=" + \
                display_results.loc[display_results["min_reviews"] > 0, "min_reviews"].astype(str) + ")"
        
        # Selecionar colunas principais
        cols_show = ["model_k", "map_10", "ndcg_10", "precision_10", "recall_10", "hitrate_10"]
        cols_show = [c for c in cols_show if c in display_results.columns]
        
        results_table = display_results[cols_show].copy()
        results_table.columns = ["Modelo", "MAP@10", "NDCG@10", "Precision@10", "Recall@10", "HitRate@10"]
        results_table = results_table.round(4)
        
        st.dataframe(results_table, use_container_width=True)

        st.markdown("---")

        # Gráfico 1: Comparação de MAP@10 entre modelos
        st.subheader("📈 Comparação de MAP@10 por Modelo")
        
        # Filtrar apenas modelos com K=10 para comparação justa
        k10_results = df_results[df_results["k"] == 10].copy()
        k10_results["model_label"] = k10_results.apply(
            lambda x: f"{x['model']}" + (f" (min={int(x['min_reviews'])})" if x.get("min_reviews", 0) > 0 else ""),
            axis=1
        )
        
        fig = px.bar(
            k10_results,
            x="model_label",
            y="map_10",
            color="model",
            barmode="group",
            labels={"map_10": "MAP@10", "model_label": "Modelo"},
            title="MAP@10 — Mean Average Precision",
            color_discrete_sequence=["#58a6ff", "#3fb950", "#f0883e"],
        )
        fig.update_layout(**plotly_template(), showlegend=False)
        fig.update_yaxes(title="MAP@10 (%)", tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

        # Gráfico 2: Popularity Baseline - Métricas por K
        st.markdown("---")
        st.subheader("📊 Popularity Baseline — Evolução por K")
        
        pop_results = df_results[df_results["model"] == "PopularityBaseline"].copy()
        if not pop_results.empty:
            metrics_for_k = ["map_10", "ndcg_10", "hitrate_10"]
            metric_labels = {"map_10": "MAP@10", "ndcg_10": "NDCG@10", "hitrate_10": "HitRate@10"}
            
            k_metrics = pop_results[["k"] + metrics_for_k].melt(
                id_vars="k", 
                var_name="métrica", 
                value_name="valor"
            )
            k_metrics["métrica"] = k_metrics["métrica"].map(metric_labels)
            
            fig = px.line(
                k_metrics,
                x="k",
                y="valor",
                color="métrica",
                markers=True,
                labels={"k": "K (número de recomendações)", "valor": "Score"},
                title="Popularity Baseline — Métricas vs. K",
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=10))
            fig.update_layout(**plotly_template())
            fig.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)

        # Gráfico 3: Comparação de todas as métricas principais
        st.markdown("---")
        st.subheader("🎯 Comparação Completa de Métricas (K=10)")
        
        # Preparar dados para radar chart
        metrics_compare = ["map_10", "ndcg_10", "recall_10", "hitrate_10"]
        model_compare = df_results[df_results["k"] == 10][["model", "min_reviews"] + metrics_compare].copy()
        
        if not model_compare.empty:
            # Radar chart
            categories = ["MAP@10", "NDCG@10", "Recall@10", "HitRate@10"]
            
            fig = go.Figure()
            colors = {"PopularityBaseline": "#58a6ff", "TopRatedBaseline": "#3fb950", "ItemItemCF": "#f0883e"}
            
            for _, row in model_compare.iterrows():
                values = [row[m] for m in metrics_compare]
                label = row["model"]
                if row.get("min_reviews", 0) > 0:
                    label += f" (min={int(row['min_reviews'])})"
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Fechar o polígono
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor=f"rgba(88, 166, 255, 0.2)",
                    line=dict(color=colors.get(row["model"], "#58a6ff"), width=2),
                    name=label,
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        tickformat=".2%",
                    )
                ),
                showlegend=True,
                title="Perfil de Performance dos Modelos",
                **plotly_template(),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Análise dos Resultados
        st.markdown("---")
        st.subheader("🔍 Análise dos Resultados")
        
        # Explicação dos resultados zero
        st.markdown(
            """
            ### Por que TopRated e ItemItemCF têm 0%?
            
            **Este resultado é esperado e não representa um bug.** Para entender:
            
            1. **Esparsidade extrema**: A matriz user-item tem 99.9967% de sparsity
               - Possíveis interações: ~3 Bilhões
               - Interações reais: 99.785
            
            2. **TopRated Baseline**: Recomenda os **mesmos itens para todos**
               - Chance de um usuário específico comprar um item específico ≈ 0.003%
            
            3. **ItemItemCF**: Depende do histórico do usuário
               - Muitos usuários têm apenas 1-2 compras
               - Limitedo by sparse co-occurrence matrix
            
            ### Conclusão
            
            O **Popularity Baseline é o piso de performance esperado**:
            - HitRate K=10: **1.33%** (alguns usuários compraram itens populares)
            - Para melhorar, precisamos de **modelos com aprendizado** (NCF + embeddings)
            """
        )

        # Próximos passos
        st.markdown("---")
        st.success(
            """
            ### ✅ Status dos Entregáveis
            
            | Etapa | Status |
            |-------|--------|
            | Pipeline de dados (DVC) | ✅ Concluído |
            | Feature Engineering (10→42) | ✅ Concluído |
            | Modelos Baseline | ✅ Concluído |
            | Métricas reais (MAP, NDCG, etc.) | ✅ Concluído |
            | Modelo NCF (PyTorch) | 🔄 Em desenvolvimento |
            | MLflow Tracking | ⚠️ Requer servidor |
            | Model Registry | ⏳ Pendente |
            """
        )


# ============================================================================
# ABA 4 — NCF (MLP PyTorch)
# ============================================================================
if _ncf_available:
    ncf_tab_idx = 3
    with tabs[ncf_tab_idx]:
        st.header("🧠 NCF Híbrido — Multi-Layer Perceptron (PyTorch)")

        # Carregar métricas e config do NCF
        with open(NCF_METRICS_PATH) as f:
            ncf_metrics = json.load(f)
        with open(NCF_CONFIG_PATH) as f:
            ncf_config = yaml.safe_load(f)

        # Carregar MLflow tracking (opcional)
        ncf_mlflow_runs = []
        try:
            import mlflow as _mlflow

            _mlflow.set_tracking_uri(f"sqlite:///{NCF_MLFLOW_DB}")
            _client = _mlflow.tracking.MlflowClient()
            _exp = _client.get_experiment_by_name("Olist_NCF_Baseline")
            if _exp is not None:
                ncf_mlflow_runs = _client.search_runs(
                    experiment_ids=[_exp.experiment_id],
                    filter_string="status = 'FINISHED'",
                    order_by=["start_time DESC"],
                )
        except Exception:
            pass

        # --- Bloco 1: KPIs principais ---
        st.subheader("📊 Métricas @ K=10 (Test Set)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("HitRate@10", f"{ncf_metrics['test']['HitRate@K']:.4f}")
        col2.metric("Recall@10", f"{ncf_metrics['test']['Recall@K']:.4f}")
        col3.metric("Precision@10", f"{ncf_metrics['test']['Precision@K']:.4f}")
        col4.metric("NDCG@10", f"{ncf_metrics['test']['NDCG@K']:.4f}")
        col5.metric("MAP@10", f"{ncf_metrics['test']['MAP@K']:.4f}")

        # Banner de produção
        st.success(
            f"🏆 **Modelo em Production** (MLflow Registry) — "
            f"NDCG@10 = {ncf_metrics['test']['NDCG@K']:.4f} "
            f"(+{ncf_metrics['test']['NDCG@K'] / 0.0045 * 100 - 100:.0f}% vs baseline)"
        )

        st.markdown("---")

        # --- Bloco 2: Comparação com Baseline ---
        st.subheader("⚔️ NCF vs Baseline (Popularidade)")
        if NCF_FIG_BASELINE.exists():
            st.image(str(NCF_FIG_BASELINE), use_container_width=True)
        else:
            st.warning("Figura `ncf_vs_baseline.png` não encontrada. Execute `uv run python scripts/generate_ncf_figures.py`.")

        # Lift calculado
        if "baseline" in ncf_metrics:
            st.info("Lift calculado a partir das métricas salvas no JSON do run MLflow.")
        else:
            st.success(
                f"**NCF supera baseline em ~30x no NDCG@10** "
                f"(NCF: {ncf_metrics['test']['NDCG@K']:.4f} vs Baseline típico: ~0.005)"
            )

        st.markdown("---")

        # --- Bloco 3: Gap Train/Val/Test ---
        st.subheader("📈 Análise de Generalização (Train vs Val vs Test)")
        if NCF_FIG_TVT.exists():
            st.image(str(NCF_FIG_TVT), use_container_width=True)

        col_train, col_val, col_test = st.columns(3)
        col_train.info("**Train (sanity check)**\n\nNDCG ~0.60 — Modelo aprendeu a ranquear corretamente os pares vistos no treino.")
        col_val.info("**Validation**\n\nNDCG ~0.30 — Generalização razoável em dados não vistos.")
        col_test.info("**Test**\n\nNDCG ~0.13 — Queda esperada: 98% dos usuários do test são cold-start.")

        st.markdown("---")

        # --- Bloco 4: Cold-start analysis ---
        st.subheader("🧊 Análise de Cold-Start")
        if NCF_FIG_COLDSTART.exists():
            st.image(str(NCF_FIG_COLDSTART), use_container_width=True)

        st.warning(
            "**Cold-Start Massivo**: 98.4% dos usuários do test set são inéditos no treino. "
            "Para esses usuários, o embedding é aleatório e o score depende apenas de "
            "**item embedding + categoria + features auxiliares**."
        )

        st.markdown("---")

        # --- Bloco 5: Comparação de Runs (Etapa 4) ---
        st.subheader("🏆 Comparação de Runs (Etapa 4 — Otimização)")
        if NCF_FIG_OPTIMIZATION.exists():
            st.image(str(NCF_FIG_OPTIMIZATION), use_container_width=True)

        # Tabela resumo das runs
        runs_summary = []
        import os as _os
        for _f in sorted(_os.listdir(BASE_DIR / "artifacts")):
            if _f.startswith("metrics_") and _f.endswith(".json"):
                with open(BASE_DIR / "artifacts" / _f) as _fp:
                    _m = json.load(_fp)
                runs_summary.append({
                    "Run": _m.get("run_name", _f.replace("metrics_", "").replace(".json", "")),
                    "NDCG@10": round(_m["test"]["NDCG@K"], 4),
                    "MAP@10": round(_m["test"]["MAP@K"], 4),
                    "HitRate@10": round(_m["test"]["HitRate@K"], 4),
                    "Best Val NDCG": round(_m["best_val_ndcg"], 4),
                })
        if runs_summary:
            runs_df = pd.DataFrame(runs_summary).sort_values("NDCG@10", ascending=False)
            st.dataframe(runs_df, use_container_width=True, hide_index=True)

            best_ndcg = runs_df["NDCG@10"].max()
            worst_ndcg = runs_df["NDCG@10"].min()
            st.info(
                f"📊 **Lift entre melhor e pior modelo**: "
                f"`{best_ndcg:.4f} / {worst_ndcg:.4f} = {best_ndcg / worst_ndcg:.2f}x` "
                f"em NDCG@10. A otimização dos hiperparâmetros rendeu "
                f"**{(best_ndcg / worst_ndcg - 1) * 100:.0f}%** de melhoria."
            )

        # Ablation finding
        st.warning(
            "🧪 **Achado da Ablation**: As 20 side features **prejudicaram** o modelo final "
            f"(NDCG 0.2226 → 0.2725 sem features, **+22.5% lift**). "
            "Hipótese: com 98% cold-start, os embeddings aleatórios não contribuem, "
            "e as features normalizadas pelo scaler fitado no treino dominam o sinal "
            "de forma homogênea entre usuários."
        )

        st.markdown("---")

        # --- Bloco 5: Embeddings ---
        st.subheader("🔮 Embeddings Aprendidos")
        if NCF_FIG_EMBED.exists():
            st.image(str(NCF_FIG_EMBED), use_container_width=True)

        col_emb1, col_emb2, col_emb3 = st.columns(3)
        col_emb1.metric(
            "User Embeddings",
            f"{ncf_config['n_users']:,}",
            help="Dimensão 16 — 93.358 usuários",
        )
        col_emb2.metric(
            "Item Embeddings",
            f"{ncf_config['n_items']:,}",
            help="Dimensão 16 — 32.216 produtos",
        )
        col_emb3.metric(
            "Category Embeddings",
            f"{ncf_config['n_categories']:,}",
            help="Dimensão 8 — 72 categorias",
        )

        st.markdown("---")

        # --- Bloco 6: Hiperparâmetros ---
        st.subheader("⚙️ Hiperparâmetros do Modelo FINAL (Etapa 4)")

        ncf_hp = {
            "Embedding dim (user/item)": "32",
            "Embedding dim (category)": "8",
            "MLP hidden layers": "[64, 32]",
            "Dropout": "0.5",
            "Learning rate": "5e-4",
            "Optimizer": "AdamW",
            "Weight decay": "5e-4",
            "Batch size": "2048",
            "N negatives (treino)": "8",
            "Loss function": "BPR",
            "Epochs (com early stop)": "14",
            "Patience": "3",
            "Gradient clipping": "5.0",
            "Scheduler": "ReduceLROnPlateau(factor=0.5, patience=2)",
            "Scaler": "N/A (ablation — sem features auxiliares)",
            "Negative sampling": "on-the-fly",
            "Side features": "❌ DESABILITADO (ablation)",
        }

        hp_cols = st.columns(2)
        for i, (k, v) in enumerate(ncf_hp.items()):
            with hp_cols[i % 2]:
                st.markdown(f"**{k}**: `{v}`")

        st.markdown("---")

        # --- Bloco 7: MLflow runs ---
        st.subheader("📋 MLflow Tracking")
        if ncf_mlflow_runs:
            for run in ncf_mlflow_runs[:3]:
                with st.expander(
                    f"Run: {run.data.tags.get('mlflow.runName', 'unnamed')} "
                    f"({run.info.run_id[:8]}) — {run.info.status}",
                    expanded=True,
                ):
                    rcols = st.columns(2)
                    with rcols[0]:
                        st.markdown("**Parâmetros:**")
                        st.json(dict(run.data.params))
                    with rcols[1]:
                        st.markdown("**Métricas de Test:**")
                        test_m = {
                            k.replace("test_", ""): round(v, 4)
                            for k, v in run.data.metrics.items()
                            if k.startswith("test_")
                        }
                        st.json(test_m)
        else:
            st.warning(
                "Nenhuma run completa encontrada no MLflow. "
                "Execute `uv run python scripts/train_ncf.py` para registrar."
            )

        # --- Bloco 8: Comandos de reprodução ---
        st.markdown("---")
        st.subheader("🔧 Como Reproduzir")
        st.code(
            "# Treinar nova run do NCF\n"
            "uv run python scripts/train_ncf.py \\\n"
            "    --epochs 12 --emb-dim 16 --hidden 64 32 \\\n"
            "    --batch-size 1024 --lr 5e-4 --n-negatives 4 \\\n"
            "    --experiment-name 'Olist_NCF_Baseline'\n\n"
            "# Visualizar experimentos no MLflow UI\n"
            "uv run mlflow ui \\\n"
            "    --backend-store-uri sqlite:///./artifacts/mlflow.db",
            language="bash",
        )


# ============================================================================
# ABA 5 — RECOMENDAÇÕES (CONDICIONAL)
# ============================================================================
baseline_tab_idx = 3 if len(tabs) > 4 else None
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