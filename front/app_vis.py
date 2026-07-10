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
BASELINE_RESULTS_PATH = BASE_DIR / "data" / "processed" / "baseline_results.csv"
TRAIN_SPLIT_PATH = BASE_DIR / "data" / "processed" / "train_split.parquet"
TEST_SPLIT_PATH = BASE_DIR / "data" / "processed" / "test_split.parquet"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FEATURES_DOC_PATH = BASE_DIR / "data" / "processed" / "FEATURES.md"
FEATURE_METADATA_PATH = BASE_DIR / "data" / "processed" / "feature_metadata.json"

# --- Caminhos dos artefatos do NCF (PyTorch) ---
# Prioridade: models/ (versionado via DVC) > artifacts/ (legado).
# Rota A (versao 6.5 do REPORT.md) moveu o modelo Production para models/.
NCF_MODEL_PRIMARY = BASE_DIR / "models" / "ncf_production.pt"  # Rota A (DVC)
NCF_MODEL_FALLBACK = BASE_DIR / "artifacts" / "ncf_Ablation_FINAL_no_aux_emb32.pt"
NCF_MODEL_FALLBACK_2 = BASE_DIR / "artifacts" / "ncf_final.pt"
NCF_MODEL_PATH = next(
    (p for p in [NCF_MODEL_PRIMARY, NCF_MODEL_FALLBACK, NCF_MODEL_FALLBACK_2] if p.exists()),
    NCF_MODEL_PRIMARY,
)
NCF_METRICS_PRIMARY = BASE_DIR / "artifacts" / "metrics_Ablation_FINAL_no_aux_emb32.json"
NCF_METRICS_FALLBACK = BASE_DIR / "artifacts" / "metrics_NCF_FINAL_emb32_h64-32_d0.5_lr5e-4.json"
NCF_METRICS_PATH = NCF_METRICS_PRIMARY if NCF_METRICS_PRIMARY.exists() else NCF_METRICS_FALLBACK
NCF_SCALER_PRIMARY = BASE_DIR / "models" / "scaler_production.pkl"  # Rota A (DVC)
NCF_SCALER_FALLBACK = BASE_DIR / "artifacts" / "scaler.pkl"
NCF_SCALER_PATH = NCF_SCALER_PRIMARY if NCF_SCALER_PRIMARY.exists() else NCF_SCALER_FALLBACK
NCF_CONFIG_PATH = BASE_DIR / "configs" / "selected_features.yaml"
NCF_BEST_CONFIG_PATH = BASE_DIR / "configs" / "ncf_best.yaml"
NCF_MLFLOW_DB = BASE_DIR / "artifacts" / "mlflow.db"

# --- URL canonica do Model Registry no DagsHub (Rota B - secao 16 do REPORT.md) ---
DAGSHUB_MLFLOW_URL = (
    "https://dagshub.com/deniscelclaro/projeto_fiap_modulo2/#/models/olist-ncf-recommender"
)
DAGSHUB_REPO_URL = "https://dagshub.com/deniscelclaro/projeto_fiap_modulo2"
PRODUCTION_MODEL_HASH = "439244cc81273d4bbc0bfa710a9142ee"  # MD5 do ncf_production.pt
NCF_FIG_BASELINE = FIGURES_DIR / "ncf_vs_baseline.png"
NCF_FIG_TVT = FIGURES_DIR / "ncf_train_val_test.png"
NCF_FIG_COLDSTART = FIGURES_DIR / "coldstart_analysis.png"
NCF_FIG_EMBED = FIGURES_DIR / "embedding_norms.png"
NCF_FIG_OPTIMIZATION = FIGURES_DIR / "ncf_optimization_comparison.png"

# --- Diretório de recomendacoes geradas pelos baselines ---
BASELINES_RECS_DIR = BASE_DIR / "artifacts" / "baselines"

# --- Configuração da página ---
st.set_page_config(
    page_title="Olist Recommender - Dashboard Analítico",
    layout="wide",
    page_icon="🛒",
)

# --- Sidebar: Modo Apresentacao + links uteis ---
with st.sidebar:
    st.markdown("### 🎛️ Controles")
    presentation_mode = st.toggle(
        "🎤 Modo Apresentação",
        value=False,
        help="Ativa um layout mais limpo para demos: esconde abas técnicas e aumenta KPIs.",
    )
    st.markdown("---")
    st.markdown("### 🔗 Links Úteis")
    st.markdown(f"[📦 Repositório GitHub](https://github.com/fabiopolli/pos-ml-eng-tech-challenge-fase-02)")
    st.markdown(f"[🏔️ DagsHub Repo]({DAGSHUB_REPO_URL})")
    st.markdown(f"[🏆 Modelo Production (MLflow)]({DAGSHUB_MLFLOW_URL})")
    st.markdown(f"[📖 REPORT.md](https://github.com/fabiopolli/pos-ml-eng-tech-challenge-fase-02/blob/main/docs/REPORT.md)")
    st.markdown("---")
    st.caption(f"🔖 Modelo MD5: `{PRODUCTION_MODEL_HASH[:8]}...`")

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
    """Carrega e empilha as recomendacoes geradas pelos baselines.

    Cada baseline gera arquivos CSV em artifacts/baselines/ no formato
    recommendations_<Model>_<Variant>_K<N>.csv. Cada arquivo tem:
      - Linha 1: header com 'top_<N>'
      - Linhas seguintes: product_id (UUID string) para cada rank
    """
    if not BASELINES_RECS_DIR.exists():
        return None

    files = sorted(BASELINES_RECS_DIR.glob("recommendations_*.csv"))
    if not files:
        return None

    frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            # Parse do nome do arquivo: recommendations_<Model>_<Variant>_K<N>.csv
            stem = fp.stem
            parts = stem.split("_")
            k_str = parts[-1]  # K10
            k = int(k_str.replace("K", ""))
            model_class = parts[1] if len(parts) > 1 else "Unknown"
            variant = "_".join(parts[2:-1]) if len(parts) > 3 else ""

            # Header='top_10' — pula a primeira linha
            df = pd.read_csv(fp, skiprows=1, header=None)
            df.columns = [f"rank_{i+1}" for i in range(df.shape[1])]
            df.insert(0, "rank0_user", df.index)
            df["baseline_model"] = model_class
            df["variant"] = variant
            df["k"] = k
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


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
# Aba de Versionamento (sempre presente - referencia DagsHub)
tab_names.append("🔖 Versionamento")
# No modo apresentacao, escondemos a aba "Sobre o Pipeline" (detalhes tecnicos)
if not presentation_mode:
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
        # KPIs principais (above the fold para apresentacao)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Interações", f"{len(df):,}")
        col2.metric("Usuários Únicos", f"{df['customer_unique_id'].nunique():,}")
        col3.metric("Produtos Únicos", f"{df['product_id'].nunique():,}")
        col4.metric("Categorias", f"{df['product_category_name_english'].nunique() if 'product_category_name_english' in df.columns else 0:,}")

        # --- Bloco "Sparsity da Matriz User-Item" (alta densidade visual) ---
        # Este bloco e o "momento dramatico" da apresentacao (98% cold-start).
        st.markdown("---")
        st.subheader("⚠️ Sparsity da Matriz User-Item (o vilão do projeto)")

        n_users_unique = df['customer_unique_id'].nunique()
        n_products_unique = df['product_id'].nunique()
        # Compras por cliente (para detectar cold-start)
        user_purchase_counts = df.groupby('customer_unique_id').size()
        n_one_buy = (user_purchase_counts == 1).sum()
        n_two_plus = (user_purchase_counts >= 2).sum()
        pct_one_buy = n_one_buy / n_users_unique * 100
        pct_two_plus = n_two_plus / n_users_unique * 100

        # KPIs: 3 numeros grandes (sparsity, clientes 1 compra, clientes >=2 compras)
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "🔴 Sparsity da Matriz",
            "99,9967%",
            delta=f"3 bi possíves · {len(df):,} observados",
            delta_color="off",
        )
        k2.metric(
            "🔴 Cold-start severo",
            f"{pct_one_buy:.1f}%",
            delta=f"{n_one_buy:,} clientes com 1 compra",
            delta_color="off",
        )
        k3.metric(
            "🟢 Clientes com histórico",
            f"{pct_two_plus:.1f}%",
            delta=f"{n_two_plus:,} clientes com ≥2 compras",
        )

        # --- Grafico-donut: 98% vs 2% (visual forte para a pausa dramatica) ---
        d1, d2 = st.columns([1, 1])
        with d1:
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["1 compra (cold-start)", "≥2 compras (com histórico)"],
                        values=[n_one_buy, n_two_plus],
                        hole=0.55,
                        marker=dict(colors=["#f85149", "#3fb950"]),
                        textinfo="label+percent",
                        textfont=dict(size=14, color="#e6edf3"),
                    )
                ]
            )
            fig.update_layout(
                title=dict(
                    text="<b>98% cold-start</b><br><sub>Apenas 2 mil clientes têm ≥2 compras</sub>",
                    x=0.5,
                    font=dict(color="#e6edf3", size=16),
                ),
                **plotly_template(),
            )
            fig.add_annotation(
                text=f"<b>{pct_two_plus:.1f}%</b>",
                x=0.5, y=0.5,
                font=dict(size=32, color="#3fb950"),
                showarrow=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        with d2:
            st.markdown("### 💡 O que isso significa?")
            st.markdown(
                f"""
                - **{pct_two_plus:.1f}%** dos clientes (~**{n_two_plus:,}**) têm
                  **histórico relevante** para alimentar features engineered.
                - **{pct_one_buy:.1f}%** dos clientes (~**{n_one_buy:,}**) compraram
                  **UMA ÚNICA VEZ** — não há passado para usar.
                - Qualquer feature dependente de histórico (gasto médio,
                  categorias preferidas) está **vazia** para a maioria.
                - Isso **forçou** 3 decisões de design no projeto (ver aba
                  *Sobre o Pipeline*).
                """
            )

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
        # Preferir IDs inteiros de interactions_fe (mais preciso)
        if df_fe is not None and "user_id" in df_fe.columns and "product_id_idx" in df_fe.columns:
            n_users = df_fe["user_id"].nunique()
            n_items = df_fe["product_id_idx"].nunique()
        elif "customer_unique_id" in df.columns and "product_id" in df.columns:
            n_users = df["customer_unique_id"].nunique()
            n_items = df["product_id"].nunique()
        else:
            n_users = n_items = n_interactions = 0
            total = 1
        n_interactions = len(df) if df is not None else len(df_fe)
        total = n_users * n_items
        sparsity = 1 - (n_interactions / total) if total > 0 else 0

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

        # Nota visual sobre o cold-start severo (motivacao dos baselines zerados)
        st.error(
            """
            ⚠️ **Por que vários baselines retornaram MAP = 0.0000?**\n
            **Cold-start severo** do dataset: 98% dos usuários do conjunto de teste nunca
            aparecem no treino (ver aba *Visão Geral* → *Sparsity da Matriz User-Item*).
            Métodos baseados em **co-ocorrência** (Item-Item CF, TopRated, TruncatedSVD)
            não conseguem gerar ranking personalizado para clientes inéditos. **Apenas
            Popularidade** (não-personalizada) gera recomendações válidas, com NDCG@10
            de **0.0045** — 60× abaixo do nosso NCF Production.
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
            r"""
            ### ✅ Status dos Entregáveis

            | Etapa | Status |
            |-------|--------|
            | Pipeline de dados (DVC) | ✅ Concluído |
            | Feature Engineering (10→42) | ✅ Concluído |
            | Baselines (Popularity, TopRated, Item-Item CF, TruncatedSVD) | ✅ Concluído |
            | Métricas reais (HitRate, Recall, Precision, NDCG, MAP) | ✅ Concluído |
            | Modelo NCF (PyTorch + BPR) | ✅ Concluído |
            | MLflow Tracking (SQLite local) | ✅ Concluído |
            | Ablation Study (no\_aux venceu) | ✅ Concluído |
            | Modelo em Production (`Ablation_FINAL_no_aux_emb32`) | ✅ Concluído |
            | Dashboard Streamlit (`front/app_vis.py`) | ✅ Concluído |
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

        # Carregar Production config (fonte canonica do modelo em prod)
        prod_hp: dict = {}
        if NCF_BEST_CONFIG_PATH.exists():
            with open(NCF_BEST_CONFIG_PATH) as _hp:
                prod_hp = yaml.safe_load(_hp) or {}

        # Carregar MLflow tracking (opcional) — busca o melhor run em todos os experimentos
        ncf_mlflow_runs = []
        try:
            import mlflow as _mlflow

            _mlflow.set_tracking_uri(f"sqlite:///{NCF_MLFLOW_DB}")
            _client = _mlflow.tracking.MlflowClient()
            _exp_ids = [
                str(e.experiment_id)
                for e in _client.search_experiments()
                if e.name != "Default"
            ]
            ncf_mlflow_runs = _client.search_runs(
                experiment_ids=_exp_ids,
                filter_string="status = 'FINISHED'",
                order_by=["metrics.test_NDCG_at_K DESC"],
                max_results=10,
            )
        except Exception:
            pass

        # --- Bloco 1: Banner hero do Production Model ---
        # Numero "uau" da apresentacao: NDCG@10 = 60x vs baseline.
        baseline_ndcg = ncf_metrics.get("baseline", {}).get("NDCG@K", 0.0045)
        ncf_ndcg = ncf_metrics['test']['NDCG@K']
        lift_x = ncf_ndcg / baseline_ndcg if baseline_ndcg > 0 else 60.6

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
                border: 2px solid #58a6ff;
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 16px;
                text-align: center;
            ">
                <div style="font-size: 0.9em; color: #7d8590; letter-spacing: 2px;
                            text-transform: uppercase; margin-bottom: 8px;">
                    🏆 Production Model (DagsHub MLflow Registry)
                </div>
                <div style="font-size: 3.5em; font-weight: 800; color: #58a6ff;
                            line-height: 1; margin-bottom: 8px;">
                    {ncf_ndcg:.4f}
                </div>
                <div style="font-size: 1em; color: #7d8590; margin-bottom: 12px;">
                    NDCG@10 · Test Set
                </div>
                <div style="display: inline-block; background: #3fb9501a; border: 1px solid #3fb950;
                            border-radius: 20px; padding: 6px 16px; color: #3fb950; font-weight: 700;
                            font-size: 1.1em;">
                    ↑ {lift_x:.1f}x vs baseline (Popularidade = {baseline_ndcg:.4f})
                </div>
                <div style="font-size: 0.75em; color: #7d8590; margin-top: 12px;">
                    <a href="{DAGSHUB_MLFLOW_URL}" target="_blank"
                       style="color: #58a6ff; text-decoration: none;">
                       Ver no DagsHub MLflow →
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Bloco 1b: KPIs com delta visual ---
        st.subheader("📊 Métricas @ K=10 (Test Set)")
        col1, col2, col3, col4, col5 = st.columns(5)
        baseline_hr = ncf_metrics.get("baseline", {}).get("HitRate@K", 0.0113)
        baseline_rec = ncf_metrics.get("baseline", {}).get("Recall@K", 0.0104)
        baseline_prec = ncf_metrics.get("baseline", {}).get("Precision@K", 0.0010)
        baseline_map = ncf_metrics.get("baseline", {}).get("MAP@K", 0.0019)
        col1.metric(
            "HitRate@10",
            f"{ncf_metrics['test']['HitRate@K']:.4f}",
            delta=f"+{(ncf_metrics['test']['HitRate@K'] / baseline_hr - 1) * 100:.0f}% vs baseline",
        )
        col2.metric(
            "Recall@10",
            f"{ncf_metrics['test']['Recall@K']:.4f}",
            delta=f"+{(ncf_metrics['test']['Recall@K'] / baseline_rec - 1) * 100:.0f}% vs baseline",
        )
        col3.metric(
            "Precision@10",
            f"{ncf_metrics['test']['Precision@K']:.4f}",
            delta=f"+{(ncf_metrics['test']['Precision@K'] / baseline_prec - 1) * 100:.0f}% vs baseline",
        )
        col4.metric(
            "NDCG@10",
            f"{ncf_metrics['test']['NDCG@K']:.4f}",
            delta=f"+{lift_x:.1f}x vs baseline",
        )
        col5.metric(
            "MAP@10",
            f"{ncf_metrics['test']['MAP@K']:.4f}",
            delta=f"+{(ncf_metrics['test']['MAP@K'] / baseline_map - 1) * 100:.0f}% vs baseline",
        )

        # Indicador discreto do hash do modelo + origem
        st.caption(
            f"🔖 Modelo em Production · MD5 `{PRODUCTION_MODEL_HASH}` · "
            f"📦 Repositório DagsHub: [{DAGSHUB_REPO_URL}]({DAGSHUB_REPO_URL})"
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
            "🧪 **Achado da Ablation**: O modelo **no_aux** (apenas embeddings) superou "
            "todas as variantes com 20 side features. Decisão: Production = "
            "`Ablation_FINAL_no_aux_emb32` (NDCG@10 = 0.2725). "
            "Hipótese: com 98% cold-start, os embeddings aleatórios não contribuem, "
            "e as features normalizadas pelo scaler fitado no treino dominam o sinal "
            "de forma homogênea entre usuários — empurrando todos os scores para a "
            "mesma região e prejudicando o ranqueamento."
        )

        st.markdown("---")

        # --- Bloco 5: Embeddings ---
        st.subheader("🔮 Embeddings Aprendidos")
        if NCF_FIG_EMBED.exists():
            st.image(str(NCF_FIG_EMBED), use_container_width=True)

        col_emb1, col_emb2, col_emb3 = st.columns(3)
        # Usar dim do config YAML do Production (32) — não do YAML base (que pode ter sido atualizado)
        prod_emb_dim = prod_hp.get("architecture", {}).get("embedding_dim", 32) if prod_hp else 32
        prod_cat_dim = prod_hp.get("architecture", {}).get("category_embedding_dim", 8) if prod_hp else 8
        col_emb1.metric(
            "User Embeddings",
            f"{ncf_config['n_users']:,}",
            help=f"Dimensão {prod_emb_dim} — {ncf_config['n_users']:,} usuários",
        )
        col_emb2.metric(
            "Item Embeddings",
            f"{ncf_config['n_items']:,}",
            help=f"Dimensão {prod_emb_dim} — {ncf_config['n_items']:,} produtos",
        )
        col_emb3.metric(
            "Category Embeddings",
            f"{ncf_config['n_categories']:,}",
            help=f"Dimensão {prod_cat_dim} — {ncf_config['n_categories']:,} categorias",
        )

        st.markdown("---")

        # --- Bloco 6: Hiperparâmetros ---
        st.subheader("⚙️ Hiperparâmetros do Modelo Production (configs/ncf_best.yaml)")

        arch = prod_hp.get("architecture", {}) if prod_hp else {}
        train_cfg = prod_hp.get("training", {}) if prod_hp else {}

        ncf_hp = {
            "Embedding dim (user/item)": str(arch.get("embedding_dim", "32")),
            "Embedding dim (category)": str(arch.get("category_embedding_dim", "8")),
            "MLP hidden layers": str(arch.get("hidden_layers", [64, 32])),
            "Dropout": str(arch.get("dropout", 0.5)),
            "Optimizer": str(train_cfg.get("optimizer", "adamw")),
            "Learning rate": str(train_cfg.get("learning_rate", 5e-4)),
            "Weight decay": str(train_cfg.get("weight_decay", 5e-4)),
            "Batch size": str(train_cfg.get("batch_size", 2048)),
            "N negatives (treino)": str(train_cfg.get("negative_samples_per_positive", 8)),
            "Loss function": str(train_cfg.get("loss", "BPR")),
            "Epochs (max com early stop)": str(train_cfg.get("epochs_max", 20)),
            "Patience": str(train_cfg.get("early_stopping_patience", 3)),
            "Gradient clipping": str(train_cfg.get("gradient_clipping_max_norm", 5.0)),
            "Scheduler": f"{train_cfg.get('scheduler', 'ReduceLROnPlateau')}(factor={train_cfg.get('scheduler_factor', 0.5)}, patience={train_cfg.get('scheduler_patience', 2)})",
            "Side features (ablation)": "❌ DESABILITADO (modelo `no_aux`)",
            "Scaler": "N/A (sem features auxiliares)",
            "Negative sampling": "on-the-fly",
            "MLflow experiment": str(prod_hp.get("artifacts", {}).get("mlflow_experiment", "Olist_NCF_Ablation")),
            "MLflow run": str(prod_hp.get("artifacts", {}).get("mlflow_best_run_name", "Ablation_FINAL_no_aux_emb32")),
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
            "# Treinar nova run do NCF (Production)\n"
            "uv run python src/train.py \\\n"
            "    --epochs 20 --emb-dim 32 --hidden 64 32 \\\n"
            "    --batch-size 2048 --lr 5e-4 --n-negatives 8 \\\n"
            "    --dropout 0.5 --no-aux\n\n"
            "# Gerar figuras do relatorio\n"
            "uv run python scripts/generate_ncf_figures.py\n\n"
            "# Visualizar experimentos no MLflow UI (SQLite local)\n"
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
        st.header("🎯 Recomendações Geradas pelos Baselines")

        if df_baseline is None or df_baseline.empty:
            st.warning(
                "⚠️ Artefatos de recomendacoes nao encontrados em `artifacts/baselines/`. "
                "Execute `uv run python src/train.py` para gerar as recomendacoes dos baselines."
            )
        else:
            # Carrega nome das categorias para enriquecer a tabela
            category_map: dict = {}
            product_id_map: dict = {}
            if df_fe is not None and "product_id_idx" in df_fe.columns:
                # product_id_idx -> category_id (ja encoded)
                cat_lookup = (
                    df_fe.drop_duplicates("product_id_idx")
                    .set_index("product_id_idx")[["category_id", "product_id"]]
                )
                product_id_map = cat_lookup["product_id"].to_dict()
            if df is not None and "product_id" in df.columns and "product_category_name_english" in df.columns:
                category_map = (
                    df.drop_duplicates("product_id")
                    .set_index("product_id")["product_category_name_english"]
                    .to_dict()
                )

            col1, col2, col3 = st.columns(3)
            col1.metric("Arquivos de Recomendacao", f"{len(df_baseline['baseline_model'].unique())} baselines")
            col2.metric("Top-N medio", f"{int(df_baseline['k'].mean())}")
            col3.metric("Variantes geradas", f"{df_baseline.groupby(['baseline_model', 'k']).ngroups}")

            st.markdown("---")
            st.subheader("📋 Inventario de Artefatos em `artifacts/baselines/`")

            inventory = (
                df_baseline.groupby(["baseline_model", "variant", "k"])
                .size()
                .reset_index(name="linhas")
            )
            st.dataframe(inventory, use_container_width=True, hide_index=True)

            # Para um baseline especifico, mostrar top-K como exemplo
            st.markdown("---")
            st.subheader("🔍 Exemplo: Top-5 Recomendacoes (Popularidade K=10)")
            pop_k10 = df_baseline[(df_baseline["baseline_model"] == "PopularityBaseline") & (df_baseline["k"] == 10)]
            if not pop_k10.empty:
                # Pegar primeiros 5 usuarios
                sample = pop_k10.head(5).copy()
                # rank_<r> contem product_id (UUID string), mapear para categoria
                display_rows = []
                for _, row in sample.iterrows():
                    recs = []
                    for r in range(1, 6):
                        pid = row.get(f"rank_{r}")
                        if pd.isna(pid):
                            continue
                        cat = category_map.get(pid, "?")
                        recs.append(f"#{r}: {str(pid)[:12]}… ({cat})")
                    display_rows.append({"user_idx": row["rank0_user"], "Top-5": " | ".join(recs)})
                st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)


# ============================================================================
# ABA 6 — VERSIONAMENTO (DagsHub + DVC + MLflow)
# ============================================================================
versioning_tab_idx = (
    3 if _ncf_available else 2 + (1 if df_baseline is not None else 0)
) + (0 if _ncf_available else 0)
# Simplifica: pega a aba imediatamente antes da final
final_tab_idx = len(tabs) - 1
versioning_tab_idx = final_tab_idx - (0 if presentation_mode else 1)

with tabs[versioning_tab_idx]:
    st.header("🔖 Versionamento de Modelos (GitHub + DVC + MLflow + DagsHub)")

    st.markdown(
        """
        Este painel é **reproduzível bit a bit** graças à orquestração entre
        **quatro ferramentas**. Cada uma resolve um problema específico e, juntas,
        entregam **reprodutibilidade científica + governança de modelos**.
        """
    )

    # Card grande: Production Model com link DagsHub
    st.markdown(
        f"""
        <div style="
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
        ">
            <div style="font-size: 0.85em; color: #7d8590; letter-spacing: 1.5px;
                        text-transform: uppercase;">
                🏆 Production Model
            </div>
            <div style="font-size: 1.5em; color: #e6edf3; font-weight: 700;
                        margin: 8px 0;">
                olist-ncf-recommender · v1
            </div>
            <div style="font-size: 0.85em; color: #7d8590; line-height: 1.6;">
                <div><b>MLflow Run:</b> <code>c65f5531564f45c583926c564985cc65</code></div>
                <div><b>MD5 (DVC):</b> <code>{PRODUCTION_MODEL_HASH}</code></div>
                <div><b>Origem:</b> <code>models/ncf_production.pt</code> + <code>models/scaler_production.pkl</code></div>
            </div>
            <div style="margin-top: 12px;">
                <a href="{DAGSHUB_MLFLOW_URL}" target="_blank"
                   style="color: #58a6ff; text-decoration: none; font-weight: 700;">
                   📦 Ver Model Registry no DagsHub →
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabela das 4 ferramentas (cards)
    st.subheader("🧩 As 4 ferramentas (cada uma resolve um problema)")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown(
            """
            **📦 GitHub** — guarda **código**
            - Scripts, configs, documentação
            - Texto puro, revisão por humanos
            - Inclui os `*.dvc` (metadados pequenos)
            """
        )
        st.markdown(
            """
            **🔗 DVC** — extensão do Git para dados/modelos
            - Versiona binários grandes (CSVs, `.pt`)
            - Guarda apenas **hashes MD5** no Git
            - `dvc push` envia os binários para o DagsHub
            """
        )
    with fcol2:
        st.markdown(
            """
            **📋 MLflow** — cartório do modelo
            - Registra cada experimento (parâmetros, métricas)
            - Model Registry com **stages** (Staging / Production / Archived)
            - Hospedado no DagsHub (gratuito)
            """
        )
        st.markdown(
            f"""
            **🏔️ DagsHub** — plataforma unificada
            - Git + bucket S3 + MLflow Tracking num lugar só
            - [Repositório]({DAGSHUB_REPO_URL})
            - [Model Registry]({DAGSHUB_MLFLOW_URL})
            """
        )

    # Comando: como reproduzir
    st.markdown("---")
    st.subheader("🔧 Como reproduzir o estado exato deste painel")
    st.code(
        """# 1. Clonar o repositório
git clone https://github.com/fabiopolli/pos-ml-eng-tech-challenge-fase-02.git
cd pos-ml-eng-tech-challenge-fase-02

# 2. Configurar DagsHub (uma unica vez)
uv run dvc remote modify --local origin auth basic
uv run dvc remote modify --local origin user <seu_usuario>
uv run dvc remote modify --local origin password <seu_token>

# 3. Baixar dados + modelos
uv run dvc pull

# 4. Abrir o dashboard
uv run streamlit run front/app_vis.py""",
        language="bash",
    )

    # Status atual (resumo visual)
    st.markdown("---")
    st.subheader("✅ Status Atual do Versionamento (2026-07-10)")
    scol1, scol2 = st.columns(2)
    with scol1:
        st.success(
            f"""
            **🔗 DVC (Rota A)** ✅ Sincronizado
            - `models/ncf_production.pt` (16 MB) — MD5 `{PRODUCTION_MODEL_HASH[:12]}...`
            - `models/scaler_production.pkl` (1.7 KB)
            - Binarios no bucket S3 do DagsHub
            """
        )
    with scol2:
        st.success(
            """
            **📋 MLflow (Rota B)** ✅ Registrado
            - Modelo `olist-ncf-recommender` v1
            - Stage: **Production**
            - Run ID: `c65f5531564f45c583926c564985cc65`
            """
        )


# ============================================================================
# ABA FINAL — SOBRE O PIPELINE
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
          ↓ interactions_fe.parquet (42 colunas, 18 features numericas)
        audit (scripts/feature_selection.py)
          ↓ Spearman |rho| > 0.95 → remove 2 features redundantes
        ↓
        train (src/train.py)
          ↓ 4 baselines (Popularity, TopRated, ItemItemCF, TruncatedSVD)
          ↓ NCF Production (Ablation_FINAL_no_aux_emb32)
          ↓ MLflow tracking (SQLite local)
        ```

        ### 🧪 Decisões Arquiteturais
        - **Split temporal** (70/15/15) obrigatório para evitar data leakage.
        - **Target encoding** para categorias com Bayesian smoothing (alpha=10).
        - **BPR Loss** com negative sampling on-the-fly para feedback implícito.
        - **Ablation**: features auxiliares **prejudicam** o modelo com 98% cold-start.
        - **Factory Method** em `src/models/factory.py` para construir modelos.
        - **Strategy Pattern** em `src/data/strategies.py` para pré-processamento.
        - **Sparsity extrema**: 99,9967% — semi-cold-start (98% dos test users inéditos).

        ### 📊 Métricas Avaliadas
        - NDCG@K (principal para ranking)
        - MAP@K
        - Recall@K
        - HitRate@K
        - Precision@K

        ### 🔧 Como Reproduzir
        ```bash
        # Pipeline completo (DVC)
        uv run dvc repro

        # Treinar modelos
        uv run python src/train.py --baselines
        uv run python src/train.py --ncf --epochs 20 --emb-dim 32

        # MLflow UI local
        uv run mlflow ui --backend-store-uri sqlite:///./artifacts/mlflow.db

        # Dashboard
        uv run streamlit run front/app_vis.py
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

    # Footer com mini-cards de "proxima aba na demonstracao" (alinhado com o script)
    if not presentation_mode:
        st.markdown("---")
        st.markdown("### 🎯 Próximo passo na demonstração")
        st.markdown(
            """
            <style>
                .next-card {
                    display: inline-block;
                    background: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 8px;
                    padding: 10px 16px;
                    margin: 4px;
                    color: #58a6ff;
                    font-weight: 600;
                    font-size: 0.95em;
                }
            </style>
            <div style="text-align: center;">
                <span class="next-card">📊 Visão Geral</span>
                <span class="next-card">🔧 Feature Engineering</span>
                <span class="next-card">🏋️ Baselines</span>
                <span class="next-card">🧠 NCF (MLP PyTorch)</span>
                <span class="next-card">🎯 Recomendações</span>
                <span class="next-card">🔖 Versionamento</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.8rem;'>"
    "Painel Analítico de Recomendação v2.0 | Olist Tech Challenge Fase 02</div>",
    unsafe_allow_html=True,
)