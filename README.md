# 🛒 Olist Recommender System

![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3+-f7931e)
![MLflow](https://img.shields.io/badge/MLflow-2.5+-0194e2)
![DVC](https://img.shields.io/badge/DVC-3.0+-945dd6)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)

> Sistema de recomendação de produtos para e-commerce baseado no dataset público **Olist Brazilian E-Commerce**, implementando pipeline MLOps completo de treinamento, tracking, versionamento e deploy.

**Tech Challenge Fase 02** — Pós-graduação em Machine Learning Engineering

---

## 📋 Sobre o Projeto

Este projeto implementa um sistema de recomendação end-to-end utilizando o dataset público da Olist (~100k pedidos, 2016-2018). O objetivo é comparar modelos baseline (Scikit-Learn) com uma rede neural de embeddings (PyTorch), seguindo práticas modernas de MLOps.

### 🎯 Objetivos Técnicos
- [x] ≥ 10.000 interações user-item (99.785 alcançadas)
- [x] Pipeline de dados com DVC (3 estágios: prepare → featurize → validate)
- [x] Feature Engineering (10 → 42 features)
- [x] 3 Baselines implementados: Popularidade, Top-Rated, Item-Item CF
- [x] Métricas reais de ranking (MAP@K, NDCG@K, Recall@K, Precision@K, Hit Rate@K)
- [x] Split temporal para avaliação (70/15/15)
- [x] **Modelo NCF com PyTorch + BPR Loss** (Etapa 4 completa)
- [x] **Tracking MLflow com 6 runs** (3 experimentos: Baseline, Optimization, Ablation)
- [x] **Modelo registrado no MLflow Model Registry — Production stage**
- [ ] Dockerfile multi-stage
- [ ] Deploy em cloud
- [ ] Model Card
- [ ] Vídeo STAR (5 min)

---

## 👥 Equipe e Responsabilidades

| Membro | Responsabilidade |
|---|---|
| **Fábio Polli** | Arquitetura, DVC, baseline Item-Item CF |
| **Bill** | DevOps, dashboard Streamlit, CI/CD, **NCF PyTorch + BPR Loss** |
| **Denis & Romário** | Pipeline de dados, feature engineering |

---

## 🏗️ Arquitetura do Projeto

```
pos-ml-eng-tech-challenge-fase-02/
├── src/
│   ├── data/                     # Módulo de dados (PyTorch)
│   │   ├── dataset.py            # ImplicitFeedbackDataset
│   │   ├── splits.py             # temporal_split()
│   │   └── preprocessing.py      # fit_scaler(), transform_features()
│   ├── models/                   # Módulo de modelos
│   │   ├── ncf.py                # NCFHybrid (PyTorch)
│   │   └── losses.py             # BPR Loss
│   ├── training/                 # Módulo de treino
│   │   ├── train.py              # Loop de treinamento + MLflow
│   │   └── evaluate.py           # Métricas @K (NDCG, MAP, HitRate, Recall, Precision)
│   ├── data_preparation.py       # Stage 1: ETL → interactions.parquet
│   ├── feature_engineering.py    # Stage 2: FE → interactions_fe.parquet
│   └── eda.py                    # Análise exploratória
├── data/
│   ├── raw/                      # CSVs originais (versionados via DVC)
│   └── processed/                # Artefatos do pipeline
│       ├── interactions.parquet  # 99.785 × 10 (user-item)
│       ├── interactions_fe.parquet # 99.785 × 42 (com features)
│       ├── id_mappings.json      # Mapeamentos para embeddings
│       └── feature_metadata.json # Metadados das features
├── notebooks/
│   ├── 00_pipeline_explanation.ipynb # Decisões narrativas
│   ├── 01_eda.ipynb              # EDA sobre interactions.parquet
│   ├── 02_eda_feature_engineered.ipynb # EDA sobre interactions_fe.parquet
│   ├── 03_baseline_training.ipynb # Treinamento e resultados dos baselines
│   └── 04_ncf_training_results.ipynb # Resultados do NCF + otimização
├── front/
│   └── app_vis.py                # Dashboard Streamlit (6 abas, dark mode)
├── scripts/
│   ├── train_ncf.py              # Script de treino NCF + MLflow
│   ├── feature_selection.py      # Análise de correlação + MI
│   ├── generate_ncf_figures.py   # Gera figuras do notebook
│   ├── generate_optimization_figure.py # Figuras de comparação de runs
│   └── check_dvc_status.py       # Status checker do pipeline DVC
├── reports/
│   ├── eda_report.md             # Relatório de EDA
│   ├── ncf_optimization_report.md # Relatório de otimização (Etapa 4)
│   ├── feature_selection_report.md # Análise de feature selection
│   └── figures/                  # 20+ visualizações do EDA + NCF
├── artifacts/                    # Artefatos do modelo
│   ├── ncf_final.pt              # Modelo NCF Production (16.1 MB)
│   ├── metrics_*.json            # Métricas de cada run
│   ├── scaler.pkl                # StandardScaler ajustado
│   └── mlflow.db                 # Tracking SQLite do MLflow
├── configs/
│   ├── selected_features.yaml    # 20 features numéricas + 3 embedding
│   └── ncf_best.yaml             # Hiperparâmetros do modelo Production
├── tests/
│   └── test_import.py            # Teste de imports
├── docs/
│   ├── GUIDE.md                  # Guia técnico de modelagem
│   ├── REPORT.md                 # Relatório de progresso detalhado
│   ├── NCF_IMPLEMENTATION_PLAN.md # Plano de implementação NCF
│   ├── CHECKLIST.md              # Checklist de entregas
│   └── DVC_REMOTE_SETUP.md       # Configuração DVC
├── CHECKLIST.md                  # Checklist completo do projeto
├── pyproject.toml                # Dependências (uv/Poetry)
├── dvc.yaml                      # Pipeline DVC (3 estágios)
└── README.md
```

---

## 🏗️ Pipeline DVC

> **Importante:** Todos os comandos DVC devem ser executados com `uv run` para garantir que as dependências do projeto sejam carregadas corretamente.

```bash
# Executar pipeline completo
uv run dvc repro

# Ou estágios individuais
uv run dvc repro --single-stage prepare    # data_preparation.py
uv run dvc repro --single-stage featurize # feature_engineering.py
uv run dvc repro --single-stage validate  # Validação de shape

# Outros comandos úteis
uv run dvc pull      # Baixar dados versionados
uv run dvc status    # Verificar status do pipeline
uv run dvc dag       # Visualizar grafo do pipeline
```

### Estágios

| Stage | Script | Saída | Shape |
|-------|--------|-------|-------|
| `prepare` | `data_preparation.py` | `interactions.parquet` | 99.785 × 10 |
| `featurize` | `feature_engineering.py` | `interactions_fe.parquet` | 99.785 × 42 |
| `validate` | inline | shape check | (99785, 42) |

---

## ⏳ Por que Split Temporal (e não Aleatório)?

O pipeline de recomendação adota **split temporal 70/15/15** (ordenado por `order_purchase_timestamp` → `days_since_reference`) em vez do split aleatório clássico de ML. Esta decisão é **deliberada** e fundamentada em três razões:

### 1. Data leakage em split aleatório
Split aleatório mistura passado e futuro nos conjuntos de treino e teste. Como em e-commerce há forte sinal temporal (sazonalidade, lançamentos, tendência de categorias), isso **superestima a performance real** do modelo. Um item comprado em 2018 não deveria estar no treino se foi comprado em 2017.

### 2. Simulação realista do cenário de produção
Em produção, o modelo é treinado com dados **até o momento atual** e recomenda pedidos **futuros**. O split temporal replica exatamente esse cenário:
- **Treino:** 70% mais antigo (`days_since_reference` baixos)
- **Validação:** 15% intermediário (early stopping, hyperparameter tuning)
- **Teste:** 15% mais recente (métrica final, simulando produção)

### 3. Sazonalidade e evolução de preferências
- Categorias como **beleza/saúde** e **brinquedos** têm picos sazonais
- Preferências de usuários mudam com o tempo (tendências, modas)
- Novos produtos entram no catálogo após o treino — modelo deve ser avaliado sem conhecê-los

### Implementação
A função `temporal_split()` em [`src/data/splits.py`](src/data/splits.py) implementa a divisão com **assertions anti-leakage**:

```python
assert train_max <= val_min, "Vazamento: treino avança sobre validação"
assert val_max <= test_min, "Vazamento: validação avança sobre teste"
```

As assertions falham imediatamente se a ordenação temporal for violada, garantindo que nenhum evento futuro contamine o treino.

### Justificativa técnica completa
Veja [`docs/GUIDE.md`](docs/GUIDE.md) §4 — "Split Temporal e Anti-Leakage" para análise técnica aprofundada (cálculo de leakage, comparação com split aleatório em toy dataset, e discussão de cold-start).

---

## 🚀 Quick Start

```bash
# 1. Clonar
git clone https://github.com/fabiopolli/pos-ml-eng-tech-challenge-fase-02.git
cd pos-ml-eng-tech-challenge-fase-02

# 2. Instalar dependências (uv)
uv sync

# 3. Pull DVC
uv run dvc pull

# 4. Executar pipeline
uv run python src/data_preparation.py
uv run python src/feature_engineering.py

# 5. Treinar baselines
uv run python src/train.py

# 6. Treinar NCF (PyTorch) — Modelo Production já disponível
uv run python scripts/train_ncf.py --epochs 15 --emb-dim 32 --hidden 64 32 \
    --dropout 0.5 --lr 5e-4 --batch-size 2048 --n-negatives 8 \
    --use-scheduler --weight-decay 5e-4 \
    --run-name "Ablation_FINAL_no_aux_emb32" \
    --experiment-name "Olist_NCF_Optimization"

# 7. Dashboard (em outro terminal)
uv run streamlit run front/app_vis.py

# 8. MLflow UI (em outro terminal)
uv run mlflow ui --backend-store-uri sqlite:///./artifacts/mlflow.db
```

---

## 📊 Estatísticas do Dataset

| Métrica | Valor |
|---|---|
| Interações | 99.785 |
| Usuários únicos | 93.358 |
| Produtos únicos | 32.216 |
| Categorias | 72 |
| Sparsity | 99,9967% |
| Features (pós-FE) | 42 |
| Período | 2016-09-15 → 2018-08-29 |

### Features Geradas (42 total)

| Categoria | Qtd | Exemplos |
|-----------|-----|----------|
| Identificadores | 5 | customer_unique_id, user_id, product_id_idx |
| Target/Sinal | 3 | review_score, has_review, purchase_count |
| Numéricas | 6 | price_log, price_to_freight_ratio, has_price_outlier |
| Temporais | 8 | purchase_month, is_weekend, days_since_reference |
| Categóricas Encodadas | 8 | category_target_enc, payment_type_* (OHE) |
| Agregações Usuário | 6 | user_total_purchases, user_recency_days |
| Agregações Produto | 6 | product_popularity, product_avg_review_score |

---

## 📦 Stack Tecnológica

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.10-3.12 |
| Gerenciador | uv + Poetry |
| Deep Learning | PyTorch ≥ 2.0 |
| ML Clássico | Scikit-Learn ≥ 1.3 |
| Dados | Pandas, NumPy, PyArrow |
| Tracking | MLflow ≥ 2.5 |
| Versionamento | DVC ≥ 3.0 |
| Dashboard | Streamlit ≥ 1.0 |
| Visualização | Plotly, Matplotlib, Seaborn |
| Logging | Loguru ≥ 0.7 |
| Lint | Ruff ≥ 0.3 |
| Testes | Pytest |

---

## 📚 Documentação

| Arquivo | Descrição |
|---------|-----------|
| [`docs/GUIDE.md`](docs/GUIDE.md) | Guia técnico de modelagem (NCF, split temporal, negative sampling) |
| [`docs/ML_EXECUTION_GUIDE.md`](docs/ML_EXECUTION_GUIDE.md) | Passo-a-passo para executar o pipeline |
| [`docs/REPORT.md`](docs/REPORT.md) | Relatório detalhado de progresso |
| [`CHECKLIST.md`](CHECKLIST.md) | Checklist completo com status de cada entrega |
| [`notebooks/00_pipeline_explanation.ipynb`](notebooks/00_pipeline_explanation.ipynb) | Explicação narrativa das decisões (para apresentação) |
| [`front/app_vis.py`](front/app_vis.py) | Dashboard Streamlit (mesmo padrão do Fase 01) |

---

## 🎯 Baselines Implementados

| Baseline | Descrição | Script |
|---------|-----------|--------|
| **Popularidade** | Produtos mais interagidos | `train_popularity_baseline()` |
| **Top-Rated** | Produtos com melhor nota média | `train_top_rated_baseline()` |
| **Item-Item CF** | Similaridade cosseno item-item | `train_item_similarity_baseline()` |
| **TruncatedSVD** | Próximo a implementar | - |

---

## 📊 Dashboard Streamlit

O dashboard [`front/app_vis.py`](front/app_vis.py) segue o mesmo padrão visual aprovado no Fase 01 (dark mode premium).

```bash
uv run streamlit run front/app_vis.py
```

**Abas (6 total):**
1. **📊 Visão Geral** — KPIs, distribuições, temporal, sparsity
2. **🔧 Feature Engineering** — Mapa de features, correlações
3. **🏋️ Baselines** — Métricas dos baselines clássicos
4. **🧠 NCF (MLP PyTorch)** — Resultados do modelo neural, hiperparâmetros, MLflow runs ⭐
5. **🎯 Recomendações** — Top-N dos baselines (quando disponíveis)
6. **ℹ️ Sobre o Pipeline** — Arquitetura e decisões

---

## 🤝 Padrões de Código

```bash
# Lint
uv run ruff check .

# Formatar
uv run ruff format .

# Testes
uv run pytest

# Pre-commit
uv run pre-commit run --all-files
```

---

## 📄 Licença

MIT License

---

**Contato**: Grupo Tech Challenge Fase 02 — Pós-graduação ML Engineering
