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
- [ ] Modelo NCF com PyTorch + BPR Loss
- [ ] Métricas reais de ranking (NDCG@K, Recall@K, MAP@K, Hit Rate@K)
- [ ] Tracking MLflow com 3+ runs
- [ ] Dockerfile multi-stage
- [ ] Deploy em cloud
- [ ] Model Card
- [ ] Vídeo STAR (5 min)

---

## 👥 Equipe e Responsabilidades

| Membro | Responsabilidade |
|---|---|
| **Fábio Polli** | Arquitetura, DVC, baseline Item-Item CF |
| **Bill** | DevOps, dashboard Streamlit, CI/CD |
| **Denis & Romário** | Pipeline de dados, feature engineering |
| **TBD** | NCF com PyTorch + BPR Loss |

---

## 🏗️ Arquitetura do Projeto

```
pos-ml-eng-tech-challenge-fase-02/
├── src/
│   ├── data_preparation.py       # Stage 1: ETL (CSVs → interactions.parquet)
│   ├── feature_engineering.py    # Stage 2: FE (interactions → interactions_fe)
│   ├── train.py                  # Baselines + MLflow tracking
│   └── eda.py                    # Análise exploratória
├── data/
│   ├── raw/                      # CSVs originais (versionados via DVC)
│   └── processed/                # Artefatos do pipeline
│       ├── interactions.parquet  # 99.785 × 10 (user-item)
│       ├── interactions_fe.parquet # 99.785 × 42 (com features)
│       ├── id_mappings.json      # Mapeamentos para embeddings
│       └── feature_metadata.json # Metadados das features
├── notebooks/
│   ├── 00_pipeline_explanation.ipynb # Decisões narrativas (para apresentação)
│   ├── 01_eda.ipynb              # EDA sobre interactions.parquet
│   ├── 02_eda_feature_engineered.ipynb # EDA sobre interactions_fe.parquet
│   └── demo.ipynb                # Demonstração dos baselines no MLflow
├── front/
│   └── app_vis.py                # Dashboard Streamlit (dark mode premium)
├── scripts/
│   └── check_dvc_status.py       # Status checker do pipeline DVC
├── reports/
│   ├── eda_report.md             # Relatório de EDA
│   └── figures/                  # 20 visualizações do EDA
├── docs/
│   ├── GUIDE.md                  # Guia técnico de modelagem
│   ├── ML_EXECUTION_GUIDE.md     # Passo-a-passo de execução
│   ├── REPORT.md                 # Relatório de progresso
│   └── DVC_REMOTE_SETUP.md      # Configuração DVC
├── CHECKLIST.md                  # Checklist completo do projeto
├── tests/
│   └── test_import.py            # Teste de imports
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

# 6. Dashboard (em outro terminal)
uv run streamlit run front/app_vis.py

# 7. MLflow UI (em outro terminal)
mlflow ui --host 127.0.0.1 --port 5000
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

**Abas:**
1. **📊 Visão Geral** — KPIs, distribuições, temporal, sparsity
2. **🔧 Feature Engineering** — Mapa de features, correlações
3. **🎯 Recomendações** — Top-N dos baselines (quando disponíveis)
4. **ℹ️ Sobre o Pipeline** — Arquitetura e decisões

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
