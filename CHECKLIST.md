# CHECKLIST — Tech Challenge Fase 02: Sistema de Recomendação

> Checklist de execução derivado do PDF `Tech_Challenge_Fase_02.pdf` e da `propostaV1.md`.
> Convenção de status: `[x]` feito · `[~]` em andamento / parcial · `[ ]` pendente.

---

## Visão Geral do Desafio

- [x] Ler e interpretar `Tech_Challenge_Fase_02.pdf`
- [x] Ler e interpretar `docs/propostaV1.md`
- [x] Validar alinhamento da proposta com o grupo (4 etapas do PDF vs. 5 fases da proposta)

### Requisitos transversais (valem em todas as etapas)

- [x] Repositório Git inicializado
- [x] `.gitignore` configurado
- [x] `.dockerignore` configurado
- [x] `.env.example` configurado
- [ ] `.env` real (NUNCA commitado) — apenas local
- [ ] Histórico de commits semântico (Conventional Commits)
- [ ] `seeds` fixados em todos os processos estocásticos (numpy, torch, sklearn, split)

---

## ETAPA 0 — Pipeline de ML (fundação executada em `notebooks/` + `src/`)

> Não está totalmente detalhada no PDF, mas é pré-requisito para a ETAPA 4.
> Documentos de referência: `docs/REPORT.md` (progresso macro), `docs/GUIDE.md` (guia técnico de modelagem), `docs/ML_EXECUTION_GUIDE.md` (passos de execução).
> ⚠️ **Alerta resolvido**: Pipeline 2 (classificação `review_score`) foi removido. Apenas o Pipeline 1 (Recomendação) permanece.

### 0.1 Escolha do dataset

- [x] Dataset selecionado: **Olist (Brazilian E-Commerce Public Dataset)**
- [x] ≥ 10.000 interações user-item: **99.785 interações** (≈ 10× o mínimo)
- [x] 93.358 usuários únicos · 32.216 produtos únicos · 72 categorias
- [x] Período coberto: 2016-09-15 → 2018-08-29
- [x] EDA inicial de contexto: `notebooks/simple-eda-sales-and-customer-patterns.ipynb`
- [ ] Avaliar datasets alternativos (Instacart / RetailRocket / MovieLens) — apenas se Olist mostrar-se inviável

### 0.2 Limpeza do dataset

- [x] Junção dos CSVs brutos Olist (orders, items, products, customers, payments, reviews, category translation)
- [x] Filtragem para pedidos `delivered` (110.840 pedidos → 99.785 interações)
- [x] Remoção de duplicados (0 linhas duplicadas no dataset final)
- [x] Agregação em pares `user × item` por `customer_unique_id` gerando `interactions.parquet`
- [x] Formato Parquet (5,66 MB) preserva tipos nativos
- [x] Relatório EDA principal: `notebooks/01_eda.ipynb` (+ versão executada `01_eda_executed.ipynb`)
- [x] Relatório escrito: `reports/eda_report.md`
- [x] 8 figuras EDA: `reports/figures/01..08_*.png`
- [x] Dicionário de dados: `data/processed/README.md`
- [x] Desativado o filtro de cold-start severo (reduzia para 2.656 linhas, abaixo do mínimo)

### 0.3 Escolha das features

- [x] Inventário de features por categoria documentado em `notebooks/02_eda_feature_engineered.ipynb`
  - [x] Identificadores (5) — incluindo mapeamentos densos para embeddings
  - [x] Target/Sinal (3) — `review_score`, `has_review`, `purchase_count`
  - [x] Numéricas (6) — `price`, `freight_value`, logs, ratios, outliers
  - [x] Temporais (8) — ano, mês, dia, hora, recência, sazonalidade
  - [x] Categóricas encodadas (8) — target encoding, frequency, OHE payment
  - [x] Agregações de usuário (6) — histórico de compras
  - [x] Agregações de produto (6) — popularidade e stats
- [x] Validação da qualidade das features: `reports/figures/19_quality_validation.png`
- [x] Metadata persistida: `data/processed/feature_metadata.json`
- [x] Catálogo completo: `data/processed/FEATURES.md`
- [x] Mapeamentos ID para embedding: `data/processed/id_mappings.json`
- [x] Features redundantes já removidas: 3 constantes + 4 colineares (`product_volume_cm3`, `order_item_id_max`, `freight_value_log`, `payment_installments_sum`)
- [ ] Revisar `feature_metadata.json` e remover features redundantes antes do MLP (rodada extra de auditoria)

### 0.4 Escolha dos modelos e estratégia de treinamento

- [x] Modelos baseline definidos (pipeline **clássico**): Gradient Boosting, Decision Tree, Random Forest, Logistic Regression
- [x] Modelos baseline definidos (pipeline **recomendação**): Popularidade, Top Rated, Item-Item CF (`src/train.py`)
- [x] Métricas clássicas: Accuracy, Precision (weighted/macro), Recall (weighted/macro), F1, CV
- [x] Validação cruzada (k-fold) nos baselines clássicos
- [x] Split treino/teste (≈ 80/20: 94.516 / 23.630 amostras) com seed fixada — **pipeline clássico**
- [ ] **Split temporal** (70/15/15) definido em `GUIDE.md` §4 — **NÃO implementado** para o pipeline de recomendação
  - [ ] Implementar `temporal_split()` sem leakage
  - [ ] Justificar no README por que temporal > aleatório para recomendação
- [ ] Definir formalmente a estratégia de avaliação **top-K** para o sistema de recomendação
  - [ ] Precision@K, Recall@K, NDCG@K, Hit Rate@K (≥ 4 métricas conforme PDF)
  - [ ] Implementação manual já esboçada em `GUIDE.md` §9.2 (`calculate_metrics_at_k`, `evaluate_model`)

### 0.5 Feature Engineering

- [x] Pipeline `src/feature_engineering.py` (Stage 2 do DVC) implementado
- [x] Saída: `data/processed/interactions_fe.parquet` (99.785 × **42** colunas)
- [x] Catálogo de features: `data/processed/FEATURES.md`
- [x] Notebook de validação: `notebooks/02_eda_feature_engineered.ipynb` (+ executado)
- [x] 12 figuras de validação FE: `reports/figures/09..20_*.png`
- [x] Comparação antes/depois: `reports/figures/20_before_after_comparison.png`
- [x] Estratégia documentada em `docs/REPORT.md` §7.5 (uso em baselines vs. MLP)

### 0.6 Treinamento dos modelos de Baseline (Scikit-Learn) — Recomendação

**Pipeline de recomendação (`src/train.py`):**
- [x] Baseline 1: **Popularidade Global** (`train_popularity_baseline`)
- [x] Baseline 2: **Top Rated** com filtro `min_reviews` (`train_top_rated_baseline`)
- [x] Baseline 3: **Item-Item CF (Cosine Similarity)** (`train_item_similarity_baseline`)
- [ ] Baseline 4: **TruncatedSVD** — definido em `GUIDE.md` §7, ainda **NÃO implementado**
- [ ] Métricas **dummy/placeholder** em `evaluate_dummy_metrics()` (valores fixos) — **NÃO calcula top-K real**
- [ ] Substituir `evaluate_dummy_metrics()` por cálculo real top-K (Recall@K, NDCG@K, MAP@K, Hit Rate@K)
- [ ] Split temporal aplicado antes de montar a matriz CSR
- [ ] Promover artefatos `data/processed/temporary_baseline_recommendations.csv` para **artefatos versionados**

### 0.7 Treinamento do MLP (PyTorch) — **PENDENTE — PRÓXIMO BLOCO CRÍTICO**

> Especificação já detalhada em `docs/GUIDE.md` §11 (NCF + BPR Loss + Embeddings + Aux Features).

- [ ] Definir arquitetura — recomendada em `GUIDE.md`:
  - [ ] `nn.Embedding` para `user_id` e `product_id_idx`
  - [ ] `nn.Embedding` menor para `category_id`
  - [ ] MLP com camadas `[128, 64]` ou `[256, 128, 64]` + ReLU + Dropout
  - [ ] Concatenação com features auxiliares normalizadas (`StandardScaler`)
- [ ] Implementar `Dataset` e `DataLoader` PyTorch sobre `interactions_fe.parquet`
- [ ] **Negative sampling on-the-fly** (1–4 negativos por positivo) — base para BPR
- [ ] Implementar **BPR Loss** (`-log(sigmoid(pos - neg))`) com `F.logsigmoid` numericamente estável
- [ ] Treino com **early stopping** monitorando validação
- [ ] Otimizador: `AdamW` (weight_decay 1e-4) ou `Adam`
- [ ] Seed global fixada (numpy + torch + python)
- [ ] Device handling: tensors movidos para GPU quando disponível
- [ ] Avaliar com as **≥ 4 métricas top-K** (Hit Rate@K, Recall@K, Precision@K, NDCG@K)
- [ ] Comparar MLP vs. SVD vs. Popularidade (tabela final em `demo.ipynb`)
- [ ] Logar params/métricas/artefatos no **MLflow** (≥ 3 runs variando hiperparâmetros)
- [ ] Promover modelo campeão para **Model Registry → Production**
- [ ] Redigir `MODEL_CARD.md` com performance, limitações e vieses

### 0.8 Limitações e vieses já identificados (insumo para o Model Card)

- [x] **Sparsidade extrema**: 99,997 % (pior que Amazon Reviews, similar a MovieLens 1M)
- [x] **Cold-start severo**: média 1,07 interação/usuário · mediana 1 · máx 14
- [x] **Long-tail de produtos**: máx 456 interações em um único produto
- [x] **Desbalanceamento de classes** no target (`review_score`): maioria 5★ e 4★
- [x] **Viés geográfico**: concentração em SP/RJ/MG (≈ 67 % das compras)
- [x] **Viés de pagamento**: 73,5 % cartão de crédito + 19,5 % boleto
- [x] **Viés temporal**: 100k pedidos concentrados em 2 anos (2016-2018)
- [x] **Carrinho unitário**: maioria esmagadora contém 1 único tipo de produto
- [ ] Quantificar magnitude dos vieses em métricas do Model Card

### 0.8 Limitações e vieses já identificados (insumo para o Model Card)

- [x] **Sparsidade extrema**: 99,997 % (pior que Amazon Reviews, similar a MovieLens 1M)
- [x] **Cold-start severo**: média 1,07 interação/usuário · mediana 1 · máx 14
- [x] **Long-tail de produtos**: máx 456 interações em um único produto
- [x] **Desbalanceamento de classes** no target (`review_score`): maioria 5★ e 4★
- [x] **Viés geográfico**: concentração em SP/RJ/MG (≈ 67 % das compras)
- [x] **Viés de pagamento**: 73,5 % cartão de crédito + 19,5 % boleto
- [x] **Viés temporal**: 100k pedidos concentrados em 2 anos (2016-2018)
- [x] **Carrinho unitário**: maioria esmagadora contém 1 único tipo de produto
- [ ] Quantificar magnitude dos vieses em métricas do Model Card

---

## ETAPA 1 — Clean Code e Estrutura (Disciplina 01) · Proposta Fase 1

> Entregável: repositório base com estrutura limpa e linting passando.

- [x] Estrutura de diretórios `src/`, `tests/`, `data/`, `models/`, `configs/`
- [~] Clean code em todo o código (funções ≤ 20 linhas, naming, SOLID)
  - [ ] Revisão geral de `src/*.py` para garantir funções ≤ 20 linhas
  - [ ] Padronizar naming conventions em todo o código
- [~] `pyproject.toml` com Poetry/uv, deps prod/dev separadas, lock file commitado
  - [x] `pyproject.toml` presente
  - [x] `poetry.lock` presente
  - [x] `uv.lock` presente
  - [ ] Garantir separação explícita `dependencies` vs. `group.dev.dependencies`
- [~] Type hints + docstrings Google style em funções públicas
  - [ ] Auditar `src/` e preencher docstrings ausentes
- [~] Linter configurado (Ruff) + pre-commit hooks
  - [x] `.pre-commit-config.yaml` presente
  - [x] Ruff configurado
  - [ ] Rodar `ruff check .` e zerar warnings
- [ ] Implementar ≥ 1 design pattern (Factory p/ modelos, Strategy p/ preprocessors)
  - [ ] Factory para instanciar modelos (baselines + MLP)
  - [ ] Strategy para preprocessors
- [ ] Módulos curtos e responsabilidades únicas (SRP)

---

## ETAPA 2 — Ambiente e Dependências (Disciplina 02) · Proposta Fase 1 (continuação)

> Entregável: projeto instalável do zero com `poetry install`.

- [x] `pyproject.toml` com deps de prod (pytorch, sklearn, mlflow) e dev (pytest, ruff)
- [x] Lock file commitado (`poetry.lock`)
- [~] Configurações externalizadas em `.env` + Pydantic Settings
  - [x] `.env.example` presente
  - [ ] Implementar `src/config.py` com `pydantic-settings`
  - [ ] Carregar configs em todos os módulos (substituir hardcoded)
- [ ] Script `scripts/validate_env.py` (valida Python, deps, env vars)
- [ ] Verificar instalação limpa em ambiente novo (`poetry install` do zero)

---

## ETAPA 3 — Containerização e Versionamento (Disciplinas 03 e 04) · Propostas Fases 2 e 4

> Entregável: pipeline reprodutível via `dvc repro` + Docker funcional.

### 3.1 Dataset e DVC

- [x] Dataset de e-commerce com ≥ 10.000 interações user-item
- [x] `.dvc/` inicializado
- [x] `dvc.yaml` presente
- [ ] Configurar DVC remote (local FS / S3 / GDrive) — `docs/DVC_REMOTE_SETUP.md` já esboçado
- [ ] Versionar dataset final (`data/*.csv` rastreados por `.dvc`)

### 3.2 Pipeline DVC

- [~] Pipeline `dvc.yaml` com ≥ 3 stages sequenciais
  - [x] `dvc.yaml` existente
  - [ ] Validar ≥ 3 stages: `preprocess` → `feature_eng` → `train` → `evaluate`
  - [ ] Tornar `dvc repro` 100% reprodutível (sem warnings)

### 3.3 Pré-processamento e Baselines (Scikit-Learn)

- [~] Pipeline de pré-processamento
  - [x] `src/data_preparation.py`
  - [x] `src/feature_engineering.py`
  - [ ] Limpar/refatorar conforme Clean Code
- [~] Baselines de Recomendação (em `src/train.py`)
  - [x] Popularity Baseline
  - [x] Top-Rated Baseline
  - [x] Item-Item CF (Cosine Similarity)
  - [ ] TruncatedSVD Baseline
- [ ] Registrar métricas de baselines para comparação

### 3.4 Docker

- [ ] Dockerfile multi-stage (`builder` deps + `runtime` app)
- [ ] Imagem final otimizada (slim/alpine, sem cache de build)
- [ ] `docker-compose.yml` com serviço de treino + MLflow server
- [ ] Testar build local: `docker build .` e `docker compose up`

### 3.5 MLflow Tracking (>= 3 runs)

- [x] `src/train.py` com integração MLflow
- [ ] Garantir ≥ 3 runs rastreados no MLflow
- [ ] Servidor MLflow rodando (via docker-compose ou local)

---

## ETAPA 4 — Rede Neural, Registry e Entrega (Disciplina 04) · Propostas Fases 3 e 5

> Entregável: repositório final + modelo no Registry + vídeo STAR.

### 4.1 Modelo Neural (PyTorch) — **especificação detalhada em `GUIDE.md` §11**

- [ ] Arquitetura **NCF (Neural Collaborative Filtering)** — baseada em `docs/GUIDE.md`:
  - [ ] `nn.Embedding(n_users, emb_dim)` para `user_id`
  - [ ] `nn.Embedding(n_items, emb_dim)` para `product_id_idx`
  - [ ] `nn.Embedding(n_categories, 8)` para `category_id` (cardinalidade menor)
  - [ ] MLP com camadas `[128, 64]` ou `[256, 128, 64]` + ReLU + Dropout
  - [ ] Concatenação com features auxiliares normalizadas via `StandardScaler`/`BatchNorm1d`
  - [ ] Inicialização Xavier nos lineares + normal(0, 0.01) nos embeddings
- [ ] **BPR Loss** implementada (`-F.logsigmoid(pos_scores - neg_scores).mean()`)
- [ ] Negative sampling on-the-fly (1–4 negativos por positivo) dentro do `DataLoader`
- [ ] Treinamento com **Early Stopping** monitorando validação
- [ ] Otimizador `AdamW` (weight_decay 1e-4) ou `Adam`
- [ ] Seed global fixada (numpy + torch + python) — atende requisito transversal
- [ ] Device handling (GPU quando disponível)
- [ ] Avaliar com **≥ 4 métricas top-K**: Hit Rate@K, Recall@K, Precision@K, NDCG@K
- [ ] Comparar performance do NCF vs. SVD vs. Popularidade (tabela final em `demo.ipynb`)
- [ ] Grades de hiperparâmetros para tunar (≥ 3 combinações distintas):
  - `emb_dim` ∈ {16, 32, 64}
  - `hidden_layers` ∈ {`[128, 64]`, `[256, 128, 64]`}
  - `dropout` ∈ {0.2, 0.3, 0.5}
  - `learning_rate` ∈ {1e-3, 5e-4, 1e-4}
  - `batch_size` ∈ {512, 1024, 2048}
  - `num_negatives` ∈ {1, 4, 10}

### 4.2 MLflow Model Registry

- [ ] Registrar modelo validado no Registry
- [ ] Transição: `None` → `Staging` → `Production`
- [ ] Documentar critério de promoção no `Model Card`

### 4.3 Model Card

- [ ] Redigir `MODEL_CARD.md` com:
  - [ ] Performance alcançada
  - [ ] Limitações do modelo
  - [ ] Vieses identificados nos dados
  - [ ] Uso pretendido e fora de escopo

### 4.4 README

- [~] `README.md` presente
- [ ] Instruções exaustivas de reprodução:
  - [ ] Pré-requisitos (Python, Poetry/Docker)
  - [ ] Instalação passo a passo
  - [ ] Como rodar pipeline (`dvc repro`)
  - [ ] Como treinar/servir modelo
  - [ ] Como acessar MLflow UI
  - [ ] Troubleshooting

### 4.5 Vídeo STAR (5 minutos)

- [ ] Roteiro STAR: Situation → Task → Action → Result
- [ ] Gravação ≤ 5 minutos
- [ ] Upload e link referenciado no README

---

## ETAPA 5 (Proposta) — Qualidade, Documentação e Apresentação Executiva

> Consolidação final e auditoria.

- [ ] Auditoria do histórico de commits (semântico, mensagens claras)
- [ ] Revisão final do `README.md`
- [ ] Revisão final do `MODEL_CARD.md`
- [ ] `docs/REPORT.md` e `docs/GUIDE.md` revisados (já existentes)
- [ ] Checklist de qualidade: `ruff`, `pytest`, `dvc repro`, `docker build` todos passando

---

## Bônus — Deploy em Nuvem (AWS / Azure / GCP) · Proposta Fase 4

- [ ] Escolher provedor (AWS / Azure / GCP)
- [ ] Dockerfile pronto para deploy
- [ ] Provisionar infraestrutura (VM / Container Service / App Runner)
- [ ] Subir imagem para registry (ECR / ACR / GCR / Docker Hub)
- [ ] Deploy e teste do container
- [ ] URL pública acessível e documentada no README

---

## Critérios de Avaliação e Pesos

| Critério                         | Peso | Status |
| :------------------------------- | :--: | :----: |
| Clean code e estrutura           | 15%  |  [~]   |
| Reprodutibilidade                | 15%  |  [~]   |
| Docker                           | 15%  |  [ ]   |
| DVC + Pipeline                   | 15%  |  [~]   |
| Rede neural (PyTorch)            | 15%  |  [ ]   |
| MLflow + Registry                | 10%  |  [~]   |
| Vídeo STAR                       | 10%  |  [ ]   |
| **Bônus:** Deploy em nuvem       |  5%  |  [ ]   |

> Legenda: `[x]` feito · `[~]` parcial · `[ ]` pendente.

---

## Como usar este checklist

1. Antes de cada sessão: revisar o que está `[ ]` ou `[~]` e definir o próximo item.
2. Ao concluir um item: marcar `[x]` e adicionar nota curta se houver decisão relevante.
3. Itens `[~]` devem ter um **dono** e **prazo informal** definidos em reunião do grupo.
4. Atualizar este arquivo sempre que o escopo mudar.

---

*Documento vivo. Última atualização: 2026-06-27 — sincronizado com `docs/REPORT.md`, `docs/GUIDE.md`, `docs/ML_EXECUTION_GUIDE.md` e `docs/DVC_REMOTE_SETUP.md`. Adicionada ETAPA 0.9 (alerta de duplicação de pipelines) e detalhamento técnico do NCF.*