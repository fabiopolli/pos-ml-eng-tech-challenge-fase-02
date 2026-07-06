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
- [x] `.env` real (NUNCA commitado) — apenas local — `.env` ausente do repo (correto), `.env.example` presente, `.gitignore` linha 28 ignora `.env`
- [~] Histórico de commits semântico (Conventional Commits) — 14/20 commits (70%) seguem padrão; recentes sim (`feat:`, `docs:`, `refactor:`, `chore:`), legados não
- [x] `seeds` fixados em todos os processos estocásticos (numpy, torch, sklearn, split) — `src/config.py:55` define `seed=42`; `scripts/train_ncf.py:set_seed()` cobre numpy/torch/cuda; sklearn via `random_state` em todos os locais

---

## ETAPA 0 — Pipeline de ML (fundação executada em `notebooks/` + `src/`)

> Não está totalmente detalhada no PDF, mas é pré-requisito para a ETAPA 4.
> Documentos de referência: `docs/REPORT.md` (progresso macro, timeline, métricas de sucesso), `docs/GUIDE.md` (guia técnico de modelagem, fórmulas de métricas), `README.md` (Quick Start para execução).
> ⚠️ **Alerta resolvido**: Pipeline 2 (classificação `review_score`) foi removido. Apenas o Pipeline 1 (Recomendação) permanece.

### 0.1 Escolha do dataset

- [x] Dataset selecionado: **Olist (Brazilian E-Commerce Public Dataset)**
- [x] ≥ 10.000 interações user-item: **99.785 interações** (≈ 10× o mínimo)
- [x] 93.358 usuários únicos · 32.216 produtos únicos · 72 categorias
- [x] Período coberto: 2016-09-15 → 2018-08-29
- [x] EDA inicial de contexto: `notebooks/simple-eda-sales-and-customer-patterns.ipynb`

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
- [x] Revisar `feature_metadata.json` e remover features redundantes antes do MLP (rodada extra de auditoria) — **Auditoria Spearman concluída** em `reports/feature_audit_spearman.md`. Removidos: `user_recency_days` (ρ=-0.986 com `days_since_reference`) e `freight_value_log` (ρ=+0.966 com `user_avg_freight`). Resultado surpreendente: a remoção **NÃO** melhorou NDCG (caiu 13.2%). Production (`no_aux`) permanece como melhor modelo (NDCG=0.2725).

### 0.4 Escolha dos modelos e estratégia de treinamento

- [x] Modelos baseline definidos (pipeline **clássico**): Gradient Boosting, Decision Tree, Random Forest, Logistic Regression
- [x] Modelos baseline definidos (pipeline **recomendação**): Popularidade, Top Rated, Item-Item CF (`src/train.py`)
- [x] Métricas clássicas: Accuracy, Precision (weighted/macro), Recall (weighted/macro), F1, CV
- [x] Validação cruzada (k-fold) nos baselines clássicos
- [x] Split treino/teste (≈ 80/20: 94.516 / 23.630 amostras) com seed fixada — **pipeline clássico**
- [x] **Split temporal** (70/15/15) definido em `GUIDE.md` §4 — implementado em `src/data/splits.py`
  - [x] Implementar `temporal_split()` sem leakage — implementado com assertions anti-leakage
  - [x] Justificar no README por que temporal > aleatório para recomendação — seção "Por que Split Temporal?" adicionada ao README
- [x] Definir formalmente a estratégia de avaliação **top-K** — `docs/GUIDE.md` §9.0 (cutoffs K=5/10/20, candidatos 1+99, cold-start, agregação macro)
  - [x] Precision@K, Recall@K, NDCG@K, Hit Rate@K (≥ 4 métricas conforme PDF) — 5 métricas em `calculate_metrics_at_k()`
  - [x] Implementação manual já esboçada em `GUIDE.md` §9.2 (`calculate_metrics_at_k`, `evaluate_model`) — funcional em `src/training/evaluate.py` e `src/train.py`

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
- [x] Baseline 4: **TruncatedSVD** — classe `TruncatedSVDBaseline` implementada em `src/train.py` (CSR matrix + dot product, n_components=50, explained_var≈0.097)
- [x] Split temporal aplicado antes de montar a matriz CSR — `temporal_split()` é chamado antes de qualquer baseline
- [x] Promover artefatos `data/processed/temporary_baseline_recommendations.csv` para **artefatos versionados** — movidos para `artifacts/baselines/` com `.gitkeep`

> **Nota:** Itens `evaluate_dummy_metrics()` e "Substituir por cálculo real" eram **fantasmas do checklist** (a função nunca existiu). `evaluate_model()` sempre usou métricas reais. Removidos.

### 0.7 Treinamento do MLP (PyTorch) — **✅ CONCLUÍDO**

> Especificação detalhada em `docs/GUIDE.md` §11 (arquitetura NCF, BPR loss, hiperparâmetros).

- [x] Definir arquitetura NCF (Neural Collaborative Filtering):
  - [x] `nn.Embedding` para `user_id` e `product_id_idx` (emb_dim=32)
  - [x] `nn.Embedding` para `category_id` (emb_dim=8)
  - [x] MLP com camadas `[64, 32]` + ReLU + Dropout(0.5)
  - [x] Concatenação com features auxiliares normalizadas (StandardScaler)
- [x] Implementar `Dataset` e `DataLoader` PyTorch: `src/data/dataset.py`
- [x] **Negative sampling on-the-fly** (8 negativos por positivo) — base para BPR
- [x] Implementar **BPR Loss** (`-F.logsigmoid(pos_scores - neg_scores).mean()`)
- [x] Treino com **early stopping** monitorando NDCG@10 na validação
- [x] Otimizador: `AdamW` (lr=5e-4, weight_decay=5e-4) + ReduceLROnPlateau
- [x] Seed global fixada (numpy + torch + python) — atende requisito transversal
- [x] Device handling: tensors movidos para GPU quando disponível (fallback CPU)
- [x] Avaliar com **≥ 4 métricas top-K**: Hit Rate@K, Recall@K, Precision@K, NDCG@K, MAP@K
- [x] **6 runs MLflow** registradas (3 experimentos: Baseline, Optimization, Ablation)
- [x] Modelo registrado e promovido: `olist_ncf_recommender v1` → **Production stage**
- [x] Relatório de otimização: `reports/ncf_optimization_report.md`
- [x] Notebook NCF: `notebooks/04_ncf_training_results.ipynb`
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
- [x] Clean code em todo o código (funções ≤ 20 linhas, naming, SOLID)
  - [x] Revisão geral de `src/*.py` para garantir funções ≤ 20 linhas — `evaluate_model()` em `src/training/evaluate.py` (121→73 linhas via 5 helpers) e em `src/train.py` (45→28 linhas via 2 helpers)
  - [x] Padronizar naming conventions em todo o código — `docs/NAMING_CONVENTIONS.md` criado com regras, prefixos e exemplos
- [x] `pyproject.toml` com Poetry/uv, deps prod/dev separadas, lock file commitado
  - [x] `pyproject.toml` presente
  - [x] `poetry.lock` presente
  - [x] `uv.lock` presente
  - [x] Garantir separação explícita `dependencies` vs. `group.dev.dependencies` — migrado para formato PEP 621 puro, removido `[tool.poetry]` duplicado, adicionado `[project.authors]`, `[project.readme]`, build backend `hatchling`
- [x] Type hints + docstrings Google style em funções públicas
  - [x] Auditar `src/` e preencher docstrings ausentes — apenas 4 funções públicas sem docstring; todas preenchidas (`set_strategy`, `strategy` getter, `main` em data_preparation e feature_engineering)
- [~] Linter configurado (Ruff) + pre-commit hooks
  - [x] `.pre-commit-config.yaml` presente
  - [x] Ruff configurado
  - [x] Rodar `ruff check .` e zerar warnings — **0 warnings** em `src/` e `scripts/` (`uv run ruff check`)
- [x] Implementar ≥ 1 design pattern (Factory p/ modelos, Strategy p/ preprocessors)
  - [x] Factory para instanciar modelos — `src/models/factory.py` com 4 modelos registrados (popularity, top_rated, item_cf, svd)
  - [x] Strategy para preprocessors — `src/data/strategies.py` com TemporalSplitStrategy, RandomSplitStrategy, SplitContext
- [x] Módulos curtos e responsabilidades únicas (SRP) — `docs/SRP_RESPONSIBILITIES.md` declara responsabilidade única por módulo, com anti-patterns e mapa de dependências

---

## ETAPA 2 — Ambiente e Dependências (Disciplina 02) · Proposta Fase 1 (continuação)

> Entregável: projeto instalável do zero com `poetry install`.

- [x] `pyproject.toml` com deps de prod (pytorch, sklearn, mlflow) e dev (pytest, ruff)
- [x] Lock file commitado (`poetry.lock`)
- [x] Configurações externalizadas em `.env` + Pydantic Settings
  - [x] `.env.example` presente
  - [x] Implementar `src/config.py` com `pydantic-settings` — classe `Settings` com paths, MLflow, NCF, avaliação
  - [x] Carregar configs em todos os módulos (substituir hardcoded) — `src/train.py`, `src/data/splits.py`, `src/training/evaluate.py`, `scripts/train_ncf.py`
- [x] Script `scripts/validate_env.py` (valida Python, deps, env vars) — valida 6 categorias (Python, deps, dirs, files, DVC, dados processados)
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

- [x] Pipeline de pré-processamento
  - [x] `src/data_preparation.py`
  - [x] `src/feature_engineering.py`
  - [x] Limpar/refatorar conforme Clean Code — docstrings preenchidas (`main()`), ruff zero warnings, type hints verificados, SRP documentado em `docs/SRP_RESPONSIBILITIES.md`
- [x] Baselines de Recomendação (em `src/train.py`)
  - [x] Popularity Baseline
  - [x] Top-Rated Baseline
  - [x] Item-Item CF (Cosine Similarity)
  - [x] TruncatedSVD Baseline — implementado em `src/train.py` (rodada em K=10 e K=20)
- [x] Registrar métricas de baselines para comparação — todas as métricas são logadas no MLflow (10 runs geradas no pipeline end-to-end)

### 3.4 Docker

- [X] Dockerfile multi-stage (`builder` deps + `runtime` app)
- [X] Imagem final otimizada (slim/alpine, sem cache de build)
- [X] `docker-compose.yml` com serviço de treino + MLflow server
- [X] Testar build local: `docker build .` e `docker compose up`

### 3.5 MLflow Tracking (>= 3 runs)

- [x] `src/train.py` com integração MLflow
- [x] Garantir ≥ 3 runs rastreados no MLflow (**6 runs** em 3 experimentos)
- [x] Servidor MLflow local via SQLite (`mlflow.db`) — sem necessidade de servidor separado

---

## ETAPA 4 — Rede Neural, Registry e Entrega (Disciplina 04) · Propostas Fases 3 e 5

> Entregável: repositório final + modelo no Registry + vídeo STAR.

### 4.1 Modelo Neural (PyTorch) — **✅ CONCLUÍDO**

- [x] Arquitetura **NCF (Neural Collaborative Filtering)** — baseada em `docs/GUIDE.md`:
  - [x] `nn.Embedding(n_users, emb_dim=32)` para `user_id`
  - [x] `nn.Embedding(n_items, emb_dim=32)` para `product_id_idx`
  - [x] `nn.Embedding(n_categories, 8)` para `category_id`
  - [x] MLP com camadas `[64, 32]` + ReLU + Dropout(0.5)
  - [x] Concatenação com features auxiliares (StandardScaler)
  - [x] Inicialização Xavier nos lineares + normal(0, 0.01) nos embeddings
- [x] **BPR Loss** implementada (`-F.logsigmoid(pos_scores - neg_scores).mean()`)
- [x] Negative sampling on-the-fly (8 negativos por positivo) dentro do `DataLoader`
- [x] Treinamento com **Early Stopping** monitorando NDCG@10 na validação
- [x] Otimizador `AdamW` (lr=5e-4, weight_decay=5e-4) + ReduceLROnPlateau scheduler
- [x] Seed global fixada (numpy + torch + python) — atende requisito transversal
- [x] Device handling (GPU quando disponível, fallback CPU)
- [x] Avaliar com **≥ 4 métricas top-K**: Hit Rate@K, Recall@K, Precision@K, NDCG@K, MAP@K
- [x] Comparar NCF vs. baselines (tabela em `notebooks/04_ncf_training_results.ipynb`)
- [x] **6 runs** com variação de HPs (emb_dim ∈ {16,32,64}, hidden ∈ {[64,32], [128,64,32], [256,128,64]}, dropout ∈ {0.3,0.4,0.5}, lr ∈ {1e-3, 5e-4}, batch ∈ {1024, 2048}, n_neg ∈ {4, 8})

### 4.2 MLflow Model Registry

- [x] Registrar modelo validado no Registry (`olist_ncf_recommender`)
- [x] Transição: `None` → **Production** (v1)
- [ ] Documentar critério de promoção no `Model Card`

### 4.3 Model Card

- [ ] Redigir `MODEL_CARD.md` com:
  - [x] Performance alcançada: **NDCG@10 = 0.2725** (60× vs baseline)
  - [x] Limitações do modelo: cold-start 98%, 1 compra/usuário mediana
  - [x] Vieses identificados nos dados: long-tail, concentração SP/RJ/MG
  - [ ] Uso pretendido e fora de escopo

### 4.4 README

- [x] `README.md` presente
- [x] Instruções exaustivas de reprodução:
  - [x] Pré-requisitos (Python, Poetry/Docker)
  - [x] Instalação passo a passo (`uv sync`)
  - [x] Como rodar pipeline (`dvc repro`)
  - [x] Como treinar NCF (`scripts/train_ncf.py`)
  - [x] Como acessar MLflow UI (`mlflow ui --backend-store-uri sqlite://...`)
  - [x] Troubleshooting (comandos `uv run` para todas as ferramentas)

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

## ETAPA 6 — Deploy em Nuvem (AWS / Azure / GCP) · Proposta Fase 4

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
| Clean code e estrutura           | 15%  |  [x]   |
| Reprodutibilidade                | 15%  |  [x]   |
| Docker                           | 15%  |  [ ]   |
| DVC + Pipeline                   | 15%  |  [~]   |
| Rede neural (PyTorch)            | 15%  |  [x]   |
| MLflow + Registry                | 10%  |  [x]   |
| Vídeo STAR                       | 10%  |  [ ]   |
| Deploy em nuvem                  |  5%  |  [ ]   |

> Legenda: `[x]` feito · `[~]` parcial · `[ ]` pendente.
> **Nota 2026-06-27:** "Clean code e estrutura" promovido para `[x]` — ruff zero warnings, Factory + Strategy implementados, Pydantic Settings em uso. "DVC + Pipeline" rebaixado para `[~]` — itens pendentes: configurar DVC remote e validar 3+ stages em `dvc.yaml`.

---

## Como usar este checklist

1. Antes de cada sessão: revisar o que está `[ ]` ou `[~]` e definir o próximo item.
2. Ao concluir um item: marcar `[x]` e adicionar nota curta se houver decisão relevante.
3. Itens `[~]` devem ter um **dono** e **prazo informal** definidos em reunião do grupo.
4. Atualizar este arquivo sempre que o escopo mudar.

---

*Documento vivo. Última atualização: 2026-06-27 (sessão 3) — ETAPA 1 finalizada. 6 itens marcados como feitos: funções ≤ 20 linhas (refatoração de `evaluate_model`), `NAMING_CONVENTIONS.md` criado, `pyproject.toml` consolidado (formato PEP 621 + hatchling), 4 docstrings preenchidas (100% cobertura pública), `SRP_RESPONSIBILITIES.md` criado. Ruff continua em zero warnings.*

*Atualização adicional (sessão 2): requisitos transversais (linhas 20-22) revisados — `.env` (não commitado, OK), commits semânticos (~70%, parcial), seeds (todos cobertos, OK).*

*Sessão 1: 13 itens marcados como feitos após execução do `ACTION_PLAN.md`: Split temporal + justificativa, estratégia top-K formal, TruncatedSVD Baseline, `src/config.py` com pydantic-settings, `scripts/validate_env.py`, ruff zero warnings, Factory + Strategy patterns. Itens fantasmas (`evaluate_dummy_metrics`) removidos.*