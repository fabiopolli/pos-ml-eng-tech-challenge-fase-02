# 🛒 Olist Recommender System

![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3+-f7931e)
![MLflow](https://img.shields.io/badge/MLflow-2.5+-0194e2)
![DVC](https://img.shields.io/badge/DVC-3.0+-945dd6)
![Poetry](https://img.shields.io/badge/Poetry-1.7+-60a5fa)
![License](https://img.shields.io/badge/license-MIT-green)

> Sistema de recomendação de produtos para e-commerce baseado no dataset público **Olist Brazilian E-Commerce**, implementando pipeline MLOps completo de treinamento, tracking, versionamento e deploy.

**Tech Challenge Fase 02** — Pós-graduação em Machine Learning Engineering

## 📋 Sobre o Projeto

Este projeto implementa um sistema de recomendação end-to-end utilizando o dataset público da Olist (100k pedidos, 2016-2018). O objetivo é comparar modelos baseline (Scikit-Learn) com uma rede neural de embeddings (PyTorch), seguindo práticas modernas de MLOps.

### 🎯 Objetivos Técnicos
- Modelo de recomendação com MLP / Embeddings (PyTorch)
- Comparação com baselines clássicos (Scikit-Learn)
- Tracking de experimentos com MLflow (mínimo 3 runs)
- Versionamento de dados com DVC (pipeline ≥ 3 estágios)
- Containerização com Docker (multi-stage) e deploy em cloud
- ≥ 4 métricas quantitativas e ≥ 10.000 interações user-item

## 👥 Equipe e Responsabilidades

| Membro | Fase | Responsabilidade |
|---|---|---|
| **Fábio Polli** | Fase 1 — Arquitetura | Estrutura inicial do projeto, Poetry, DVC |
| **Bill** | Fase 4 — DevOps | Docker, deploy em cloud, CI/CD |
| **Denis & Romário** | Fase 2 — Dados + Baselines | Pipeline de dados, modelos baseline |
| **TBD** | Fase 3 — Deep Learning | MLP/NCF com PyTorch |
| **TBD** | Fase 5 — Documentação | Model Card, vídeo STAR, README |

## 📅 Cronograma

| Fase | Período | Status |
|---|---|---|
| 1. Arquitetura | 07/06 - 13/06 | ✅ Concluída |
| 2. Dados + Baselines | 14/06 - 27/06 | 🔄 Em andamento |
| 3. Deep Learning | 28/06 - 04/07 | ⏳ Pendente |
| 4. DevOps | 05/07 - 09/07 | ⏳ Pendente |
| 5. Documentação | 10/07 - 13/07 | ⏳ Pendente |

## 📊 Status do Projeto

- [x] Repositório estruturado com Poetry + DVC
- [x] Pipeline de preparação de dados (99.785 interações, 42 features)
- [x] EDA completa (2 notebooks + 20 visualizações)
- [x] Feature engineering documentado
- [ ] Modelos baseline (Popularidade, TruncatedSVD, Item-Item CF)
- [ ] Modelo NCF com PyTorch + BPR
- [ ] Tracking MLflow com 3+ runs
- [ ] Dockerfile multi-stage
- [ ] Deploy em cloud
- [ ] Model Card
- [ ] Vídeo STAR (5 min)

## 🏗️ Arquitetura do Projeto

```
pos-ml-eng-tech-challenge-fase-02/
├── src/                          # Código fonte
│   ├── data_preparation.py       # Pipeline ETL (CSVs → parquet)
│   ├── eda.py                    # Análise exploratória
│   └── feature_engineering.py    # Engenharia de features
├── data/
│   ├── raw/                      # CSVs originais (versionados via DVC)
│   └── processed/                # Artefatos processados (versionados via DVC)
├── notebooks/                    # Jupyter notebooks
├── reports/                      # Relatórios e figuras do EDA
├── docs/                         # Documentação adicional
├── tests/                        # Testes unitários
├── configs/                      # Configurações
├── models/                       # Modelos treinados (DVC)
├── pyproject.toml                # Poetry
├── poetry.lock                   # Lock file
├── .pre-commit-config.yaml       # Pre-commit hooks
└── README.md
```

## 🚀 Quick Start

### Pré-requisitos
- [Poetry](https://python-poetry.org/) (gerenciador de dependências)
- Python 3.10-3.12
- Git
- DVC

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/fabiopolli/pos-ml-eng-tech-challenge-fase-02.git
cd pos-ml-eng-tech-challenge-fase-02

# 2. Instalar dependências com Poetry
poetry install

# 3. Configurar DVC remote (após clonar)
dvc pull

# 4. Ativar ambiente virtual
poetry shell

# 5. Executar pipeline de dados
poetry run python src/data_preparation.py
poetry run python src/feature_engineering.py
```

## 📦 Stack Tecnológica

| Categoria | Tecnologia | Versão |
|---|---|---|
| Linguagem | Python | 3.10-3.12 |
| Gerenciador | Poetry | 1.7+ |
| Deep Learning | PyTorch | ≥ 2.0 |
| ML Clássico | Scikit-Learn | ≥ 1.3 |
| Dados | Pandas, NumPy, PyArrow | latest |
| Tracking | MLflow | ≥ 2.5 |
| Versionamento | DVC | ≥ 3.0 |
| Logging | Loguru | ≥ 0.7 |
| Lint/Format | Ruff | ≥ 0.3 |
| Testes | Pytest | ≥ 7.4 |

## 📚 Documentação Adicional

- [`docs/REPORT.md`](./docs/REPORT.md) — Relatório técnico detalhado (447 linhas)
- [`docs/GUIDE.md`](./docs/GUIDE.md) — Guia para treinamento dos modelos (699 linhas)
- [`docs/propostaV1.md`](./docs/propostaV1.md) — Plano de ação original em 5 fases
- [`docs/Tech_Challenge_Fase_02.pdf`](./docs/Tech_Challenge_Fase_02.pdf) — PDF original do desafio
- [`reports/eda_report.md`](./reports/eda_report.md) — Relatório de EDA
- [`reports/figures/`](./reports/figures/) — 20 visualizações do EDA

## 📊 Estatísticas do Dataset

| Métrica | Valor |
|---|---|
| Total de interações | 99.785 |
| Usuários únicos | 93.358 |
| Produtos únicos | 32.216 |
| Categorias únicas | 72 |
| Sparsity | 99,9967% |
| Features (após FE) | 42 |
| Período | 2016-09-15 → 2018-08-29 |

## 🤝 Contribuindo

1. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
2. Commit suas mudanças (`git commit -m 'feat: Minha nova feature'`)
3. Push para a branch (`git push origin feature/MinhaFeature`)
4. Abra um Pull Request

### Padrões de Código
- Format: `poetry run ruff format .`
- Lint: `poetry run ruff check .`
- Type check: `poetry run mypy src/`
- Testes: `poetry run pytest`
- Pre-commit: `poetry run pre-commit run --all-files`

## 📄 Licença

MIT License — ver [`LICENSE`](./LICENSE)

---

**Contato**: Grupo Tech Challenge Fase 02 — Pós-graduação ML Engineering
