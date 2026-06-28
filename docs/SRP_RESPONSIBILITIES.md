# SRP — Single Responsibility Principle por Módulo

> **Status:** Vigente a partir de 2026-06-27
> **Princípio:** Cada módulo deve ter **uma única razão para mudar** (Robert C. Martin, *Clean Architecture*)

Este documento declara explicitamente a responsabilidade de cada módulo em `src/`. Se uma mudança requer editar mais de um módulo para uma única feature, há violação de SRP.

---

## Mapa de Responsabilidades

### 📁 `src/`

| Arquivo | Responsabilidade Única | NÃO deve conter |
|---|---|---|
| `src/config.py` | Centralizar configurações via Pydantic Settings | Lógica de ML, I/O, transformações |
| `src/data_preparation.py` | Pipeline ETL: carregar CSVs Olist brutos → `interactions.parquet` | Treino, métricas, MLflow |
| `src/feature_engineering.py` | Gerar features engineered → `interactions_fe.parquet` | Treino, baseline, scoring |
| `src/eda.py` | Análise exploratória e geração de relatório Markdown | Mutação de dados, treino |

### 📁 `src/data/`

| Arquivo | Responsabilidade | NÃO deve conter |
|---|---|---|
| `src/data/splits.py` | Função `temporal_split()` com assertions anti-leakage | Treino, métricas |
| `src/data/dataset.py` | `ImplicitFeedbackDataset` (PyTorch DataLoader para BPR) | Modelos, métricas |
| `src/data/preprocessing.py` | `fit_scaler()`, `transform_features()`, `save_scaler_stats()` | Split, treino |
| `src/data/strategies.py` | Strategy Pattern para splits (Temporal vs Random) | Modelos, datasets |

### 📁 `src/models/`

| Arquivo | Responsabilidade | NÃO deve conter |
|---|---|---|
| `src/models/ncf.py` | Arquitetura `NCFHybrid` (embeddings + MLP + aux features) | Treino, dados, métricas |
| `src/models/losses.py` | `bpr_loss()` e variantes de ranking loss | Modelos, dados |
| `src/models/factory.py` | Factory Method para instanciar baselines | Treino, métricas, datasets |

### 📁 `src/training/`

| Arquivo | Responsabilidade | NÃO deve conter |
|---|---|---|
| `src/training/evaluate.py` | Métricas @K (HitRate, Recall, Precision, NDCG, MAP) + `evaluate_model()` | Treino, modelos, dados |

### 📁 `src/` (scripts de treinamento direto)

| Arquivo | Responsabilidade | NÃO deve conter |
|---|---|---|
| `src/train.py` | Treino de baselines clássicos (Popularity, TopRated, ItemItemCF, TruncatedSVD) + MLflow logging | NCF, PyTorch |

---

## Fluxo de Dependências (sem ciclos)

```
src/data_preparation.py
        ↓
src/feature_engineering.py
        ↓
src/data/{dataset,splits,preprocessing,strategies}
        ↓
src/models/{ncf,losses,factory} ← src/train.py (baselines)
        ↓
src/training/evaluate.py
        ↓
scripts/train_ncf.py → MLflow Model Registry
```

**Invariantes:**
- ✅ Módulos em `data/` podem importar de `config.py`
- ✅ Módulos em `models/` podem importar de `data/` e `config.py`
- ❌ Módulos em `data/` NÃO devem importar de `models/` ou `training/`
- ❌ Módulos em `models/` NÃO devem importar de `training/`

## Violações Comuns a Evitar

### ❌ Anti-pattern 1: "God Module"
```python
# ERRADO: src/utils.py
def load_data(): ...      # → data_preparation
def train_model(): ...    # → train
def compute_metrics(): ...# → training/evaluate
def save_to_mlflow(): ... # → scripts/train_ncf
```

### ❌ Anti-pattern 2: "Leaky Abstraction"
```python
# ERRADO: modelo conhece detalhes do split
class NCFHybrid:
    def fit(self, train_df, val_df):  # ← acoplamento com split
        pass
```

### ✅ Pattern Correto: Composição
```python
# CERTO: cada módulo tem responsabilidade clara
context = PreprocessingContext(TemporalSplitStrategy())
train_df, val_df, test_df = context.execute(df)

model = model_factory("svd")
model.fit(train_df)

metrics = evaluate_model(model, val_df, ...)
```

## Como Auditar SRP

### 1. Pergunte: "Por que este módulo mudaria?"
- `data_preparation.py` muda quando **schema do Olist** muda
- `feature_engineering.py` muda quando **lógica de FE** evolui
- `train.py` muda quando **novo baseline** é adicionado
- `evaluate.py` muda quando **nova métrica @K** é introduzida

Se duas dessas mudanças afetam o mesmo arquivo → SRP violado.

### 2. Verifique Acoplamento
```bash
# Quantos imports cada módulo tem de outros módulos src/?
uv run python -c "
import ast
from pathlib import Path
from collections import Counter

deps = Counter()
for f in Path('src').rglob('*.py'):
    if '__pycache__' in str(f): continue
    tree = ast.parse(f.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('src.'):
            deps[f.relative_to('src').as_posix()] += len(node.names)
for mod, n in sorted(deps.items(), key=lambda x: -x[1])[:10]:
    print(f'  {n:>3} deps: {mod}')
"
```

**Saúde esperada:**
- `data_preprocessing.py`: 2-3 deps
- `models/ncf.py`: 1-2 deps
- `training/evaluate.py`: 1-2 deps (config + torch)
- `train.py`: 3-4 deps (config + data + models)

## Histórico de Refatorações SRP

| Data | Módulo | Antes | Depois |
|---|---|---|---|
| 2026-06-27 | `src/training/evaluate.py` | `evaluate_model()` monolítico (121 linhas) | 5 helpers privados ≤ 23 linhas cada |
| 2026-06-27 | `src/train.py` | `evaluate_model()` monolítico (45 linhas) | 2 helpers privados ≤ 19 linhas cada |
| 2026-06-27 | `src/models/factory.py` | Inline class instantiation | `model_factory(name, **kwargs)` |

---

*Documento vivo. Última atualização: 2026-06-27*
