# Plano de Ação — Ações Genuinamente Pendentes

> Foco exclusivo: 10 ações pendentes do escopo do Tech Challenge.
> Atualizado em 2026-06-27.
> **Convenção: nenhum código alterado nesta fase — apenas avaliação e planejamento.**

---

## 1. Resumo Geral

| Prioridade | Qtd | Ações |
|---|---|---|
| **Alta** | 3 | `src/config.py`, TruncatedSVD Baseline, `ruff check .` |
| **Média** | 6 | Justificativa temporal no README, estratégia top-K documentada, Factory, Strategy, `validate_env.py`, CSV baselines |
| **Baixa** | 1 | Promover CSVs de baselines para `artifacts/` |

---

## 2. Avaliação — O Que Já Está Feito (Não mexer)

| # | CHECKLIST diz | Realidade |
|---|---|---|
| 2 | `temporal_split()` sem leakage — **NÃO implementado** | ✅ Implementado em `src/data/splits.py` com assertions anti-leakage |
| 3 | Split temporal (70/15/15) — **NÃO implementado** | ✅ Mesma função acima |
| 6 | Precision@K, Recall@K, NDCG@K, HitRate@K (≥ 4 métricas) | ✅ 5 métricas em `calculate_metrics_at_k()` |
| 7 | Implementação manual em GUIDE.md §9.2 | ✅ `evaluate_model()` e `calculate_metrics_at_k()` funcionais |
| 9 | Métricas dummy em `evaluate_dummy_metrics()` | ❌ Função nunca existiu — **fantasma no CHECKLIST** |
| 10 | Substituir dummy por métricas reais | ❌ Não há o que substituir — métricas reais já existem |
| 11 | Split temporal antes da CSR | ✅ `temporal_split()` chamado antes de tudo em `src/train.py` |

**Ação necessária:** Marcar os itens 2, 3, 6, 7, 11 como `[x]` no CHECKLIST.md e remover os itens 9 e 10 (fantasmas).

---

## 3. Ações Genuinamente Pendentes — Plano Detalhado

### 3.1 🔴 Alta — `src/config.py` com `pydantic-settings`

**Status real:** Arquivo não existe. `pydantic-settings>=2.2.0` está em `pyproject.toml` mas nunca foi usado.

**Plano de execução:**

1. Criar `src/config.py`:
   ```python
   from pydantic_settings import BaseSettings, SettingsConfigDict
   from pathlib import Path

   class Settings(BaseSettings):
       model_config = SettingsConfigDict(
           env_file=".env",
           env_file_encoding="utf-8",
           case_sensitive=False,
       )
       # Paths
       data_dir: Path = Path("data/processed")
       artifacts_dir: Path = Path("artifacts")
       # MLflow
       mlflow_uri: str = "sqlite:///./artifacts/mlflow.db"
       experiment_name: str = "Olist_Recommender"
       # Treino
       seed: int = 42
       device: str = "cuda"
       # NCF
       ncf_emb_dim: int = 32
       ncf_hidden: list[int] = [64, 32]
       ncf_dropout: float = 0.5
       ncf_lr: float = 5e-4
       ncf_batch_size: int = 2048
       n_epochs: int = 15
       n_negatives: int = 8
       # Avaliação
       k_eval: int = 10
       n_neg_eval: int = 99
   ```

2. Substituir constantes hardcoded nos módulos:
   - `src/train.py`: paths, `K_VALUES`, `MLFLOW_URI`
   - `scripts/train_ncf.py`: `emb_dim`, `hidden_dims`, `lr`, `batch_size`, `n_epochs`
   - `src/data/splits.py`: `train_size`, `val_size` (70/15/15)
   - `src/training/evaluate.py`: `k`, `n_neg_eval`, `device`

3. Importar nos módulos: `from src.config import Settings; settings = Settings()`

**Dependências:** Nenhuma.
**Complexidade:** Média | **Estimativa:** 2-3h

---

### 3.2 🔴 Alta — TruncatedSVD Baseline

**Status real:** Esboço existe em `docs/GUIDE.md` §7, mas código não foi implementado em `src/train.py`.

**Plano de execução:**

1. Implementar classe `TruncatedSVDBaseline` em `src/train.py`:
   ```python
   class TruncatedSVDBaseline:
       def __init__(self, n_components: int = 50, random_state: int = 42):
           self.n_components = n_components
           self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
           self.user_factors: np.ndarray = None
           self.item_factors: np.ndarray = None
           self.user_idx_map: dict = None
           self.item_idx_map: dict = None
           self.rated_items: dict = None  # user_id -> set of rated item ids

       def fit(self, train_df: pd.DataFrame, user_col: str, item_col: str):
           # Construir CSR matrix
           users = train_df[user_col].unique()
           items = train_df[item_col].unique()
           self.user_idx_map = {u: i for i, u in enumerate(users)}
           self.item_idx_map = {p: i for i, p in enumerate(items)}
           # CSR: (user_idx, item_idx) -> count
           row = train_df[user_col].map(self.user_idx_map).values
           col = train_df[item_col].map(self.item_idx_map).values
           val = np.ones(len(train_df))
           n_users, n_items = len(users), len(items)
           self.rated_items = train_df.groupby(user_col)[item_col].apply(set).to_dict()
           # ...
           # Fit SVD
           self.user_factors, self.item_factors = svd_factors

       def recommend(self, user_id: str, top_k: int = 10) -> list[tuple]:
           if user_id not in self.user_idx_map:
               return []
           u_idx = self.user_idx_map[user_id]
           scores = self.user_factors[u_idx] @ self.item_factors.T
           rated = self.rated_items.get(user_id, set())
           for rated_item in rated:
               if rated_item in self.item_idx_map:
                   scores[self.item_idx_map[rated_item]] = -np.inf
           top_indices = np.argsort(scores)[::-1][:top_k]
           return [(list(self.item_idx_map.keys())[i], scores[i]) for i in top_indices]
   ```

2. Integrar no pipeline de treino (após baselines 1-3):
   ```python
   svd_model = TruncatedSVDBaseline(n_components=50)
   svd_model.fit(train_df, user_col="customer_unique_id", item_col="product_id")
   svd_metrics = evaluate_model(svd_model, ...)
   mlflow.log_metrics(svd_metrics, step=epoch)
   ```

3. Registrar métricas no MLflow (nome do run: `TruncatedSVD`)

4. Comparar NCF vs todos baselines (popularity, top_rated, item_cf, svd)

**Dependências:** `src/config.py` (para n_components e seed).
**Complexidade:** Média | **Estimativa:** 2-3h

---

### 3.3 🔴 Alta — `ruff check .` e zerar warnings

**Status real:** Ruff está em `pyproject.toml` (`[project.optional-dependencies].dev`) mas **não está instalado** como executável no ambiente (`error: Failed to spawn: ruff`). Nunca foi executado.

**Plano de execução:**

1. Instalar ruff no ambiente:
   ```bash
   uv pip install ruff
   # ou adicionar a pyproject.toml e rodar uv sync
   ```

2. Executar auditoria inicial:
   ```bash
   ruff check src/ scripts/ --output-format=text
   ```

3. Resolver warnings por categoria:
   - `F` (pyflakes): importações não usadas, variáveis indefinidas
   - `E` (pycodestyle): linhas > 88 caracteres, imports mal formatados
   - `I` (isort): imports fora de ordem
   - `UP` (pyupgrade): f-strings, `typing.List` → `list`

4. Rodar com `--fix` para correções automáticas:
   ```bash
   ruff check . --fix
   ```

5. Integrar no pre-commit hook (opcional):
   ```bash
   uv pip install pre-commit
   # criar .pre-commit-config.yaml
   ```

**Dependências:** Nenhuma.
**Complexidade:** Média | **Estimativa:** 2-3h

---

### 3.4 🟡 Média — Justificar no README por que temporal > aleatório

**Status real:** `README.md` menciona split temporal mas não justifica.

**Plano de execução:**

1. Adicionar seção no README: "Por que Split Temporal?"
   - **Data leakage:** split aleatório mistura passado e futuro na avaliação, superestimando performance
   - **Simulação realista:** treinar com dados passados → prever futuros pedidos
   - **Cenário e-commerce:** sazonalidade, evolução de preferências, novos produtos
   - **Referência:** `docs/GUIDE.md` §4 já contém justificação técnica completa — usar como base

**Dependências:** Nenhuma.
**Complexidade:** Baixa | **Estimativa:** 30min

---

### 3.5 🟡 Média — Definir formalmente estratégia de avaliação top-K

**Status real:** Métricas existem no código (`calculate_metrics_at_k()`) mas não estão documentadas formalmente.

**Plano de execução:**

1. Documentar formalmente em `docs/GUIDE.md` §9:
   - **Cutoffs:** K=5 (conservador), K=10 (padrão indústria), K=20 (leniente)
   - **Estratégia de candidatos:** 1 positivo + 99 negativos aleatórios por usuário
   - **Cold-start:** usuários sem histórico no treino são filtrados da avaliação
   - **Agregação:** média macro sobre todos os usuários
   - **Métricas:** HitRate@K, Recall@K, Precision@K, NDCG@K, MAP@K

**Dependências:** Nenhuma.
**Complexidade:** Baixa | **Estimativa:** 1h

---

### 3.6 🟡 Média — Factory para modelos

**Status real:** Não existe. Cada baseline é instanciado inline no `src/train.py`.

**Plano de execução:**

1. Criar `src/models/factory.py`:
   ```python
   from src.models.popularity import PopularityBaseline
   from src.models.top_rated import TopRatedBaseline
   from src.models.item_cf import ItemItemCF
   from src.train import TruncatedSVDBaseline

   def model_factory(name: str, **kwargs):
       """Instancia um modelo de recomendação pelo nome."""
       models = {
           "popularity": PopularityBaseline,
           "top_rated": TopRatedBaseline,
           "item_cf": ItemItemCF,
           "svd": TruncatedSVDBaseline,
       }
       if name not in models:
           raise ValueError(f"Modelo desconhecido: {name}. Opções: {list(models.keys())}")
       return models[name](**kwargs)
   ```

2. Refatorar `src/train.py` para usar `model_factory`:
   ```python
   for baseline_name in ["popularity", "top_rated", "item_cf", "svd"]:
       model = model_factory(baseline_name, random_state=SEED)
       model.fit(train_df, ...)
       metrics = evaluate_model(model, ...)
   ```

3. Opcional: usar Enum para os nomes dos modelos

**Dependências:** `TruncatedSVD Baseline` (item 3.2), `src/config.py` (item 3.1).
**Complexidade:** Alta | **Estimativa:** 4-6h

---

### 3.7 🟡 Média — Strategy para preprocessors

**Status real:** Não existe. Lógica de split e feature engineering está inline.

**Plano de execução:**

1. Criar `src/data/strategies.py`:
   ```python
   from abc import ABC, abstractmethod
   import pandas as pd

   class SplitStrategy(ABC):
       @abstractmethod
       def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
           ...

   class TemporalSplitStrategy(SplitStrategy):
       def __init__(self, train_size=0.70, val_size=0.15, time_col="days_since_reference"):
           self.train_size = train_size
           self.val_size = val_size
           self.time_col = time_col

       def split(self, df):
           from src.data.splits import temporal_split
           return temporal_split(df, time_col=self.time_col,
                                  train_size=self.train_size,
                                  val_size=self.val_size)

   class RandomSplitStrategy(SplitStrategy):
       def __init__(self, train_size=0.70, val_size=0.15, seed=42):
           self.train_size = train_size
           self.val_size = val_size
           self.seed = seed

       def split(self, df):
           from sklearn.model_selection import train_test_split
           train, temp = train_test_split(df, train_size=self.train_size,
                                            random_state=self.seed)
           val, test = train_test_split(temp, train_size=self.val_size/(1-self.train_size),
                                         random_state=self.seed)
           return train, val, test

   class PreprocessingContext:
       def __init__(self, strategy: SplitStrategy):
           self.strategy = strategy

       def execute(self, df):
           return self.strategy.split(df)
   ```

2. Refatorar `src/train.py` para usar `PreprocessingContext`:
   ```python
   from src.data.strategies import TemporalSplitStrategy, PreprocessingContext

   context = PreprocessingContext(TemporalSplitStrategy())
   train_df, val_df, test_df = context.execute(df)
   ```

**Dependências:** Nenhuma.
**Complexidade:** Alta | **Estimativa:** 4-6h

---

### 3.8 🟡 Média — `scripts/validate_env.py`

**Status real:** Arquivo não existe.

**Plano de execução:**

1. Criar `scripts/validate_env.py`:
   ```python
   #!/usr/bin/env python3
   """Valida que o ambiente tem todas as dependências e configurações necessárias."""

   import sys
   from pathlib import Path

   def check_python_version():
       required = (3, 12)
       actual = sys.version_info[:2]
       if actual < required:
           print(f"[✗] Python {required[0]}.{required[1]}+ necessário, {actual[0]}.{actual[1]} encontrado")
           return False
       print(f"[✓] Python {actual[0]}.{actual[1]}")
       return True

   def check_dependencies():
       required = ["torch", "sklearn", "mlflow", "pandas", "numpy", "scipy", "pydantic_settings"]
       missing = []
       for pkg in required:
           try:
               __import__(pkg)
               print(f"[✓] {pkg}")
           except ImportError:
               print(f"[✗] {pkg} não instalado")
               missing.append(pkg)
       return len(missing) == 0

   def check_directories():
       required_dirs = ["data/processed", "artifacts"]
       all_ok = True
       for d in required_dirs:
           p = Path(d)
           if p.exists():
               print(f"[✓] {d}")
           else:
               print(f"[✗] {d} não existe")
               all_ok = False
       return all_ok

   def check_dvc():
       from pathlib import Path
       if Path(".dvc").exists():
           print("[✓] DVC inicializado")
           return True
       print("[✗] DVC não inicializado")
       return False

   if __name__ == "__main__":
       results = [
           check_python_version(),
           check_dependencies(),
           check_directories(),
           check_dvc(),
       ]
       if all(results):
           print("\n[OK] Ambiente válido!")
           sys.exit(0)
       else:
           print("\n[ERRO] Ambiente inválido — corrija os itens acima")
           sys.exit(1)
   ```

**Dependências:** Nenhuma.
**Complexidade:** Baixa | **Estimativa:** 1h

---

### 3.9 🟡 Média — Promover CSV de baselines para `artifacts/`

**Status real:** Não há arquivo `data/processed/temporary_baseline_recommendations.csv` no repositório.

**Plano de execução:**

1. Verificar se existem CSVs de baselines em `data/processed/`:
   ```bash
   ls data/processed/*.csv
   ```

2. Se existirem:
   - Mover para `artifacts/` com timestamp: `artifacts/baselines/YYYYMMDD/`
   - Adicionar ao `.gitignore`
   - Criar `artifacts/baselines/.gitkeep`

3. Documentar: "outputs de baselines vão para `artifacts/`"

**Dependências:** Nenhuma.
**Complexidade:** Baixa | **Estimativa:** 30min

---

## 4. Ordem de Execução Recomendada

```
FASE 1 (Fundacional — sem dependências)
├── 3.4 README: justificativa temporal    (30min)
├── 3.5 GUIDE.md §9: estratégia top-K     (1h)
├── 3.8 validate_env.py                    (1h)

FASE 2 (Infraestrutura)
├── 3.1 src/config.py                      (2-3h)  ← Depende: Fase 1
└── 3.3 ruff check e zero warnings         (2-3h)

FASE 3 (Pipeline)
├── 3.2 TruncatedSVD Baseline              (2-3h)  ← Depende: 3.1
└── 3.9 CSVs para artifacts                 (30min)

FASE 4 (Design Patterns)
├── 3.6 Factory para modelos                (4-6h)  ← Depende: 3.2
└── 3.7 Strategy para preprocessors         (4-6h)
```

---

## 5. Atualizações Sugeridas no CHECKLIST.md

Itens para marcar como `[x]` (feitos):
- `[0.4] Split temporal (70/15/15) — NÃO implementado` → ✅ Implementado
- `[0.4] Implementar temporal_split() sem leakage` → ✅ Implementado
- `[0.4] Precision@K, Recall@K, NDCG@K, HitRate@K` → ✅ Implementado
- `[0.4] Implementação manual em GUIDE.md §9.2` → ✅ Implementado
- `[0.6] Split temporal aplicado antes de CSR` → ✅ Implementado

Itens para remover (fantasmas — nunca existiram):
- `[0.6] Métricas dummy em evaluate_dummy_metrics()`
- `[0.6] Substituir evaluate_dummy_metrics() por cálculo real`
