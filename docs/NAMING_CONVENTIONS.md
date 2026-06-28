# Convenções de Nomenclatura — Olist Recommender System

> **Status:** Vigente a partir de 2026-06-27
> **Aplicação:** Todo código Python em `src/`, `scripts/`, `tests/` e `notebooks/`
> **Lint:** Validado por `ruff` (`uv run ruff check src/ scripts/`)

Este documento define as convenções de nomenclatura adotadas no projeto. Aplicar consistentemente é parte da ETAPA 1 (Clean Code e Estrutura).

---

## 1. Regras Gerais

| Tipo | Convenção | Exemplo |
|---|---|---|
| **Módulos** | `snake_case` (lowercase) | `data_preparation.py`, `train_ncf.py` |
| **Pacotes** | `snake_case` curto, sem underscores | `data`, `models`, `training` |
| **Classes** | `PascalCase` | `PopularityBaseline`, `NCFHybrid`, `SplitContext` |
| **Funções / Métodos** | `snake_case` | `temporal_split()`, `fit()`, `recommend()` |
| **Variáveis** | `snake_case` | `train_df`, `n_users`, `mlflow_uri` |
| **Constantes** | `UPPER_SNAKE_CASE` | `K_VALUES`, `MIN_USER_INTERACTIONS` |
| **Privados** | `_snake_case` (underscore prefix) | `_execute()`, `_build_item_lookups()` |
| **Type aliases** | `PascalCase` | `UserItemsMap = dict[int, set[int]]` |

## 2. Prefixos Semânticos

| Prefixo | Significado | Exemplo |
|---|---|---|
| `is_`, `has_`, `should_` | Booleanos | `is_weekend`, `has_review`, `filter_cold_start` |
| `n_` | Contagem | `n_users`, `n_epochs`, `n_negatives`, `n_components` |
| `k_` | Cutoff K (ranking) | `k_eval`, `k_values` |
| `lr_` | Learning rate | `lr=5e-4` |
| `df_` | DataFrame | `df_train`, `df_test`, `df_eval` |
| `_` | Privado | `_init_results_dict`, `_aggregate_metrics` |
| `ml_` | Machine Learning | `ml_model`, `ml_pipeline` |

## 3. Nomes Específicos do Domínio

| Conceito | Nome padrão | Notas |
|---|---|---|
| Identificador de usuário | `user_id` | Tipo: `int` (codificado) |
| Identificador original de usuário | `customer_unique_id` | Coluna original do Olist |
| Identificador de produto | `product_id` | Tipo: `str` (original) |
| Índice embedding de produto | `product_id_idx` | Tipo: `int` (codificado) |
| Identificador de categoria | `category_id` | Tipo: `int` (codificado) |
| Timestamp de compra | `order_purchase_timestamp` | Coluna original |
| Dias desde referência | `days_since_reference` | Feature temporal usada no split |
| Aux features | `_AUX_COLS`, `aux_*` | Features normalizadas para MLP |
| Item rankeado | `top_k_idx`, `recommended` | Output de `model.recommend()` |

## 4. Sufixos Recomendados

| Sufixo | Uso | Exemplo |
|---|---|---|
| `_df` | DataFrame | `train_df`, `test_df`, `val_df` |
| `_np` | NumPy array | `scores_np`, `labels_np` |
| `_tensor` ou `_t` | PyTorch tensor | `user_tensor`, `item_t` |
| `_map` | Dict de mapeamento | `user_items_map`, `idx_to_item` |
| `_baseline` | Classe de baseline | `PopularityBaseline`, `ItemItemCF` |
| `_strategy` | Strategy pattern | `TemporalSplitStrategy` |
| `_factory` | Factory pattern | `model_factory()` |

## 5. Padrões Especiais (Clean Code)

### Booleanos
✅ `enable_mlflow`, `filter_cold_start`, `has_history`
❌ `mlflow_flag`, `cold_start_filter`, `history`

### Funções
- Devem ser **verbos** ou **frases verbais**: `fit()`, `predict()`, `evaluate_model()`
- Evitar nomes genéricos: ❌ `process()`, `handle()`, `do_stuff()`

### Classes
- Devem ser **substantivos**: `NCFHybrid`, `UserItemsMap`
- Evitar sufixos redundantes: ❌ `ClassNCF`, `ManagerClass`

### Constantes
- Apenas módulos no topo: `K_VALUES = [5, 10, 20]`
- Não usar `const_` ou `k_` como prefixo (snake_case UPPER já é suficiente)

## 6. Auditoria Automática

Ruff aplica automaticamente várias dessas regras via os seguintes checks:

| Regra | Ruff Code | Descrição |
|---|---|---|
| Nomes de variáveis | `N802`, `N803`, `N806` | Funções/variáveis devem ser lowercase |
| Imports | `I001`, `I002` | Imports ordenados |
| Variáveis não usadas | `F841`, `RUF059` | Detecta `_` patterns |
| Type hints | `UP006`, `UP007` | `dict` em vez de `Dict`, `X \| None` em vez de `Optional[X]` |

Para auditar manualmente:
```bash
uv run ruff check src/ scripts/ --select N --output-format=concise
```

## 7. Exemplos Práticos no Projeto

### Bom (seguindo convenções)
```python
# src/train.py
class PopularityBaseline:
    def __init__(self, k: int = 10):
        self.k = k
        self.popular_items: list = []

    def fit(self, df: pd.DataFrame, item_col: str = "product_id") -> "PopularityBaseline":
        counts = df[item_col].value_counts()
        self.popular_items = counts.head(self.k).index.tolist()
        return self

    def recommend(self, user_id: str, user_items: set, n: int | None = None) -> list:
        n = n or self.k
        return self.popular_items[:n]


K_VALUES = [5, 10, 20]  # Constante no topo do módulo
```

### Ruim (violando convenções)
```python
class Popularity_baseline:  # ❌ snake_case em classe
    def __init__(self, K=10):  # ❌ K maiúsculo, sem type hint
        self.K = K
        self.PopularItems = []  # ❌ PascalCase em variável

    def Fit(self, DataFrame):  # ❌ PascalCase em método, sem type hints
        pass
```

## 8. Exceções Documentadas

Por design, algumas funções são **longas por necessidade**:

| Função | Arquivo | Linhas | Justificativa |
|---|---|---|---|
| `temporal_split()` | `src/data/splits.py` | 56 | Lógica de split com assertions anti-leakage |
| `main()` | Scripts entry points | 86 | Orquestração de pipelines |
| `eda.py` funções | `src/eda.py` | 40-90 | Geração de relatórios (one-shot) |

Estas exceções são **aceitáveis** desde que:
- Função única entrada do módulo (`main()`)
- Lógica linear e declarativa (sem branches complexos)
- Com documentação adequada

## 9. Checklist de Nomenclatura ao Adicionar Código

Antes de abrir PR, valide:

- [ ] Classes em `PascalCase`
- [ ] Funções/variáveis em `snake_case`
- [ ] Constantes em `UPPER_SNAKE_CASE`
- [ ] Booleanos com prefixo `is_/has_/should_`
- [ ] Contagens com prefixo `n_`
- [ ] DataFrames com sufixo `_df`
- [ ] Privados com prefixo `_`
- [ ] Type hints em todas as assinaturas públicas
- [ ] `ruff check .` passa

---

*Documento vivo. Última atualização: 2026-06-27*
