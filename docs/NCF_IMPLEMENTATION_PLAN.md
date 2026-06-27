# Plano de Implementação: NCF Híbrido com Side-Information

## Visão Geral

Implementação completa de um sistema de recomendação baseado em **Neural Collaborative Filtering (NCF)** com side-information, utilizando PyTorch, DVC e MLflow. O plano está dividido em **4 etapas sequenciais**, cada uma com entregáveis bem definidos.

**Dataset:** `data/processed/interactions_fe.parquet` (99.785 interações × 42 features)

**Arquitetura Alvo:**
```
[User ID] → Embed(32) ──────┐
[Product ID] → Embed(32) ───┤
[Category ID] → Embed(8) ───┤
[20 Aux Features] → Scaler ─┴── Concat → MLP → Score
```

---

## Etapa 1 — Seleção e Análise de Features

**Objetivo:** Selecionar e validar as features que alimentarão o modelo NCF.

### 1.1. Inventário de Features Disponíveis (42 colunas)

| Categoria | Features | Cardinalidade |
|-----------|----------|---------------|
| **Identificadores** | `user_id`, `product_id_idx`, `category_id` | 93k / 32k / 72 |
| **Feedback** | `review_score`, `has_review` | Contínua / Binária |
| **Preço/Logística** | `price`, `price_log`, `freight_value`, `freight_value_log`, `price_to_freight_ratio`, `has_price_outlier` | Contínua |
| **Temporal** | `purchase_year`, `purchase_month`, `purchase_day_of_week`, `purchase_hour`, `purchase_day_of_month`, `is_weekend`, `is_holiday_season`, `days_since_reference` | Discreta |
| **Categoria** | `category_target_enc`, `category_frequency`, `category_is_top10`, `category_is_rare` | Contínua/Binária |
| **Pagamento** | `payment_type_credit_card`, `payment_type_boleto`, `payment_type_voucher`, `payment_type_debit_card` | Binária (OHE) |
| **Agregações User** | `user_total_purchases`, `user_avg_price`, `user_avg_freight`, `user_purchase_span_days`, `user_recency_days`, `user_review_rate` | Contínua |
| **Agregações Product** | `product_popularity`, `product_unique_buyers`, `product_avg_review_score`, `product_avg_price`, `product_avg_freight`, `product_review_rate` | Contínua |

### 1.2. Análise de Correlação e Multicolinearidade

**Ações:**
- Gerar matriz de correlação completa (Spearman para não-lineares)
- Identificar pares com `|r| > 0.9`:
  - `user_avg_price` ↔ `price` (r=0.99) — remover `user_avg_price`
  - `product_avg_price` ↔ `price` (r=1.00) — remover `product_avg_price`
  - `product_unique_buyers` ↔ `product_popularity` (r=1.00) — remover `product_unique_buyers`
  - `user_review_rate` ↔ `has_review` (r=0.99) — remover `user_review_rate`
- Manter features com maior interpretabilidade preditiva

### 1.3. Análise de Importância (Feature Selection)

**Métodos combinados:**
1. **Mutual Information** — capturar relações não-lineares com o target
2. **Random Forest feature importance** — validação cruzada
3. **Análise univariada** — distribuição e variância

### 1.4. Features Finais Selecionadas (20 features)

```python
NUMERIC_FEATURES = [
    # Preço/Logística (4)
    'price_log', 'freight_value_log', 'price_to_freight_ratio',
    'has_price_outlier',
    
    # Temporal (3) - agregadas
    'days_since_reference', 'is_weekend', 'is_holiday_season',
    
    # Categoria (2)
    'category_target_enc', 'category_frequency',
    
    # Pagamento (4) - já OHE
    'payment_type_credit_card', 'payment_type_boleto',
    'payment_type_voucher', 'payment_type_debit_card',
    
    # User Profile (4)
    'user_total_purchases', 'user_avg_freight',
    'user_purchase_span_days', 'user_recency_days',
    
    # Product Profile (3)
    'product_popularity', 'product_avg_review_score',
    'product_review_rate',
]
```

**Entregáveis Etapa 1:**
- [ ] Notebook `04_feature_selection.ipynb` com análise de correlação
- [ ] Lista final de features em `configs/selected_features.yaml`
- [ ] Justificativa documentada para cada exclusão

---

## Etapa 2 — Feature Engineering e Pipeline de Dados

**Objetivo:** Construir o pipeline PyTorch-ready com normalização, split temporal e negative sampling.

### 2.1. Pré-processamento Numérico

**StandardScaler (fit apenas no treino):**
- Aplicar em todas as 20 features numéricas
- Salvar scaler em `artifacts/scaler.pkl`
- Verificar que `train_mean ≈ 0` e `train_std ≈ 1` pós-transform

**Tratamento de outliers:**
- `has_price_outlier` já é flag — manter
- `days_since_reference`: já está em escala de dias (não aplicar log)
- Winsorização opcional para `user_recency_days` (p99)

### 2.2. Split Temporal Estratificado

**Proporções:** 70% train / 15% val / 15% test

**Algoritmo:**
```python
def temporal_split(df, time_col, train_size=0.70, val_size=0.15):
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    return df_sorted.iloc[:train_end], df_sorted.iloc[train_end:val_end], df_sorted.iloc[val_end:]
```

**Validações rigorosas:**
- `assert train_max_date <= val_min_date` (sem leakage)
- `assert val_max_date <= test_min_date`
- Verificar que todos os 93k usuários aparecem em pelo menos um split

### 2.3. Negative Sampling

**Estratégia:** On-the-fly no DataLoader (não persistir negativos)

```python
class ImplicitFeedbackDataset(Dataset):
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        neg_item = np.random.choice(self.all_items)
        while neg_item in self.user_items[user]:
            neg_item = np.random.choice(self.all_items)
        return user, pos_item, neg_item, aux_features
```

**Parâmetros:**
- `n_negatives = 1` por positivo (suficiente para BPR)
- `num_workers = 4` no DataLoader
- `pin_memory = True` se GPU disponível

### 2.4. Validação do Pipeline

**Sanity checks:**
- [ ] Dimensões dos tensores: `(batch, 1)` para IDs, `(batch, 20)` para aux
- [ ] Tipos: `torch.long` para IDs, `torch.float32` para features
- [ ] Sem NaN/Inf nas features normalizadas
- [ ] Negative sampling nunca retorna item já visto pelo usuário

**Entregáveis Etapa 2:**
- [ ] `src/data/dataset.py` com `ImplicitFeedbackDataset`
- [ ] `src/data/splits.py` com `temporal_split()`
- [ ] `src/data/preprocessing.py` com `fit_scaler()` / `transform_features()`
- [ ] `artifacts/scaler.pkl` salvo
- [ ] `artifacts/feature_stats.json` (mean, std, min, max)
- [ ] Testes unitários em `tests/test_dataset.py`

---

## Etapa 3 — Modelo NCF Baseline (Primeira Versão)

**Objetivo:** Implementar e treinar a primeira versão funcional do NCF híbrido.

### 3.1. Arquitetura do Modelo

```python
class NCFHybrid(nn.Module):
    def __init__(self, n_users, n_items, n_categories, n_aux_features,
                 emb_dim=32, cat_emb_dim=8,
                 hidden=[128, 64, 32], dropout=0.3):
        super().__init__()
        
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.cat_emb = nn.Embedding(n_categories, cat_emb_dim)
        
        input_dim = (emb_dim * 2) + cat_emb_dim + n_aux_features
        
        layers = []
        for h in hidden:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    
    def forward(self, user, item, category, aux_features):
        u = self.user_emb(user)
        i = self.item_emb(item)
        c = self.cat_emb(category)
        x = torch.cat([u, i, c, aux_features], dim=-1)
        return self.mlp(x).squeeze(-1)
```

### 3.2. Loss Function (BPR)

```python
def bpr_loss(pos_scores, neg_scores):
    return -F.logsigmoid(pos_scores - neg_scores).mean()
```

**Justificativa:** Maximiza diferença entre score positivo e negativo. Adequada para feedback implícito (não temos nota, apenas compra).

### 3.3. Loop de Treinamento

**Hiperparâmetros iniciais (conservadores):**
- `emb_dim = 16`
- `hidden = [64, 32]`
- `dropout = 0.3`
- `lr = 1e-3`
- `batch_size = 1024`
- `epochs = 20`
- `weight_decay = 1e-5`
- `patience = 3` (early stopping)

**Estrutura do loop:**
```python
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        user, pos_item, neg_item, cat_pos, cat_neg, aux = batch
        pos_scores = model(user, pos_item, cat_pos, aux)
        neg_scores = model(user, neg_item, cat_neg, aux)
        loss = bpr_loss(pos_scores, neg_scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validação
    val_metrics = evaluate(model, val_loader, k=10)
    early_stopping(val_metrics['NDCG@10'])
```

### 3.4. Métricas de Avaliação

**4 métricas obrigatórias:**
1. **Recall@10** — fração de itens relevantes encontrados
2. **NDCG@10** — qualidade do ranking
3. **Hit Rate@10** — pelo menos um acerto no top-10
4. **MAP@10** — precisão média ponderada por posição

**Implementação:**
- Funções vetorizadas em `src/training/evaluate.py`
- Validação em par (positivo, negativo) — usar 99 negativos por usuário para avaliar ranking real

### 3.5. MLflow Tracking (Run #1)

```python
with mlflow.start_run(run_name="NCF_baseline_v1"):
    mlflow.log_params({
        "emb_dim": 16, "hidden": [64, 32], "dropout": 0.3,
        "lr": 1e-3, "batch_size": 1024, "n_negatives": 1
    })
    mlflow.log_metrics({"Recall@10": ..., "NDCG@10": ..., "HitRate@10": ..., "MAP@10": ...})
    mlflow.pytorch.log_model(model, "model")
```

**Critérios de sucesso Etapa 3:**
- [ ] Loss converge (não explode, não estagna)
- [ ] Train NDCG@10 > val NDCG@10 (sinal de aprendizado)
- [ ] Métricas de validação superiores ao baseline Most Popular
- [ ] 1 run registrada no MLflow com sucesso

**Entregáveis Etapa 3:**
- [ ] `src/models/ncf.py` — classe `NCFHybrid`
- [ ] `src/models/losses.py` — `bpr_loss`
- [ ] `src/training/train.py` — loop completo
- [ ] `src/training/evaluate.py` — métricas @K
- [ ] `scripts/train_ncf.py` — entry point CLI
- [ ] Modelo baseline treinado salvo em `artifacts/ncf_v1.pt`
- [ ] Run MLflow registrada

---

## Etapa 4 — Otimização e Refinamento

**Objetivo:** Atingir a melhor performance possível via tuning sistemático e ablation.

### 4.1. Hyperparameter Tuning (Runs #2 e #3 no MLflow)

**Grid de busca focado (3 runs obrigatórias):**

| Run | emb_dim | hidden | dropout | lr | batch_size |
|-----|---------|--------|---------|-----|------------|
| 1 (baseline) | 16 | [64, 32] | 0.3 | 1e-3 | 1024 |
| 2 (mais capacidade) | 32 | [128, 64, 32] | 0.3 | 1e-3 | 1024 |
| 3 (regularizado) | 64 | [256, 128, 64] | 0.4 | 5e-4 | 2048 |

**Variáveis monitoradas entre runs:**
- Convergence speed (épocas até convergir)
- Train/val gap (sinal de overfitting)
- NDCG@10 final
- Tempo de treinamento por época

### 4.2. Técnicas de Otimização Avançada

**a) Learning Rate Scheduling:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)
```

**b) Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**c) Weight Decay (L2):**
- Aumentar para `1e-4` se overfitting severo
- Aplicar apenas em `mlp`, não em embeddings (preserva semântica)

**d) Embedding Dropout:**
```python
nn.Dropout(0.1)  # aplicado após embedding lookup
```

### 4.3. Ablation Study

**Comparar modelos:**
1. NCF só com embeddings (sem aux features) — isolar ganho das features
2. NCF com embeddings + top-10 features (mutual info)
3. NCF com todas as 20 features (modelo final)

### 4.4. Análise de Erros

**Diagnósticos pós-treino:**
- Distribuição de scores para positivos vs negativos
- Análise por segmento de usuário (1 compra, 2-5 compras, 6+)
- Análise por categoria (top-10 vs raras)
- Cold-start: usuários do test sem treino

**Fallback strategy:**
- Para usuários novos: recomendar Top-K popularidade global
- Para produtos novos: embedding inicialização aleatória + flag

### 4.5. Promoção no MLflow Model Registry

```python
# Registrar melhor modelo
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, "olist_ncf_recommender")

# Promover para Production
client.transition_model_version_stage(
    name="olist_ncf_recommender",
    version=best_version,
    stage="Production"
)
```

**Critérios de sucesso Etapa 4:**
- [ ] 3 runs completas no MLflow com hiperparâmetros variados
- [ ] Melhor modelo identificado e promovido para `Production`
- [ ] NDCG@10 ≥ 0.10 (meta mínima, superar baselines clássicos)
- [ ] Ablation study documentado mostrando valor das side features
- [ ] Análise de erros com hipóteses para iterações futuras

**Entregáveis Etapa 4:**
- [ ] `configs/ncf_best.yaml` — hiperparâmetros do modelo final
- [ ] `artifacts/ncf_final.pt` — modelo promovido
- [ ] `reports/ncf_optimization_report.md` — comparação de runs
- [ ] `notebooks/05_ncf_results.ipynb` — visualizações e ablation
- [ ] Model Registry atualizado com versão Production

---

## Cronograma e Dependências

```
Etapa 1 ──► Etapa 2 ──► Etapa 3 ──► Etapa 4
  Features   Pipeline    Baseline    Tuning
   (2h)       (3h)        (4h)        (6h)
```

**Total estimado:** ~15 horas de desenvolvimento iterativo.

## Estrutura Final de Arquivos

```
pos-ml-eng-tech-challenge-fase-02/
├── src/
│   ├── data/
│   │   ├── dataset.py          # ImplicitFeedbackDataset
│   │   ├── splits.py           # temporal_split
│   │   └── preprocessing.py    # scaler, transformações
│   ├── models/
│   │   ├── ncf.py              # NCFHybrid
│   │   └── losses.py           # BPR
│   ├── training/
│   │   ├── train.py            # Loop principal
│   │   └── evaluate.py         # Métricas @K
│   └── utils/
│       └── mlflow_utils.py
├── configs/
│   ├── selected_features.yaml  # Lista final 20 features
│   └── ncf_best.yaml           # HP do modelo final
├── artifacts/
│   ├── scaler.pkl
│   ├── feature_stats.json
│   ├── ncf_v1.pt               # Baseline
│   └── ncf_final.pt            # Modelo production
├── reports/
│   └── ncf_optimization_report.md
├── notebooks/
│   ├── 04_feature_selection.ipynb
│   └── 05_ncf_results.ipynb
├── scripts/
│   └── train_ncf.py           # CLI entry point
├── tests/
│   ├── test_dataset.py
│   ├── test_losses.py
│   └── test_metrics.py
└── docs/
    └── NCF_IMPLEMENTATION_PLAN.md  # Este arquivo
```

## Critérios de Aceitação Global

- [ ] **Reprodutibilidade:** `uv run python scripts/train_ncf.py` reproduz modelo final
- [ ] **Performance:** NDCG@10 superior ao baseline TruncatedSVD
- [ ] **Tracking:** 3+ runs no MLflow com variação documentada
- [ ] **Produção:** Modelo registrado e promovido no MLflow Registry
- [ ] **Qualidade:** Ruff + mypy + pytest passando
- [ ] **Documentação:** `notebooks/05_ncf_results.ipynb` com análise completa

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Overfitting por sparsity | Alto | Dropout 0.3-0.5, weight decay, early stopping |
| Cold-start severo | Médio | Embedding inicialização Xavier, fallback popularidade |
| Treinamento instável | Médio | Gradient clipping, learning rate scheduling |
| Tempo de treinamento | Médio | Reduzir `emb_dim`, batch size maior |
| Memory issues (matriz esparsa) | Baixo | Manter uso de `csr_matrix` para baselines |

## Referências

- He et al. (2017). *Neural Collaborative Filtering*. arXiv:1708.05031
- Rendle et al. (2009). *BPR: Bayesian Personalized Ranking*. UAI 2009
- Aggarwal (2016). *Recommender Systems: The Textbook*, Chapter on CF
- PyTorch Docs: `torch.nn.Embedding`, `torch.optim.AdamW`