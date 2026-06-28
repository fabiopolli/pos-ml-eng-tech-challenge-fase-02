# Auditoria de Features — Análise de Correlação Spearman (Round 2)

**Data:** 2026-06-27
**Método:** Correlação de Spearman pairwise (não-linear, robusta a outliers)
**Threshold:** `|ρ| > 0.95` → candidata a remoção
**Dataset:** `data/processed/interactions_fe.parquet` (99.785 linhas × 42 colunas)

---

## 1. Escopo da Auditoria

Esta é a **segunda rodada** de feature selection (Round 1 = revisão manual de constantes/colineares
durante o FE inicial; ver `feature_metadata.json` → `audit_history[0]`).

**Features auditadas:** 20 numéricas em `configs/selected_features.yaml` (3 embeddings separadas).

## 2. Pares com `|ρ| > 0.95` (Remoção Recomendada)

| Par | Spearman ρ | Decisão | Justificativa |
|---|---|---|---|
| `days_since_reference` ↔ `user_recency_days` | **−0.9861** | Remover `user_recency_days` | Proxies temporais redundantes; `days_since_reference` é o timestamp canônico usado no split temporal |
| `freight_value_log` ↔ `user_avg_freight` | **+0.9657** | Remover `freight_value_log` | Ambos medem frete; `user_avg_freight` é uma agregação de usuário mais robusta |

## 3. Pares com `0.85 < |ρ| ≤ 0.95` (Warning — Mantidos)

| Par | Spearman ρ | Decisão | Justificativa |
|---|---|---|---|
| `payment_type_credit_card` ↔ `payment_type_boleto` | −0.8840 | Manter ambos | OHE columns — multicolinearidade é esperada (1-p) |

## 4. Ações Aplicadas

- ✅ `configs/selected_features.yaml`: `numeric_features` reduzido de 20 → **18 features**
- ✅ `src/training/evaluate.py:_AUX_COLS`: lista reduzida de 20 → **18 features**
- ✅ `data/processed/feature_metadata.json`: adicionado `audit_history[1]` com metadados da rodada

## 5. Re-treinamento do NCF (15 épocas, demais HPs idênticas)

| Modelo | Features | NDCG@K | HitRate@K | Recall@K | MAP@K |
|---|---|---|---|---|---|
| **NCF_FINAL** (original) | 3 emb + 20 aux | 0.2226 | 0.3993 | 0.3914 | 0.1725 |
| **Audit_Spearman_18feat** (novo) | 3 emb + 18 aux | 0.1932 | 0.3413 | 0.3322 | 0.1547 |
| **Ablation_FINAL** (produção atual) | 3 emb only (sem aux) | **0.2725** | 0.4949 | 0.4886 | 0.2081 |

## 6. Conclusões

### 6.1 Remoção das 2 features correlacionadas **NÃO melhorou** a performance

Surpreendentemente, remover `user_recency_days` e `freight_value_log` causou uma **queda de 13.2% no NDCG@K**
(de 0.2226 → 0.1932). Possíveis explicações:

- **Sinal não-linear:** Spearman captura monotonia, mas o NCF aprende relações não-lineares.
  As duas features removidas podem carregar **sinais distintos em regiões específicas** do espaço latente.
- **Ruído vs. sinal:** A correlação alta não implica redundância para o modelo — o gradiente da rede pode
  explorar dimensões que a correlação linear não revela.
- **Regularização implícita:** Mais features podem ajudar a rede a convergir para soluções mais estáveis,
  mesmo que linearmente redundantes.

### 6.2 O modelo Production (`no_aux`) é o melhor

A descoberta mais importante: o **modelo Production atual (`Ablation_FINAL_no_aux_emb32`)**,
que usa **apenas os 3 embeddings** (sem nenhuma feature auxiliar), supera todas as variantes
com aux features:

- vs 20 features: **+22.5% NDCG@K**
- vs 18 features: **+41.0% NDCG@K**

Isto está alinhado com a literatura: modelos neurais baseados em embedding (NCF) frequentemente
superam baselines que usam features engineered quando há **alta esparsidade** (99.997% no Olist)
e **sinal de preferência fraco** (98% dos usuários têm 1 compra).

### 6.3 Recomendação

**Manter `Ablation_FINAL_no_aux_emb32` como Production.** As features auxiliares em geral não
contribuem para o modelo final.

**Reverter a remoção das 2 features de `_AUX_COLS` e `selected_features.yaml`** (opcional),
uma vez que o modelo Production não as usa. A remoção atual não causa impacto negativo em
produção, mas reduz a cobertura para futuros experimentos que desejem usar aux features.

## 7. Métricas-Chave do Item de Checklist

| Métrica | Valor |
|---|---|
| Features removidas (Spearman > 0.95) | 2 (`user_recency_days`, `freight_value_log`) |
| Features mantidas (Spearman 0.85-0.95) | 2 (`payment_type_credit_card`, `payment_type_boleto`) — OHE |
| Δ NDCG@K (20 → 18 features) | **−13.2%** (pior, não melhor) |
| Best NDCG@K (Production no_aux) | **0.2725** |
| Lift vs baseline popularidade | **60×** (sem aux) ou **43×** (com aux) |

## 8. Artefatos Gerados

- `configs/selected_features.yaml` (atualizado)
- `src/training/evaluate.py` (atualizado)
- `data/processed/feature_metadata.json` (atualizado com `audit_history`)
- `artifacts/metrics_Audit_Spearman_18feat.json` (métricas do re-treino)
- `artifacts/metrics_Audit_Spearman_18feat_E15.json` (métricas do re-treino, 15 epochs)
- `artifacts/ncf_Audit_Spearman_18feat.pt` (modelo serializado)
- `reports/feature_audit_spearman.md` (este relatório)
