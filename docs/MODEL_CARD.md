# Model Card — Sistema de Recomendação Olist E-Commerce

## 1. Visão Geral

### Modelos Documentados
- **NCFHybrid** — Rede neural neural collaborative filtering com PyTorch para recomendação de produtos.
- **TruncatedSVD** — Decomposição SVD como baseline de produção.
- **Popularidade, Top-Rated, Item-Item CF** — Baselines de referência.

### Objetivo
Prever e recomendar produtos para clientes de e-commerce baseado no histórico de interações (implicit feedback) para aumentar conversão e satisfação do cliente.

### Tipo de Problema
Ranking implícito de itens: recomendar top-K produtos mais relevantes para cada usuário baseado em embeddings aprendidos e similaridade de item.

### Responsáveis
- **Fábio Polli** —  RM373870.
- **Willian Kopp** — RM370703.
- **Romário Silva** — RM372243.
- **Denis Barros Melo** — RM370416.

### Data de Criação
Julho 2026

### Versão do Documento
1.0

## 2. Uso do Modelo

### Aplicações Recomendadas
- Geração de ranking de recomendações em tempo real na plataforma e-commerce.
- Personalização de feeds de produtos para cada usuário.
- Priorização de produtos em campanhas de marketing direcionadas.
- Análise de preferências de categorias de produtos por segmento de usuário.

### Aplicações Não Recomendadas
- Uso em domínios fora de e-commerce (ex.: redes sociais, streaming) sem re-treinamento.
- Decisões automáticas de alto impacto (ex.: desativação de contas) sem revisão humana.
- Cenários com dados muito antigos ou com distribuição muito diferente do treino (2016-2018).
- Recomendações para usuários com histórico muito limitado sem estratégia de cold-start.

### Consumo via API
- **Endpoint**: `POST /api/v1/recommend`
- **Host padrão**: `http://localhost:8000`
- **Payload de exemplo**:
  ```json
  {
    "user_id": 12345,
    "top_k": 10,
    "exclude_categories": ["eletrônicos"]
  }
  ```
- **Resposta esperada**:
  ```json
  {
    "user_id": 12345,
    "recommendations": [
      {"product_id": 54321, "rank": 1, "score": 0.45},
      {"product_id": 54322, "rank": 2, "score": 0.42},
      ...
    ],
    "model_version": "Ablation_FINAL_no_aux_emb32",
    "timestamp": "2026-07-04T10:30:00Z"
  }
  ```

## 3. Dados

### Fonte e Contexto
- **Dataset**: Olist Brazilian E-Commerce Public Dataset
- **Localização**: `data/raw/`
- **Período**: 2016-2018 (dados históricos públicos)
- **Volume**: 99.785 interações user-item (pedidos) | 93.358 usuários únicos | 32.216 produtos | 72 categorias

### Variáveis Utilizadas (Production)
**Embeddings (aprendidos)**:
- `user_id` — Embedding do usuário (dim=32)
- `product_id_idx` — Embedding do produto (dim=32)
- `category_id` — Embedding da categoria (dim=8)

**Features auxiliares** (ablação: removidos em production):
- Antes: 20 features (price_log, freight_value_log, review_score, etc.)
- Atual: **sem features auxiliares** — apenas embeddings

### Distribuição de Interações
- **Treino**: 69.849 (70%)
- **Validação**: 14.968 (15%)
- **Teste**: 14.968 (15%)
- **Padrão de split**: Temporal (baseado em `days_since_reference`)
- **Cold-start**: 98.4% dos usuários no teste são inéditos no treino

### Pré-processamento
- **Schema validation** com Pydantic
- **Negative sampling on-the-fly** (8 negativos por positivo positivo)
- **BPR Loss** para ranking implícito
- **Gradient clipping** (max_norm=5.0)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)
- Implementação em `src/data/preprocessing.py` e `src/training/train.py`

## 4. Métricas de Avaliação

### Métricas Principais (Test Set — Run Production)
- **NDCG@10**: 0.2725
- **MAP@10**: 0.2081
- **Recall@10**: 0.4886
- **HitRate@10**: 0.4949
- **Precision@10**: 0.0509

### Comparativo de Modelos

| Modelo | NDCG@10 | MAP@10 | Recall@10 | HitRate@10 | Lift vs Popularidade |
|--------|---------|--------|-----------|------------|---------------------|
| NCFHybrid (Production ⭐) | **0.2725** | **0.2081** | **0.4886** | **0.4949** | **60.6×** |
| NCFHybrid (com side features) | 0.2226 | 0.1725 | — | 0.3993 | 49.2× |
| TruncatedSVD | 0.1829 | 0.1459 | — | 0.3242 | 40.4× |
| Top-Rated Baseline | 0.1634 | 0.1151 | — | 0.3311 | 36.1× |
| Item-Item CF | — | — | — | — | — |
| Popularidade (baseline) | 0.0045 | 0.0031 | 0.0096 | 0.0099 | 1.0× |

### Curvas de Aprendizado
- **Train NDCG@10**: 0.5827 (sanity check — modelo aprendeu padrões vistos)
- **Validation NDCG@10**: 0.3506 (generalização em warm-start users)
- **Test NDCG@10**: 0.2725 (queda esperada por cold-start massivo)
- **Train/Val/Test gap**: consistente com 98% de cold-start no teste

### Métrica de Negócio
- **Taxa de hit**: ~49.5% das recomendações contêm pelo menos 1 produto futuro
- **Lift estimado**: 60.6× melhor que baseline de popularidade
- Impacto em receita deve ser validado em A/B test antes de roll-out completo

## 5. Arquitetura de Modelo

### NCFHybrid (Production)
- **Framework**: PyTorch
- **Estrutura**:
  - Embedding Layer (user: dim=32, product: dim=32, category: dim=8)
  - Concatenação de embeddings
  - MLP tower: [64, 32]
  - ReLU com Dropout(0.5) entre layers
  - Output: dot product (user_emb · item_emb) para score final
- **Hiperparâmetros principais**:
  - Learning rate: 5e-4
  - Batch size: 2048
  - Épocas máx: 20
  - Otimizador: AdamW
  - Weight decay: 5e-4
  - Loss: BPR (Bayesian Personalized Ranking)
  - Early stopping: patience=3, métrica=val_NDCG@10

### TruncatedSVD (Baseline)
- **Framework**: scikit-learn
- **Parâmetros**: `n_components=64`, `random_state=42`
- **Procedimento**: SVD sobre matriz user-item, scoring por similaridade cosseno

### Baselines Adicionais
- **Popularidade**: ranking por frequência de ocorrência no treino
- **Top-Rated**: ranking por média de review_score
- **Item-Item CF**: similaridade cosseno entre vetores bag-of-words de usuários que compraram cada item

### Dependências Principais
- PyTorch (2.0+)
- scikit-learn (1.3+)
- pandas, numpy
- MLflow (2.5+)
- DVC (3.0+)
- Streamlit (visualização)

## 6. Limitações e Riscos

### Limitações Técnicas
- Treinado especificamente em dados 2016-2018; generalização para períodos mais recentes pode ser limitada.
- Cold-start massivo (98.4% usuários desconhecidos no teste) reduz performance absoluta; estratégia de fallback necessária.
- Tamanho pequeno do dataset (99k interações) em relação a sistemas reais de produção (milhões).
- Ausência de contexto temporal recente e dados de sessão (apenas histórico agregado).
- Não captura mudanças de preferência ao longo do tempo.

### Vieses Identificados
- **Por categoria**: produtos em categorias populares tendem a ser recomendados mais frequentemente.
- **Por usuário**: usuários com muito histórico de compra recebem recomendações mais personalizadas; novos usuários recebem recomendações genéricas.
- **Cold-start**: novo usuário sem histórico receberá ranking de popularidade ou fallback para baseline.
- Risco de "filter bubble" — usuários verão majoritariamente categorias similares ao seu histórico.

### Riscos de Privacidade
- Embeddings de usuários podem ser invertidos para inferir preferências privadas.
- Recomenda-se anonimização de user_id em logs e auditorias.
- Compliance com LGPD/GDPR para retenção e deleção de dados de usuário.

### Cenários de Falha
- Falha no carregamento de embeddings → fallback para TruncatedSVD
- Entrada com `user_id` não visto → estratégia cold-start (recomendações populares)
- Queda de performance superior a 10% em NDCG → disparar re-treinamento e revisão

## 7. Monitoramento e Manutenção

### Monitoramento
- **MLflow Tracking** para métricas de treino, validação e teste (experiment: `Olist_NCF_Optimization`)
- **Streamlit Dashboard** (`front/app_vis.py`) para exploração de recomendações
- **Métricas operacionais**:
  - Latência P95 < 100 ms (embedding lookup + forward pass)
  - Cobertura de produtos: % de produtos únicos recomendados (evitar monopolização)
  - Hit rate semanal mantido estável > 0.48
  - Taxa de cold-start detectada e tratada

### Re-treinamento
- **Frequência**: mensal ou quando hit rate cair abaixo de 0.45
- **Procedimento sugerido**:
  1. Coletar novos logs de interação (últimos 30 dias)
  2. Executar `python scripts/train_ncf.py --config configs/ncf_best.yaml`
  3. Avaliar run em MLflow UI
  4. Se NDCG@10 > 0.27, promover para Staging
  5. A/B test em 10% do tráfego por 1 semana
  6. Se lift positivo, promover para Production

### Versionamento
- Modelos registrados no **MLflow Model Registry** (`Ablation_FINAL_no_aux_emb32` em Production)
- Fallback local em `models/ncf_final.pt` quando MLflow Registry indisponível
- Histórico de runs rastreável em MLflow UI

## 8. Contato

- **Documentação técnica**: `docs/GUIDE.md`, `docs/REPORT.md`, `reports/ncf_optimization_report.md`
- **Repositório**: GitHub (ver GUIDE.md para setup)
- **Dashboard**: `streamlit run front/app_vis.py`
- **Suporte**: abrir issue no repositório para incidentes ou melhorias

---

**Última atualização**: Julho 2026 | **Modelo em Produção**: `Ablation_FINAL_no_aux_emb32`
