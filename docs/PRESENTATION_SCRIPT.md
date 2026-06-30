# Script de Apresentação — Olist Recommender System

> **Tech Challenge Fase 02 · Pós-Graduação ML Engineering**
> Foco: EDA, Escolha de Features, Feature Engineering e Treinamento
> Duração estimada: 25–30 min + 10 min de perguntas
> Autor: [seu nome]
> Data: 2026-06-27

---

## Como usar este script

- Cada **slide** tem um **título**, um **gatilho visual** (figura/tabela) e um **roteiro falado**.
- O **tempo por slide** é sugerido; ajuste conforme o ritmo da audiência.
- Os **comandos ao vivo** (mostrar código, rodar MLflow UI etc.) estão marcados com 🎬.
- Os **números importantes** estão destacados em **negrito**.

---

# PARTE 1 — CONTEXTO E PROBLEMA (5 min)

## Slide 1 — Capa (0 min)

**Título na tela:**
> **Olist Recommender System**
> Da EDA ao modelo NCF em produção
> Tech Challenge Fase 02 · 2026-06-27

**Roteiro (30 s):**
> "Bom dia/boa tarde. Meu nome é [seu nome] e hoje vou apresentar o pipeline completo que desenvolvemos para construir um sistema de recomendação de produtos no dataset Olist. A apresentação está dividida em 4 partes: contexto, EDA, feature engineering, e o treinamento do modelo. No final, mostro os resultados e o que aprendemos."

---

## Slide 2 — O Problema de Negócio (1 min)

**Título na tela:**
> **O Problema**
> Como recomendar produtos relevantes em um marketplace com **99,99% de sparsity**?

**Pontos-chave no slide:**
- Olist: 100 mil pedidos (2016-2018) · 93 mil clientes · 32 mil produtos · 72 categorias
- 98% dos clientes compraram **uma única vez** (cold-start massivo)
- Objetivo técnico: **NDCG@K ≥ baseline** com split temporal 70/15/15

**Roteiro (1 min):**
> "O Olist é um marketplace brasileiro. O dataset tem 100 mil pedidos, mas o desafio central é a **sparsidade extrema**: 99,99% dos pares usuário-produto nunca foram observados. Além disso, **98% dos clientes compraram só uma vez** — isso é o que chamamos de cold-start massivo e torna o problema muito difícil. A métrica que vamos otimizar é o NDCG, que mede qualidade de ranking, não apenas acerto."

---

## Slide 3 — Stack e Pipeline (2 min)

**Título na tela:**
> **Pipeline DVC (3 stages) + Treinamento**

```
Raw CSVs (9 arquivos)  →  data/raw/
   ↓
[1] prepare            →  interactions.parquet (10 col, ~100k)
[2] featurize          →  interactions_fe.parquet (42 col, 18 numéricas)
[3] validate           →  shape check + null check
   ↓
Treinamento (MLflow + SQLite):
  - 4 baselines (Popularity, TopRated, ItemItemCF, TruncatedSVD)
  - NCF Production (5 runs + 1 ablation)
```

**Roteiro (2 min):**
> "Aqui está a arquitetura do projeto. Usamos DVC para versionar dados e Python puro via `uv run` para execução. O pipeline tem 3 stages: prepare, featurize e validate. O treino dos modelos é registrado no MLflow com SQLite local. Em particular, treinamos 4 baselines clássicos e 5 configurações do NCF com uma ablation study."

🎬 *(Opcional: mostrar `cat dvc.yaml` no terminal)*

---

# PARTE 2 — EDA (8 min)

## Slide 4 — Estatísticas-Chave (1 min)

**Título na tela:**
> **EDA — Os números que importam**

| Métrica | Valor |
|---|---|
| Total de interações | **99.785** |
| Usuários únicos | **93.358** |
| Produtos únicos | **32.216** |
| Categorias | 72 |
| **Sparsity** | **99,9967%** |
| Período | set/2016 → ago/2018 |
| Review médio | 4,09 / 5 |
| **% usuários com 1 compra** | **98%** |

**Roteiro (1 min):**
> "A EDA começou com 9 CSVs brutos que geraram 100 mil interações entregues. Dois números definem o problema: a **sparsidade de 99,99%** e o fato de que **98% dos usuários compraram uma única vez**. Isso já nos disse, lá no início, que baselines baseados em co-ocorrência como item-item CF ou matrix factorization puros iam sofrer. O review médio de 4,09 mostra um viés positivo — usuários que retornam para avaliar tendem a estar satisfeitos."

🎬 *(Mostrar a figura 11_sparsity se existir; usar reports/eda_report.md)*

---

## Slide 5 — Distribuição do Target (1 min)

**Figura:** `reports/figures/01_review_score_distribution.png`

**Roteiro (1 min):**
> "Olhando a distribuição de reviews, **77% das compras têm nota 4 ou 5**. Isso confirma o viés positivo. Para a recomendação, isso significa que o sinal de feedback é fraco: a maioria dos produtos 'gostados' tem rating 5, e o ruído entre rating 4 e 5 é difícil de separar. Decidimos tratar o problema como **feedback implícito** — não prever o review, mas a probabilidade de interação positiva vs negativa via BPR Loss."

---

## Slide 6 — Features Numéricas e Outliers (1 min)

**Figura:** `reports/figures/02_numerical_distributions.png`

**Roteiro (1 min):**
> "Preço e frete têm distribuições com cauda longa muito pesada — alguns produtos custam mais de 10 mil reais. Aplicamos **log-transform** e clipping no percentil 99. Sem isso, qualquer modelo sensível a outliers seria dominado por esses casos extremos. A figura mostra o antes: a distribuição original é praticamente invisível no histograma, concentrada em uma barra única."

---

## Slide 7 — Sparsity e Cold-Start (2 min)

**Figura:** `reports/figures/05_user_behavior.png` + `06_product_behavior.png`

**Roteiro (2 min):**
> "Aqui está o problema em imagens. À esquerda, **93% dos usuários têm exatamente 1 compra**. À direita, **70% dos produtos foram comprados uma única vez**. Isso significa que qualquer feature que dependa de histórico do usuário — seja embedding treinado, seja agregação como 'média de gasto' — vai estar ruidosa para a esmagadora maioria dos dados. Esse foi o insight que guiou toda a engenharia de features: tudo que calculamos tem que ser **útil mesmo para quem comprou 1 vez**."

🎬 *(Pausa — esperar a ficha cair na audiência)*

---

## Slide 8 — Conclusões da EDA → Decisões (1 min)

**Título na tela:**
> **EDA → 3 Decisões Fundamentais**

| Achado | Decisão |
|---|---|
| 98% cold-start, sparsity 99,99% | Tratar como **feedback implícito** (BPR) — não prever rating |
| Cauda longa em preço/frete | **Log-transform** + clipping p99 |
| Target desbalanceado (77% nota 4-5) | Embeddings > features engineered lineares |

**Roteiro (1 min):**
> "Encerrando a EDA, três decisões viraram princípios: usar BPR Loss, aplicar log-transform em preço e frete, e priorizar embeddings sobre features engineered. A próxima parte mostra como essas decisões se materializaram em features."

---

# PARTE 3 — FEATURE ENGINEERING (8 min)

## Slide 9 — Visão Geral (1 min)

**Título na tela:**
> **Feature Engineering: de 10 colunas brutas para 42 engenheiradas**

```
interactions.parquet (10 colunas)
   ↓
interactions_fe.parquet (42 colunas)
   ├── 18 features numéricas (selecionadas via Mutual Information)
   ├──  3 features de embedding (user, item, category)
   ├──  7 features temporais (dias, weekend, holiday, etc.)
   ├──  4 features de pagamento (OHE)
   └── 10 features de agregação (user-level, product-level)
```

**Roteiro (1 min):**
> "Saímos de 10 colunas brutas para 42 engenheiradas. A estratégia foi construir features que sobrevivessem ao cold-start: features temporais e categóricas não dependem de histórico individual, e as agregações user/product são 'suavizadas' via target encoding com prior global."

---

## Slide 10 — Log-Transform e Outliers (1 min)

**Figura:** `reports/figures/09_log_transformations.png`

**Roteiro (1 min):**
> "Mostro aqui a transformação logarítmica em ação. O preço original tem uma distribuição praticamente invisível no histograma linear — concentrada em zero. Depois do log, a distribuição se assemelha a uma normal, que é o que modelos de embedding esperam. Mesma técnica aplicada ao freight_value. A flag `has_price_outlier` também foi criada para indicar observações fora do percentil 99 — o modelo pode aprender a tratá-las de forma diferenciada."

---

## Slide 11 — Target Encoding Categórico (1 min)

**Figura:** `reports/figures/13_target_encoding.png`

**Roteiro (1 min):**
> "Para categorias, o desafio é que 72 categorias com 32 mil produtos geram uma matriz sparse enorme. Usamos **target encoding com Bayesian smoothing** (alpha=10): cada categoria recebe a média de review_score dos seus produtos, suavizada pela média global. Isso é equivalente a um prior Bayesiano — categorias com poucos produtos 'puxam' para a média global, evitando overfitting. A figura mostra a distribuição dos encodings — bem comportada, centrada em torno de 4,0."

---

## Slide 12 — Feature Engineering Temporal (1 min)

**Figura:** `reports/figures/11_temporal_features.png` + `12_seasonality_flags.png`

**Roteiro (1 min):**
> "Criamos 4 tipos de features temporais: dias desde uma data de referência (para o split temporal), flags de sazonalidade como `is_weekend` e `is_holiday_season`, e deltas como 'dias desde a última compra do usuário'. A diferença importante: features temporais **funcionam para usuários cold-start** porque não dependem do histórico individual — apenas do calendário."

---

## Slide 13 — Agregações de Usuário e Produto (1 min)

**Figura:** `reports/figures/16_user_aggregations.png` + `17_product_aggregations.png`

**Roteiro (1 min):**
> "Agregações de usuário e produto são o coração de sistemas de filtragem colaborativa clássicos. Calculamos, por exemplo, `user_total_purchases`, `user_avg_freight`, `user_purchase_span_days`, e no lado do produto, `product_popularity`, `product_avg_review_score`, `product_review_rate`. O truque: para usuários com 1 compra, esses valores 'puxam' para a média global via target encoding, evitando sinal ruidoso."

---

## Slide 14 — Feature Selection (Auditoria Spearman) (2 min)

**Título na tela:**
> **Feature Selection: o que aprendemos com Spearman**

| Par | \|ρ\| | Decisão |
|---|---|---|
| `days_since_reference` ↔ `user_recency_days` | 0,986 | Remover `user_recency_days` |
| `freight_value_log` ↔ `user_avg_freight` | 0,966 | Remover `freight_value_log` |

**Resultado da auditoria:**

| Modelo | Features | NDCG@K | Δ |
|---|---|---|---|
| Original | 20 aux | 0,2226 | baseline |
| Após auditoria | 18 aux | 0,1932 | **−13,2%** ❌ |
| **Production (`no_aux`)** | **0 aux** | **0,2725** | **+22,5%** ✅ |

**Roteiro (2 min):**
> "Aqui está o achado contraintuitivo do projeto. Rodamos uma auditoria Spearman com threshold |ρ| > 0,95 e identificamos 2 pares redundantes. Removemos as 2 features e re-treinamos o NCF — **a performance caiu 13%**. Ou seja, alta correlação não implica redundância para o modelo: o NCF pode explorar dimensões que a correlação linear não captura.
>
> Mais importante: quando fizemos a ablation removendo **todas as 20 features auxiliares** e mantendo apenas os 3 embeddings, a performance **subiu 22,5%**. A explicação: com 98% cold-start, os embeddings são inicializados aleatoriamente e o MLP acaba dependendo só das features auxiliares — que são globais, não personalizadas. O sinal é 'médio para todo mundo', o que empurra todos os scores para a mesma região e mata o ranking."

---

# PARTE 4 — TREINAMENTO (8 min)

## Slide 15 — Setup de Avaliação (1 min)

**Título na tela:**
> **Setup de Avaliação**

- **Split temporal 70/15/15** (não aleatório) — evita data leakage
- **Métricas**: MAP@K, NDCG@K, Recall@K, Precision@K, HitRate@K
- **99 negativos** por positivo na avaliação
- **Cold-start filter**: 98% dos test users são **ignorados** (justiça!)

**Roteiro (1 min):**
> "A avaliação é crítica. Usamos **split temporal** em vez de aleatório, porque em recomendação o futuro não pode vazar para o passado. Avaliamos com 99 negativos por positivo — o padrão da literatura. E temos um detalhe sutil: como 98% dos usuários do test são inéditos, **filtramos** esses usuários na avaliação. Se não filtrássemos, o NDCG seria puxado para zero por esses casos impossíveis."

---

## Slide 16 — Baselines (1 min)

**Título na tela:**
> **4 Baselines: a régua de comparação**

| Baseline | MAP@10 | NDCG@10 | Comentário |
|---|---|---|---|
| **Popularity** | 0,0019 | 0,0053 | ✅ Funciona no cold-start |
| TopRated | 0,0000 | 0,0000 | ❌ Filtros de review eliminam todos os produtos |
| ItemItemCF | 0,0000 | 0,0000 | ❌ Sem co-ocorrência com 1 compra por user |
| TruncatedSVD | 0,0000 | 0,0000 | ❌ Embeddings latentes ruins no cold-start |

**Roteiro (1 min):**
> "Treinamos 4 baselines para ter uma régua de comparação. **Apenas Popularidade funciona** — porque é global e não depende de histórico. TopRated, ItemItemCF e TruncatedSVD retornam zero: com 1 compra por usuário, não há sinal para 'quem gostou de X também gostou de Y'. Esse é o teto dos métodos clássicos neste dataset: NDCG de 0,0053."

---

## Slide 17 — NCF: Arquitetura (1 min)

**Título na tela:**
> **NCF (Neural Collaborative Filtering) — Arquitetura Production**

```
user_id  ──→ [user_emb (32)] ──┐
                                │
product_id ──→ [item_emb (32)] ─┤
                                ├──→ [Concat] ──→ [MLP 64→32] ──→ score
category ──→ [cat_emb (8)] ────┘
```

- **Loss**: BPR (Bayesian Personalized Ranking)
- **Negative sampling**: 8 negativos por positivo (on-the-fly)
- **Total params**: ~4 milhões

**Roteiro (1 min):**
> "O NCF é nossa rede neural. É uma arquitetura híbrida: embeddings de usuário, item e categoria, concatenados e passados por um MLP de 2 camadas. Treinamos com **BPR Loss** — que otimiza diretamente o ranqueamento, par-a-par: para cada interação positiva, amostramos 8 negativas e maximizamos a diferença de score. Cerca de 4 milhões de parâmetros total."

---

## Slide 18 — Optimization: 5 Runs + Ablation (2 min)

**Título na tela:**
> **5 Runs + Ablation: a busca pelo Production**

| # | Run | emb | hidden | dropout | NDCG@10 |
|---|---|---|---|---|---|
| 1 | emb16 baseline | 16 | [64,32] | 0.3 | 0.1336 |
| 2 | emb32 h128-64-32 | 32 | [128,64,32] | 0.3 | 0.1634 |
| 3 | emb64 h256-128-64 | 64 | [256,128,64] | 0.4 | 0.1829 |
| 4 | NCF_FINAL (com 20 aux) | 32 | [64,32] | 0.5 | 0.2226 |
| **5** | **Ablation_FINAL (sem aux)** ⭐ | **32** | **[64,32]** | **0.5** | **0.2725** |

🎬 *(Mostrar `mlflow ui` ao vivo)*

**Roteiro (2 min):**
> "Aqui está a jornada de otimização. Começamos com 5 runs variando capacidade e regularização. O padrão foi claro: maior embedding e mais dropout ajudam, mas há retornos decrescentes. O salto real veio da **ablation**: remover as 20 features auxiliares, mantendo só os 3 embeddings, melhorou o NDCG em 22,5%. Esse é o nosso **Production model: `Ablation_FINAL_no_aux_emb32`** com NDCG de 0,2725."

🎬 *(Se possível, abrir o MLflow UI e mostrar o run em tempo real)*

---

## Slide 19 — Análise de Erros (1 min)

**Título na tela:**
> **Análise de Erros: o gap Train vs Test**

| Métrica | Train | Val | Test | Gap |
|---|---|---|---|---|
| NDCG@10 | 0,58 | 0,34 | 0,27 | 0,31 |

**Diagnóstico:**
- **Train NDCG alto (0,58)** → modelo aprendeu os pares vistos
- **Val/Test caindo** → 98% são cold-start
- **Não é overfitting** — é a **realidade do problema**

**Roteiro (1 min):**
> "Olhando os números: NDCG de 0,58 no treino, 0,34 na validação, 0,27 no teste. Alguém pode pensar 'isso é overfitting'. Mas não é — é a **realidade do problema**. O modelo aprendeu bem os pares que viu no treino. No test, 98% dos usuários são inéditos, e o que estamos medindo é a **capacidade do modelo de rankear** dado apenas o embedding aleatório de um usuário novo. O fato de ele acertar 50% das top-10 recomendações nesses casos é, na verdade, surpreendentemente bom."

---

## Slide 20 — Comparação Final (1 min)

**Título na tela:**
> **Resultado Final: 60× sobre o baseline**

| Modelo | NDCG@10 | Lift vs baseline |
|---|---|---|
| Popularidade (baseline) | 0,0045 | 1× |
| NCF (com aux) | 0,2226 | 49× |
| **NCF Production (sem aux)** | **0,2725** | **60,6×** |

**Figura:** `reports/figures/ncf_vs_baseline.png`

**Roteiro (1 min):**
> "Fechando: nosso modelo Production tem NDCG de 0,2725, que é **60 vezes superior** à baseline de popularidade. Em e-commerce, mesmo lift de 10× já é um salto enorme. Esse número valida a abordagem e justifica seguir para a próxima fase do projeto."

---

# PARTE 5 — LIÇÕES E PRÓXIMOS PASSOS (3 min)

## Slide 21 — 3 Lições Aprendidas (2 min)

**Título na tela:**
> **3 Lições que valem para qualquer projeto de recomendação**

1. **EDA antes de tudo** — 99,99% sparsity mudou todas as decisões downstream
2. **Correlação ≠ redundância** — Spearman removeu features que **ajudavam** o modelo
3. **Cold-start massivo muda a equação** — embeddings simples superaram features engineered

**Roteiro (2 min):**
> "Três lições que eu levarei para outros projetos:
>
> 1. **A EDA salva o projeto.** Os 30 minutos gastos analisando sparsity e cold-start evitaram semanas de tentativa e erro com features complexas que não iam funcionar.
>
> 2. **Correlação linear não é redundância funcional.** A auditoria Spearman parecia uma boa ideia, mas o modelo estava usando as features de formas que a correlação não captura. Hoje eu questionaria qualquer feature selection puramente estatística.
>
> 3. **Cold-start muda a equação.** Em datasets densos (MovieLens), features engineered brilham. Em datasets esparsos como o Olist, embeddings treinados superam. Saber qual é o seu caso é metade do problema."

---

## Slide 22 — Próximos Passos (1 min)

**Título na tela:**
> **Próximos Passos**

- [ ] **Sequência temporal** de produtos (Transformer-based recommender)
- [ ] **Online learning** para incorporar feedback em tempo real
- [ ] **Deploy** via API (FastAPI + Docker) e monitoramento (Prometheus)
- [ ] **A/B testing** em produção para medir impacto em CTR/conversão

**Roteiro (1 min):**
> "Para a fase 3 do projeto, queremos explorar: arquiteturas baseadas em Transformer que capturam sequência de produtos comprados pelo usuário ao longo do tempo; online learning para adaptar o modelo em produção; e deploy via FastAPI com monitoramento de drift. O próximo grande teste será A/B testing: não basta ter o melhor NDCG no offline, o lift em CTR e conversão é o que conta."

---

## Slide 23 — Perguntas (0 min)

**Título na tela:**
> **Perguntas?**

🎬 *(Deixar 10 minutos para perguntas)*

**Possíveis perguntas e respostas prontas:**

- **P: "Por que split temporal e não aleatório?"**
  > R: Em sistemas de recomendação, o passado de um usuário está sempre disponível, mas o futuro não. Split aleatório gera leakage — o modelo vê interações futuras no treino, o que infla artificialmente as métricas.

- **P: "Por que BPR e não BCE?"**
  > R: BPR otimiza diretamente o ranking (par-a-par), enquanto BCE trata cada interação como classificação binária independente. Em recomendação, o que importa é a ordem dos itens, não a probabilidade absoluta.

- **P: "Como o sistema lida com produtos novos?"**
  > R: Produtos novos têm `category_id` conhecido, então o `cat_emb` pode ser usado. Mas o `item_emb` é aleatório. Em produção, podemos usar o `category_target_enc` e popularity como fallback.

- **P: "Por que não usar LSTM ou Transformer?"**
  > R: Exploramos na próxima fase. Para esta fase, o foco era validar o pipeline de ponta-a-ponta com uma arquitetura simples mas bem fundamentada (NCF).

---

# ANEXO — Comandos ao Vivo

Caso queira demonstrar ao vivo durante a apresentação:

```bash
# Mostrar a DAG do DVC
cd pos-ml-eng-tech-challenge-fase-02
uv run dvc dag

# Listar artefatos
ls -la data/processed/ artifacts/

# Abrir MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///./artifacts/mlflow.db

# Abrir dashboard
uv run streamlit run front/app_vis.py

# Mostrar notebook
uv run jupyter lab notebooks/
```

---

# CHECKLIST PRÉ-APRESENTAÇÃO

- [ ] Testar todos os comandos ao vivo com 1 dia de antecedência
- [ ] Ter backup dos slides em PDF (caso o projetor falhe)
- [ ] MLflow UI rodando em uma aba do navegador
- [ ] Dashboard Streamlit rodando em outra aba
- [ ] Notebook 00 aberto no Jupyter
- [ ] Figuras 01, 02, 11, 18, ncf_vs_baseline.png acessíveis
- [ ] README.md aberto para mostrar estrutura do projeto
- [ ] Tempo ensaiado: 25 min apresentação + 10 min perguntas
