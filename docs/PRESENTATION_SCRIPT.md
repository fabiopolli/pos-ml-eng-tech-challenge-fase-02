# Script de Apresentação — Olist Recommender System

> **Tech Challenge Fase 02 · Pós-Graduação ML Engineering**
> Formato: **Walkthrough guiado pelo dashboard Streamlit** (`front/app_vis.py`)
> Duração estimada: 25–30 min + 10 min de perguntas
> Foco: EDA, escolha de features, feature engineering e treinamento dos modelos

---

## Como usar este script

- Este é um **roteiro de demonstração ao vivo** baseado em clicks no dashboard.
- Cada **seção** corresponde a uma **aba ou sub-bloco** do dashboard.
- Cada bloco tem 3 colunas:
  - 👁️ **O QUE MOSTRAR** → ação exata na UI (clique, scroll, etc.)
  - 🗣️ **O QUE DIZER** → narração para a audiência
  - 💡 **DICA** → cuidado ou truque visual
- Os **números importantes** estão em **negrito**.
- Os marcadores 🎬 indicam **comandos de terminal** para alternar entre ferramentas.

---

## 0. Preparação (5 min antes)

🎬 Abra **3 terminais** lado a lado:

```bash
# Terminal 1 — Dashboard principal
cd pos-ml-eng-tech-challenge-fase-02
uv run streamlit run front/app_vis.py
# → abre automaticamente em http://localhost:8501

# Terminal 2 — MLflow Tracking (deixe aberto em outra aba)
uv run mlflow ui --backend-store-uri sqlite:///./artifacts/mlflow.db
# → abre automaticamente em http://localhost:5000

# Terminal 3 — auxiliar (para responder perguntas)
```

🎬 Redimensione a janela do navegador para **1280×800** ou maior — o dashboard usa 4 colunas lado a lado.

💡 **DICA:** Se o Streamlit perguntar sobre coleta de telemetria, clique em "No thanks" no canto inferior direito.

---

# ROTEIRO PRINCIPAL

## SEÇÃO 1 — Abertura (2 min)

👁️ **O QUE MOSTRAR:** Fique na aba **📊 Visão Geral**. Aponte para o cabeçalho e os 4 KPIs do topo.

🗣️ **O QUE DIZER:**
> "Boa tarde a todos. Hoje vou apresentar o sistema de recomendação que desenvolvemos para o dataset Olist. Vamos usar este dashboard interativo como fio condutor — vocês vão ver os dados reais, as features que engenheiramos e os resultados dos modelos treinados. Tudo está em `front/app_vis.py`, escrito em Streamlit, e lê os artefatos direto do pipeline DVC.
>
> Aqui no topo temos os 4 KPIs principais: **99.785 interações**, **93.358 usuários únicos**, **32.216 produtos** e **72 categorias**. Antes de entrar no detalhe técnico, quero que vocês guardem esses números."

---

## SEÇÃO 2 — EDA: Target e Cold-Start (4 min)

👁️ **O QUE MOSTRAR:** Ainda na aba **📊 Visão Geral**, scroll para a seção **"Distribuição do `review_score`"**. Aponte para os dois gráficos de barras (volume e percentual).

🗣️ **O QUE DIZER:**
> "A primeira coisa que olhamos foi a distribuição do target — o `review_score`. **77% das compras têm nota 4 ou 5** — um viés positivo forte. Isso já nos disse uma coisa importante: o sinal de feedback está concentrado nas notas altas, e o ruído entre 4 e 5 é difícil de separar. Por isso, decidimos tratar o problema como **feedback implícito** — não prever o review, mas ranquear itens via BPR Loss."

👁️ **O QUE MOSTRAR:** Scroll até **"Comportamento Temporal"** e **"Sparsity da Matriz User-Item"** (3 métricas grandes no final).

🗣️ **O QUE DIZER:**
> "Aqui está o problema central. A sparsity da matriz user-item é de **99,9967%** — quase todos os pares usuário-produto nunca foram observados. E, como o gráfico de comportamento de usuários mostra, **98% dos usuários compraram uma única vez**. Isso é o que chamamos de cold-start massivo.
>
> Por que isso importa? Qualquer feature que dependa de histórico do usuário — seja embedding treinado, seja 'média de gasto' — vai estar **ruidosa para 98% dos dados**. Esse foi o insight que guiou toda a engenharia de features: tudo que construímos precisa funcionar **mesmo para quem comprou uma vez**."

💡 **DICA:** Faça uma pausa de 2 segundos após dizer "98% dos usuários compraram uma única vez". Espere a ficha cair.

---

## SEÇÃO 3 — EDA: Features Numéricas (2 min)

👁️ **O QUE MOSTRAR:** Ainda na aba **📊 Visão Geral**, scroll para **"📈 Distribuições Numéricas (price, freight_value)"** e depois **"📐 Análise de Cauda Longa (Escala Logarítmica)"**.

🗣️ **O QUE DIZER:**
> "Aqui estão as duas features numéricas brutas: preço e frete. Olhem a escala — algumas observações passam de **R$ 10 mil** e quebram qualquer histograma linear. Por isso, na fase de feature engineering, aplicamos **log-transform** em ambas. O gráfico da direita mostra a mesma distribuição em escala log — vira uma normal, que é o que o modelo espera.
>
> Também criamos uma flag `has_price_outlier` para marcar observações fora do percentil 99. O modelo pode aprender a tratá-las de forma diferenciada."

---

## SEÇÃO 4 — EDA: Categorias e Comportamento (2 min)

👁️ **O QUE MOSTRAR:** Ainda na aba **📊 Visão Geral**, scroll para **"🛍️ Top 15 Categorias Mais Vendidas"** e **"📅 Comportamento Temporal"**.

🗣️ **O QUE DIZER:**
> "As top 15 categorias concentram grande parte das vendas — fenômeno típico de e-commerce chamado 'cabeça longa'. E olhem o comportamento temporal: as vendas estão concentradas em 2017-2018, com sazonalidade clara em novembro (Black Friday) e dezembro (Natal). Usamos essas duas informações para construir **flags de sazonalidade** (`is_holiday_season`, `is_weekend`) que funcionam para usuários cold-start, porque dependem só do calendário, não do histórico individual."

---

## SEÇÃO 5 — Feature Engineering: Visão Geral (3 min)

👁️ **O QUE MOSTRAR:** Clique na aba **🔧 Feature Engineering** (segunda aba). Aponte para **"📋 Mapa de Features por Categoria"** e **"🔄 Comparativo Antes/Depois da FE"**.

🗣️ **O QUE DIZER:**
> "Aqui está a fase de feature engineering. Saímos de 10 colunas brutas para 42 engenheiradas. A tabela mostra o mapa de features por categoria: numéricas, temporais, categóricas, agregações e embeddings. O comparativo Antes/Depois resume o trabalho: **18 features numéricas, 7 temporais, 4 pagamentos OHE, e 10 agregações de usuário/produto**."

👁️ **O QUE MOSTRAR:** Aponte para **"🏆 Categorias com Maior Target Encoding"**.

🗣️ **O QUE DIZER:**
> "Para lidar com as 72 categorias, usamos **target encoding com Bayesian smoothing** (alpha=10). Cada categoria recebe a média de review_score dos seus produtos, suavizada pela média global. Categorias com poucos produtos 'puxam' para a média global, evitando overfitting. O gráfico mostra as top categorias por nota média."

---

## SEÇÃO 6 — Feature Engineering: Correlação e Spearman (3 min)

👁️ **O QUE MOSTRAR:** Ainda na aba **🔧 Feature Engineering**, scroll até **"🔗 Mapa de Correlação (Features Numéricas)"**.

🗣️ **O QUE DIZER:**
> "Aqui está o mapa de correlação. Cores quentes indicam correlação alta. Vejam que existem vários pares com correlação maior que 0,9 — `price` e `price_log` (r=1,0), `freight_value` e `freight_value_log` (r=1,0), e vários outros. Esses foram eliminados na primeira rodada de feature selection (redundâncias lineares óbvias)."

👁️ **O QUE MOSTRAR:** Mude para a aba **🧠 NCF** (terceira aba), depois **"🏆 Comparação de Runs (Etapa 4 — Otimização)"** — a tabela que mostra 5 runs + 1 ablation.

🗣️ **O QUE DIZER:**
> "Mas aqui está o achado contraintuitivo do projeto. Rodamos uma **segunda rodada** de feature selection usando **Spearman** com threshold |ρ| > 0,95 — isso identificou 2 pares redundantes: `user_recency_days` vs `days_since_reference` (ρ=−0,986) e `freight_value_log` vs `user_avg_freight` (ρ=0,966). Removemos essas 2 features e re-treinamos o NCF — **a performance caiu 13,2%** no NDCG.
>
> Ou seja, **correlação linear não implica redundância funcional**. O modelo estava usando essas features de formas que a correlação não captura."

---

## SEÇÃO 7 — Treinamento: Baselines (3 min)

👁️ **O QUE MOSTRAR:** Clique na aba **🏋️ Baselines** (terceira aba). Aponte para **"📊 Configuração do Split Temporal"** e a **"📋 Tabela de Resultados"**.

🗣️ **O QUE DIZER:**
> "Aqui é a régua de comparação. Treinamos 4 baselines clássicos: Popularity, TopRated, ItemItemCF e TruncatedSVD. O split é **temporal 70/15/15** — não aleatório, porque em recomendação o futuro não pode vazar para o passado. Avaliamos com 99 negativos por positivo.
>
> Vejam a tabela: **apenas Popularidade funciona** (MAP=0,0019, NDCG=0,0053). TopRated, ItemItemCF e TruncatedSVD retornam **zero** em todas as métricas. Por quê? Porque com **1 compra por usuário, não há co-ocorrência** — o sinal que esses métodos precisam simplesmente não existe."

👁️ **O QUE MOSTRAR:** Aponte para **"📈 Comparação de MAP@10 por Modelo"** e **"🎯 Comparação Completa de Métricas (K=10)"**.

🗣️ **O QUE DIZER:**
> "Os gráficos mostram isso visualmente. PopularityBaseline é o único com barra visível. Os outros são todos zero — o teto dos métodos clássicos neste dataset é NDCG=0,0053."

💡 **DICA:** Este é o momento de **preparar o terreno** para a próxima seção: "Mas a abordagem neural muda completamente esse jogo."

---

## SEÇÃO 8 — NCF: Métricas @ K=10 (3 min)

👁️ **O QUE MOSTRAR:** Clique na aba **🧠 NCF (MLP PyTorch)** (quarta aba). Aponte para **"📊 Métricas @ K=10 (Test Set)"** — os 5 grandes KPIs no topo.

🗣️ **O QUE DIZER:**
> "E aqui está o salto. O NCF Production atinge **NDCG@10 = 0,2725**, **HitRate = 0,4949** e **MAP = 0,2081** no test set. Em particular, **49% dos top-10 contém pelo menos um item relevante** para o usuário. Isso é uma transformação qualitativa em relação aos baselines."

👁️ **O QUE MOSTRAR:** Aponte para **"⚔️ NCF vs Baseline (Popularidade)"**.

🗣️ **O QUE DIZER:**
> "O banner no topo do NCF mostra: **60,6× superior** à baseline de popularidade no NDCG. Em e-commerce, mesmo um lift de 10× já é enorme. Esse número valida que a abordagem neural de embeddings é a escolha certa para datasets esparsos como o Olist."

---

## SEÇÃO 9 — NCF: Generalização e Cold-Start (2 min)

👁️ **O QUE MOSTRAR:** Ainda na aba **🧠 NCF**, scroll para **"📈 Análise de Generalização (Train vs Val vs Test)"** e **"🧊 Análise de Cold-Start"**.

🗣️ **O QUE DIZER:**
> "Aqui está uma análise de erro honesta. O NDCG no treino é 0,58, na validação 0,34, no test 0,27. Alguém pode pensar 'isso é overfitting'. Mas **não é** — é a realidade do problema.
>
> No test, **98% dos usuários são inéditos** no treino. O que estamos medindo é a capacidade do modelo de rankear itens dado apenas o **embedding aleatório** de um usuário novo. O fato de ele acertar 50% das top-10 recomendações nesses casos é surpreendentemente bom."

---

## SEÇÃO 10 — NCF: Otimização e Hiperparâmetros (2 min)

👁️ **O QUE MOSTRAR:** Volte para **"🏆 Comparação de Runs"** (na mesma aba NCF).

🗣️ **O QUE DIZER:**
> "Aqui está a jornada de otimização. Começamos com 5 runs variando capacidade e regularização. O padrão: maior embedding ajuda até certo ponto (16→32→64), mais dropout ajuda, e learning rate + scheduler são cruciais.
>
> Mas o **salto real** veio da linha 5: a **ablation** removendo todas as 20 features auxiliares. NDCG subiu de 0,2226 para **0,2725** — **+22,5%**. A hipótese: com 98% cold-start, o MLP estava dependendo só das features globais, que 'puxam' todos os scores para a mesma região e matam o ranking."

👁️ **O QUE MOSTRAR:** Scroll para **"⚙️ Hiperparâmetros do Modelo Production"** (configs/ncf_best.yaml).

🗣️ **O QUE DIZER:**
> "Estes são os hiperparâmetros finais: embedding 32, MLP [64, 32], dropout 0,5, AdamW com lr 5e-4, BPR loss com 8 negativos por positivo. Está tudo no `configs/ncf_best.yaml` — fonte canônica do que está em produção."

💡 **DICA:** Se alguém perguntar, **NÃO** tente mostrar a tabela completa de HPs (é longa). Apenas mencione: "Estão todos no `configs/ncf_best.yaml`, o modelo tem 4 milhões de parâmetros, e levou 5 minutos para treinar no nosso setup".

---

## SEÇÃO 11 — Recomendações Geradas (2 min)

👁️ **O QUE MOSTRAR:** Clique na aba **🎯 Recomendações** (quinta aba). Aponte para **"📋 Inventário de Artefatos"** e **"🔍 Exemplo: Top-5 Recomendações"**.

🗣️ **O QUE DIZER:**
> "Para fechar a parte técnica, aqui está o output dos baselines: **10 arquivos CSV** em `artifacts/baselines/` com top-K recomendações. A tabela mostra o inventário completo: Popularity, TopRated, ItemItemCF, TruncatedSVD para K=5, 10 e 20.
>
> A tabela no final mostra o **top-5 do Popularity** para os primeiros 5 usuários — note que as categorias se concentram nas mais vendidas (cama_mesa_banho, beleza_saude, etc.), que é exatamente o esperado de um baseline global."

💡 **DICA:** Se alguém perguntar "mas o NCF também gera recomendações?", responda: "O NCF está em `artifacts/ncf_Ablation_FINAL_no_aux_emb32.pt` — para gerar top-K de um usuário específico, basta carregar o checkpoint e fazer o scoring, como mostrado no `notebooks/demo.ipynb`."

---

## SEÇÃO 12 — Sobre o Pipeline (1 min)

👁️ **O QUE MOSTRAR:** Clique na aba **ℹ️ Sobre o Pipeline** (última aba). Aponte para a seção de **Pipeline DVC** e o **Status dos Entregáveis**.

🗣️ **O QUE DIZER:**
> "Para fechar, este é o resumo do pipeline. Usamos DVC para versionar os 3 estágios de dados. O `dvc.yaml` declara prepare, featurize e validate. A reprodutibilidade é total: rodar `uv run dvc repro` reconstrói tudo do zero.
>
> O status dos entregáveis mostra: **9 itens concluídos**, de pipeline de dados a dashboard Streamlit. Para deploy em produção, os próximos passos são: FastAPI + Docker, monitoramento com Prometheus, e A/B testing."

---

# PERGUNTAS E RESPOSTAS PREPARADAS

💡 **DICA:** Tenha o **dashboard aberto** durante o Q&A. Quando alguém perguntar sobre uma métrica específica, você pode clicar na aba correspondente e mostrar o número ao vivo.

### P1: "Por que split temporal e não aleatório?"

🗣️ "Em sistemas de recomendação, o passado de um usuário está sempre disponível, mas o futuro não. Split aleatório gera **data leakage** — o modelo vê interações futuras no treino, o que infla artificialmente as métricas. O split temporal reflete o cenário real de produção: o modelo de hoje recomenda para o usuário de hoje, baseado no histórico até ontem."

👁️ Mostre: aba **🏋️ Baselines** → "Configuração do Split Temporal".

### P2: "Por que BPR e não BCE?"

🗣️ "BPR (Bayesian Personalized Ranking) **otimiza diretamente o ranking**, par-a-par: para cada interação positiva, amostramos 8 negativas e maximizamos a diferença de score. BCE (Binary Cross-Entropy) trata cada interação como classificação binária independente. Em recomendação, o que importa é a **ordem dos itens**, não a probabilidade absoluta — por isso BPR é superior."

👁️ Mostre: aba **🧠 NCF** → "Hiperparâmetros" → Loss: BPR.

### P3: "Como o sistema lida com produtos novos?"

🗣️ "Produtos novos têm `category_id` conhecido, então o `cat_emb` pode ser usado. Mas o `item_emb` é inicializado aleatoriamente. Em produção, usaríamos um **fallback baseado em popularidade** da categoria. Para o próximo passo do projeto, queremos implementar **online learning** que ajusta o embedding do produto assim que ele recebe a primeira interação."

### P4: "Por que não usar LSTM ou Transformer?"

🗣️ "Excelente pergunta. Para esta fase, o foco era validar o pipeline end-to-end com uma arquitetura simples mas bem fundamentada (NCF). Transformer-based recommenders como SASRec ou BERT4Rec capturam **sequência temporal de produtos** comprados pelo usuário — isso é o próximo passo. Eles brilham quando há histórico de múltiplas compras, mas como 98% dos usuários têm 1 compra aqui, o ganho esperado seria marginal."

### P5: "O que aconteceu com o TopRated? Por que zero?"

🗣️ "TopRated recomenda produtos com **maior review_score médio**. Mas nosso dataset tem **77% das notas entre 4 e 5** — quase todo produto é 'top rated'. E quando aplicamos o filtro de mínimo de reviews (5 ou 15), a maioria dos produtos é eliminada. O resultado é lista vazia para quase todos os usuários, especialmente os cold-start."

👁️ Mostre: aba **🏋️ Baselines** → tabela de resultados → TopRated com zeros.

### P6: "Quanto tempo levou para treinar o NCF?"

🗣️ "O NCF Production completo (20 epochs com early stop) leva cerca de **5 a 10 minutos** no CPU. O bottleneck é o scoring em batch durante a validação — temos 93 mil usuários e 32 mil itens. Em GPU seria 10× mais rápido. O modelo tem 4 milhões de parâmetros (16 MB no disco), o que é trivial para produção."

### P7: "Por que o nome `Ablation_FINAL_no_aux_emb32`?"

🗣️ "O nome tem história. `Ablation` porque foi o resultado de uma **ablation study** que removemos features. `no_aux` significa **sem features auxiliares** (só os 3 embeddings). `emb32` é o tamanho do embedding. E `FINAL` é porque foi o último experimento antes de promover para produção. É um nome longo, mas é autoexplicativo."

---

# ANEXO — Comandos Úteis Durante a Apresentação

## Se perguntarem "como reproduzir isso?"

🎬 Abra um terminal e rode:

```bash
# Pipeline completo (DVC)
uv run dvc repro

# Baselines
uv run python src/train.py

# NCF Production (5 min no CPU)
uv run python scripts/train_ncf.py \
    --ablation-no-aux \
    --epochs 20 \
    --emb-dim 32 \
    --hidden 64 32 \
    --dropout 0.5 \
    --lr 5e-4 \
    --use-scheduler

# Ver experimentos
uv run mlflow ui --backend-store-uri sqlite:///./artifacts/mlflow.db
```

## Se perguntarem "onde está o código?"

```bash
# Pipeline
cat src/data_preparation.py
cat src/feature_engineering.py

# Modelos
cat src/models/ncf.py
cat src/models/factory.py
cat src/models/losses.py

# Treino
cat src/train.py
cat scripts/train_ncf.py
```

## Se perguntarem "qual a arquitetura?"

```bash
# Resumo do modelo Production
cat configs/ncf_best.yaml
```

## Se perguntarem "como funciona o MLflow?"

🎬 Mostre a aba **🧠 NCF** → **"📋 MLflow Tracking"** no dashboard. Há uma tabela com os 4 experimentos e os top runs.

🎬 Ou abra `http://localhost:5000` no navegador e clique em **Experiments** → **Olist_NCF_Optimization**.

---

# CHECKLIST PRÉ-APRESENTAÇÃO (T-30 min)

- [ ] **Streamlit rodando** em `http://localhost:8501` (validar que carrega sem erro)
- [ ] **MLflow UI rodando** em `http://localhost:5000` (deixe em aba separada)
- [ ] **6 abas testadas**: 📊 Visão Geral → 🔧 FE → 🏋️ Baselines → 🧠 NCF → 🎯 Recomendações → ℹ️ Sobre
- [ ] **4 figuras-chave acessíveis** (caso queira mostrar PNG separado):
  - `reports/figures/01_review_score_distribution.png`
  - `reports/figures/05_user_behavior.png`
  - `reports/figures/18_correlation_heatmap.png`
  - `reports/figures/ncf_vs_baseline.png`
- [ ] **3 terminais abertos**: dashboard, MLflow, auxiliar
- [ ] **README.md aberto** no editor (para mostrar estrutura do projeto)
- [ ] **Notebook 00 aberto** no Jupyter (opcional, para mostrar pipeline)
- [ ] **Tempo ensaiado**: 25 min apresentação + 10 min perguntas
- [ ] **Água na mesa** 😄

---

# TIMING SUGERIDO

| Seção | Duração | Acumulado |
|-------|---------|-----------|
| 1. Abertura | 2 min | 0:02 |
| 2. EDA: Target + Cold-Start | 4 min | 0:06 |
| 3. EDA: Features Numéricas | 2 min | 0:08 |
| 4. EDA: Categorias + Temporal | 2 min | 0:10 |
| 5. FE: Visão Geral | 3 min | 0:13 |
| 6. FE: Correlação + Spearman | 3 min | 0:16 |
| 7. Baselines | 3 min | 0:19 |
| 8. NCF: Métricas @ K=10 | 3 min | 0:22 |
| 9. NCF: Generalização | 2 min | 0:24 |
| 10. NCF: Otimização | 2 min | 0:26 |
| 11. Recomendações | 2 min | 0:28 |
| 12. Sobre o Pipeline | 1 min | 0:29 |
| **Perguntas** | **10 min** | **0:39** |

Se a audiência estiver mais interessada em uma parte específica, **não tenha medo de pular ou estender**. O dashboard permite que vocês voltem a qualquer aba sob demanda.
