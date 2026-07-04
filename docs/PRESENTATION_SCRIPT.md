# Script de Apresentação — Olist Recommender System

> **Tech Challenge Fase 02 · Pós-Graduação ML Engineering**
> Formato: **Walkthrough guiado pelo dashboard Streamlit** (`front/app_vis.py`)
> Duração estimada: 30–35 min + 10 min de perguntas
> Audiência: **grupo não-especialista** (termos técnicos sempre vêm com analogia)
> Foco: EDA, escolha de features, feature engineering e treinamento dos modelos

---

## Como usar este script

- Roteiro de **demonstração ao vivo** baseado em clicks no dashboard
- Cada **seção** corresponde a uma **aba ou sub-bloco** do dashboard
- Cada bloco tem 4 colunas:
  - 👁️ **O QUE MOSTRAR** → ação exata na UI
  - 🗣️ **O QUE DIZER** → bullet points curtos para a fala
  - 📖 **GLOSSÁRIO** → definição simples de termos técnicos
  - 💡 **DICA** → cuidado ou truque visual
- **Números importantes** em **negrito**
- 🎬 → comandos de terminal | 🧠 → insight conceitual

---

## 0. Preparação (5 min antes)

🎬 Abrir **3 terminais** lado a lado:

- **Terminal 1** — Dashboard: `cd pos-ml-eng-tech-challenge-fase-02 && uv run streamlit run front/app_vis.py` (→ `localhost:8501`)
- **Terminal 2** — MLflow: `uv run mlflow ui --backend-store-uri sqlite:///./artifacts/mlflow.db` (→ `localhost:5000`)
- **Terminal 3** — auxiliar (para responder perguntas)

🎬 Redimensionar navegador para **1280×800** ou maior

💡 **DICA:** Se o Streamlit perguntar sobre telemetria, clicar "No thanks"

---

# PARTE 1 — O PROBLEMA (5 min)

## SEÇÃO 1 — Abertura (2 min)

👁️ **O QUE MOSTRAR:** Ficar na aba **📊 Visão Geral**. Apontar para o cabeçalho e os 4 KPIs do topo.

📖 **GLOSSÁRIO:**
- **Sistema de recomendação** — software que sugere produtos (ex: Amazon, Netflix, TikTok)
- **Marketplace** — site que conecta vários vendedores a vários compradores (Olist, Mercado Livre)
- **KPIs** — os 4 números grandes no topo, os "sinais vitais" do dataset

🗣️ **O QUE DIZER:**
- Apresentação do sistema de recomendação do **Olist** (marketplace brasileiro)
- Objetivo: dado um cliente, sugerir **produtos que ele provavelmente quer comprar**
- Exemplos do dia-a-dia: Netflix, Amazon, Spotify — todos usam ML para aprender padrões de gosto
- **Fio condutor**: dashboard interativo em `front/app_vis.py`
- **4 KPIs principais** no topo:
  - **99.785** compras
  - **93.358** clientes únicos
  - **32.216** produtos
  - **72** categorias
- **⚠️ Nota de transparência**: o dataset do Olist **não é ideal** para esse tipo de problema
  - 99k compras, mas 98% cold-start = dataset pequeno, esparso, pouca repetição
  - Trouxe **muitos desafios** que vamos discutir ao longo da apresentação
  - Resultado final: **60× melhor que o baseline**, mas **modesto em termos absolutos** vs sistemas em datasets mais densos
  - Vou ser **honesto** sobre isso em cada etapa

---

## SEÇÃO 2 — Cold-Start: o vilão do projeto (3 min)

👁️ **O QUE MOSTRAR:** Na aba **📊 Visão Geral**, scroll para **"Distribuição do `review_score`"** (2 gráficos de barras: volume e percentual)

📖 **GLOSSÁRIO:**
- **Feedback explícito** — quando o usuário **diz** o que achou (nota, like) — é o `review_score` aqui
- **Sparsity (esparsidade)** — planilha gigante cliente×produto com quase todas as células vazias
- **Cold-start** — sistema sabe **pouco ou nada** sobre alguém (cliente novo, produto novo) — como pedir dica de restaurante para quem acabou de chegar na cidade

🗣️ **O QUE DIZER (target + viés positivo):**
- Primeira análise: **distribuição das notas** (feedback explícito)
- **77% das compras têm nota 4 ou 5** — viés fortíssimo para o positivo
- Lição: sinal de feedback está concentrado nas notas altas; ruído entre 4 e 5 é difícil de separar
- **Decisão**: tratar como **feedback implícito** — em vez de prever a nota, prever **probabilidade de interação**
- Exemplo: Netflix não tenta adivinhar se você vai dar 5 estrelas, tenta adivinhar se você vai **clicar**

👁️ **O QUE MOSTRAR:** Scroll para **"Sparsity da Matriz User-Item"** (3 métricas grandes no final)

🗣️ **O QUE DIZER (sparsity + cold-start):**
- **Verdadeiro vilão do projeto**: sparsity de **99,9967%**
- **Analogia da sala de aula**:
  - 93 mil alunos, 32 mil livros
  - 93 mil × 32 mil = **3 bilhões de pares possíveis**
  - Apenas 99.785 observados — o resto é buraco
- **98% dos usuários compraram uma única vez** — cold-start massivo
- **Por que importa**: qualquer feature que dependa de histórico (média de gasto, categorias preferidas) está **vazia** para 98% dos clientes
- **Insight que guiou tudo**: features precisam funcionar **mesmo para quem comprou uma vez**; quem compra 2× já é "veterano"

🧠 **INSIGHT:** "Cold-start massivo diferencia recomendação de outras áreas de ML. Em classificação de imagens, cada exemplo tem muitas features (pixels). Aqui, a maioria dos clientes tem **ZERO exemplos** além da própria interação que estamos prevendo."

💡 **DICA:** Pausar 2 segundos após "98% dos usuários compraram uma única vez". Esperar a ficha cair.

---

# PARTE 2 — EDA: ENTENDENDO OS DADOS (8 min)

## SEÇÃO 3 — Features Numéricas e a Cauda Longa (2 min)

👁️ **O QUE MOSTRAR:** Na aba **📊 Visão Geral**, scroll para **"📈 Distribuições Numéricas"** e **"📐 Análise de Cauda Longa"**

📖 **GLOSSÁRIO:**
- **Cauda longa (long tail)** — distribuição com maioria de valores baixos e pouquíssimos valores extremos
- **Outlier** — valor que foge muito do padrão (ex: R$ 10 mil onde a mediana é R$ 80)
- **Log-transform** — aplicar logaritmo: R$ 10.000 → 4, R$ 100 → 2, R$ 10 → 1 (espreme grandes, estica pequenos)
- **Percentil 99** — valor abaixo do qual estão 99% dos dados

🗣️ **O QUE DIZER:**
- Duas features numéricas brutas: **preço** e **frete**
- Gráfico da esquerda: quase tudo no canto inferior; algumas observações passam de **R$ 10 mil** = **cauda longa**
- **Por que é problema**: modelos de ML são sensíveis a escala — R$ 10.000 "puxa" a média e o modelo acha que ele é 100× mais importante
- **Solução**: **log-transform** — R$ 10.000 vira 4, R$ 100 vira 2, R$ 10 vira 1 → diferença de magnitude desaparece
- Gráfico da direita (escala log): distribuição vira **simétrica** (o que o modelo espera)
- Flag `has_price_outlier` para marcar observações fora do percentil 99 — "esse aqui é especial, vê o que acha"

---

## SEÇÃO 4 — Categorias e Comportamento Temporal (2 min)

👁️ **O QUE MOSTRAR:** Na aba **📊 Visão Geral**, scroll para **"🛍️ Top 15 Categorias Mais Vendidas"** e **"📅 Comportamento Temporal"**

📖 **GLOSSÁRIO:**
- **One-Hot Encoding (OHE)** — transformar coluna categórica em várias colunas binárias (0/1)
- **Sazonalidade** — padrão que se repete no tempo (Black Friday, Natal, fim de semana)

🗣️ **O QUE DIZER:**
- **Top 15 categorias**: `cama_mesa_banho`, `beleza_saúde`, `esporte_lazer` dominam
- **Cabeça longa** de e-commerce: 15 categorias concentram a maioria; outras 57 dividem o resto
- **Comportamento temporal**:
  - Vendas concentradas em 2017-2018
  - Picos claros em **novembro** (Black Friday) e **dezembro** (Natal)
- **Viram features**: flags `is_holiday_season`, `is_weekend`
- **Por que é útil**: funcionam **mesmo para usuários cold-start** — dependem só do calendário, não do histórico individual
- Exemplo: cliente que nunca comprou, se comprar no Natal, recebe o boost de "época natalina"

---

## SEÇÃO 5 — Conclusões da EDA → Decisões de Design (1 min)

👁️ **O QUE MOSTRAR:** Scroll até o final da aba **📊 Visão Geral**, parar antes de mudar de aba

🗣️ **O QUE DIZER (3 decisões):**
- Três decisões viraram **princípios de design**:
  1. **Tratar como feedback implícito (BPR Loss)** — nota tem muito viés positivo; o que importa é ranquear
  2. **Log-transform em preço e frete** — neutralizar cauda longa
  3. **Priorizar embeddings sobre features engineered** — com 98% cold-start, features que dependem de histórico são ruído para a maioria

🗣️ **O QUE DIZER (reflexão honesta sobre o dataset):**
- Essas 3 decisões **não foram "escolhas sofisticadas"** — foram **adaptativas**
- Em dataset denso (MovieLens, com centenas de avaliações por usuário), a abordagem clássica (user-item matrix factorization) funciona bem
- Mas o Olist nos **forçou a repensar tudo**
- Trabalhar com 99 mil compras e 98% cold-start é, em essência, **trabalhar com 2 mil usuários relevantes**
- É um dataset **pequeno, esparso**, e essa limitação guiou cada decisão técnica daqui em diante

---

# PARTE 3 — FEATURE ENGINEERING (8 min)

## SEÇÃO 6 — Visão Geral: de 10 para 42 colunas (2 min)

👁️ **O QUE MOSTRAR:** Clicar na aba **🔧 Feature Engineering**. Apontar para **"📋 Mapa de Features por Categoria"** e **"🔄 Comparativo Antes/Depois"**

📖 **GLOSSÁRIO:**
- **Feature engineering** — criar novas colunas a partir dos dados brutos (parte mais importante e demorada de um projeto de ML)
- **Embedding** — vetor denso de números (ex: 32 dimensões) que representa uma entidade; usuários parecidos ficam "perto" no espaço
- **Target encoding** — substituir categoria pela **média do target** dessa categoria (cuidado com target leakage)
- **Bayesian smoothing** — evitar que categorias raras tenham valores extremos (ex: 1 produto com nota 5 ≠ "média 5,0")

🗣️ **O QUE DIZER:**
- Fase mais artesanal: **feature engineering** — criar novas colunas
- Saímos de **10 colunas brutas para 42 engenheiradas**
- **Mapa de features**:
  - **18 numéricas** (log de preço, log de frete, etc.)
  - **7 temporais** (dia, mês, flags de feriado)
  - **4 pagamentos OHE** (one-hot encoding: cada tipo vira uma coluna 0/1)
  - **10 agregações** de usuário/produto
  - **3 embeddings** (usuário, item, categoria) — 32, 32 e 8 dimensões
- **O que é embedding**: cada entidade vira um **vetor de 32 números** aprendido do zero
- Usuários parecidos → vetores parecidos → moram "perto" num espaço de 32 dimensões

---

## SEÇÃO 7 — Target Encoding: a média com cautela (2 min)

👁️ **O QUE MOSTRAR:** **"🏆 Categorias com Maior Target Encoding"** e depois **"🔗 Mapa de Correlação"**

📖 **GLOSSÁRIO:**
- **Target leakage** — modelo "vê" informação do futuro ou do target durante o treino, gerando métricas infladas

🗣️ **O QUE DIZER (target encoding):**
- Para `category_id` (72 valores), one-hot encoding geraria matriz enorme de zeros
- Usamos **target encoding com Bayesian smoothing**:
  - Cada categoria recebe a **média de review_score** dos seus produtos
  - Categorias com poucos produtos "puxam" para a média global
- **Analogia do professor**: aluno com 1 prova e nota 10 não é "nota 10" definitiva — professor espera mais provas
- `alpha=10` no código = "fator de paciência"
- **Risco grave: target leakage**:
  - Se a média da categoria for calculada usando o **test set**, o modelo **cola** — vê respostas do teste no treino
  - Solução: calcular média apenas no **train set** e aplicar ao test
  - É o **split temporal** salvando a gente

👁️ **O QUE MOSTRAR:** **"🔗 Mapa de Correlação"** no final da aba

🗣️ **O QUE DIZER (correlação):**
- Mapa de correlação entre features numéricas
- Cores quentes (vermelho/laranja) = correlação alta; frias (azul) = baixa/negativa
- Vários pares com correlação 0,9 ou mais:
  - `price` e `price_log` (r=1,0 — mesma informação)
  - `freight_value` e `freight_value_log` (r=1,0)
  - e outros
- **Redundância linear óbvia** — duas features carregando o mesmo sinal
- Já removemos na primeira rodada de feature selection
- Mas e as redundâncias **sutis**? É o que vem agora

---

## SEÇÃO 8 — Auditoria Spearman: o achado contraintuitivo (3 min)

👁️ **O QUE MOSTRAR:** Mudar para a aba **🧠 NCF**, depois **"🏆 Comparação de Runs"** (tabela com 5 runs + 1 ablation)

📖 **GLOSSÁRIO:**
- **Spearman ρ (rho)** — correlação monotônica (captura qualquer curva crescente/decrescente, não só linear)
- **Ablation study** — remover um componente de cada vez para ver o quanto ele contribuía
- **|ρ| > 0,95** — correlação quase perfeita

🗣️ **O QUE DIZER (auditoria Spearman):**
- Depois de remover redundâncias óbvias, rodamos **2ª rodada** com Spearman (|ρ| > 0,95)
- Mais robusto a outliers que Pearson
- **2 pares identificados**:
  - `user_recency_days` vs `days_since_reference` (ρ = **−0,986**) — mesma info, sinal contrário
  - `freight_value_log` vs `user_avg_freight` (ρ = **+0,966**) — frete do produto e frete médio do usuário, redundantes
- Removemos e re-treinamos o NCF → **performance caiu 13,2%** no NDCG
- **Lição profunda**: **correlação linear não é redundância funcional**
  - Spearman captura monotonia; modelo neural aprende relações **não-lineares** que a correlação não vê
  - Features removidas podem carregar sinal sutil em certas regiões do espaço latente
- **Achado mais contraintuitivo** (última linha da tabela): ao remover **todas as 20 features auxiliares** mantendo só os 3 embeddings → NDCG **sobe 22,5%**
- **Explicação**: com 98% cold-start, MLP acaba dependendo só das features globais — "médias para todo mundo"; todos os scores ficam parecidos e o ranking morre

🧠 **INSIGHT:** "Trade-off central de sistemas com cold-start massivo: features engineered dão **sinal global** (média); embeddings dão **sinal personalizado** (mas precisam de histórico). Quando histórico é raro, **menos é mais**."

🗣️ **O QUE DIZER (reflexão honesta):**
- Preciso ser transparente: esse resultado — "a ablation superou o modelo com features" — **não é um sinal de excelência técnica**
- Em dataset denso e bem comportado, a expectativa é que mais features engineered **ajudem** o modelo
- O fato de aqui elas **prejudicarem** é sintoma de que o dataset é **pequeno demais** para o modelo aprender representações úteis a partir delas
- Em outras palavras, **o modelo está "vencendo por desistência"** — as features não conseguiram competir com a aleatoriedade dos embeddings porque não havia dados suficientes para treiná-las bem

---

# PARTE 4 — TREINAMENTO DOS MODELOS (10 min)

## SEÇÃO 9 — Setup de Avaliação: split temporal e cold-start (2 min)

👁️ **O QUE MOSTRAR:** Clicar na aba **🏋️ Baselines**. Apontar para **"📊 Configuração do Split Temporal"**

📖 **GLOSSÁRIO:**
- **Split temporal** — dividir dados por **tempo** (treino = antigo, test = novo), não aleatório
- **Data leakage** — modelo "ver" informação do futuro durante o treino
- **NDCG (Normalized Discounted Cumulative Gain)** — métrica de **qualidade do ranking**; acerto na posição 1 vale muito, na 10 vale pouco
- **HitRate@K** — "o item relevante apareceu em algum lugar do top-K?"

🗣️ **O QUE DIZER:**
- Antes dos modelos, entender **como avaliamos**
- Setup: **split temporal 70/15/15** — não aleatório
- **Por que temporal**: em produção, modelo de hoje recomenda para usuário de hoje baseado no histórico até ontem
  - Se test set for amostra aleatória, modelo "verá" interações futuras do mesmo usuário → **data leakage** → métricas infladas
- **Métrica principal**: **NDCG@10**
  - Item certo na posição 1 → vale muito
  - Item certo na posição 10 → vale pouco
  - Mede **qualidade do ranking**, não apenas acerto
- Avaliamos com **99 negativos por positivo**: para cada item comprado, mostramos 99 não-comprados; medimos se o item certo ficou no topo

---

## SEÇÃO 10 — Baselines: a régua de comparação (3 min)

👁️ **O QUE MOSTRAR:** **"📋 Tabela de Resultados"** e depois **"📈 Comparação de MAP@10 por Modelo"**

📖 **GLOSSÁRIO:**
- **Baseline** — modelo simples usado como ponto de comparação ("o sofisticado é melhor que o simples?")
- **Popularity Baseline** — recomenda itens mais comprados globalmente; não personaliza nada
- **TopRated** — itens com maior review_score médio (com filtro de mínimo de N reviews)
- **Item-Item Collaborative Filtering (CF)** — "quem comprou X também comprou Y"; similaridade entre produtos
- **TruncatedSVD** — fatoração de matrizes; base do algoritmo do prêmio Netflix
- **MAP (Mean Average Precision)** — média da precisão em cada posição onde aparece item relevante

🗣️ **O QUE DIZER (4 baselines):**
- 4 baselines — modelos **simples** que servem como **régua de comparação**
- Pergunta: o modelo neural sofisticado é **realmente** melhor que esses?
- **1. Popularidade** — itens mais comprados globalmente; não personaliza → MAP=0,0019, NDCG=0,0053
- **2. TopRated** — maior review médio; filtro de mínimo 5 ou 15 reviews → **tudo zero**
- **3. Item-Item CF** — "quem comprou X também comprou Y"; similaridade entre produtos → **tudo zero**
- **4. TruncatedSVD** — fatorar matriz usuário-item em matrizes menores; base do prêmio Netflix → **tudo zero**
- **Por que 3 dos 4 zeraram**: cold-start massivo — 98% dos usuários compraram 1 vez
  - Não existe "quem comprou X também comprou Y" para usuário com 1 item
  - Não existem "fatores latentes" para usuário com 1 interação
  - Único que sobrevive: **Popularity** (é **global**, não depende de histórico individual)

🗣️ **O QUE DIZER (reflexão honesta):**
- Preciso pausar e ser direto: **3 dos 4 baselines zerarem não é "o problema é difícil"** — é o problema sendo **impossível** para essas técnicas neste dataset
- Item-Item CF **recomendou livros na Amazon por uma década** e ganhou o prêmio Netflix
- TruncatedSVD é o **estado da arte em fatoração de matrizes há 20 anos**
- Eles não são fracos — simplesmente **não foram feitos** para datasets onde 98% dos usuários compraram uma única vez
- Popularity Baseline (NDCG=0,0053) é tecnicamente o **melhor modelo possível que não usa deep learning** neste dataset
- É o **teto** — e ele é **muito baixo**
- Sinal claro de que o Olist **não é o dataset ideal** para benchmarks de recomendação
- Explica por que a literatura acadêmica usa **MovieLens, Amazon Reviews 5-core, Yelp** — que filtram agressivamente para garantir densidade

💡 **DICA:** Preparar terreno: "Mas a abordagem neural muda completamente esse jogo. Vamos ver."

---

## SEÇÃO 11 — NCF: a virada com Embeddings (3 min)

👁️ **O QUE MOSTRAR:** Clicar na aba **🧠 NCF (MLP PyTorch)**. Apontar para **"📊 Métricas @ K=10 (Test Set)"** (5 KPIs grandes no topo)

📖 **GLOSSÁRIO:**
- **NCF (Neural Collaborative Filtering)** — substitui fatoração de matrizes por uma **rede neural** que aprende a função de scoring
- **Embedding** — pense num mapa: cada cidade tem coordenadas (lat, long) — 2 números; usuários com gostos parecidos ficam perto no mapa; embeddings são a mesma coisa, mas com 32 dimensões
- **MLP (Multi-Layer Perceptron)** — rede neural clássica com várias camadas densas; aqui: 3 embeddings concatenados (32+32+8=72) → 2 camadas (64 → 32 → 1 score)
- **BPR Loss (Bayesian Personalized Ranking)** — otimiza **o ranqueamento** par-a-par; para cada positivo, amostra 8 negativos; maximiza diferença de score

🗣️ **O QUE DIZER (arquitetura + resultado):**
- **A virada**: NCF é a versão **neural** dos baselines
- Em vez de fatorar matrizes, treinamos uma **rede neural** que aprende a dar **scores** para pares (usuário, item)
- **Arquitetura**:
  - **3 embeddings** (usuário 32D, item 32D, categoria 8D) concatenados
  - Passam por **MLP de 2 camadas** (64 → 32 → 1)
  - Output = score
  - Treinamos com **BPR Loss**: para cada positivo, mostramos 8 negativas; modelo aprende a dar scores **maiores** para positivos
- **Resultados**: **NDCG@10 = 0,2725** — cinquenta vezes melhor que o baseline!
- **49% dos top-10 contém pelo menos um item relevante** — transformação qualitativa

🗣️ **O QUE DIZER (reflexão honesta sobre o número absoluto):**
- Vou ser honesto: **NDCG = 0,2725 não é um número grande**
- Em sistemas academicamente competitivos, você encontra NDCG na faixa de **0,4 a 0,6** — estamos bem abaixo
- **Mas o contexto importa**:
  - Esse número é **60× maior** que o baseline mais forte (Popularity, 0,0045)
  - E o baseline mais forte, em datasets densos, costuma estar na faixa de **0,15-0,25**
  - Ou seja, nosso baseline é **10× mais fraco** que o baseline de um dataset denso
- **Traduzindo**: se o nosso baseline fosse 0,05, o NCF precisaria atingir ~0,40 para ser competitivo
- Aqui, o NCF atinge 0,27 partindo de um baseline de 0,005
- **Lift relativo** é enorme, mas o **número absoluto** carrega a marca do dataset fraco
- **Em produção**: NDCG de 0,27 com 49% de HitRate já é **utilizável** — suficiente para uma home page com recomendações relevantes

---

## SEÇÃO 12 — Generalização: por que a métrica no test é menor (2 min)

👁️ **O QUE MOSTRAR:** Na aba **🧠 NCF**, scroll para **"📈 Análise de Generalização"** e **"🧊 Análise de Cold-Start"**

📖 **GLOSSÁRIO:**
- **Overfitting** — modelo "decorou" o treino em vez de "aprender o padrão"
- **Generalization gap** — diferença entre performance no treino e no test (quanto menor, melhor)
- **Cold-start (de novo)** — 98% dos usuários do test são **inéditos**, nunca vistos no treino

🗣️ **O QUE DIZER:**
- **Números**: NDCG no treino = 0,58, validação = 0,34, test = 0,27
- Alguém pode pensar: "isso é **overfitting!**"
- Mas **não é** — é a **realidade do problema**
- No treino, modelo viu o usuário comprando o item → acertou
- No test, **98% dos usuários são inéditos**:
  - Nunca apareceram no treino
  - Embedding é **inicializado aleatoriamente**
  - Rede nunca ajustou esses valores
- O que estamos medindo é a capacidade do modelo de **ranquear bem** mesmo com **embeddings aleatórios**
- **Analogia**: aluno que perdeu todas as aulas ainda chuta algumas certas
- **50% de acerto no top-10** com "embeddings aleatórios" é surpreendentemente bom
- Análise de cold-start no dashboard mostra: maior parte do test é filtrada por "usuário não estava no treino"
- Estamos medindo o que **realmente** podemos recomendar em produção — **usuários novos**

---

## SEÇÃO 13 — Otimização e Hiperparâmetros (2 min)

👁️ **O QUE MOSTRAR:** Voltar para **"🏆 Comparação de Runs"** (mesma aba NCF)

📖 **GLOSSÁRIO:**
- **Hiperparâmetro** — configuração do modelo **não aprendida** (escolhida por nós)
- **Grid search / ablation** — experimentar combinações de hiperparâmetros
- **Dropout** — técnica de regularização que "desliga" N% dos neurônios durante o treino
- **Learning rate** — "tamanho do passo" que o modelo dá ao aprender
- **AdamW** — variante do otimizador Adam com **weight decay** (decaimento de pesos)

🗣️ **O QUE DIZER (5 runs + ablation):**
- **Jornada de otimização**: 5 runs com combinações de hiperparâmetros
- **Padrões observados**:
  - Maior embedding ajuda até certo ponto (16 → 32 → 64)
  - Mais dropout ajuda (0,3 → 0,5)
  - **Learning rate com scheduler** (reduz passo quando performance estagna) é crucial
- **Salto real — linha 5**: ablation removendo todas as 20 features auxiliares
  - NDCG: 0,2226 → 0,2725 = **+22,5%**
  - Em datasets com cold-start massivo, **menos é mais**

👁️ **O QUE MOSTRAR:** Scroll para **"⚙️ Hiperparâmetros do Modelo Production"** (configs/ncf_best.yaml)

🗣️ **O QUE DIZER (HPs finais):**
- **HPs finais**:
  - **Embedding 32** (user/item)
  - **MLP [64, 32]**
  - **Dropout 0,5**
  - **AdamW com lr 5e-4**
  - **BPR loss com 8 negativos por positivo**
- Tudo documentado em `configs/ncf_best.yaml` — **fonte canônica** do que está em produção
- Modelo: **4 milhões de parâmetros**, **16 MB no disco**, **5-10 min** para treinar no CPU

💡 **DICA:** Se perguntarem detalhes de HPs: "Estão todos no `configs/ncf_best.yaml`. Ponto principal: configuração **mais simples** que ainda superou tudo — só 4M de parâmetros, nada absurdo."

---

# PARTE 5 — FECHAMENTO (5 min)

## SEÇÃO 14 — Recomendações Geradas (1 min)

👁️ **O QUE MOSTRAR:** Clicar na aba **🎯 Recomendações**. Apontar para **"📋 Inventário de Artefatos"** e **"🔍 Exemplo: Top-5 Recomendações"**

📖 **GLOSSÁRIO:**
- **Top-K** — lista dos K itens mais recomendados para um usuário (padrão K=10)

🗣️ **O QUE DIZER:**
- Output dos baselines: **10 arquivos CSV** em `artifacts/baselines/` com top-K recomendações
- **Inventário completo**: Popularity, TopRated, ItemItemCF, TruncatedSVD para K=5, 10 e 20
- **Exemplo**: top-5 do Popularity para os primeiros 5 usuários
  - Categorias se concentram nas mais vendidas — esperado de baseline global
- NCF em si está no checkpoint `ncf_Ablation_FINAL_no_aux_emb32.pt`
- Gera recomendações via scoring, como mostrado no `notebooks/demo.ipynb`

---

## SEÇÃO 15 — Sobre o Pipeline (1 min)

👁️ **O QUE MOSTRAR:** Clicar na aba **ℹ️ Sobre o Pipeline**. Apontar para **"Pipeline DVC"** e **"Status dos Entregáveis"**

📖 **GLOSSÁRIO:**
- **DVC (Data Version Control)** — sistema que versiona **dados e pipelines** parecido com o Git
- **Reprodutibilidade** — garantir que outra pessoa (ou você daqui 6 meses) consegue rodar o projeto do zero

🗣️ **O QUE DIZER:**
- Resumo do pipeline: **DVC** versiona os 3 estágios de dados
- `dvc.yaml` declara: **prepare**, **featurize**, **validate**
- **Reprodutibilidade total**: `uv run dvc repro` reconstrói tudo do zero a partir dos CSVs brutos
- **Status dos entregáveis**: **9 itens concluídos** (de pipeline de dados a dashboard Streamlit)
- Para **produção**, próximos passos:
  - **FastAPI + Docker**
  - **Monitoramento com Prometheus**
  - **A/B testing** para medir impacto em CTR (taxa de clique) e conversão

---

## SEÇÃO 16 — 4 Lições que Valem para Qualquer Projeto (2 min)

👁️ **O QUE MOSTRAR:** Ficar na aba **ℹ️ Sobre o Pipeline** (ou simplesmente fechar o navegador para olhar a audiência)

🗣️ **O QUE DIZER (4 lições):**
- Quatro lições que eu levo desse projeto — servem para qualquer sistema de recomendação:

- **1. EDA antes de tudo**
  - 30 minutos analisando sparsity e cold-start evitaram semanas de tentativa e erro
  - Se pulássemos direto para o modelo, teríamos escolhido arquitetura que não funciona nesse dataset

- **2. Correlação linear não é redundância funcional**
  - Auditoria Spearman parecia boa ideia, mas modelo usava features de formas que correlação não captura
  - Hoje eu questionaria qualquer feature selection puramente estatística — sem o **teste empírico** de treinar o modelo, não dá para saber

- **3. Cold-start massivo muda a equação**
  - Em datasets densos (MovieLens, Amazon), features engineered brilham
  - Em datasets esparsos como o Olist, embeddings treinados superam
  - **Saber qual é o seu caso é metade do problema**

- **4. A escolha do dataset importa mais do que a escolha do modelo**
  - Gastamos horas ajustando hiperparâmetros, fazendo ablation studies
  - Resultado final NDCG=0,27 ainda é modesto
  - Se tivéssemos começado com **MovieLens 25M** (25 milhões de ratings, 162 mil usuários com média de 162 ratings cada), o **mesmo NCF** provavelmente passaria de NDCG=0,4 sem tuning
  - Olist foi escolhido por ser **dataset público brasileiro** e por ser didático
  - Mas **didático ≠ fácil**
  - Para quem vai começar projeto de recomendação: **invistam tempo escolhendo o dataset antes de escolher o modelo**

- **Obrigado. Perguntas?**

---

# PERGUNTAS E RESPOSTAS PREPARADAS

💡 **DICA:** Manter o **dashboard aberto** durante o Q&A. Quando alguém perguntar sobre métrica específica, clicar na aba correspondente.

### P1: "Por que split temporal e não aleatório?"

📖 **GLOSSÁRIO:** **Data leakage** — modelo "ver" informação do futuro durante o treino

🗣️ "Em sistemas de recomendação, o passado de um usuário está sempre disponível, mas o futuro não":
- Imagine um modelo em produção hoje: ele recomenda com base no histórico até ontem
- Se durante o treino o modelo viu interações do **futuro**, as métricas ficam artificialmente infladas
- O test set deixou de ser "futuro" e virou "passado"
- O split temporal reflete o cenário **real**

👁️ Mostrar: aba **🏋️ Baselines** → "Configuração do Split Temporal"

---

### P2: "Por que BPR e não BCE?"

📖 **GLOSSÁRIO:**
- **BCE (Binary Cross-Entropy)** — função de perda mais comum; "a resposta certa é 0 ou 1?"
- **BPR (Bayesian Personalized Ranking)** — otimiza a **ordem** entre pares (positivo vs negativo)

🗣️ "BPR otimiza o **ranqueamento** par-a-par":
- Para cada interação positiva, amostramos 8 negativas
- Maximizamos a **diferença de score**
- BCE trata cada interação como classificação binária independente — "este usuário gostou deste item? sim/não"
- Em recomendação, o que importa é a **ordem** dos itens, não a probabilidade absoluta
- Por isso BPR é superior

👁️ Mostrar: aba **🧠 NCF** → "Hiperparâmetros" → Loss: BPR

---

### P3: "Como o sistema lida com produtos novos?"

📖 **GLOSSÁRIO:** **Cold-start de produto** — análogo ao cold-start de usuário, mas do lado do produto

🗣️ "Produtos novos têm `category_id` conhecido, então o `cat_emb` pode ser usado":
- Mas o `item_emb` é inicializado aleatoriamente
- Hoje, usaríamos um **fallback baseado em popularidade da categoria** para produtos novos
- Para o próximo passo do projeto, queremos implementar **online learning** que ajusta o embedding do produto assim que ele recebe a primeira interação

---

### P4: "Por que não usar LSTM ou Transformer?"

📖 **GLOSSÁRIO:**
- **LSTM (Long Short-Term Memory)** — rede neural para sequências, mantém "memória"
- **Transformer** — arquitetura moderna (a do GPT); mecanismo de atenção, olha todas as palavras ao mesmo tempo
- **SASRec, BERT4Rec** — modelos de recomendação baseados em Transformer, capturam **sequência temporal** de produtos

🗣️ "Excelente pergunta. Para esta fase, o foco era validar o pipeline end-to-end com arquitetura simples mas bem fundamentada (NCF)":
- Transformer-based recommenders (SASRec, BERT4Rec) capturam a **sequência** de produtos comprados pelo usuário — próximo passo
- Brilham quando há histórico de múltiplas compras
- Como 98% dos usuários têm 1 compra aqui, o ganho esperado seria **marginal**
- Em outro dataset (Netflix), Transformer **dominaria**

---

### P5: "O que aconteceu com o TopRated? Por que zero?"

🗣️ "TopRated recomenda produtos com **maior review_score médio**, com filtro de mínimo de reviews (5 ou 15)":
- Nosso dataset tem **77% das notas entre 4 e 5** — quase todo produto é "top rated" no sentido fraco
- Quando aplicamos o filtro de mínimo de reviews, a maioria dos produtos é eliminada (70% dos produtos têm 1 compra só)
- Resultado: **lista vazia** para quase todos os usuários, especialmente os cold-start
- É um sintoma do **viés positivo** que mencionei na EDA

👁️ Mostrar: aba **🏋️ Baselines** → tabela de resultados → TopRated com zeros

---

### P6: "Quanto tempo levou para treinar o NCF?"

🗣️ "O NCF Production completo (até 20 epochs com early stop) leva cerca de **5 a 10 minutos** no CPU":
- O gargalo é o **scoring em batch durante a validação**
- Temos 93 mil usuários e 32 mil itens → para cada usuário precisamos ranquear 32 mil itens
- Em GPU seria **10× mais rápido**
- Modelo: **4 milhões de parâmetros** (16 MB no disco)
- Trivial para produção — um modelo GPT tem **100 mil vezes mais**

---

### P7: "Por que o nome `Ablation_FINAL_no_aux_emb32`?"

🗣️ "O nome tem história":
- **Ablation** → resultado de uma **ablation study** (removemos features)
- **no_aux** → sem features auxiliares (só os 3 embeddings)
- **emb32** → tamanho do embedding (32 dimensões)
- **FINAL** → último experimento antes de promover para produção
- É um nome longo, mas **autoexplicativo** — qualquer pessoa que chegar ao repositório entende o que é o modelo só pelo nome do arquivo

---

### P8: "Vocês vão conseguir colocar isso em produção?"

🗣️ "Boa pergunta. Resposta curta: **sim, mas com ressalvas**":
- Modelo está pronto tecnicamente — checkpoint salvo, métricas validadas, dashboard para visualização
- O que falta é **infraestrutura de produção**:
  - **API REST** (FastAPI) para servir recomendações
  - **Containerização** (Docker) para isolar o ambiente
  - **Monitoramento** (Prometheus + Grafana) para detectar drift
  - **A/B testing** para medir impacto em CTR/conversão
- Para um Tech Challenge, o modelo de ML é a parte visível
- Em produção **90% do trabalho é engenharia de software** — pipelines de dados, latência, resiliência, observabilidade
- Nosso próximo passo é exatamente esse: pegar o modelo e empacotar tudo

---

### P9: "O dataset do Olist foi uma boa escolha? Vocês escolheriam de novo?"

📖 **GLOSSÁRIO:** **Dataset público** — dados disponibilizados abertamente para a comunidade (Olist é dos mais usados no Brasil)

🗣️ "Vou ser direto: **o Olist foi a escolha possível, não a ideal**":
- Escolhemos por ser (1) público, (2) brasileiro, (3) real, com problemas reais
- Mas, em termos de **benchmarking**, ele é desafiador demais para o objetivo de mostrar que um modelo "funciona bem"
- **O que eu faria diferente hoje**: usaria **MovieLens 25M** (25M ratings, 162k usuários com 162 ratings médios) ou **Amazon Reviews 5-core** (filtrado para cada usuário/produto ter ≥5 interações)
- Esses datasets têm cold-start **resolvido por construção** — permitem focar no **modelo** em vez de lutar contra os dados
- Dito isso, trabalhar com dataset difícil trouxe **aprendizados que dataset fácil não traria**:
  - Valor de uma boa EDA
  - Importância de questionar feature selection estatística
  - Coragem de remover features que pareciam úteis
- **Resposta final**: o Olist foi desafiador, mas nos ensinou mais do que um dataset fácil teria ensinado

---

### P10: "O resultado final é bom? Valeu a pena?"

🗣️ "Vou separar em duas respostas: **técnica** e **de aprendizado**:

**Tecnicamente**:
- NDCG=0,27 é **modesto em absoluto**, mas **excelente em relativo** (60× sobre o baseline)
- Em produção, esse número é **utilizável** — 50% dos top-10 têm pelo menos um item relevante
- Em e-commerce real, isso já é o suficiente para gerar **impacto de receita**

**De aprendizado**:
- Valeu **muito a pena**
- Construir sistema de recomendação end-to-end — da EDA ao modelo em produção, passando por baselines, ablation, MLflow e deploy do dashboard — é o tipo de experiência que **não se consegue lendo**
- Os trade-offs que enfrentamos (cold-start, target leakage, correlação vs redundância, escolha de loss function) são exatamente o que se vê em **entrevistas técnicas** e em **projetos reais**

**Resposta curta**: o resultado técnico é modesto por causa do dataset, mas o resultado educacional é excelente. E esse é o ponto que mais importa nesse Tech Challenge."

---

# ANEXO — Comandos Úteis Durante a Apresentação

## Se perguntarem "como reproduzir isso?"

🎬 Abrir terminal e rodar:

- **Pipeline completo (DVC)**: `uv run dvc repro`
- **Baselines**: `uv run python src/train.py`
- **NCF Production (5-10 min no CPU)**:
  ```bash
  uv run python scripts/train_ncf.py \
      --ablation-no-aux --epochs 20 --emb-dim 32 \
      --hidden 64 32 --dropout 0.5 --lr 5e-4 --use-scheduler
  ```
- **Ver experimentos**: `uv run mlflow ui --backend-store-uri sqlite:///./artifacts/mlflow.db`

## Se perguntarem "onde está o código?"

- **Pipeline de dados**: `cat src/data_preparation.py`, `cat src/feature_engineering.py`
- **Modelos**: `cat src/models/ncf.py` (arquitetura), `cat src/models/factory.py` (Factory), `cat src/models/losses.py` (BPR)
- **Treino**: `cat src/train.py` (baselines), `cat scripts/train_ncf.py` (NCF)
- **Avaliação**: `cat src/training/evaluate.py` (métricas @K)

## Se perguntarem "qual a arquitetura?"

- **Resumo do modelo Production**: `cat configs/ncf_best.yaml`

## Se perguntarem "como funciona o MLflow?"

- 🎬 Mostrar aba **🧠 NCF** → **"📋 MLflow Tracking"** no dashboard (tabela com 4 experimentos e top runs)
- 🎬 Ou abrir `http://localhost:5000` no navegador → **Experiments** → **Olist_NCF_Optimization**

---

# CHECKLIST PRÉ-APRESENTAÇÃO (T-30 min)

- [ ] **Streamlit rodando** em `http://localhost:8501` (validar que carrega sem erro)
- [ ] **MLflow UI rodando** em `http://localhost:5000` (deixar em aba separada)
- [ ] **6 abas testadas**: 📊 Visão Geral → 🔧 FE → 🏋️ Baselines → 🧠 NCF → 🎯 Recomendações → ℹ️ Sobre
- [ ] **4 figuras-chave acessíveis** (caso queira mostrar PNG separado):
  - `reports/figures/01_review_score_distribution.png`
  - `reports/figures/05_user_behavior.png`
  - `reports/figures/18_correlation_heatmap.png`
  - `reports/figures/ncf_vs_baseline.png`
- [ ] **3 terminais abertos**: dashboard, MLflow, auxiliar
- [ ] **README.md aberto** no editor (para mostrar estrutura do projeto)
- [ ] **Notebook 00 aberto** no Jupyter (opcional, para mostrar pipeline)
- [ ] **Tempo ensaiado**: 30 min apresentação + 10 min perguntas
- [ ] **Água na mesa** 😄

---

# TIMING SUGERIDO

| # | Tópico | Duração | Acumulado |
|---|--------|---------|-----------|
| 1 | Abertura | 2 min | 0:02 |
| 2 | Cold-Start: o vilão | 3 min | 0:05 |
| 3 | Features Numéricas | 2 min | 0:07 |
| 4 | Categorias + Temporal | 2 min | 0:09 |
| 5 | Conclusões EDA | 1 min | 0:10 |
| 6 | FE: visão geral | 2 min | 0:12 |
| 7 | Target Encoding | 2 min | 0:14 |
| 8 | Auditoria Spearman | 3 min | 0:17 |
| 9 | Setup de avaliação | 2 min | 0:19 |
| 10 | Baselines | 3 min | 0:22 |
| 11 | NCF: a virada | 3 min | 0:25 |
| 12 | Generalização | 2 min | 0:27 |
| 13 | Otimização | 2 min | 0:29 |
| 14 | Recomendações | 1 min | 0:30 |
| 15 | Sobre o pipeline | 1 min | 0:31 |
| 16 | 4 lições | 2 min | 0:33 |
| **Q&A** | **Perguntas** | **10 min** | **0:43** |

> Se a audiência estiver mais interessada em uma parte específica, **não tenha medo de pular ou estender**. O dashboard permite voltar a qualquer aba sob demanda.
