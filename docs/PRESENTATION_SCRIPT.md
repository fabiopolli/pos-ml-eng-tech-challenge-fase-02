# Script de Apresentação — Olist Recommender System

> **Tech Challenge Fase 02 · Pós-Graduação ML Engineering**
> Formato: **Walkthrough guiado pelo dashboard Streamlit** (`front/app_vis.py`)
> Duração estimada: 30–35 min + 10 min de perguntas
> Audiência: **grupo não-especialista** (termos técnicos sempre vêm com analogia)
> Foco: EDA, escolha de features, feature engineering e treinamento dos modelos

---

## Como usar este script

- Este é um **roteiro de demonstração ao vivo** baseado em clicks no dashboard.
- Cada **seção** corresponde a uma **aba ou sub-bloco** do dashboard.
- Cada bloco tem 4 colunas:
  - 👁️ **O QUE MOSTRAR** → ação exata na UI (clique, scroll, etc.)
  - 🗣️ **O QUE DIZER** → narração para a audiência (com analogias e exemplos)
  - 📖 **GLOSSÁRIO** → definição simples de termos técnicos
  - 💡 **DICA** → cuidado ou truque visual
- Os **números importantes** estão em **negrito**.
- Os marcadores 🎬 indicam **comandos de terminal** para alternar entre ferramentas.
- 🧠 indica um **insight conceitual** que vale a pena enfatizar.

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

# PARTE 1 — O PROBLEMA (5 min)

## SEÇÃO 1 — Abertura (2 min)

👁️ **O QUE MOSTRAR:** Fique na aba **📊 Visão Geral**. Aponte para o cabeçalho e os 4 KPIs do topo.

📖 **GLOSSÁRIO:**
- **Sistema de recomendação**: software que sugere produtos/itens para um usuário. Exemplos clássicos: "Quem comprou X também comprou Y" da Amazon, a lista "Para Você" do TikTok, ou "Filmes similares" da Netflix.
- **Marketplace**: site que conecta vários vendedores a vários compradores (Olist, Mercado Livre, Amazon).
- **KPIs** (Key Performance Indicators): os 4 números grandes no topo são os "sinais vitais" do dataset.

🗣️ **O QUE DIZER:**
> "Boa tarde a todos. Hoje vou mostrar o sistema de recomendação que construímos para o **Olist**, um marketplace brasileiro. O objetivo dele é simples: dado um cliente, sugerir **produtos que ele provavelmente quer comprar**.
>
> Vocês já usam sistemas assim todos os dias — quando a Netflix sugere um filme, quando a Amazon mostra 'Clientes que compraram X também compraram Y', ou quando o Spotify monta a playlist 'Descobertas da Semana'. Por trás de cada um há um modelo de machine learning que aprende os padrões de gosto de cada usuário.
>
> No nosso caso, vamos usar este dashboard interativo como fio condutor. Aqui no topo temos os **4 KPIs principais** do dataset: **99.785 compras**, **93.358 clientes únicos**, **32.216 produtos diferentes** e **72 categorias** (cama/mesa/banho, beleza/saúde, etc.). Guardem esses números — eles são a base de tudo."

---

## SEÇÃO 2 — Cold-Start: o vilão do projeto (3 min)

👁️ **O QUE MOSTRAR:** Ainda na aba **📊 Visão Geral**, scroll para a seção **"Distribuição do `review_score`"**. Aponte para os dois gráficos de barras (volume e percentual).

📖 **GLOSSÁRIO:**
- **Feedback explícito**: quando o usuário **diz** o que achou (nota, like, comentário). É o `review_score` aqui — a nota de 1 a 5 que o cliente dá.
- **Sparsity (esparsidade)**: imagine uma planilha gigante com **uma linha por cliente** e **uma coluna por produto**. Cada célula diz se o cliente comprou aquele produto. **Quase todas as células estão vazias** (porque ninguém compra nem 1% dos 32 mil produtos). Sparsity de 99,99% significa que 99,99% das células estão vazias.
- **Cold-start**: quando o sistema sabe **pouco ou nada** sobre alguém (cliente novo, produto novo). É como pedir dica de restaurante para alguém que acabou de chegar na cidade — você não tem histórico de gostos dele.

🗣️ **O QUE DIZER:**
> "A primeira coisa que olhamos foi a distribuição das notas — o **feedback explícito** que os clientes deixam. Vejam: **77% das compras têm nota 4 ou 5** — um viés fortíssimo para o positivo. Isso já nos deu a primeira lição: o sinal de feedback está concentrado nas notas altas, e o ruído entre 4 e 5 é difícil de separar.
>
> Decidimos então tratar o problema como **feedback implícito** — em vez de prever a nota (que é ruidosa), prevemos a **probabilidade de interação**. Quem viu o item, quem comprou, quem voltou. É o que a Netflix faz: ela não tenta adivinhar se você vai dar 5 estrelas para um filme, ela tenta adivinhar se você vai **clicar** nele."

👁️ **O QUE MOSTRAR:** Scroll até **"Comportamento Temporal"** e **"Sparsity da Matriz User-Item"** (3 métricas grandes no final).

🗣️ **O QUE DIZER:**
> "Agora vejam o verdadeiro vilão do projeto: a **sparsity da matriz user-item é 99,9967%**. Para visualizar: imagine uma sala de aula com 93 mil alunos e 32 mil livros. Se cada aluno lesse em média 1 livro, teríamos 93 mil 'interações aluno-livro'. Mas a sala tem 93 mil × 32 mil = **3 bilhões de pares possíveis**. Apenas 99.785 deles foram 'observados'. O resto é buraco.
>
> E como o gráfico de comportamento mostra, **98% dos usuários compraram uma única vez**. Isso é **cold-start massivo**. Por que isso importa? Qualquer feature que dependa de histórico — 'média de gasto', 'categorias preferidas' — vai estar **vazia** para 98% dos clientes.
>
> Esse foi o insight que guiou toda a engenharia: tudo que construímos precisa funcionar **mesmo para quem comprou uma vez**. Quem compra duas vezes já é 'veterano' no nosso dataset."

🧠 **INSIGHT:** "Cold-start massivo é o que diferencia recomendação de outras áreas de ML. Em classificação de imagens, cada exemplo tem muitas features (pixels). Aqui, **a maioria dos clientes tem ZERO exemplos** além da própria interação que estamos prevendo."

💡 **DICA:** Faça uma pausa de 2 segundos após dizer "98% dos usuários compraram uma única vez". Espere a ficha cair.

---

# PARTE 2 — EDA: ENTENDENDO OS DADOS (8 min)

## SEÇÃO 3 — Features Numéricas e a Cauda Longa (2 min)

👁️ **O QUE MOSTRAR:** Ainda na aba **📊 Visão Geral**, scroll para **"📈 Distribuições Numéricas (price, freight_value)"** e depois **"📐 Análise de Cauda Longa (Escala Logarítmica)"**.

📖 **GLOSSÁRIO:**
- **Cauda longa (long tail)**: numa distribuição de preços, a maioria dos produtos está na faixa de R$ 20-200, mas existem alguns produtos com R$ 5.000, R$ 10.000. Esses valores extremos são a "cauda" — raros mas muito altos.
- **Outlier**: um valor que foge muito do padrão. R$ 10 mil num dataset onde a mediana é R$ 80 é um outlier.
- **Log-transform**: aplicar logaritmo a uma variável. O efeito prático: **espreme** os valores grandes e **estica** os pequenos. R$ 10.000 vira 4 (log10), R$ 100 vira 2, R$ 10 vira 1. A distribuição vira mais simétrica.
- **Percentil 99**: o valor abaixo do qual estão 99% dos dados. Os 1% maiores são 'outliers pelo percentil 99'.

🗣️ **O QUE DIZER:**
> "Aqui estão as duas features numéricas brutas: **preço** e **freito**. Olhem a escala da esquerda: quase tudo está concentrado no canto inferior esquerdo, e algumas observações passam de **R$ 10 mil**. Isso é a tal **cauda longa** — pouquíssimos produtos caríssimos, muitíssimos produtos baratos.
>
> Por que isso é um problema? Modelos de machine learning são sensíveis a **escala**. Se o preço médio é R$ 100, um produto de R$ 10.000 'puxa' a média e o modelo acha que ele é 100× mais importante. Para resolver, aplicamos **log-transform**: o logaritmo de R$ 10.000 é 4, o de R$ 100 é 2, o de R$ 10 é 1. A diferença de magnitude desaparece, e a distribuição vira **simétrica**, como o gráfico da direita mostra.
>
> Também criamos uma flag `has_price_outlier` para marcar observações fora do percentil 99. É como dizer para o modelo: 'esse aqui é especial, vê o que acha'."

---

## SEÇÃO 4 — Categorias e Comportamento Temporal (2 min)

👁️ **O QUE MOSTRAR:** Ainda na aba **📊 Visão Geral**, scroll para **"🛍️ Top 15 Categorias Mais Vendidas"** e **"📅 Comportamento Temporal"**.

📖 **GLOSSÁRIO:**
- **One-Hot Encoding (OHE)**: transformar uma coluna categórica (tipo de pagamento) em várias colunas binárias (0 ou 1). Ex: pagamento = 'cartão de crédito' vira `[1, 0, 0, 0]`. É a forma padrão de dar categorias para o modelo.
- **Sazonalidade**: padrão que se repete ao longo do tempo (Black Friday em novembro, Natal em dezembro, mais vendas em fim de semana).

🗣️ **O QUE DIZER:**
> "Vejam o gráfico de top 15 categorias: **cama_mesa_banho**, **beleza_saúde** e **esporte_lazer** dominam. É o clássico **'cabeça longa'** de e-commerce: 15 categorias concentram a maioria das vendas, e as outras 57 dividem o resto.
>
> O gráfico temporal mostra outro padrão importante: as vendas estão concentradas em **2017-2018**, com picos claros em **novembro** (Black Friday) e **dezembro** (Natal). Isso virou uma feature: criamos flags como `is_holiday_season` e `is_weekend`. Por que isso é útil? Porque **essas features funcionam mesmo para usuários cold-start** — elas dependem só do calendário, não do histórico individual. Um cliente que nunca comprou antes, se comprar no Natal, recebe o boost de 'época natalina'."

---

## SEÇÃO 5 — Conclusões da EDA → Decisões de Design (1 min)

👁️ **O QUE MOSTRAR:** Scroll até o final da aba **📊 Visão Geral** e pare por um momento antes de mudar de aba.

🗣️ **O QUE DIZER:**
> "Fechando a EDA, três decisões viraram **princípios de design** que carregamos para o resto do projeto:
>
> 1. **Tratar como feedback implícito (BPR Loss)** — porque a nota tem muito viés positivo, e o que importa é ranquear.
> 2. **Log-transform em preço e frete** — para neutralizar a cauda longa.
> 3. **Priorizar embeddings sobre features engineered** — porque com 98% cold-start, features que dependem de histórico do usuário são ruído para a maioria."

---

# PARTE 3 — FEATURE ENGINEERING (8 min)

## SEÇÃO 6 — Visão Geral: de 10 para 42 colunas (2 min)

👁️ **O QUE MOSTRAR:** Clique na aba **🔧 Feature Engineering** (segunda aba). Aponte para **"📋 Mapa de Features por Categoria"** e **"🔄 Comparativo Antes/Depois da FE"**.

📖 **GLOSSÁRIO:**
- **Feature engineering**: o processo de **criar novas colunas** a partir dos dados brutos para ajudar o modelo a aprender melhor. É considerado a parte **mais importante** e mais demorada de um projeto de ML.
- **Embedding**: vetor denso de números (ex: 32 dimensões) que representa uma entidade (usuário, produto, categoria). É como se cada usuário tivesse um "endereço" num espaço de 32 dimensões, e usuários parecidos ficassem perto.
- **Target encoding**: substituir uma categoria pela **média do target** (ex: review médio) dessa categoria. É um truque esperto, mas precisa de cuidado para não vazar informação do futuro.
- **Bayesian smoothing (suavização Bayesiana)**: técnica para evitar que categorias raras tenham valores extremos. Imagine que uma categoria tem só 1 produto com nota 5 — em vez de dizer 'essa categoria tem média 5,0', dizemos 'tem média entre 4,5 e 5,0, dependendo de quanta certeza temos'.

🗣️ **O QUE DIZER:**
> "Agora entramos na fase mais artesanal: **feature engineering** — criar novas colunas a partir dos dados brutos. Saímos de **10 colunas brutas para 42 engenheiradas**.
>
> A tabela no topo mostra o mapa: 18 numéricas (log de preço, log de frete, etc.), 7 temporais (dia, mês, flags de feriado), 4 pagamentos em **OHE** (One-Hot Encoding: cada tipo vira uma coluna 0/1), e 10 agregações de usuário/produto. E no fim, **3 embeddings**: um para usuário, um para produto, um para categoria.
>
> O que é embedding? É um truque clássico: cada entidade vira um **vetor de 32 números** que o modelo aprende do zero. Usuários parecidos vão parar com vetores parecidos. É como se cada usuário morasse num 'endereço' em um espaço de 32 dimensões, e o modelo aprendesse quem mora perto de quem."

---

## SEÇÃO 7 — Target Encoding: a média com cautela (2 min)

👁️ **O QUE MOSTRAR:** Aponte para **"🏆 Categorias com Maior Target Encoding"** e em seguida **"🔗 Mapa de Correlação"**.

📖 **GLOSSÁRIO:**
- **Target leakage**: quando o modelo 'vê' informação do futuro ou do target durante o treino, gerando métricas infladas que não se reproduzem em produção. Ex: usar a média geral de reviews (que inclui o test set) para fazer encoding.

🗣️ **O QUE DIZER:**
> "Para a coluna `category_id` (que tem 72 valores possíveis), usar one-hot encoding geraria uma matriz enorme de zeros. Usamos então **target encoding com Bayesian smoothing**: cada categoria recebe a **média de review_score** dos seus produtos, mas com um truque — categorias com poucos produtos 'puxam' para a média global.
>
> É como um professor dando nota: se um aluno fez só 1 prova e tirou 10, o professor não vai dizer 'esse aluno é nota 10' — vai esperar mais provas. O `alpha=10` no nosso código é esse 'fator de paciência'.
>
> Mas tem um **risco grave** aqui: o **target leakage**. Se a média da categoria for calculada usando o **test set**, o modelo 'cola' — vê respostas do teste no treino. Por isso, calculamos a média apenas no **train set** e aplicamos ao test. Isso é o **split temporal** salvando a gente."

👁️ **O QUE MOSTRAR:** Aponte para **"🔗 Mapa de Correlação (Features Numéricas)"** no final da aba.

🗣️ **O QUE DIZER:**
> "Aqui está o mapa de correlação entre as features numéricas. Cores quentes (vermelho/laranja) = correlação alta; cores frias (azul) = correlação baixa ou negativa. Vejam vários pares com correlação 0,9 ou mais: `price` e `price_log` (r=1,0 — são literalmente a mesma informação), `freight_value` e `freight_value_log` (r=1,0), e outros.
>
> Isso é **redundância linear óbvia** — duas features carregando o mesmo sinal. Já removemos essas na primeira rodada de feature selection. Mas e as redundâncias mais sutis? É o que vamos ver agora."

---

## SEÇÃO 8 — Auditoria Spearman: o achado contraintuitivo (3 min)

👁️ **O QUE MOSTRAR:** Mude para a aba **🧠 NCF** (quarta aba), depois **"🏆 Comparação de Runs (Etapa 4 — Otimização)"** — a tabela que mostra 5 runs + 1 ablation.

📖 **GLOSSÁRIO:**
- **Spearman ρ (rho)**: correlação de Spearman. Diferente da correlação de Pearson (que mede relação **linear**), Spearman mede relação **monotônica** — se X sobe, Y também sobe? Não importa se a relação é linear, exponencial, ou qualquer curva monotônica. É mais robusto a outliers.
- **Ablation study**: na área de deep learning, é o processo de **remover um componente de cada vez** para ver o quanto ele contribuía. É a forma de provar 'isso aqui é importante'.
- **|ρ| > 0,95**: o valor absoluto da correlação é maior que 0,95 (correlação quase perfeita).

🗣️ **O QUE DIZER:**
> "Agora vem um dos achados mais interessantes do projeto. Depois de remover as redundâncias óbvias, rodamos uma **segunda rodada** de feature selection com Spearman (|ρ| > 0,95), que é mais robusto a outliers que Pearson. Identificamos 2 pares:
>
> - `user_recency_days` vs `days_since_reference` (ρ = **−0,986**) — basicamente a mesma informação, em sinal contrário.
> - `freight_value_log` vs `user_avg_freight` (ρ = **+0,966**) — frete do produto e frete médio do usuário, redundantes.
>
> Removemos essas 2 features e re-treinamos o NCF. Resultado? **Performance caiu 13,2%** no NDCG. Ou seja, o modelo **piorou**.
>
> A lição é profunda: **correlação linear não é redundância funcional**. Spearman captura monotonia, mas o modelo neural aprende relações **não-lineares** que a correlação não vê. As features removidas podem carregar sinal sutil em certas regiões do espaço latente.
>
> E o achado mais contraintuitivo vem na última linha da tabela: quando **removemos todas as 20 features auxiliares** e mantemos só os 3 embeddings, o NDCG **sobe 22,5%**. A explicação: com 98% cold-start, os embeddings são inicializados aleatórios e o MLP acaba dependendo só das features globais — que são 'médias para todo mundo'. Todos os scores ficam parecidos, e o ranking morre."

🧠 **INSIGHT:** "Esse é o **trade-off central** de sistemas com cold-start massivo: features engineered dão sinal global (média), embeddings dão sinal personalizado (mas precisam de histórico). Quando histórico é raro, **menos é mais**."

---

# PARTE 4 — TREINAMENTO DOS MODELOS (10 min)

## SEÇÃO 9 — Setup de Avaliação: split temporal e cold-start (2 min)

👁️ **O QUE MOSTRAR:** Clique na aba **🏋️ Baselines** (terceira aba). Aponte para **"📊 Configuração do Split Temporal"**.

📖 **GLOSSÁRIO:**
- **Split temporal**: dividir os dados por **tempo**, não aleatoriamente. Treino = dados antigos, test = dados novos. É o que simula produção: o modelo de hoje recomenda para o usuário de hoje.
- **Data leakage**: o modelo 'ver' informação do futuro durante o treino. Split aleatório causa isso (mistura passado e futuro), split temporal evita.
- **NDCG (Normalized Discounted Cumulative Gain)**: métrica que mede **qualidade do ranking**, não apenas acerto. Um acerto na posição 1 vale muito, na posição 10 vale pouco. É a métrica padrão em sistemas de recomendação.
- **HitRate@K**: 'o item relevante apareceu **em algum lugar** do top-K?'. Simples e intuitivo.

🗣️ **O QUE DIZER:**
> "Antes de falar dos modelos, vamos entender **como avaliamos**. O setup é **split temporal 70/15/15** — não aleatório. Por que? Porque em produção, o modelo de hoje recomenda para o usuário de hoje baseado no histórico até ontem. Se o test set for uma amostra aleatória, o modelo 'verá' interações futuras do mesmo usuário — **data leakage** — e as métricas ficam artificialmente infladas.
>
> A métrica principal é o **NDCG@10** (Normalized Discounted Cumulative Gain). A ideia é simples: se o item certo aparece na **posição 1** do ranking, vale muito. Se aparece na **posição 10**, vale pouco. É uma métrica de **qualidade do ranking**, não apenas de acerto.
>
> Avaliamos com **99 negativos por positivo**: para cada item que o usuário comprou, mostramos 99 itens que ele **não** comprou, e medimos se o item certo ficou no topo."

---

## SEÇÃO 10 — Baselines: a régua de comparação (3 min)

👁️ **O QUE MOSTRAR:** Aponte para **"📋 Tabela de Resultados"** e depois **"📈 Comparação de MAP@10 por Modelo"**.

📖 **GLOSSÁRIO:**
- **Baseline**: modelo **simples** usado como ponto de comparação. A pergunta é sempre: 'o modelo sofisticado é **melhor** que o simples?'.
- **Popularity Baseline**: recomenda os itens **mais comprados globalmente** para todo mundo. Não personaliza nada — todo usuário recebe a mesma lista.
- **TopRated**: recomenda os itens com **maior review_score médio**. Mas tem um filtro: 'mínimo de N reviews', para evitar recomendar produtos com 1 review só de 5 estrelas.
- **Item-Item Collaborative Filtering (CF)**: a ideia clássica da Amazon — 'quem comprou X também comprou Y'. Calcula **similaridade entre produtos** baseada em quem os comprou junto. Se você comprou um livro de Python e muitas pessoas que compraram esse livro também compraram 'Fluent Python', o sistema recomenda 'Fluent Python'.
- **TruncatedSVD**: técnica de **redução de dimensionalidade** da álgebra linear. Pega a matriz gigante usuário-item e a decompõe em matrizes menores que capturam os 'fatores latentes' (gostos escondidos). É a base do algoritmo Surprise da Netflix.
- **MAP (Mean Average Precision)**: média da precisão em cada posição onde aparece um item relevante. Parecido com NDCG, mas com fórmula diferente.

🗣️ **O QUE DIZER:**
> "Treinamos 4 baselines — modelos **simples** que servem como régua de comparação. A pergunta é: será que o modelo neural sofisticado é **realmente** melhor que esses?
>
> **1. Popularidade** — recomenda os itens mais comprados globalmente. Não personaliza nada, todo mundo recebe a mesma lista. Resultado: MAP=0,0019, NDCG=0,0053.
>
> **2. TopRated** — recomenda os itens com maior review médio. Mas com filtro: mínimo de 5 ou 15 reviews, para evitar recomendar produtos com 1 review de 5 estrelas. Resultado: **tudo zero**.
>
> **3. Item-Item CF** — a ideia clássica da Amazon: 'quem comprou X também comprou Y'. Calcula similaridade entre produtos. Resultado: **tudo zero**.
>
> **4. TruncatedSVD** — fatore a matriz usuário-item em matrizes menores (técnica de álgebra linear) e use os 'fatores latentes' como features. É a base do algoritmo do prêmio Netflix. Resultado: **tudo zero**.
>
> Por que 3 dos 4 baselines zeraram? Lembra do cold-start massivo: **98% dos usuários compraram uma vez**. Não existe 'quem comprou X também comprou Y' para um usuário que comprou 1 item. Não existem 'fatores latentes' para um usuário com 1 interação. O único baseline que sobrevive é o **Popularity**, porque ele é **global** — não depende de histórico individual."

💡 **DICA:** Este é o momento de **preparar o terreno** para a próxima seção: "Mas a abordagem neural muda completamente esse jogo. Vamos ver."

---

## SEÇÃO 11 — NCF: a virada com Embeddings (3 min)

👁️ **O QUE MOSTRAR:** Clique na aba **🧠 NCF (MLP PyTorch)** (quarta aba). Aponte para **"📊 Métricas @ K=10 (Test Set)"** — os 5 grandes KPIs no topo.

📖 **GLOSSÁRIO:**
- **NCF (Neural Collaborative Filtering)**: técnica que **substitui** a fatoração de matrizes clássica por uma **rede neural**. Em vez de aprender fatores latentes diretamente, a rede aprende a função de scoring.
- **Embedding**: para entender melhor, pense num mapa. Cada cidade tem coordenadas (latitude, longitude) — 2 números. Usuários com gostos parecidos ficam perto no mapa. Embeddings são a mesma coisa, mas com 32 dimensões: cada usuário vira um ponto num espaço de 32D, e usuários parecidos ficam perto.
- **MLP (Multi-Layer Perceptron)**: rede neural clássica com várias camadas densas. Aqui: pega os 3 embeddings concatenados (32+32+8=72 números) e passa por 2 camadas (64 → 32 → 1 score).
- **BPR Loss (Bayesian Personalized Ranking)**: função de perda que otimiza **o ranqueamento**, par-a-par. Para cada item que o usuário gostou, amostramos 8 que ele não gostou, e treinamos o modelo a dar score **maior** para o positivo.

🗣️ **O QUE DIZER:**
> "E aqui está a virada. O **NCF (Neural Collaborative Filtering)** é a versão neural dos baselines. Em vez de fatorar matrizes, treinamos uma **rede neural** que aprende a dar **scores** para pares (usuário, item).
>
> A arquitetura é simples: **3 embeddings** (usuário com 32 dimensões, item com 32, categoria com 8) são concatenados e passam por um **MLP de 2 camadas** (64 → 32 → 1). O output é um score. Treinamos com **BPR Loss**: para cada interação positiva, mostramos 8 negativas, e o modelo aprende a dar scores **maiores** para os positivos.
>
> E os resultados? **NDCG@10 = 0,2725** — cinquenta vezes melhor que o baseline de popularidade! **49% dos top-10 contém pelo menos um item relevante**. Isso é uma transformação qualitativa."

---

## SEÇÃO 12 — Generalização: por que a métrica no test é menor (2 min)

👁️ **O QUE MOSTRAR:** Ainda na aba **🧠 NCF**, scroll para **"📈 Análise de Generalização (Train vs Val vs Test)"** e **"🧊 Análise de Cold-Start"**.

📖 **GLOSSÁRIO:**
- **Overfitting**: o modelo 'decorou' o treino em vez de 'aprender o padrão'. Ex: aluno que memoriza as respostas da prova anterior mas não entende a matéria.
- **Generalization gap**: a diferença entre a performance no treino e no test. Quanto menor, melhor.
- **Cold-start (de novo)**: 98% dos usuários do test são **inéditos** — nunca vistos no treino. O modelo precisa ranquear **sem ter aprendido nada** sobre eles.

🗣️ **O QUE DIZER:**
> "Vejam os números: NDCG no treino é 0,58, na validação 0,34, no test 0,27. Alguém pode pensar: 'isso é overfitting!'. Mas **não é** — é a realidade do problema.
>
> No treino, o modelo viu o usuário comprando o item — ele acertou. No test, **98% dos usuários são inéditos**: nunca apareceram no treino. O embedding deles é **inicializado aleatoriamente** e a rede nunca ajustou esses valores. O que estamos medindo é a capacidade do modelo de **ranquear bem** mesmo com embeddings aleatórios.
>
> É como dar uma prova para um aluno que perdeu todas as aulas: ele ainda pode chutar algumas certas. No nosso caso, **50% de acerto no top-10** com 'embeddings aleatórios' é surpreendentemente bom.
>
> A análise de cold-start no dashboard mostra exatamente isso: a maior parte do test set é filtrada por 'usuário não estava no treino'. Estamos medindo o que **realmente** podemos recomendar em produção — usuários novos."

---

## SEÇÃO 13 — Otimização e Hiperparâmetros (2 min)

👁️ **O QUE MOSTRAR:** Volte para **"🏆 Comparação de Runs"** (na mesma aba NCF).

📖 **GLOSSÁRIO:**
- **Hiperparâmetro**: configuração do modelo que **não** é aprendida — é escolhida por nós. Exemplos: tamanho do embedding (16, 32, 64?), taxa de dropout (0,3? 0,5?), learning rate.
- **Grid search / ablation**: experimentar combinações de hiperparâmetros para ver qual dá melhor resultado. É demorado, mas necessário.
- **Dropout**: técnica de regularização que **desliga** aleatoriamente N% dos neurônios durante o treino. Força a rede a não depender de um único neurônio, melhorando a generalização.
- **Learning rate**: o 'tamanho do passo' que o modelo dá ao aprender. Grande demais = pula o ótimo; pequeno demais = demora demais.
- **AdamW**: variante do otimizador Adam com **weight decay** (decaimento de pesos) — ajuda a regularização.

🗣️ **O QUE DIZER:**
> "Aqui está a jornada de otimização. Fizemos 5 runs com combinações de hiperparâmetros, e em cada uma medimos o NDCG. O padrão: maior embedding ajuda até certo ponto (16→32→64), mais dropout ajuda (0,3→0,5), e o **learning rate com scheduler** (que reduz o passo quando a performance estagna) é crucial.
>
> Mas o salto real veio da **linha 5**: a **ablation** removendo todas as 20 features auxiliares. NDCG subiu de 0,2226 para 0,2725 — **+22,5%**. Já falei disso, mas vale repetir: em datasets com cold-start massivo, **menos é mais**."

👁️ **O QUE MOSTRAR:** Scroll para **"⚙️ Hiperparâmetros do Modelo Production"** (configs/ncf_best.yaml).

🗣️ **O QUE DIZER:**
> "Estes são os hiperparâmetros finais: **embedding 32, MLP [64, 32], dropout 0,5, AdamW com lr 5e-4, BPR loss com 8 negativos por positivo**. Está tudo documentado no `configs/ncf_best.yaml` — a **fonte canônica** do que está em produção. O modelo tem 4 milhões de parâmetros, ocupa 16 MB no disco, e leva cerca de 5-10 minutos para treinar no CPU."

💡 **DICA:** Se alguém perguntar detalhes de HPs: "Estão todos no `configs/ncf_best.yaml`. O ponto principal é: escolhemos a configuração **mais simples** que ainda superou tudo — o modelo de produção tem só 4 milhões de parâmetros, nada absurdo."

---

# PARTE 5 — FECHAMENTO (5 min)

## SEÇÃO 14 — Recomendações Geradas (1 min)

👁️ **O QUE MOSTRAR:** Clique na aba **🎯 Recomendações** (quinta aba). Aponte para **"📋 Inventário de Artefatos"** e **"🔍 Exemplo: Top-5 Recomendações"**.

📖 **GLOSSÁRIO:**
- **Top-K**: lista dos K itens mais recomendados para um usuário. K=10 é o padrão, mas pode ser K=5, K=20.

🗣️ **O QUE DIZER:**
> "Para fechar a parte técnica, aqui está o output dos baselines: **10 arquivos CSV** em `artifacts/baselines/` com as top-K recomendações. A tabela mostra o inventário completo: Popularity, TopRated, ItemItemCF, TruncatedSVD para K=5, 10 e 20.
>
> O exemplo mostra o **top-5 do Popularity** para os primeiros 5 usuários. Note que as categorias se concentram nas mais vendidas — é o esperado de um baseline global. O NCF em si está no checkpoint `ncf_Ablation_FINAL_no_aux_emb32.pt` e gera recomendações via scoring, como mostrado no `notebooks/demo.ipynb`."

---

## SEÇÃO 15 — Sobre o Pipeline (1 min)

👁️ **O QUE MOSTRAR:** Clique na aba **ℹ️ Sobre o Pipeline** (última aba). Aponte para a seção de **Pipeline DVC** e o **Status dos Entregáveis**.

📖 **GLOSSÁRIO:**
- **DVC (Data Version Control)**: sistema que versiona **dados e pipelines** parecido com o Git. Permite reproduzir experimentos com datasets antigos.
- **Reprodutibilidade**: garantir que outra pessoa (ou você mesmo daqui a 6 meses) consegue rodar o projeto do zero e obter os mesmos resultados.

🗣️ **O QUE DIZER:**
> "Para fechar, este é o resumo do pipeline. Usamos **DVC** para versionar os 3 estágios de dados. O `dvc.yaml` declara prepare, featurize e validate. A reprodutibilidade é total: rodar `uv run dvc repro` reconstrói tudo do zero a partir dos CSVs brutos.
>
> O status dos entregáveis mostra: **9 itens concluídos**, de pipeline de dados a dashboard Streamlit. Para deploy em produção, os próximos passos são: FastAPI + Docker, monitoramento com Prometheus, e A/B testing para medir impacto em CTR (taxa de clique) e conversão."

---

## SEÇÃO 16 — 3 Lições que Valem para Qualquer Projeto (2 min)

👁️ **O QUE MOSTRAR:** Fique na aba **ℹ️ Sobre o Pipeline**, scroll para a seção **"Próximos Passos"** se houver, ou simplesmente feche o navegador se preferir olhar para a audiência.

🗣️ **O QUE DIZER:**
> "Para fechar, três lições que eu levo desse projeto — e que servem para qualquer sistema de recomendação:
>
> **1. EDA antes de tudo.** Os 30 minutos gastos analisando sparsity e cold-start evitaram semanas de tentativa e erro. Se a gente tivesse pulado direto para o modelo, teríamos escolhido uma arquitetura que não funciona nesse dataset.
>
> **2. Correlação linear não é redundância funcional.** A auditoria Spearman parecia uma boa ideia, mas o modelo estava usando as features de formas que a correlação não captura. Hoje eu questionaria qualquer feature selection puramente estatística — sem o **teste empírico** de treinar o modelo, não dá para saber.
>
> **3. Cold-start massivo muda a equação.** Em datasets densos (MovieLens, Amazon), features engineered brilham. Em datasets esparsos como o Olist, embeddings treinados superam. **Saber qual é o seu caso é metade do problema**.
>
> Obrigado. Perguntas?"

---

# PERGUNTAS E RESPOSTAS PREPARADAS

💡 **DICA:** Tenha o **dashboard aberto** durante o Q&A. Quando alguém perguntar sobre uma métrica específica, você pode clicar na aba correspondente e mostrar o número ao vivo.

### P1: "Por que split temporal e não aleatório?"

📖 **GLOSSÁRIO:**
- **Data leakage (de novo)**: o modelo 'ver' informação do futuro durante o treino.

🗣️ "Em sistemas de recomendação, o passado de um usuário está sempre disponível, mas o futuro não. Imagine um modelo em produção hoje: ele recomenda com base no histórico até ontem. Se durante o treino o modelo viu interações do futuro, as métricas ficam artificialmente infladas — o **test set** deixou de ser 'futuro' e virou 'passado'. O split temporal reflete o cenário real."

👁️ Mostre: aba **🏋️ Baselines** → "Configuração do Split Temporal".

---

### P2: "Por que BPR e não BCE?"

📖 **GLOSSÁRIO:**
- **BCE (Binary Cross-Entropy)**: a função de perda mais comum. Para cada exemplo, pergunta: 'a resposta certa é 0 ou 1?'. Usada em classificação binária.
- **BPR (Bayesian Personalized Ranking)**: otimiza a **ordem** entre pares (positivo vs negativo), não a probabilidade de cada um.

🗣️ "BPR otimiza o **ranqueamento** par-a-par: para cada interação positiva, amostramos 8 negativas e maximizamos a diferença de score. BCE trata cada interação como classificação binária independente — 'este usuário gostou deste item? sim/não'. Em recomendação, o que importa é a **ordem** dos itens, não a probabilidade absoluta de cada um. Por isso BPR é superior."

👁️ Mostre: aba **🧠 NCF** → "Hiperparâmetros" → Loss: BPR.

---

### P3: "Como o sistema lida com produtos novos?"

📖 **GLOSSÁRIO:**
- **Cold-start de produto**: análogo ao cold-start de usuário, mas do lado do produto. Produto novo no catálogo, sem histórico.

🗣️ "Produtos novos têm `category_id` conhecido, então o `cat_emb` pode ser usado. Mas o `item_emb` é inicializado aleatoriamente. Hoje, usaríamos um **fallback baseado em popularidade da categoria** para produtos novos. Para o próximo passo do projeto, queremos implementar **online learning** que ajusta o embedding do produto assim que ele recebe a primeira interação."

---

### P4: "Por que não usar LSTM ou Transformer?"

📖 **GLOSSÁRIO:**
- **LSTM (Long Short-Term Memory)**: tipo de rede neural que processa **sequências** (textos, séries temporais), mantendo 'memória' do que viu antes.
- **Transformer**: arquitetura moderna (a mesma do GPT) que processa sequências com **mecanismo de atenção** — em vez de ler palavra por palavra, olha para todas as palavras ao mesmo tempo.
- **SASRec, BERT4Rec**: modelos de recomendação baseados em Transformer, que capturam a **sequência temporal** de produtos comprados.

🗣️ "Excelente pergunta. Para esta fase, o foco era validar o pipeline end-to-end com uma arquitetura simples mas bem fundamentada (NCF). Transformer-based recommenders como SASRec ou BERT4Rec capturam a **sequência** de produtos comprados pelo usuário — isso é o próximo passo. Eles brilham quando há histórico de múltiplas compras, mas como 98% dos usuários têm 1 compra aqui, o ganho esperado seria marginal. Em outro dataset — Netflix, por exemplo — Transformer dominaria."

---

### P5: "O que aconteceu com o TopRated? Por que zero?"

🗣️ "TopRated recomenda produtos com **maior review_score médio**, mas com filtro de mínimo de reviews (5 ou 15). Nosso dataset tem **77% das notas entre 4 e 5** — quase todo produto é 'top rated' no sentido fraco. E quando aplicamos o filtro de mínimo de reviews, a maioria dos produtos é eliminada (lembrem: 70% dos produtos têm 1 compra só). O resultado é lista vazia para quase todos os usuários, especialmente os cold-start. É um sintoma do **viés positivo** que mencionei na EDA."

👁️ Mostre: aba **🏋️ Baselines** → tabela de resultados → TopRated com zeros.

---

### P6: "Quanto tempo levou para treinar o NCF?"

🗣️ "O NCF Production completo (até 20 epochs com early stop) leva cerca de **5 a 10 minutos** no CPU. O gargalo é o **scoring em batch durante a validação**: temos 93 mil usuários e 32 mil itens, então para cada usuário precisamos ranquear 32 mil itens. Em GPU seria 10× mais rápido. O modelo tem 4 milhões de parâmetros (16 MB no disco), o que é trivial para produção — um modelo GPT tem 100 mil vezes mais."

---

### P7: "Por que o nome `Ablation_FINAL_no_aux_emb32`?"

🗣️ "O nome tem história. **Ablation** porque foi o resultado de uma **ablation study** que removemos features. **no_aux** significa **sem features auxiliares** (só os 3 embeddings). **emb32** é o tamanho do embedding (32 dimensões). E **FINAL** é porque foi o último experimento antes de promover para produção. É um nome longo, mas é autoexplicativo — qualquer pessoa que chegar ao repositório entende o que é o modelo só pelo nome do arquivo."

---

### P8: "Vocês vão conseguir colocar isso em produção?"

🗣️ "Boa pergunta. A resposta curta: **sim, mas com ressalvas**. O modelo está pronto tecnicamente — checkpoint salvo, métricas validadas, dashboard para visualização. O que falta é **infraestrutura de produção**:
>
> - **API REST** (FastAPI) para servir recomendações
> - **Containerização** (Docker) para isolar o ambiente
> - **Monitoramento** (Prometheus + Grafana) para detectar drift
> - **A/B testing** para medir impacto em CTR/conversão
>
> Para um Tech Challenge, o modelo de ML é a parte visível, mas em produção **90% do trabalho é engenharia de software** — pipelines de dados, latência, resiliência, observabilidade. O nosso próximo passo é exatamente esse: pegar o modelo e empacotar tudo."

---

# ANEXO — Comandos Úteis Durante a Apresentação

## Se perguntarem "como reproduzir isso?"

🎬 Abra um terminal e rode:

```bash
# Pipeline completo (DVC)
uv run dvc repro

# Baselines
uv run python src/train.py

# NCF Production (5-10 min no CPU)
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
# Pipeline de dados
cat src/data_preparation.py
cat src/feature_engineering.py

# Modelos
cat src/models/ncf.py        # arquitetura NCF
cat src/models/factory.py    # Factory Method
cat src/models/losses.py     # BPR Loss

# Treino
cat src/train.py             # baselines
cat scripts/train_ncf.py     # NCF

# Avaliação
cat src/training/evaluate.py # métricas @K
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
- [ ] **Tempo ensaiado**: 30 min apresentação + 10 min perguntas
- [ ] **Água na mesa** 😄

---

# TIMING SUGERIDO

| Seção | Tópico | Duração | Acumulado |
|-------|--------|---------|-----------|
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
| 16 | 3 lições | 2 min | 0:33 |
| **Perguntas** | **Q&A** | **10 min** | **0:43** |

Se a audiência estiver mais interessada em uma parte específica, **não tenha medo de pular ou estender**. O dashboard permite que vocês voltem a qualquer aba sob demanda.
