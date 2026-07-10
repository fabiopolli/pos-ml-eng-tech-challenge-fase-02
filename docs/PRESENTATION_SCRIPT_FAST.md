# Script de Apresentação — PARTE 1 (2 min)

> **Tech Challenge Fase 02 · Pós-Graduação ML Engineering**
> Formato: **Walkthrough ultra-rápido guiado pelo dashboard Streamlit** (`front/app_vis.py`)
> Duração: **2 minutos** (Parte 1 — Problema + Cold-Start + Decisões de Design)
> Método: **STAR** (Situation → Task → Action → Result)

---

## Como usar este script

- Cada bloco tem **5 colunas**:
  - ⏱️ **TEMPO** → tempo acumulado desde o início
  - 👁️ **O QUE MOSTRAR** → ação exata na UI (clique/aba/seção)
  - 🎯 **STAR** → em qual fase do método você está
  - 🗣️ **O QUE DIZER** → fala curta, cronometrada
  - 💡 **DICA** → cuidado visual ou de tom
- **Números importantes** em **negrito**
- Foco: **convencer** em 2 min, não aprofundar

---

## 🎬 Preparação (10 s antes)

👁️ Dashboard já aberto em **localhost:8501** na aba **📊 Visão Geral**, scroll no topo (4 KPIs visíveis).

---

# MÉTODO STAR — 2 MINUTOS

---

## ⏱️ 0:00 → 0:25 (25 s) — SITUATION

👁️ Mostrar: **4 KPIs grandes no topo da aba Visão Geral** (sem scroll).

🎯 **STAR:** SITUATION (contexto e dor)

🗣️ "Olist é um marketplace brasileiro. O desafio: **dado um cliente, recomendar produtos** que ele provavelmente quer comprar — igual Netflix ou Amazon fazem.

Olhem os números no topo do dashboard:
- **99.785** compras
- **93.358** clientes
- **32.216** produtos
- **72** categorias

Parece muito. Mas tem uma armadilha que descobrimos na EDA e que **guiou todas as decisões técnicas** do projeto."

💡 Tom: direto, sem pressa; olhar para a câmera antes de olhar o dashboard.

---

## ⏱️ 0:25 → 1:05 (40 s) — TASK

👁️ Scroll curto para **"Sparsity da Matriz User-Item"**: 3 KPIs grandes + **gráfico-donut** lado a lado com a mensagem **"98% cold-start"** no centro.

🎯 **STAR:** TASK (o verdadeiro problema a resolver)

🗣️ "O verdadeiro vilão do projeto: **sparsity de 99,9967%**.

Vou dar um número concreto:
- 93 mil clientes × 32 mil produtos = **3 bilhões de pares possíveis**
- Apenas **99.785 observados** — o resto é buraco
- E pior: **98% dos clientes compraram UMA ÚNICA vez**

Na prática, estamos trabalhando com **2 mil clientes relevantes** em um mar de 93 mil.

Olhem o gráfico-donut aqui ao lado: **98% em vermelho** (1 compra só) e **2% em verde** (≥2 compras, com histórico).

Isso muda tudo. Qualquer feature que dependa de histórico do cliente — gasto médio, categorias preferidas — está **vazia** para 98% deles. É como pedir dica de restaurante para quem acabou de chegar na cidade: não há histórico para usar."

💡 Pausar 1 s após "98% dos clientes compraram UMA ÚNICA vez". Deixar a ficha cair. Apontar para o donut.

---

## ⏱️ 1:05 → 1:35 (30 s) — ACTION

👁️ Sem mudar de aba. Apenas gesticular em direção à tela e olhar a câmera.

🎯 **STAR:** ACTION (as 3 decisões que tomamos por causa do problema)

🗣️ "Diante disso, tomamos **3 decisões de design** que apresentaremos ao longo da demonstração:

1. **Tratar como feedback implícito** — em vez de prever a nota de 1 a 5 (77% das notas são 4 ou 5, sinal muito ruidoso), prever **probabilidade de interação**. Netflix não tenta adivinhar sua nota, tenta adivinhar se você vai **clicar**.

2. **Log-transform em preço e frete** — neutralizar a cauda longa. Algumas compras passam de R$ 10 mil onde a mediana é R$ 80. Sem essa transformação, o modelo acharia que R$ 10 mil é 100× mais importante.

3. **Priorizar embeddings sobre features engineered** — counterintuitive, mas em dataset com 98% cold-start, **menos features globais = menos ruído para a maioria**.

Essas decisões estão refletidas no pipeline e no dashboard que demonstraremos a seguir."

💡 Falar olhando para a câmera; dashboard fica de apoio visual.

---

## ⏱️ 1:35 → 2:00 (25 s) — RESULT

👁️ Trocar para a aba **🧠 NCF (MLP PyTorch)**. Apontar para o **banner hero** gigante no topo (gradiente azul + fonte 3.5em + badge `↑ 60.6x vs baseline`).

🎯 **STAR:** RESULT (resultado concreto + honestidade sobre o teto)

🗣️ "Resultado dessas decisões: o modelo final de Neural Collaborative Filtering atingiu **NDCG@10 = 0,2725** — **60× melhor que o baseline** mais forte (Popularidade, 0,0045).

Olhem este banner aqui em cima, com o **número grande** e o badge **`↑ 60.6x vs baseline`**. Ele já deixa claro qual é o resultado, sem precisar fazer a conta na hora.

Mas serei honesto: **0,27 é modesto** em termos absolutos. Em datasets densos como MovieLens, o esperado é 0,4 a 0,6. A marca do dataset fraco continua lá.

Por isso, ao longo da demonstração do dashboard, vamos mostrar **a jornada técnica completa** — baselines que zeraram por causa do cold-start, a escolha do NCF com embeddings, a otimização de hiperparâmetros e a ablation que revelou o achado contraintuitivo do projeto."

💡 Tom: encerramento limpo; olhar firme para a câmera no fechamento.

---

## 📋 Resumo visual do STAR

| Fase      | Tempo     | Mensagem central                                    | Referência visual no painel |
| --------- | --------- | --------------------------------------------------- | --------------------------- |
| Situation | 0:00–0:25 | "Problema: recomendar produtos no Olist"            | 4 KPIs principais (topo) |
| Task      | 0:25–1:05 | "Sparsity 99,99% + 98% cold-start = desafio real"   | Bloco "Sparsity" + **donut** 98% |
| Action    | 1:05–1:35 | "3 decisões: implícito + log + embeddings primeiro" | Sem UI (fala para câmera) |
| Result    | 1:35–2:00 | "NDCG 0,27 = 60× baseline;局限 do dataset"          | **Banner hero** (aba NCF) |

---

## 🎛️ Sidebar — Modo Apresentação (atalho útil)

**Antes de iniciar a apresentação (10 segundos antes):**

1. Clicar no ícone `>` no canto superior esquerdo do dashboard para abrir a **sidebar**.
2. Ativar o toggle **🎤 Modo Apresentação**.
3. Pronto: a aba `ℹ️ Sobre o Pipeline` desaparece automaticamente, deixando apenas as abas relevantes para a demo.

💡 Por que isso importa: durante a apresentação, **menos é mais**. Esconder a aba técnica final reduz a chance de a banca navegar para algo que você não quer falar.

---

## 🎯 Frases de escape (se o tempo apertar)

- Se cortar depois de 1:30 (só até Action):
  > "Resultado: NDCG@10 = 0,2725 — 60× acima do baseline. Próximo bloco entra no detalhe técnico."

- Se cortar depois de 1:05 (só até Task):
  > "Sparsity de 99,99% e 98% cold-start moldaram 3 decisões de design que vou detalhar agora."

---

## 🎬 BLOCO 2 — Versionamento de Modelos (1 minuto)

> Este bloco é independente do STAR anterior e pode ser inserido em qualquer momento da apresentação quando a banca perguntar "como vocês versionam o modelo?". Duração alvo: **60 segundos**.

---

## ⏱️ 2:00 → 2:25 (25 s) — Por que 4 ferramentas?

👁️ Mostrar: **abrir o GitHub do projeto** (URL visível) e em paralelo uma aba do **DagsHub** com a página `projeto_fiap_modulo2/#/models`.

🎯 **Objetivo:** explicar a separação de responsabilidades em 30 s.

🗣️ "Uma dúvida comum: por que **quatro** ferramentas — GitHub, DVC, MLflow e DagsHub — em vez de uma só?

A resposta é separação de responsabilidades:

- **GitHub** guarda **código** — scripts, configs, documentação. É pequeno, texto puro, feito para revisão por humanos.
- **DVC** é a **extensão do Git para dados e modelos** — ele versiona binários grandes (CSVs de 100 MB, pesos PyTorch de 16 MB) que o Git não consegue lidar, mas guarda só **hashes** no repositório.
- **MLflow** é o **cartório do modelo** — registra cada experimento com parâmetros, métricas e qual versão virou Production.
- **DagsHub** é o **terreno comum** — hospeda tudo isso num lugar só, com bucket S3, MLflow Tracking e Git integrado, sem custo para projetos públicos.

Quatro ferramentas, **quatro papéis distintos**, uma plataforma unificada."

💡 Ritmo: falar rápido nos nomes das ferramentas, devagar na frase final.

---

## ⏱️ 2:25 → 2:50 (25 s) — Como o modelo de produção é versionado

👁️ **Ativar o "🎤 Modo Apresentação"** na sidebar (toggle no canto esquerdo). Trocar para a aba **🔖 Versionamento** — ela já tem um card grande do Production Model com link clicável para o DagsHub, e blocos visuais `✅ DVC Sincronizado` e `✅ MLflow Registrado`.

🎯 **Objetivo:** mostrar o fluxo ponta a ponta em 25 s.

🗣️ "Na prática, o fluxo do nosso modelo Production tem **dois trilhos paralelos**, e o painel tem uma **aba dedicada** para isso (apontar):

**Trilho 1 — DVC (reprodutibilidade científica):**
1. Treinamos o NCF e salvamos `ncf_final.pt` localmente.
2. Rodamos `dvc add`, que gera um `*.dvc` com o hash MD5 do arquivo.
3. Damos `git commit` apenas dos metadados pequenos.
4. `dvc push` envia o binário de 16 MB para o bucket S3 do DagsHub.
5. Resultado: qualquer pessoa, em qualquer máquina, faz `git clone` + `dvc pull` e recebe **exatamente o mesmo modelo** — bit a bit. *Olhem aqui embaixo, o MD5 é `439244cc...`, está visível na aba.*

**Trilho 2 — MLflow (governança de modelos):**
1. Rodamos `scripts/register_model_dagshub.py`.
2. O modelo é logado como `olist-ncf-recommender` v1.
3. Promovemos automaticamente para o stage **Production**.
4. Resultado: histórico de versões, comparação entre runs e **auditoria** de quem promoveu o quê e quando. *E tem link clicável no card — clicar abre direto no DagsHub, na página do modelo em Production.*"

💡 O destaque do momento: clicar no link `📦 Ver Model Registry no DagsHub` mostrando o `olist-ncf-recommender` v1 em Production. É quando a banca vê tudo conectado.

---

## ⏱️ 2:50 → 3:00 (10 s) — Fechamento

👁️ Voltar olhar para a câmera.

🎯 **Objetivo:** frase de impacto que amarra o tema.

🗣️ "Resumindo em uma frase: **GitHub cuida do código, DVC dos dados, MLflow do modelo, e o DagsHub costura tudo isso numa única plataforma**. É assim que garantimos que o modelo que está no dashboard é o mesmo que foi treinado, com as mesmas métricas, na mesma data."

💡 Tom: encerramento, sem pressa.

---

## 📋 Resumo visual do bloco de versionamento

| Ferramenta | Papel                                | O que guarda                          |
| ---------- | ------------------------------------ | ------------------------------------- |
| **GitHub** | Código + docs + revisão              | Scripts, configs, `*.dvc`, `dvc.lock` |
| **DVC**    | Hashes + binários grandes            | CSVs e modelos `.pt` em bucket S3     |
| **MLflow** | Cartório do modelo                   | Runs, métricas, Model Registry        |
| **DagsHub**| Plataforma unificada                 | Git + bucket + MLflow Tracking        |

---

## 🎯 Frases de escape (se o tempo apertar no bloco de versionamento)

- Se cortar depois de 2:50 (sem fechamento):
  > "GitHub + DVC versionam dados e código; MLflow + DagsHub registram e promovem o modelo. É isso."

- Se cortar depois de 2:25 (só o "por que quatro ferramentas"):
  > "Cada ferramenta resolve um problema que as outras não resolvem bem — juntas, dão reprodutibilidade e governança."

---

## 💡 Dicas finais

- **Não ler números** — mostrá-los no dashboard e falar por cima
- **Não usar jargão** sem analogia ("sparsity" → "3 bilhões de pares possíveis, 99 mil observados")
- **Não prometer** NDCG alto — falar com honestidade sobre o dataset
- **Encerrar olhando para a câmera** com firmeza na última frase