# 🚀 Guia de Execução de Machine Learning (Baselines)

Este documento descreve o passo a passo para reproduzir o pipeline de dados, executar os treinamentos dos modelos Baseline e validar as recomendações utilizando a arquitetura do projeto.

---

## 1. Setup Inicial do Ambiente e Dados

Este projeto utiliza o **`uv`** (Astral) como gerenciador de pacotes rápido. Todo o controle de dependências está mapeado no arquivo `pyproject.toml`.

### Instalação das Dependências
Para instalar os pacotes (como `pandas`, `scikit-learn`, `mlflow`, etc.) e inicializar o ambiente virtual em milissegundos, execute na raiz do projeto:
```powershell
uv pip install -r pyproject.toml

```

### Estrutura de Dados Locais

Para esta etapa, a execução ocorre localmente. Certifique-se de que os arquivos originais do dataset do Olist (formato `.csv`) estejam descompactados e alocados no seguinte diretório antes de iniciar o pipeline:
`data/raw/`

---

## 2. Pipeline de Dados

Antes de treinar os modelos, é necessário processar os dados brutos e gerar as *features*. Execute os scripts na ordem abaixo:

1. **Preparação de Dados:** Junta as tabelas e consolida as interações (`interactions.parquet`).
```powershell
uv run src/data_preparation.py

```


2. **Feature Engineering:** Trata *cold-start* e gera 45 variáveis adicionais para futura utilização na arquitetura neural (`interactions_fe.parquet`).
```powershell
uv run src/feature_engineering.py

```



---

## 3. Treinamento dos Modelos Baseline

Para servir de fundação comparativa para os modelos avançados, construímos três abordagens simples:

1. **Popularity Baseline:** Recomenda os produtos que geraram o maior número absoluto de interações.
2. **Top Rated Baseline:** Recomenda os produtos com a maior nota média de *review*.
3. **Item-Item Similarity (Collaborative Filtering):** Cria uma matriz esparsa de Co-ocorrência/Cosseno.

### Executando os Experimentos no MLflow

A orquestração do treinamento consiste em 3 *runs* variando os hiperparâmetros limitadores (como `top_n` e quantidade mínima de avaliações).

**Passo A:** Inicie o servidor do MLflow em um terminal secundário:

```powershell
uv run mlflow server --host 127.0.0.1 --port 5000

```

*A interface gráfica ficará disponível para acompanhamento em `http://127.0.0.1:5000`.*

**Passo B:** Execute o script de treinamento no terminal principal:

```powershell
uv run src/train.py

```

*O script salvará os artefatos de recomendação e as métricas (ex: NDCG@10) automaticamente.*

---

## 4. Avaliação e Resultados (Notebook de Demonstração)

A seleção do melhor modelo para simulação de "Produção" e o consumo do artefato para gerar inferências está documentada interativamente no Jupyter Notebook.

1. Abra o arquivo **`notebooks/demo.ipynb`**.
2. Certifique-se de selecionar o kernel Python referente ao ambiente `.venv` do projeto.
3. Execute as duas células do notebook:
* A **primeira célula** varre o histórico do MLflow automaticamente em busca da Run com a maior pontuação de *NDCG*.
* A **segunda célula** faz o download do arquivo de recomendações vencedor e realiza um *INNER JOIN* com a tabela original de produtos para demonstrar o `product_id` e sua respectiva categoria em um DataFrame legível de exibição.

---

## 5. Entendendo a Métrica Principal (NDCG@10)

Para avaliar a qualidade dos nossos modelos no MLflow, utilizamos o **NDCG@10** (*Normalized Discounted Cumulative Gain at 10*). Esta é a métrica padrão-ouro na indústria para motores de busca e sistemas de recomendação.

Em um e-commerce como o Olist, não basta apenas sugerir produtos relevantes; **a ordem importa**. O usuário precisa ver os itens que mais deseja logo nas primeiras posições da vitrine. O NDCG traduz esse comportamento para a matemática através de 5 pilares:

* **Gain (Ganho):** O modelo pontua apenas se recomendar um produto que o usuário considera relevante (ex: um item que ele clicaria ou compraria).
* **Cumulative (Cumulativo):** A pontuação é a soma de todos os acertos dentro da lista recomendada.
* **Discounted (Desconto/Penalidade):** É o coração da métrica. O modelo sofre uma redução de pontos drástica se colocar um item muito relevante nas últimas posições. Um acerto na posição 1 vale muito mais do que um acerto na posição 10.
* **Normalized (Normalizado):** Como alguns usuários interagem com muitos produtos e outros com poucos, a pontuação é dividida pelo cenário "perfeito" (Ideal DCG), garantindo que a nota final de todas as recomendações fique na **escala de 0 a 1**.
* **@10 (Corte):** Avaliamos apenas o Top 10. Na vida real, o cliente raramente rola a página para ver dezenas de itens. Se o produto certo estiver na 11ª posição, o modelo é penalizado.

### O que esperar dos resultados?
Em nossos testes iniciais de **Baseline** (baseados em médias globais e popularidade), é esperado que o NDCG@10 atinja valores na casa de `0.08` a `0.10`. Este é o nosso "piso". Na próxima fase do projeto, com a implementação da rede neural **NCF (Neural Collaborative Filtering) em PyTorch**, o objetivo é superar essa marca personalizando o ranqueamento para o comportamento único de cada usuário.