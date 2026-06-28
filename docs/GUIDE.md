# Guia Técnico: Treinamento de Modelos - Olist Recommender System

## 1. Boas-vindas e Visão Geral

Olá Romário! Boas-vindas à etapa de modelagem do nosso Sistema de Recomendação da Olist. Este guia foi elaborado especialmente para você, com o objetivo de ser o seu mapa principal durante o desenvolvimento, treinamento e avaliação dos modelos de recomendação do projeto.

O projeto visa construir um sistema capaz de recomendar os produtos mais relevantes para os usuários, com base no histórico de compras e interações na plataforma Olist. Já passamos pelas fases de coleta e engenharia de features, e agora o desafio é construir modelos robustos que aprendam as preferências dos nossos clientes.

**Artefatos Prontos para Uso:**
- **Dataset:** `data/processed/interactions_fe.parquet` (99.785 linhas × 42 colunas) - Contém todas as interações e features já tratadas.
- **Mapeamentos:** `data/processed/id_mappings.json` - Dicionários que mapeiam IDs originais (strings) para índices numéricos contínuos (0 a N-1).
- **Ambiente:** `pyproject.toml` já configurado com PyTorch, Scikit-Learn, MLflow e dependências de dados.

**O Que Você Precisa Entregar ao Final:**
- Implementação de pelo menos 3 modelos *baseline* para servir de base comparativa.
- Implementação do modelo principal baseado em PyTorch (Neural Collaborative Filtering - NCF).
- Rotinas completas de treinamento e avaliação usando métricas de rankeamento.
- Rastreamento dos experimentos (pelo menos 3 *runs* variando hiperparâmetros) no MLflow.
- Um modelo final promovido para "Production" no MLflow Model Registry.
- Um notebook de demonstração (`demo.ipynb`) rodando inferência e mostrando os resultados.

---

## 2. Setup do Ambiente

Antes de colocar a mão no código, precisamos garantir que o seu ambiente local esteja devidamente configurado e rodando as versões corretas das bibliotecas.

### 2.1. Ativando o Ambiente Virtual

Estamos utilizando o `uv` para gerenciamento de dependências pela sua altíssima velocidade. No terminal, execute os seguintes comandos na raiz do projeto:

```bash
# Caso o ambiente virtual ainda não exista, crie-o:
uv venv

# Ative o ambiente virtual (Linux/macOS)
source .venv/bin/activate

# Ative o ambiente virtual (Windows PowerShell)
# .venv\Scripts\activate

# Sincronize as dependências definidas no pyproject.toml
uv pip sync pyproject.toml
```

### 2.2. Verificação das Dependências Instaladas

Verifique se as bibliotecas essenciais foram instaladas corretamente executando:

```bash
python -c "import torch, sklearn, mlflow, pandas; print(f'PyTorch: {torch.__version__} | Sklearn: {sklearn.__version__} | MLflow: {mlflow.__version__} | Pandas: {pandas.__version__}')"
```

A lista principal de bibliotecas que você irá utilizar ao longo do guia:
- `pandas`, `numpy`, `scipy` e `pyarrow`: Manipulação de dados e operações com matrizes esparsas.
- `scikit-learn`: Para os modelos baseline (TruncatedSVD, NearestNeighbors) e pré-processamento.
- `torch` (PyTorch): Para o modelo de deep learning (NCF).
- `mlflow`: Para tracking de experimentos e log de métricas/modelos.

### 2.3. Executando o MLflow UI Local

Para acompanhar seus experimentos em tempo real, deixe o servidor do MLflow rodando em uma aba separada do terminal:

```bash
# Ative o ambiente e inicie o servidor MLflow
source .venv/bin/activate
mlflow ui --host 127.0.0.1 --port 5000
```

Acesse no seu navegador: `http://127.0.0.1:5000`.

---

## 3. Carregamento do Dataset

O dataset pronto para consumo está no formato Parquet, que é eficiente e preserva os tipos de dados. Ele contém 99.785 interações.

### 3.1. Carregando os Dados

Use o snippet abaixo para carregar e inspecionar os dados. Note a importância de validar as dimensões:

```python
import pandas as pd
from pathlib import Path
import json

# Definir caminhos
DATA_DIR = Path('data/processed')
DATASET_PATH = DATA_DIR / 'interactions_fe.parquet'
MAPPING_PATH = DATA_DIR / 'id_mappings.json'

# Carregar o Parquet
df = pd.read_parquet(DATASET_PATH)

print(f"Shape do Dataset: {df.shape}")
# Esperado: Shape do Dataset: (99785, 42)

# Visão geral das colunas disponíveis
print("Colunas do Dataset:")
print(df.columns.tolist())
```

### 3.2. Estrutura das 42 Colunas

O dataset é rico e possui diversas informações. As colunas se dividem basicamente em:
1. **Identificadores Originais:** `user_id_raw`, `product_id_raw`, `order_id` (não use no treino).
2. **Identificadores Numéricos (Índices):** `user_id`, `product_id_idx`, `category_id`. **Estes são os IDs que devem ir para as camadas de Embedding.**
3. **Target/Feedback Explícito:** `review_score` (1 a 5). *Nota: Para o BPR Loss trataremos qualquer interação como feedback implícito positivo (1).*
4. **Features de Contexto/Compra:** `order_purchase_timestamp`, `price`, `freight_value`.
5. **Features Auxiliares (Agregações):** `user_total_orders`, `product_popularity`, `category_avg_rating`, etc.

### 3.3. Carregando os Mapeamentos

Os dicionários `id_mappings.json` são cruciais na hora da inferência, para traduzir o índice do modelo de volta para o ID real do produto.

```python
with open(MAPPING_PATH, 'r') as f:
    mappings = json.load(f)

# Acessando mapeamentos (exemplo: product_idx -> product_raw_id)
# Importante para converter a recomendação final para o usuário
product_idx_to_id = {int(k): v for k, v in mappings['product_to_idx'].items()}
# Observação: Geralmente no json temos string como chave.
```

---

## 4. Divisão Temporal (Train/Val/Test Split)

**⚠️ MUITO IMPORTANTE: Em sistemas de recomendação baseados em e-commerce, o split de dados DEVE ser temporal, e NUNCA puramente aleatório.**

### 4.1. Por Que Divisão Temporal?

Se utilizarmos um `train_test_split` aleatório do Scikit-Learn, corremos um grande risco de **Data Leakage** (Vazamento de Dados). Podemos acidentalmente usar as compras futuras de um usuário para prever o passado. Na vida real, treinamos o modelo com dados passados para prever compras no mês seguinte. Nossa validação deve simular esse cenário.

### 4.2. Implementação Sugerida (70% / 15% / 15%)

Devemos ordenar todo o dataset por `order_purchase_timestamp` e encontrar as datas de corte (cutoff dates) que dividem o dataset nas proporções 70% Treino, 15% Validação e 15% Teste.

```python
import pandas as pd
import numpy as np

def temporal_split(df: pd.DataFrame, time_col: str = 'order_purchase_timestamp', 
                   train_size: float = 0.70, val_size: float = 0.15) -> tuple:
    """
    Realiza o split temporal do dataset baseado em uma coluna de timestamp.
    """
    # 1. Garantir ordenação temporal global
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    
    n_total = len(df_sorted)
    train_end = int(n_total * train_size)
    val_end = int(n_total * (train_size + val_size))
    
    # 2. Dividir base
    train_data = df_sorted.iloc[:train_end].copy()
    val_data = df_sorted.iloc[train_end:val_end].copy()
    test_data = df_sorted.iloc[val_end:].copy()
    
    # 3. Identificar as datas de corte (para log e validação)
    train_max_date = train_data[time_col].max()
    val_min_date, val_max_date = val_data[time_col].min(), val_data[time_col].max()
    test_min_date = test_data[time_col].min()
    
    print(f"Treino: {len(train_data)} registros (Até {train_max_date})")
    print(f"Validação: {len(val_data)} registros (De {val_min_date} a {val_max_date})")
    print(f"Teste: {len(test_data)} registros (A partir de {test_min_date})")
    
    # Validação rigorosa de leakage
    assert train_max_date <= val_min_date, "Vazamento: Treino avança sobre Validação"
    assert val_max_date <= test_min_date, "Vazamento: Validação avança sobre Teste"
    
    return train_data, val_data, test_data

# Executando o split
train_df, val_df, test_df = temporal_split(df, time_col='order_purchase_timestamp')
```

---

## 5. Negative Sampling

### 5.1. O Conceito
Sistemas de recomendação de e-commerce sofrem com o problema do "feedback implícito". Nós sabemos o que o usuário comprou (positivo), mas **não sabemos o que ele ativamente desgosta**. A não-compra não significa rejeição; pode significar que ele apenas não viu o produto.

Para treinar uma função de Loss como a BPR (Bayesian Personalized Ranking), precisamos de amostras negativas: itens que o usuário não comprou. O modelo aprenderá a classificar o item comprado mais alto do que o item não comprado.

### 5.2. Estratégia de Amostragem
A literatura recomenda amostrar **entre 1 a 4 exemplos negativos para cada exemplo positivo**. 

```python
import numpy as np

def generate_negative_samples(user_interactions, all_items, n_negatives=4):
    """
    Gera amostras negativas aleatórias para um conjunto de interações de um usuário.
    user_interactions: Set contendo os itens com os quais o usuário já interagiu
    all_items: Array ou lista com todos os IDs de itens disponíveis
    n_negatives: Quantidade de itens negativos a gerar por positivo
    """
    negatives = []
    # Assumindo que chamamos essa função para cada (user, item_positivo)
    for _ in range(n_negatives):
        neg_item = np.random.choice(all_items)
        # Garantir que não sorteou um item que o usuário já interagiu
        while neg_item in user_interactions:
            neg_item = np.random.choice(all_items)
        negatives.append(neg_item)
    return negatives
```
*Dica de Performance:* Em PyTorch, o negative sampling geralmente é feito *on-the-fly* no `Dataset` ou `DataLoader` em cada época, garantindo que o modelo veja negativos diferentes ao longo do tempo.

---

## 6. Baseline 1 — Popularidade Global

Este é o baseline mais rudimentar. Se falharmos em superar isso com um modelo complexo, algo está profundamente errado na nossa modelagem.

### 6.1. Conceito
Recomendar para todos os usuários os Top-K itens mais vendidos globalmente, independentemente de quem seja o usuário. Não há personalização.

### 6.2. Implementação
```python
def get_popularity_baseline(train_data: pd.DataFrame, k: int = 10):
    """
    Retorna os K produtos mais frequentes no conjunto de treino.
    """
    # Conta a frequência de compras por produto no treino
    item_counts = train_data['product_id_idx'].value_counts()
    
    # Pega os Top K
    top_k_items = item_counts.head(k).index.tolist()
    
    return top_k_items

# Uso: As recomendações são sempre as mesmas para qualquer user_id
popular_items = get_popularity_baseline(train_df, k=10)
print(f"Top 10 Baseline de Popularidade: {popular_items}")
```

---

## 6.5 Dummy Model (Sanity Check)

### 6.5.1 O que é e por que é importante
- **Definição técnica**: Um Dummy Model (ou baseline trivial) é o modelo mais simples possível, geralmente implementado sem aprendizado de máquina ou utilizando apenas heurísticas básicas (como aleatoriedade, média aritmética ou popularidade global).
- **Por que é fundamental para validação de modelos reais**: Estabelece um referencial mínimo obrigatório. Qualquer esforço adicional para treinar e otimizar um algoritmo avançado só se justifica se seu desempenho for comprovadamente superior ao do modelo ingênuo.
- **Como ele serve como "limite inferior" de performance**: Ele define o "piso" aceitável das métricas de avaliação. Se um modelo real apresentar desempenho inferior ou igual a um Dummy Model, isso invariavelmente indica problemas críticos na modelagem (por exemplo, bugs de implementação, vazamento de features, hiperparâmetros inadequados ou dados não representativos).

### 6.5.2 Tipos de Dummy Models para Recomendação

Abaixo apresentamos uma comparação das estratégias de modelos triviais mais utilizadas:

| Dummy Model | Estratégia Principal | Ponto Forte | Limitação | Aplicação Típica |
|:---|:---|:---|:---|:---|
| **Random Recommender** | Recomenda itens aleatórios do catálogo. | Simplicidade extrema. | Desempenho analítico quase zero. | Referência absoluta de limite inferior. |
| **Most Popular** | Recomenda os itens mais populares globalmente. | Excelente desempenho em dados enviesados. | Falta de personalização. | *Cold-start* e heurística de fallback. |
| **Mean Predictor** | Prediz o `review_score` médio global. | Captura a tendência central do dataset. | Não distingue usuários ou itens. | Regressão e predição de nota. |
| **DummyRegressor** | Predição de médias/medianas via Sklearn. | Fácil integração ao pipeline padronizado. | Restrito a predições numéricas. | Baseline para métricas de erro. |

### 6.5.3 Implementação Prática

Código Python completo para cada tipo:

```python
# 1. Random Recommender
import numpy as np

class RandomRecommender:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.items = None
    
    def fit(self, df, user_col='user_id', item_col='product_id_idx'):
        self.items = df[item_col].unique()
        self.rng = np.random.default_rng(self.random_state)
        return self
    
    def recommend(self, user_id, top_k=10):
        return self.rng.choice(self.items, size=top_k, replace=False)


# 2. Most Popular Recommender
class MostPopularRecommender:
    def __init__(self):
        self.popular_items = None
    
    def fit(self, df, item_col='product_id_idx'):
        self.popular_items = df[item_col].value_counts().index.tolist()
        return self
    
    def recommend(self, user_id, top_k=10):
        return self.popular_items[:top_k]


# 3. Sklearn DummyRegressor (para prever review_score)
from sklearn.dummy import DummyRegressor

dummy_mean = DummyRegressor(strategy='mean')
dummy_median = DummyRegressor(strategy='median')
dummy_constant = DummyRegressor(strategy='constant', constant=3.0)
```

### 6.5.4 Métricas Esperadas

- **Random Recommender**: 
  - Recall@10 ≈ 0,001 (essencialmente aleatório)
  - Hit Rate@10 ≈ 0,01 (1% de chance de acertar 1 entre 100 possíveis)
  - Esperado: TODOS os modelos reais DEVEM superar isso significativamente.

- **Most Popular**:
  - Recall@10 ≈ 0,005-0,02 (depende da concentração)
  - Hit Rate@10 ≈ 0,05-0,15
  - Pode surpreender em datasets com forte viés de popularidade.

- **Mean Predictor**:
  - RMSE ≈ desvio padrão global do review_score (~1,3)
  - MAE ≈ ~1,0
  - Qualquer modelo real deve ter RMSE menor.

### 6.5.5 Por que isso é importante para o projeto

1. **Validação de bugs**: se um modelo "sofisticado" apresentar performance pior que o Dummy, certamente há um bug na implementação.
2. **Referência em papers**: artigos acadêmicos consistentemente utilizam modelos Dummy como base de comparação.
3. **Demonstra ROI**: justifica o custo computacional e de desenvolvimento de modelos mais complexos perante as partes interessadas.
4. **Sanity check para MLflow**: garante que a infraestrutura de rastreamento de experimentos esteja reportando valores coerentes.

### 6.5.6 Integração com MLflow

```python
import mlflow

# Registrar Dummy como baseline de referência
with mlflow.start_run(run_name='dummy_random'):
    mlflow.set_tag('model_type', 'dummy')
    mlflow.log_metric('recall_at_10', 0.001)
    mlflow.log_metric('hit_rate_at_10', 0.01)
    mlflow.log_metric('ndcg_at_10', 0.005)
```

### 6.5.7 Checklist de Validação

- [ ] DummyRegressor (mean) implementado e avaliado
- [ ] Random Recommender implementado e avaliado
- [ ] Most Popular implementado e avaliado
- [ ] Resultados registrados no MLflow
- [ ] Todos os modelos reais superam Dummy nas 4+ métricas
- [ ] Gráfico comparativo Dummy vs Modelos Reais gerado

---

## 7. Baseline 2 — TruncatedSVD (PRIMÁRIO)

Este é o nosso **Baseline Principal**. O TruncatedSVD (Decomposição em Valores Singulares Truncada) é uma técnica clássica e poderosa de fatoração de matrizes.

### 7.1. Construção da Matriz User-Item
Precisamos primeiro transformar nosso dataset tabular em uma matriz esparsa onde as linhas são usuários, colunas são itens e o valor é 1.0 (compra).

### 7.2. Implementação Completa

```python
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np

# 1. Identificar dimensões globais para evitar OutOfBounds
# É importante usar os IDs máximos de TODO o dataset original 'df' 
# para que os índices no treino e teste se alinhem perfeitamente.
n_users = df['user_id'].max() + 1
n_items = df['product_id_idx'].max() + 1

# 2. Construir Matriz Esparsa (Apenas com dados de TREINO)
# Usaremos 1.0 como flag implícita de que a interação ocorreu
row_indices = train_df['user_id'].values
col_indices = train_df['product_id_idx'].values
data = np.ones(len(train_df))  # Feedback implícito

user_item_matrix = csr_matrix(
    (data, (row_indices, col_indices)),
    shape=(n_users, n_items)
)

print(f"Sparsity da matriz: {100 * (1 - user_item_matrix.nnz / (n_users * n_items)):.4f}%")

# 3. Aplicar SVD
# Hiperparâmetro a ser tunado: n_components (ex: 50, 100, 200)
k_components = 100
svd = TruncatedSVD(n_components=k_components, random_state=42)

# user_embeddings shape: (n_users, k_components)
user_embeddings = svd.fit_transform(user_item_matrix)

# item_embeddings shape: (n_items, k_components)
item_embeddings = svd.components_.T  

# 4. Função de Recomendação
def recommend_svd(user_id, user_emb, item_emb, k=10):
    # Produto escalar entre vetor do usuário e de todos os itens
    user_vector = user_emb[user_id]
    scores = np.dot(item_emb, user_vector)
    
    # Obter os índices dos K maiores scores
    # argsort ordena crescente. Pegamos os últimos K e invertemos.
    top_k_indices = np.argsort(scores)[-k:][::-1]
    return top_k_indices

# Teste para o usuário de ID 0
recs_svd = recommend_svd(0, user_embeddings, item_embeddings, k=10)
print(f"SVD Recomendações User 0: {recs_svd}")
```

---

## 8. Baseline 3 — Item-Item CF (Cosine Similarity)

### 8.1. Conceito
O Filtro Colaborativo Item-Item assume que "usuários que compraram X também compraram Y". Para isso, calculamos a similaridade entre as colunas (itens) da nossa matriz esparsa. Se dois itens foram comprados frequentemente pelos mesmos usuários, seus vetores serão similares.

### 8.2. Implementação

```python
from sklearn.metrics.pairwise import cosine_similarity

# Usaremos a mesma user_item_matrix do SVD, mas faremos a transposta.
# item_user_matrix tem shape: (n_items, n_users)
item_user_matrix = user_item_matrix.T

def get_similar_items(target_item_id, item_user_mat, k=10):
    """
    Encontra os K itens mais similares a um target_item usando similaridade de cosseno.
    """
    target_vector = item_user_mat[target_item_id]
    
    # Calcula cosseno entre target e todos os itens
    # Retorna shape (1, n_items)
    similarities = cosine_similarity(target_vector, item_user_mat)[0]
    
    # Ordenar ignorando o próprio item (que terá similaridade 1.0)
    top_k_indices = np.argsort(similarities)[-(k+1):][::-1]
    top_k_indices = [idx for idx in top_k_indices if idx != target_item_id][:k]
    
    return top_k_indices

# Para recomendar para um usuário com base no último item que ele comprou
last_item_bought = train_df[train_df['user_id'] == 0]['product_id_idx'].iloc[-1]
recs_item_item = get_similar_items(last_item_bought, item_user_matrix, k=10)
print(f"Itens Similares ao Item {last_item_bought}: {recs_item_item}")
```

---

## 9. Métricas de Avaliação

Como os sistemas de recomendação retornam uma *lista ordenada* de sugestões, métricas tradicionais de classificação (Accuracy) não funcionam bem. Precisamos de métricas de ranking baseadas no top-K. Recomendo escrever essas funções do zero.

### 9.0. Estratégia Formal de Avaliação Top-K

Esta subseção define **oficialmente** como o sistema é avaliado, evitando ambiguidades na interpretação de resultados.

#### Cutoffs (K)
Avaliamos em **três níveis de cutoff** para analisar a sensibilidade do modelo:
- **K = 5** (conservador) — útil para cenários de UI com pouco espaço (e.g., carousel principal)
- **K = 10** (padrão da indústria) — referência para comparação com literatura (He et al. 2017, NCF)
- **K = 20** (leniente) — útil para "ver mais" / infinite scroll

#### Estratégia de candidatos (negative sampling na avaliação)
Para cada usuário do conjunto de teste, geramos um pool de candidatos:
- **1 item positivo** (verdadeiro ground-truth no teste)
- **99 itens negativos** (amostrados aleatoriamente entre os itens NÃO consumidos pelo usuário no treino nem no teste)

Total: **100 candidatos por usuário**. O modelo ranqueia estes 100 itens e avaliamos se o positivo aparece no Top-K.

Esta estratégia:
- Reduz custo computacional vs. avaliar sobre todo o catálogo (~32k itens)
- É equivalente à estratégia **uniform negative sampler** em BPR loss
- Reproduz cenário real onde apresentamos N itens ranqueados ao usuário

#### Cold-start
Usuários que **não aparecem no treino** são filtrados da avaliação:
- Sem embeddings treinados, o modelo não consegue gerar ranking confiável
- Excluí-los evita inflar artificialmente as métricas com zeros

#### Agregação
- **Média macro** sobre todos os usuários do teste: cada usuário tem o mesmo peso
- (Alternativa seria média micro ponderada por nº de itens relevantes — não usamos)
- Reportamos **média ± desvio padrão** quando há ≥ 30 usuários; apenas média caso contrário

#### Métricas calculadas
Cinco métricas oficiais:
1. **HitRate@K** — cobertura: 1 se ≥1 hit, 0 caso contrário
2. **Recall@K** — completude: quantos relevantes foram recuperados
3. **Precision@K** — eficiência: quantos dos K são relevantes
4. **NDCG@K** — qualidade posicional: desconto logarítmico
5. **MAP@K** — média das precisões em cada posição de hit

#### Protocolo de reporting
Cada run no MLflow registra as 5 métricas em três Ks (5, 10, 20) → **15 valores por run**, mais:
- `n_users_evaluated`: nº de usuários efetivamente usados na avaliação
- `cold_start_rate`: % de usuários filtrados por cold-start
- `eval_time_seconds`: tempo de inferência (para SLA de produção)

### 9.1. Fórmulas e Explicações
- **Hit Rate@K:** 1 se pelo menos um item relevante estiver no top K, 0 caso contrário. (Mede a taxa de sucesso geral).
- **Recall@K:** (Itens relevantes no Top K) / (Total de itens relevantes do usuário). Quantos dos itens desejados nós conseguimos encontrar.
- **Precision@K:** (Itens relevantes no Top K) / K. Das K sugestões, quantas eram úteis.
- **NDCG@K:** Normalized Discounted Cumulative Gain. Valoriza **a posição** do acerto. Acertar na posição 1 vale muito mais do que acertar na posição 10, devido ao desconto logarítmico.
- **MAP@K:** Mean Average Precision. Calcula a precisão média em todos os pontos de "hit", recompensando acertos no topo da lista.

### 9.2. Implementação Manual

```python
import numpy as np

def calculate_metrics_at_k(recommended_list, true_items_set, k=10):
    """
    Calcula Recall, Precision, Hit Rate, e NDCG considerando os Top-K itens recomendados.
    
    recommended_list: Lista ordenada dos k IDs de itens recomendados.
    true_items_set: Set de IDs de itens que o usuário efetivamente interagiu no Teste.
    """
    rec_k = recommended_list[:k]
    hits = [1 if item in true_items_set else 0 for item in rec_k]
    num_hits = sum(hits)
    
    # 1. Hit Rate
    hit_rate = 1.0 if num_hits > 0 else 0.0
    
    # 2. Precision e Recall
    precision = num_hits / k
    # Evitar divisão por zero caso o usuário não tenha compras no teste
    recall = num_hits / len(true_items_set) if len(true_items_set) > 0 else 0.0
    
    # 3. NDCG
    dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
    
    # Calcula Ideal DCG (DCG se todos os itens relevantes estivessem no topo)
    ideal_hits = [1] * min(len(true_items_set), k)
    idcg = sum([1 / np.log2(idx + 2) for idx, hit in enumerate(ideal_hits)])
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {'HitRate@K': hit_rate, 'Recall@K': recall, 'Precision@K': precision, 'NDCG@K': ndcg}

def evaluate_model(model_predict_fn, test_data_df, k=10):
    """
    Itera sobre os usuários do Teste, gera recomendações e tira a média das métricas.
    """
    test_users = test_data_df['user_id'].unique()
    
    all_metrics = {'HitRate@K': [], 'Recall@K': [], 'Precision@K': [], 'NDCG@K': []}
    
    for uid in test_users:
        # Pega as compras reais do usuário no conjunto de teste
        true_items = set(test_data_df[test_data_df['user_id'] == uid]['product_id_idx'].tolist())
        if not true_items:
            continue
            
        # O modelo deve fornecer a lista
        recs = model_predict_fn(uid, k=k)
        
        metrics = calculate_metrics_at_k(recs, true_items, k=k)
        for metric_name in all_metrics:
            all_metrics[metric_name].append(metrics[metric_name])
            
    # Retorna médias globais
    return {k: np.mean(v) for k, v in all_metrics.items()}
```

---

## 10. Tracking com MLflow

O MLflow é nossa fonte da verdade. É obrigatório registrar pelo menos 3 rodadas (*runs*) no MLflow variando hiperparâmetros.

### 10.1. Workflow do MLflow no Python

O script de treino do NCF ou SVD deve englobar esse fluxo:

```python
import mlflow

# Defina a URI onde o servidor está rodando e o nome do experimento
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Olist_NCF_Recommender")

def train_and_log(params):
    with mlflow.start_run(run_name=f"NCF_emb_{params['emb_dim']}_lr_{params['lr']}"):
        
        # 1. Log dos hiperparâmetros
        mlflow.log_params(params)
        
        # 2. Inicializar, Treinar modelo...
        # model = NCF(...)
        # train(model, ...)
        
        # 3. Avaliar modelo
        # test_metrics = evaluate_model(...)
        test_metrics = {'Recall@10': 0.15, 'NDCG@10': 0.08} # placeholder
        
        # 4. Logar as métricas calculadas
        mlflow.log_metrics(test_metrics)
        
        # 5. Logar modelo localmente (Salvar pesos PyTorch)
        # mlflow.pytorch.log_model(model, "pytorch-ncf-model")
        
        print("Run finalizado.")

# Exemplo de Grid Search manual
for emb_dim in [16, 32, 64]:
    train_and_log({'emb_dim': emb_dim, 'lr': 1e-3, 'batch_size': 1024})
```

### 10.2. Promovendo para Production
Após realizar os testes e visualizar as métricas na UI, vá até a aba "Models" no MLflow UI, registre o melhor modelo associando um nome (ex: `olist_ncf_recommender`) e mude a *stage* dele para **Production**. 

---

## 11. Modelo Principal — Neural Collaborative Filtering (NCF)

O NCF substitui o produto escalar interno tradicional por uma rede neural (Multi-Layer Perceptron), permitindo que o modelo aprenda interações não lineares complexas entre usuários e itens.

### 11.1. Arquitetura

Diagrama conceitual da arquitetura que implementaremos:

```text
 User ID ----> User Embedding Layer ------\
                                           \
 Item ID ----> Item Embedding Layer --------> Concat() --> MLP (ReLU + Dropout) --> Output Score
                                           /
 Categoria --> Cat Embedding Layer -------/
                                         /
 Aux Features -> (StandarScaler) -------/
```
* **Justificativa:** Embeddings transformam variáveis categóricas esparsas em vetores densos de features latentes. O MLP mapeia a concatenação desses vetores para um score preditivo.

### 11.2. Loss Function — BPR (Bayesian Personalized Ranking)

Por lidarmos com feedback implícito, otimizar com Erro Quadrático Médio (MSE) não faz sentido. A Loss BPR busca maximizar a distância entre o score de um item positivo (comprado) e o score de um item negativo (amostrado aleatoriamente).

**Fórmula:** `Loss = -log(sigmoid(score_positivo - score_negativo))`

### 11.3. Implementação PyTorch Completa

Aqui está o código base para construir a arquitetura NCF:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, n_users, n_items, n_categories, 
                 emb_dim=32, hidden=[128, 64], dropout=0.3,
                 n_aux_features=20):
        super().__init__()
        
        # Dicionários de Embeddings (Lookup Tables)
        self.user_emb = nn.Embedding(num_embeddings=n_users, embedding_dim=emb_dim)
        self.item_emb = nn.Embedding(num_embeddings=n_items, embedding_dim=emb_dim)
        
        # Categoria geralmente precisa de menos dimensões por ter cardinalidade menor
        self.cat_emb = nn.Embedding(num_embeddings=n_categories, embedding_dim=8)
        
        # Definir a camada MLP
        input_dim = (emb_dim * 2) + 8 + n_aux_features
        layers = []
        
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
            
        # Camada de saída (apenas 1 neurônio emitindo o logit score sem sigmoid final)
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Inicialização customizada (Opcional, mas ajuda a convergir rápido)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, user, item, category, aux_features):
        """
        Recebe tensores batch (batch_size, 1)
        """
        u_vec = self.user_emb(user)
        i_vec = self.item_emb(item)
        c_vec = self.cat_emb(category)
        
        # Concatena no eixo das features
        x = torch.cat([u_vec, i_vec, c_vec, aux_features], dim=-1)
        
        # Passa pelo MLP e comprime a última dimensão
        output = self.mlp(x).squeeze(-1)
        return output

def bpr_loss(pos_scores, neg_scores):
    """
    BPR Loss customizada
    - pos_scores: predições para itens que o usuário comprou
    - neg_scores: predições para itens negativos
    """
    # Usando log_sigmoid numericamente estável
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    return loss
```

#### Loop de Treinamento Base (Esboço do `Trainer`)
O otimizador ideal é o `AdamW` (com weight decay) ou `Adam`.

```python
# Pseudo-código de treinamento
# model = NCF(...)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# Para cada época:
#   Para cada batch no DataLoader:
#      pos_scores = model(user, item_pos, cat_pos, aux_pos)
#      neg_scores = model(user, item_neg, cat_neg, aux_neg)
#      loss = bpr_loss(pos_scores, neg_scores)
#      
#      optimizer.zero_grad()
#      loss.backward()
#      optimizer.step()
```

### 11.4. Hiperparâmetros para Tunar

Para as 3 runs obrigatórias no MLflow, crie combinações dessa grade:

| Hiperparâmetro | Impacto | Valores Sugeridos para Teste |
| :--- | :--- | :--- |
| `embedding_dim` | Capacidade representacional latente. | `[16, 32, 64]` |
| `hidden_layers` | Capacidade do MLP em aprender não-linearidades. | `[(128, 64), (256, 128, 64)]` |
| `dropout` | Reduz overfitting, essencial pois NCF overfitta rápido. | `[0.2, 0.3, 0.5]` |
| `learning_rate` | Velocidade de descida no gradiente. | `[1e-3, 5e-4, 1e-4]` |
| `batch_size` | Estabilidade do gradiente. Batch maior é melhor pra BPR.| `[512, 1024, 2048]` |
| `num_negatives`| Amostras negativas criadas por cada item positivo. | `[1, 4, 10]` |

---

## 12. Features Auxiliares no MLP

No NCF tradicional os embeddings fazem todo o trabalho. Porém, como temos um belo dataset de features numéricas e categóricas adicionais preparadas previamente, não podemos ignorá-las!

**Quais features incluir:**
- Agregações: `user_total_orders`, `product_popularity`, `freight_value` médio.
- Temporalidades: Dia da semana da compra, mês.

**Boas Práticas e Tratamentos:**
- **Normalização Obrigatória:** As variáveis numéricas *precisam* passar por um `StandardScaler` ou `MinMaxScaler` do Sklearn antes de virarem tensores. Se uma variável for de magnitude alta (ex: Preço=1500) misturada com um Embedding de peso ~0.01, ela irá monopolizar os gradientes da rede, gerando `NaN` ou não convergindo.
- **Concatenação:** Conforme mostrado no código `forward()` do PyTorch acima, apenas empilhe (use `torch.cat`) as features normalizadas com o final dos tensores de embedding.
- **Cardinalidade Extrema:** Features textuais e descritivas (se houverem) devem ficar de fora ou ser agrupadas.

---

## 13. Avaliação Comparativa Final

Após concluir os desenvolvimentos, você precisará gerar um relatório final comparativo no seu `demo.ipynb` mostrando como o Deep Learning performou frente ao SVD, seguindo o padrão desta tabela de exemplo (espera-se que NCF supere SVD no Recall/NDCG).

| Modelo | Recall@10 | NDCG@10 | Hit Rate@10 | Vantagem Principal |
| :--- | :--- | :--- | :--- | :--- |
| Popularidade | 0.051 | 0.021 | 0.082 | Não tem cold-start; baseline ótimo de sanidade. |
| Item-Item CF | 0.103 | 0.045 | 0.150 | Recomendação fácil de explicar ("compraram X compram Y"). |
| TruncatedSVD | 0.180 | 0.102 | 0.220 | Extremamente rápido no treino, robusto para matrizes esparsas. |
| **NCF (BPR Loss)** | **0.245** | **0.155** | **0.315** | Captura relações não lineares, ingere metadata e features de contexto. |

*Discuta no notebook os trade-offs: NCF requer muito mais poder computacional e tuning de parâmetros em relação ao SVD.*

---

## 14. Troubleshooting (Problemas Comuns)

No meio do treinamento, bugs aparecem. Aqui está o seu *cheat sheet* para resolução rápida de problemas:

1. **A Loss de Treino cai para 0, mas a Loss de Validação sobe descontroladamente.**
   * *Diagnóstico:* Overfitting clássico.
   * *Solução:* Aumente o `dropout` no MLP (ex: para 0.5), aplique `weight_decay` de `1e-4` no otimizador Adam, e diminua `embedding_dim`. Implemente Early Stopping monitorando a validação.

2. **A Loss mal se altera; as predições do modelo são aleatórias.**
   * *Diagnóstico:* Underfitting ou problemas na taxa de aprendizado.
   * *Solução:* Otimizador pode estar preso. Teste aumentar o `learning_rate` para `1e-3` ou `5e-3`. A arquitetura pode estar simples demais. Aumente o tamanho das camadas ocultas.

3. **Cold-Start Extremo (Usuários do Teste sem interações prévias no Treino).**
   * *Diagnóstico:* Usuários invisíveis no modelo.
   * *Solução:* O modelo SVD retornará lixo/erros; o NCF usará um vetor embedding aleatório. A solução ideal em produção é usar uma heurística "fallback" que retorna a Baseline de Popularidade para esses usuários.

4. **Sparse Error / Matriz Esparsa Falhando.**
   * *Diagnóstico:* Você pode ter tentado usar uma matriz `dense` em Numpy que requisitou > 100GB de RAM.
   * *Solução:* Certifique-se de que está operando estritamente com estruturas `scipy.sparse.csr_matrix` até a inserção no `TruncatedSVD`.

5. **MLflow Connection Refused.**
   * *Diagnóstico:* Servidor UI caiu.
   * *Solução:* No terminal dedicado, reinicie o `mlflow ui --host 127.0.0.1 --port 5000`.

6. **PyTorch acusando erro de tipo ou Device mismatches.**
   * *Solução:* Certifique-se de mandar os inputs (`user.to(device)`, `item.to(device)`) antes de inseri-los no modelo. Assegure que as labels e features numéricas sejam `torch.float32`, enquanto IDs categóricos (para os Embeddings) sejam `torch.long`.

7. **IndexError na camada de Embedding do PyTorch.**
   * *Diagnóstico:* Um ID de Produto ou Usuário no set de teste é maior que a variável `num_embeddings` alocada.
   * *Solução:* Certifique-se de usar `max_id + 1` de **TODO** o dataset unificado como dimensão no `nn.Embedding(n_items, ...)`.

---

## 15. Checklist de Entrega

- [ ] Split Temporal foi aplicado sem vazamento (Train/Val/Test).
- [ ] Implementado Baseline 1: Popularidade Global.
- [ ] Implementado Baseline 2: TruncatedSVD.
- [ ] Implementado Baseline 3: Item-Item CF.
- [ ] Funções de Métrica (Recall@K, NDCG@K, etc) codificadas e validadas manualmente.
- [ ] Implementado PyTorch NCF com embeddings, MLP e injeção de features extras.
- [ ] Função BPR Loss implementada e utilizada no loop.
- [ ] Pelo menos 3 rodadas executadas e variação registrada no MLflow.
- [ ] Modelo de melhor performance registrado e promovido para *Production* no MLflow.
- [ ] Tabela comparativa finalizada preenchendo as métricas para todos os algoritmos.
- [ ] O código Python foi linteado (`ruff check .` e `mypy .`) e possui tipagem adequada.
- [ ] Mínimo de testes unitários criados (ex: função de métricas e negative sampling).
- [ ] Notebook `demo.ipynb` construído rodando inferências prontas visualmente.

---

## 16. Cronograma Sugerido

Este é um cronograma ideal de 10 dias úteis para ajudar você a cadenciar as entregas sem surtar com o PyTorch:

- **Dia 1-2:** Setup do ambiente com UV, carga dos dados, implementação rigorosa do Split temporal e código base do Baseline Popularidade.
- **Dia 3-4:** Construção da matriz CSR, implementação do Baseline TruncatedSVD e o Item-Item. Obtenção do primeiro conjunto de métricas para referência.
- **Dia 5-7:** O bloco denso. Codificação das classes PyTorch do NCF, do BPR Loss, negative sampling batch-wise, rodar a versão básica (só embeddings) e finalmente injetar as features auxiliares. Tuning massivo do modelo.
- **Dia 8-9:** Integração com o MLflow, documentação da matriz de experimentos, busca pelo *Sweet Spot* dos hiperparâmetros, análise das métricas cruzadas.
- **Dia 10:** Documentação do código, ruff, pytest, polimento final no `demo.ipynb`. Fechamento e Deploy!

---

## 17. Referências e Recursos

Aprofundamento sugerido (leitura não obrigatória, mas de ouro):

1. **Paper NCF:** He, X. et al. (2017). *Neural Collaborative Filtering*. [https://arxiv.org/abs/1708.05031](https://arxiv.org/abs/1708.05031) - O paper pai da arquitetura que você está implementando.
2. **Paper BPR:** Rendle, S. et al. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. O racional de matemática Bayesiana por trás da loss.
3. **MLflow Docs:** [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) - Como logar dados específicos.
4. **PyTorch Docs:** [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) - Revise como a matriz de lookup latente aloca gradientes.
5. **Livro Base:** Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Ótimo material caso você empaque em teoria (Capítulo sobre Collaborative Filtering).

Bom trabalho, Romário! Qualquer dúvida, olhe o log do MLflow ou a tabela de Troubleshooting antes de escalonar a dúvida. Vai dar bom! 🚀
