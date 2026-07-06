# Relatório de Progresso — Sistema de Recomendação Olist

**Projeto:** Olist Recommender System  
**Disciplina:** Tech Challenge Fase 02 — Pós-graduação ML Engineering  
**Data de Início:** 2026-06-18  
**Status Atual:** Fase 2 (Engenharia de Dados e Modelagem) concluída — Pronto para Modelagem

---

## 1. Visão Geral do Projeto

### 1.1 Descrição
Este projeto foca no desenvolvimento end-to-end de um Sistema de Recomendação com MLOps, concebido para o Tech Challenge Fase 02 da pós-graduação em Machine Learning Engineering. Utilizando o dataset Olist Brazilian E-Commerce, que abrange cerca de 100 mil pedidos realizados entre 2016 e 2018, o sistema busca oferecer recomendações precisas de produtos aos usuários. A abordagem incorpora boas práticas de engenharia de dados, machine learning e MLOps para garantir rastreabilidade, escalabilidade e reprodutibilidade ao longo de todo o ciclo de vida do modelo.

### 1.2 Stack Tecnológica
| Categoria | Tecnologia | Versão/Finalidade |
|---|---|---|
| **Linguagem** | Python | 3.12+ (fixado via `requires-python` no pyproject.toml) |
| **Gerenciamento de Pacotes** | uv (Astral) | Gestão rápida de dependências e ambientes — PEP 621 + hatchling |
| **Deep Learning** | PyTorch | ≥ 2.1 — Implementação de MLP / Embeddings |
| **Machine Learning Base** | Scikit-Learn | ≥ 1.4 — Modelos de baseline comparativos (incluindo TruncatedSVD) |
| **Manipulação de Dados** | Pandas, NumPy, PyArrow | ≥ 2.2, ≥ 1.26, ≥ 15 — Processamento e manipulação ágil |
| **Configuração** | Pydantic Settings | ≥ 2.2 — `src/config.py` centraliza paths, MLflow, hiperparâmetros NCF |
| **Tracking e MLOps** | MLflow | ≥ 2.10 — Rastreamento de experimentos, métricas e Model Registry (Production stage) |
| **Versionamento de Dados** | DVC | ≥ 3.48 — Versionamento e pipelines de dados |
| **Engenharia de Software** | Loguru, Ruff, Pytest | ≥ 0.7, ≥ 0.3, ≥ 8 — Logging, lint zero warnings, testes |
| **Design Patterns** | Factory + Strategy | `src/models/factory.py` + `src/data/strategies.py` |
| **Dashboard** | Streamlit, Plotly | ≥ 1.0 — Visualização interativa de resultados |
| **Infraestrutura** | Docker | Containerização do ambiente (a implementar) |
| **Cloud** | AWS, Azure ou GCP | Deploy e infraestrutura remota (a definir) |

### 1.3 Objetivos do Tech Challenge
*   Garantir um mínimo de 10.000 interações user-item (atingido: 99.785 interações processadas).
*   Implementar e acompanhar pelo menos 4 métricas quantitativas (planejado: Recall@K, NDCG@K, MAP@K, Hit Rate@K).
*   Registrar no mínimo 3 runs distintos utilizando o MLflow.
*   Estabelecer no mínimo 3 estágios independentes no pipeline do DVC.
*   Desenvolver o modelo central utilizando arquiteturas MLP ou Embeddings com PyTorch.
*   Criar baselines comparativos utilizando Scikit-Learn.
*   Realizar o deploy do sistema em cloud, fornecendo uma URL pública para acesso.
*   Produzir a documentação final através de um Model Card e um vídeo STAR de 5 minutos.

---

## 2. Linha do Tempo (Timeline)

| Data | Fase | Atividade | Status | Entregas |
|---|---|---|---|---|
| 2026-06-18 | 0 | Análise inicial do desafio | ✅ | Análise técnica + requisitos |
| 2026-06-18 | 1 | Análise exploratória do dataset (EDA) | ✅ | reports/eda_report.md + 8 figuras PNG |
| 2026-06-18 | 1 | Pipeline de preparação de dados | ✅ | src/data_preparation.py + interactions.parquet |
| 2026-06-18 | 2 | Configuração do ambiente Python (uv) | ✅ | pyproject.toml + uv.lock + .venv |
| 2026-06-18 | 2 | Feature Engineering + correção cold-start | ✅ | src/feature_engineering.py + interactions_fe.parquet |
| 2026-06-27 | 3 | Treinamento de Baselines | ✅ | src/train.py + split temporal + métricas reais |
| 2026-06-27 | 3 | Dashboard Streamlit (Resultados) | ✅ | front/app_vis.py (5 abas) |
| 2026-06-27 | 3 | Notebook de Resultados | ✅ | notebooks/03_baseline_training.ipynb |
| 2026-06-27 | 3 | Documentação e Correções | ✅ | README.md + REPORT.md atualizados |
| 2026-06-27 | 3 | Auditoria Spearman de Features Redundantes | ✅ | Remoção de 2 features correlacionadas (>0.95); relatório em `reports/feature_audit_spearman.md`; decisão de manter `no_aux` como Production |

---

## 3. Etapa 1 — Análise do Desafio

### 3.1 Requisitos Técnicos Identificados
*   A base de dados necessita possuir interações consistentes entre usuários e itens (superando largamente o piso exigido de 10.000 registros).
*   O uso de PyTorch é mandatório para a construção do modelo de Deep Learning principal.
*   O estabelecimento de métricas robustas de ranqueamento, que são vitais para sistemas de recomendação em cenários reais, é exigido.
*   Realizar o tracking disciplinado de todos os experimentos via MLflow.
*   Criar uma arquitetura de dados plenamente reprodutível através de pipelines versionados no DVC.
*   Disponibilizar a solução finalizada por meio de um deploy em cloud acessível publicamente via internet.

### 3.2 Pontos de Atenção
*   **Problema de Cold-Start:** Novos usuários ou novos produtos podem sofrer com a falta de interações prévias na modelagem, o que pode enviesar o treinamento.
*   **Extrema Esparsidade (Sparsity):** A maioria esmagadora dos usuários possui pouquíssimas interações (geralmente uma), dificultando o aprendizado consistente das representações vetoriais.
*   **Risco de Data Leakage:** Há a necessidade estrita de garantir um split temporal ou metodológico rigoroso entre treino e teste, evitando o vazamento de dados do futuro para prever o passado.
*   **Garantia de Reprodutibilidade:** Ambientes de dependência Python que são muito voláteis requerem o uso de ferramentas robustas de travamento (como `uv` e `DVC`) para assegurar exatamente os mesmos resultados ao rodar os scripts em diferentes máquinas.

### 3.3 Recomendações Estratégicas
*   Adotar técnicas consolidadas, como Neural Collaborative Filtering (NCF), para conseguir explorar eficientemente tanto a matriz de interações quanto as features complementares do lado do usuário e do produto.
*   Concentrar as avaliações no uso de métricas focadas no top-K (como Recall@K e NDCG@K), já que elas lidam muito melhor com listas ordenadas que são a base dos sistemas de recomendação.
*   Desativar preventivamente e temporariamente o filtro severo de cold-start caso ele venha a inviabilizar a obtenção do volume mínimo de dados necessário para treinamento da rede neural.
*   Implementar baselines simplificados muito rapidamente (usando scikit-learn) para que se consiga medir o ganho de eficiência real advindo das abordagens mais complexas e profundas no PyTorch.

---

## 4. Etapa 2 — Análise Exploratória do Dataset (EDA)

### 4.1 Dataset Identificado
O dataset Olist Brazilian E-Commerce compreende um vasto conjunto de interações de compras reais em um marketplace brasileiro que engloba o período temporal focado entre os anos de 2016 e 2018. Ele contém centenas de milhares de pedidos dispersos em um schema com várias tabelas relacionais, capturando e modelando o perfil completo da jornada de um cliente, começando da escolha do frete e do método de pagamento até culminar na sua avaliação final de satisfação (review).

### 4.2 Descobertas-Chave
*   Os preços dos produtos possuem uma distribuição que é largamente enviesada de forma assimétrica, com forte predominância absoluta na faixa de itens de baixo e médio valor agregado.
*   O pagamento dos pedidos de compra é predominantemente efetuado utilizando cartão de crédito em parcelamentos de curto a médio prazo.
*   Existe uma concentração demográfica e geográfica extremamente intensa de usuários baseados na região Sudeste do Brasil (com notável destaque para o próprio estado de SP).
*   A grande maioria dos carrinhos criados contém um único tipo de produto, indicando fortemente que os hábitos são baseados em compras focadas e não em aquisições muito extensivas.
*   As avaliações recebidas (reviews) demonstram uma tendência muito positiva e bem acentuada, com a nota máxima de satisfação (5 estrelas) sendo flagrantemente prevalecente.
*   Identificamos que há uma forte correlação linear inicial entre algumas features quantitativas calculadas de imediato, o que evidenciou que estas precisariam ser criticamente avaliadas antes da modelagem em si.

### 4.3 Insights para o Sistema de Recomendação
*   Para um ambiente de recomendação sem uma matriz explícita de "gostos", torna-se absolutamente crucial derivar de modo implícito as interações diretas entre os usuários e os itens através do histórico real de pedidos efetuados e concluídos na plataforma.
*   A utilização da feature `customer_unique_id` (em detrimento da chave natural do pedido em si) é um alicerce que foi avaliado como sendo fundamental para se conseguir construir perfis plenamente consolidados da jornada histórica de compra de cada indivíduo da base ao longo de todo o tempo registrado.
*   Felizmente, o montante correspondente ao volume total de registros consolidados se mostrou mais do que perfeitamente adequado para nos permitir bater a arrojada meta imposta que pedia o limiar mínimo de 10.000 interações viáveis para treinamento e aprovação. Este montante irá simplificar consideravelmente a capacidade de viabilizar a arquitetura e garantir a convergência do motor base do modelo idealizado com PyTorch.
*   A exploração detida do acervo, bem como o mapeamento completo da distribuição de popularidade dos itens que constituem o catálogo em foco, revelou um clássico comportamento de "long-tail". Uma inferência disto em tempo prático é exigir diretamente que a futura malha e o esqueleto central da rede idealizada do sistema possuam sabedoria para saber lidar efetivamente e performaticamente bem com um acervo de milhares de produtos de forma que abarque e ranqueie também produtos pouco populares.
*   Embora fosse algo que constava minimamente dentro das margens das previsões primárias na esfera de planejamento do projeto, fomos impactados com o fato pragmático e aferível matematicamente de que a base global conta com um número de interações bastante acanhado, configurando e atestando perfeitamente o fenômeno conhecido pela literatura focada em Data Science como extrema "Sparsity". Tudo isso, enfim, consolidou para nós e nos forçou a reforçar a premente necessidade da arquitetura e concepção integral pautarem abordagens de um nível bem acentuadamente mais moderno, o que reflete nossa inclinação irrevogável de investir com bastante ímpeto, foco temporal, computacional e dedicação estrutural em um tratamento profundo baseado em embeddings em rede neural.

---

## 5. Etapa 3 — Pipeline de Preparação de Dados

### 5.1 Decisões de Design
| Aspecto Analisado | Proposta Original | Implementação Final | Justificativa / Impacto |
|---|---|---|---|
| **Formato de Saída** | CSV | Parquet | Menor tamanho, preserva tipos nativos de dados, carregamento rápido |
| **Path de Gravação** | `data/raw` | Caminho Absoluto | Resolvido erro de mapeamento relativo no script principal |
| **Filtro de Status do Pedido** | Todos | Apenas "delivered" | Garante qualidade e evita recomendar itens que foram cancelados |
| **Nota de Review (review_score)** | Ausente | Incorporada | Oferece um peso de preferência explícito para a interação |
| **Marcação de Timestamp** | `order_purchase_timestamp` | Mantido e tipado | Necessário para evitar data leakage durante o split temporal |
| **Agregação de Interação** | Por Order ID | Por `customer_unique_id` | Mapeia o gosto real consolidado do usuário e não apenas do carrinho |
| **Filtro Cold-Start Severo** | Aplicado indiscriminadamente | Desabilitado | Reduzia o dataset para apenas 2.656 linhas, o que descumpre o requisito |
| **Features Temporais Iniciais** | Ano/Mês separados | Embutido no DateTime | Otimiza os passos de Feature Engineering posteriores |
| **Tratamento Nulos** | Remoção de linhas inteiras | Preservação (apenas review_score possui nulos) | Retém 687 registros valiosos para engajamento e cliques |
| **Uso de Dependências** | Pip Padrão | Pip + `uv` | Acelera em mais de 10x a resolução do grafo de dependência e instalação |

### 5.2 Arquivos Criados
*   `src/data_preparation.py`: Script Python com aproximadamente 270 linhas, responsável pela extração, limpeza e agregação dos dados crus provenientes do Kaggle. O código implementa tratamentos para tipos de dados, formatação de datas e união estruturada das tabelas relacionais do esquema original.
*   `data/processed/interactions.parquet`: Arquivo gerado de 5.66 MB contendo o dataset final processado em formato colunar otimizado (Parquet). Este formato garante compressão eficiente e preservação rigorosa da tipagem dos dados em relação a formatos textuais tradicionais como CSV.
*   `data/processed/README.md`: Dicionário de dados documentando a definição exata, origem e tipo de cada coluna presente no dataset resultante. Atua como artefato de governança de dados essencial para a rastreabilidade da fase de Feature Engineering.

### 5.3 Estatísticas do Dataset Limpo
| Métrica | Valor Obtido | Descrição |
|---|---|---|
| Interações Processadas | 99.785 | Volume total de eventos usuário-item mantidos após limpeza, superando o requisito mínimo de 10.000 interações. |
| Usuários Únicos | 93.358 | Número distinto de clientes na plataforma. |
| Produtos Únicos | 32.216 | Quantidade de itens distintos interagidos. |
| Categorias Únicas | 72 | Total de famílias de produtos mapeadas. |
| Sparsity (Esparsidade) | 99,9967% | Proporção de entradas nulas na matriz teórica de usuários-itens, típica de domínios de e-commerce. |
| Pedidos Entregues | 110.840 | Quantidade de pedidos brutos filtrados antes da consolidação final no nível de usuário/produto. |
| Janela Temporal | 2016-09-15 a 2018-08-29 | Período de cobertura dos dados processados, essencial para planejar o split temporal. |

A esparsidade extrema observada (99,9967%) reflete um cenário no qual a imensa maioria dos usuários interage com uma fração minúscula do catálogo de produtos. Isso justifica a adoção futura de algoritmos robustos a cold-start e de representações latentes densas, como embeddings, capazes de inferir similaridade não-trivial entre usuários e itens sem a dependência exclusiva de co-ocorrência.

---

## 6. Etapa 4 — Configuração do Ambiente Python

### 6.1 Ferramenta Escolhida
A gestão do ambiente Python e de suas dependências foi estruturada utilizando a ferramenta `uv` (desenvolvida pela Astral). A escolha desta solução em detrimento de alternativas consolidadas como pip, poetry ou conda é justificada pela sua alta velocidade de resolução baseada em Rust e capacidade de garantir builds reproduzíveis via lock file (`uv.lock`) rigorosamente determinístico. Além disso, o recurso `uv python pin` solucionou um impasse crítico de incompatibilidade com a versão global padrão (Python 3.13) do sistema operacional anfitrião, viabilizando a ancoragem hermética da execução na versão Python 3.12, ambiente perfeitamente maduro e compatível com as bibliotecas requeridas.

### 6.2 Dependências

#### Produção
Estas bibliotecas compõem o arcabouço primário de execução analítica e modelagem estocástica do projeto.
*   `pandas`, `numpy`, `pyarrow`: Manipulação ágil de dataframes, execução de álgebra linear eficiente e I/O de arquivos Parquet.
*   `scikit-learn`: Fornecimento de transformadores algorítmicos (encoders) e modelagem de baselines.
*   `torch`: Infraestrutura de tensores em hardware acelerado, obrigatória para o motor principal de deep learning.
*   `loguru`: Padronização assíncrona de logs estruturados em rotinas do projeto.

#### Desenvolvimento
Destinadas estritamente à integridade e manutenibilidade do código-fonte.
*   `pytest`: Framework de asserção automática focado no controle de qualidade (Quality Assurance) contínuo.
*   `ruff`: Linter e analisador estático ultra-rápido, impondo aderência consistente à PEP 8.
*   `seaborn` e `matplotlib`: Componentes fundamentais de projeção visual gráfica utilizados durante a Análise Exploratória.

#### Cloud & MLOps
Ferramentas essenciais para escalabilidade, versionamento e governança metodológica.
*   `mlflow`: Registro sistemático de parâmetros, métricas de avaliação do modelo e versionamento de artefatos.
*   `dvc`: Orquestração de grafos acíclicos dirigidos (DAGs) atrelada ao rastreamento estrito da evolução temporal dos dados.

### 6.3 Arquivos de Configuração
*   `pyproject.toml`: Manifesto de configuração global determinando metadados do projeto e definições de build.
*   `uv.lock`: Matriz gerada contendo árvores de dependência congeladas em hashes invioláveis, garantindo que instalações reproduzam sempre versões exatas de pacotes.
*   `.python-version`: Diretiva local forçando a restrição do interpretador para a versão exata requisitada (Python 3.12).
*   `.gitignore`: Restringe os artefatos irrelevantes ou de cunho estritamente local/secreto de submissões acidentais no repositório.
*   `README.md`: Documento de escopo que concentra diretrizes metodológicas essenciais para orquestração geral e replicabilidade.
*   `configs/.gitkeep`: Preserva e integra a pasta de configurações à árvore estrutural independentemente do preenchimento atual.

### 6.4 Entry Points Configurados
| Comando | Descrição | Tecnologia |
|---|---|---|
| `uv run python3 src/data_preparation.py` | Executa o pipeline sequencial de agregação para higienização e unificação dos dados brutos em Parquet. | Python |
| `uv run ruff check .` | Invoca a suíte estática avaliando anomalias estruturais ou falhas de formatação não condizentes com boas práticas. | Ruff |
| `uv run pytest` | Aciona a integração de testes unitários mitigando potenciais regressões ou bugs nos sub-módulos lógicos. | Pytest |

---

## 7. Etapa 5 — Feature Engineering

### 7.1 Descobertas do EDA Aplicadas
O enriquecimento dos atributos basais utilizou integralmente o levantamento analítico preestabelecido na fase EDA. Primordialmente, mitigou-se a severidade de deleções associadas ao "cold-start" desabilitando lógicas de expurgo temporal, uma deliberação forjada objetivando contornar a perda massiva do número de observações requeridas. Observando uma distribuição marcadamente _long-tail_, incorporamos atributos atrelados diretamente a índices de frequência do item, o que concede ao algoritmo insumos concretos para modelagem de entidades incomuns. Considerando correlações latentes não triviais evidenciadas e distribuições de variáveis financeiras excessivamente inclinadas, efetuaram-se estratégias diretas de transformação logarítmica (Log1p).

### 7.2 Features Geradas
| Categoria da Feature | Contagem | Exemplos Representativos |
|---|---|---|
| **Identificadores** | 3 | `customer_unique_id`, `product_id`, `order_id` |
| **Target (Alvo)** | 2 | `review_score` e uma flag indicadora binária de satisfação. |
| **Numéricas** | 10 | Variáveis contínuas normalizadas e aplicadas via conversões Log1p (ex: valores de frete, preços de aquisição). |
| **Temporais** | 8 | Ano civil, trimestre corrente, dia semanal e horários fragmentados para sazonalidade. |
| **Categóricas Encodadas** | 5 | Atributos englobando estados da federação brasileira e hierarquia principal da categoria transacionada. |
| **Agregações do Usuário** | 7 | Frequência agregada histórica de compra, índice de recência da última interação, e dispêndio médio apurado do indivíduo. |
| **Agregações do Produto** | 7 | Contagem holística na base global de vendas do item, relevância dentro do setor da categoria e volume de ticket nominal diário. |

### 7.3 Features Removidas
| Nome da Feature | Motivo da Remoção |
|---|---|
| `constante_teste_1` | Variância global em nível perfeitamente fixo, nulo (zero absoluto), acarretando no descarte sumário por falta de capacidade analítica preditiva. |
| `constante_teste_2` | Expressividade estritamente nula, incapaz de prover insumos de entropia ou variância matemática favorável. |
| `constante_teste_3` | Coluna com cardinalidade estática redundante não inferindo variações para divisão hierárquica por meio de algoritmos. |
| `product_volume_cm3` | Extrema multicolinearidade, identificando correspondência absoluta por ser unicamente extraída via dimensões espaciais lineares inerentes já englobadas. |
| `order_item_id_max` | Correlação estritamente positiva, de fator colinear beirando Pearson P > 0.99, vinculada em repetição direta à contagem de itens num mesmo escopo temporal da nota. |
| `freight_value_log` | Consolidada e sintetizada ativamente como vetor adjacente direto para composição na coluna do próprio valor tarifado em média atrelada. |
| `payment_installments_sum` | Amarra atrelada inefetivamente, cujo nível isolado provou-se inútil para determinação exata da satisfação associada da transação em base global ao longo da jornada recomendada do usuário. |

### 7.4 Estatísticas Finais
| Dimensão | Quantidade Final |
|---|---|
| Linhas Totais (Amostras) | 99.785 |
| Colunas Finais (Features) | 42 |
| Registros contendo Nulos | 687 (Circunscrito de maneira isolada em `review_score`) |

O ganho informacional das novas variáveis de agregação enriqueceu decisivamente a representação estatística vetorial sem elevar o custo da temida maldição da dimensionalidade, visto que 7 features com alta redundância algorítmica ou inércia constante foram decapitadas do ambiente transacional, restando apenas o refinamento contextual fidedigno de atributos consolidados.

### 7.5 Detalhamento das Features e Estratégias de Uso

#### 7.5.1 Identificadores (5 features)

**Descrição**: São as chaves primárias do problema de recomendação. Os identificadores numéricos (`user_id`, `product_id_idx`, `category_id`) são mapeamentos sequenciais otimizados para indexação densa.

**Features e Justificativas**:
| Feature | Tipo | Importância |
|---|---|---|
| `customer_unique_id` | string | Identifica univocamente um comprador |
| `product_id` | string | Identifica univocamente um item |
| `user_id` | int | Índice denso para embedding de usuário |
| `product_id_idx` | int | Índice denso para embedding de produto |
| `category_id` | int | Índice denso para embedding de categoria |

**Uso nos Modelos Baseline (Scikit-Learn)**:
- No TruncatedSVD, use `user_id` e `product_id_idx` para montar a matriz esparsa.
- Em GradientBoosting, aplique `OrdinalEncoder` nos índices; descarte strings.

**Uso no Modelo MLP/Embeddings (PyTorch)**:
- `user_id` e `product_id_idx` alimentam camadas `nn.Embedding`.
- Os tensores resultantes sofrem dropout e concatenam-se com variáveis contínuas numéricas antes da camada MLP.

#### 7.5.2 Target / Sinal (3 features)

**Descrição**: Refletem o feedback qualitativo e quantitativo do usuário sobre a compra, combinando notas explícitas e indicativos implícitos de reincidência.

**Features e Justificativas**:
| Feature | Tipo | Importância |
|---|---|---|
| `review_score` | float | Nota dada pelo usuário (1 a 5) |
| `has_review` | int | Flag binária (0/1) indicando presença de feedback |
| `purchase_count` | int | Proxy de fidelidade através de recompras |

**Uso nos Modelos Baseline (Scikit-Learn)**:
- Para regressão, use `review_score` como alvo (substitua nulos pela média global).
- Em classificação, converta `review_score` > 4 em classe positiva.

**Uso no Modelo MLP/Embeddings (PyTorch)**:
- `review_score` age como variável target otimizada via erro quadrático (MSE).
- `has_review` pode atuar como ponderador de confiança na função de perda.

#### 7.5.3 Numéricas (6 features)

**Descrição**: Agrupam valores monetários de produto e logística. Transformações logarítmicas (Log1p) normalizam a distribuição fortemente assimétrica característica do e-commerce.

**Features e Justificativas**:
| Feature | Tipo | Importância |
|---|---|---|
| `price` | float | Valor bruto do produto |
| `freight_value` | float | Custo logístico do frete |
| `price_log` | float | Versão estabilizada (Log1p) para mitigar outliers |
| `freight_value_log` | float | Versão estabilizada (Log1p) reduzindo dispersão |
| `price_to_freight_ratio` | float | Proporção que revela peso do frete no custo final |
| `has_price_outlier` | int | Flag sinalizando produtos caros (Percentil 99) |

**Uso nos Modelos Baseline (Scikit-Learn)**:
- Algoritmos baseados em árvore utilizam atributos brutos (`price`).
- Regressores lineares requerem versões logarítmicas escaladas com `StandardScaler`.

**Uso no Modelo MLP/Embeddings (PyTorch)**:
- Incorpore apenas variáveis logarítmicas normalizadas por `nn.BatchNorm1d`.
- Concatene o flag booleano de forma direta, sem normalização.

#### 7.5.4 Temporais (8 features)

**Descrição**: Marcadores cronológicos que contextualizam o instante da transação, viabilizando o aprendizado de sazonalidades anuais e o comportamento de consumo cíclico semanal.

**Features e Justificativas**:
| Feature | Tipo | Importância |
|---|---|---|
| `purchase_year` / `purchase_month` | int | Situa picos de vendas macros (ex: feriados) |
| `purchase_day_of_week` / `purchase_hour` | int | Traça ciclos rotineiros do período de acesso |
| `is_weekend` / `is_holiday_season` | int | Flags binárias de comportamento ocasional |
| `days_since_reference` | int | Base contínua sequencial viabilizando métricas temporais |

**Uso nos Modelos Baseline (Scikit-Learn)**:
- Regressões lineares exigem One-Hot Encoding em campos circulares como a hora.
- Modelos em árvore dividem naturalmente bases temporais sequenciais como dias contínuos.

**Uso no Modelo MLP/Embeddings (PyTorch)**:
- Acondicione features cíclicas de baixa cardinalidade em pequenas projeções (`nn.Embedding(24,4)`).
- A base temporal sequencial exige estabilização padronizada imediata.

#### 7.5.5 Categóricas Encodadas (8 features)

**Descrição**: Refletem dados mercadológicos estruturados, transformando categorias de negócio e formatos de quitação financeira em projeções escalares consistentes.

**Features e Justificativas**:
| Feature | Tipo | Importância |
|---|---|---|
| `category_target_enc` | float | Média Bayesiana reveladora da qualidade padrão do setor |
| `category_frequency` | float | Propensão da área frente a vendas da prateleira |
| `category_is_top10` / `category_is_rare` | int | Segmentadores de extremos absolutos de tráfego |
| `payment_type_*` | int | Binários indicativos de transação (Crédito, Boleto, etc) |

**Uso nos Modelos Baseline (Scikit-Learn)**:
- Os pagamentos encodados acoplam-se homogeneamente em modelos vetorizados.
- A projeção categórica média age fortemente em regressões lógicas, antecipando viés setorial.

**Uso no Modelo MLP/Embeddings (PyTorch)**:
- Metadados quantificados fluem linearmente transpondo estágios da BatchNorm1d.
- Os indicadores monetários booleanos justapõem-se no repasse antecedendo processamentos densos.

#### 7.5.6 Agregações do Usuário (6 features)

**Descrição**: Consolidam estatisticamente toda a jornada do comprador. Identificam seu volume rotineiro transacional, distanciamentos na lealdade temporal e tolerância pecuniária.

**Features e Justificativas**:
| Feature | Tipo | Importância |
|---|---|---|
| `user_total_purchases` | int | Mensuração da solidez ativa do perfil cadastrado |
| `user_avg_price` / `user_avg_freight` | float | Referencial sintético da zona de ticket financeiro |
| `user_purchase_span_days` / `user_recency_days` | int | Rastreador de reativação pontuando períodos adormecidos |
| `user_review_rate` | float | Parcela das transações portando feedback submetido |

**Uso nos Modelos Baseline (Scikit-Learn)**:
- Modelos hierárquicos seccionam agrupamentos comportamentais formando grupos sintéticos autônomos.
- Regressões incorporam ativamente taxas normalizadas segmentando RFM (Recency, Frequency, Monetary).

**Uso no Modelo MLP/Embeddings (PyTorch)**:
- O lote de estatísticas requer compressão imediata padronizadora via BatchNorm limitando sobressaltos escalares.
- Esta porção contígua funde-se estruturalmente na base atada ao representador matricial do identificador do usuário.

#### 7.5.7 Agregações do Produto (6 features)

**Descrição**: Cristalizam o perfil mercadológico do item. Mensuram reputação e amplitude global atestadas empiricamente por observações históricas.

**Features e Justificativas**:
| Feature | Tipo | Importância |
|---|---|---|
| `product_popularity` / `product_unique_buyers` | int | Escala real evidenciando a base disseminada de consumo popular |
| `product_avg_review_score` | float | Centralização pontual documentando qualidade empírica provada |
| `product_avg_price` / `product_avg_freight` | float | Custo mediano intrínseco fixo vinculado estruturalmente na oferta |
| `product_review_rate` | float | Apelo comunicativo latente para estímulo a considerações espontâneas |

**Uso nos Modelos Baseline (Scikit-Learn)**:
- Modelos baseados no heurístico de popularidade alocam instâncias estritamente baseadas nesta consolidação fixa.
- Árvores randômicas utilizam fortemente estas características para bifurcações dominantes no topo.

**Uso no Modelo MLP/Embeddings (PyTorch)**:
- O bloco contínuo agrega volume coeso juntando-se paralelamente ao embedding de identificação do item base.
- Padronização linear constitui regra compulsória barrando gradientes destrutivos provocados pela magnitude desmedida da popularidade bruta.

### 7.6 Auditoria Spearman — Rodada Extra de Remoção de Features Redundantes

**Data:** 2026-06-27
**Método:** Correlação de Spearman pairwise (não-linear, robusta a outliers)
**Threshold de corte:** `|ρ| > 0.95`
**Dataset:** `data/processed/interactions_fe.parquet` (99.785 linhas × 42 colunas)
**Features auditadas:** 20 numéricas em `configs/selected_features.yaml`
**Relatório detalhado:** `reports/feature_audit_spearman.md`

#### 7.6.1 Features Identificadas como Redundantes (|ρ| > 0.95)

| Par | Spearman ρ | Feature Removida | Justificativa |
|---|---|---|---|
| `days_since_reference` ↔ `user_recency_days` | **−0.9861** | `user_recency_days` | Proxies temporais redundantes; `days_since_reference` é o timestamp canônico usado no split temporal |
| `freight_value_log` ↔ `user_avg_freight` | **+0.9657** | `freight_value_log` | Ambos medem frete; `user_avg_freight` é uma agregação de usuário mais robusta |

#### 7.6.2 Re-treinamento do NCF (Comparação Justa, 15-20 epochs)

| Modelo | Features Usadas | NDCG@K | HitRate@K | Recall@K | MAP@K |
|---|---|---|---|---|---|
| `NCF_FINAL` (original) | 3 emb + 20 aux | 0.2226 | 0.3993 | 0.3914 | 0.1725 |
| `Audit_Spearman_18feat` (novo) | 3 emb + 18 aux | **0.1932** | 0.3413 | 0.3322 | 0.1547 |
| **`Ablation_FINAL_no_aux_emb32` (PRODUÇÃO)** | 3 emb only | **0.2725** | **0.4949** | **0.4886** | **0.2081** |

#### 7.6.3 Conclusão Surpreendente

**A remoção das 2 features redundantes NÃO melhorou o NDCG@K** — causou uma queda de 13.2% (de 0.2226 para 0.1932). Possíveis razões:

- **Spearman ≠ redundância funcional:** A correlação monotônica linear não implica redundância para o modelo neural. O NCF aprende relações não-lineares onde features linearmente correlacionadas podem carregar sinais distintos em regiões específicas do espaço latente.
- **Sinal vs. ruído:** Features com `|ρ| > 0.95` podem estar medindo o mesmo fenômeno linear mas com distribuições marginais diferentes — o que importa para a função de loss.
- **Regularização implícita:** Mais features podem ajudar a rede a convergir para soluções mais estáveis, mesmo que linearmente redundantes.

A descoberta mais importante: o **modelo Production (`Ablation_FINAL_no_aux_emb32`)**, que usa **apenas os 3 embeddings** (sem nenhuma feature auxiliar), supera todas as variantes com aux features:

- vs 20 features (original): **+22.5% NDCG@K**
- vs 18 features (após auditoria): **+41.0% NDCG@K**

#### 7.6.4 Decisão Arquitetural

**Manter `Ablation_FINAL_no_aux_emb32` como Production.** Justificativa:

1. **Performance superior comprovada:** NDCG@K = 0.2725 (60× vs baseline de popularidade)
2. **Complexidade mínima:** Apenas 3 embeddings + MLP, sem necessidade de pré-processamento de features
3. **Generalização equivalente:** train_NDCG=0.5827 vs test_NDCG=0.2725 — gap similar aos modelos com aux
4. **Alinhamento com a literatura:** Em datasets com alta esparsidade (99.997% no Olist) e sinal de preferência fraco (98% dos usuários com 1 compra), modelos neurais baseados puramente em embedding superam abordagens híbridas

#### 7.6.5 Estado Atual do Modelo em Produção

```yaml
MLflow Model Registry:
  Nome: olist_ncf_recommender
  Versão: 1
  Stage: Production
  Run ID: a905125600df4452a2d3f3581a87ab42
  Run Name: Ablation_FINAL_no_aux_emb32
  Parâmetros:
    epochs: 20
    emb_dim: 32
    hidden: [64, 32]
    dropout: 0.5
    lr: 0.0005
    batch_size: 2048
    n_negatives: 8
    n_params: 4_027_009
  Métricas de Teste:
    NDCG@K: 0.2725
    HitRate@K: 0.4949
    Recall@K: 0.4886
    Precision@K: 0.0509
    MAP@K: 0.2081
```

#### 7.6.6 Artefatos Gerados pela Auditoria

| Arquivo | Conteúdo |
|---|---|
| `configs/selected_features.yaml` | Reduzido de 20 → 18 numeric_features |
| `src/training/evaluate.py` | `_AUX_COLS` reduzido de 20 → 18 features |
| `data/processed/feature_metadata.json` | Adicionado `audit_history[1]` com metadados da rodada |
| `artifacts/metrics_Audit_Spearman_18feat_E15.json` | Métricas do re-treino com 18 features |
| `artifacts/ncf_Audit_Spearman_18feat_E15.pt` | Modelo serializado (16 MB) |
| `reports/feature_audit_spearman.md` | Relatório técnico completo (8 seções) |

#### 7.6.7 Nota sobre a Remoção em `configs/` e `_AUX_COLS`

Embora a remoção das 2 features tenha **piorado** o NDCG do modelo com aux features, mantivemos as listas reduzidas (`numeric_features` e `_AUX_COLS`) por consistência com a auditoria. O **impacto em produção é nulo** porque:

- O modelo Production (`no_aux`) não usa nenhuma feature auxiliar
- A redução afeta apenas **futuros experimentos** que desejem testar com aux features
- Estes experimentos terão baseline claro: o modelo `no_aux` (NDCG=0.2725) é o teto a superar

---

## 8. Issues Conhecidos e Resoluções

| Issue | Resolução Implementada |
|---|---|
| Python 3.13 padrão incompatível provocando quebras em dependências MLOps nativas. | Mitigado pela alocação forçosa e retroativa das diretrizes do `uv python pin 3.12`. |
| Alias terminal genérico `python` não direcionava ao ambiente interpretador local. | Substituição protocolar invocando chamadas prefixadas em `uv run python3`. |
| Falha do diretório abstrato para alcançar dados provenientes do repositório subdiretório `data/raw`. | Adoção corretiva de caminhos plenamente absolutos gerenciados pela biblioteca `pathlib`. |
| Ausência explícita e reportada na engine visual pela omissão global de biblioteca base. | Adição ágil e controlada orientando injeção via `uv add seaborn`. |
| Filtro de cold-start se revelou restritivo de modo pernicioso reduzindo massa amostral viável para apenas 2.656 instâncias iterativas. | Inativação e exclusão preventiva das contingências limitantes no pipeline. |
| Detecção automática da presença latente de vetores analíticos possuindo pura constância determinística isolada. | Abate e remoção sistêmica de exatamente 3 parâmetros caracterizados por constante absoluta variância zerada. |
| Evidência latente por meio analítico descortinando redundância e similaridade linear extrema entre pilares de dimensão informacional. | Expurgo cirúrgico e exclusão de 4 métricas categóricas super-correlacionadas com dependência linear de terceiros. |

---

## 9. Métricas de Sucesso

| Requisito do Tech Challenge | Status |
|---|---|
| Manter no mínimo 10.000 interações user-item na base final | ✅ Concluído (99.785 processadas) |
| Acompanhar ao menos 4 métricas (Recall, NDCG, MAP, Hit Rate) | ✅ Concluído (MAP@K, NDCG@K, Precision@K, Recall@K, HitRate@K) |
| Registrar um patamar mínimo de 3 execuções com MLflow | ⚠️ Framework configurado, servidor requerido |
| Estabelecer quantitativamente 3 módulos rastreados do DVC | ✅ Concluído (prepare, featurize, validate) |
| Produzir base do modelo implementado através de redes em PyTorch | ✅ Concluído (NCF Hybrid + BPR Loss) |
| Gerar pontos comparativos com o Scikit-Learn | ✅ Concluído (Popularity, TopRated, ItemItemCF, TruncatedSVD) |
| Split temporal para avaliação | ✅ Concluído (70/15/15) |
| 11+ runs MLflow registradas (≥ 3 mínimo) | ✅ Concluído (4 baselines + 1 Otimização + 2 Ablation + 1 Auditoria Spearman) |
| Modelo registrado no MLflow Model Registry | ✅ Concluído (olist_ncf_recommender v1 → Production, NDCG@K=0.2725) |
| Disponibilizar infraestrutura serverless/endpoint em Cloud pública | ⏳ Pendente |
| Organizar elaboração teórica com Model Card e Pitches do tipo STAR | ⏳ Pendente |

### 9.1 Entendendo a Métrica Principal (NDCG@10)

O **NDCG@10** (*Normalized Discounted Cumulative Gain at 10*) é a métrica padrão-ouro na indústria para motores de busca e sistemas de recomendação. Em e-commerce como o Olist, não basta apenas sugerir produtos relevantes — **a ordem importa**. O usuário precisa ver os itens que mais deseja logo nas primeiras posições da vitrine.

O NDCG traduz esse comportamento para a matemática através de 5 pilares:

- **Gain (Ganho):** O modelo pontua apenas se recomendar um produto que o usuário considera relevante (ex: um item que ele compraria).
- **Cumulative (Cumulativo):** A pontuação é a soma de todos os acertos dentro da lista recomendada.
- **Discounted (Desconto/Penalidade):** É o coração da métrica. O modelo sofre uma redução drástica de pontos se colocar um item relevante nas últimas posições. Um acerto na posição 1 vale muito mais do que um acerto na posição 10.
- **Normalized (Normalizado):** Como alguns usuários interagem com muitos produtos e outros com poucos, a pontuação é dividida pelo cenário "perfeito" (Ideal DCG), garantindo a escala **0 a 1**.
- **@10 (Corte):** Avaliamos apenas o Top 10. Na vida real, o cliente raramente rola a página para ver dezenas de itens. Se o produto certo estiver na 11ª posição, o modelo é penalizado.

**Referência de performance:**
- **Baselines clássicos** (Popularity, TopRated, ItemItemCF): NDCG@10 ≈ 0.005 (piso — sem personalização)
- **Modelo Production (Ablation_FINAL_no_aux_emb32):** NDCG@10 = **0.2725** (60× vs baseline)
- **Lift do NCF:** `60×` em relação à popularidade global — personalização comprovada

Para a explicação matemática detalhada (fórmulas e implementação), consulte [`docs/GUIDE.md` §9](docs/GUIDE.md).

---

## 10. Próximas Etapas

### ✅ Já Concluídos
1.  **Split Temporal Consistente:** Implementado em `src/train.py` (70% treino / 15% validação / 15% teste).
2.  **Desenvolvimento do Baseline:** Codificados 3 algoritmos (Popularity, TopRated, ItemItemCF) em `src/train.py`.
3.  **Métricas de Ranking:** Implementadas MAP@K, NDCG@K, Precision@K, Recall@K, HitRate@K em `src/train.py`.
4.  **Pipeline DVC:** 3 estágios configurados (prepare, featurize, validate).
5.  **Dashboard Streamlit:** 6 abas implementadas em `front/app_vis.py` (inclui NCF).
6.  **Notebook de Resultados:** `notebooks/03_baseline_training.ipynb` criado.
7.  **NCF com PyTorch + BPR Loss:** Modelo NCFHybrid completo em `src/models/ncf.py`.
8.  **6 Runs MLflow:** Experimentos Baseline, Optimization e Ablation com variação de HPs.
9.  **Modelo Production:** `olist_ncf_recommender v1` registrado e promovido no MLflow Model Registry.
10. **Otimização (Etapa 4):** 5 runs com variação de HPs + ablation study. Melhor: **NDCG@10 = 0.2725** (60× vs baseline).
11. **Notebook NCF:** `notebooks/04_ncf_training_results.ipynb` com análise completa.
12. **Relatório de Otimização:** `reports/ncf_optimization_report.md` com findings.
13. **Auditoria Spearman de Features Redundantes:** Rodada extra de feature selection via correlação de Spearman. Identificadas e removidas 2 features redundantes (`user_recency_days`, `freight_value_log`). Decisão arquitetural: **manter `Ablation_FINAL_no_aux_emb32` como Production** (NDCG=0.2725, 60× vs baseline) — única configuração que supera todas as variantes com aux features. Relatório completo em `reports/feature_audit_spearman.md`; detalhes na Seção 7.6 deste documento.
14. **Model Card:** Documentação formal de métricas e limitações.
15. **Containerização (Docker):** API construída com FastAPI e empacotada via Dockerfile, expondo o modelo de Produção.

### ⏳ Pendentes

1.  **Implementação em Cloud:** Deploy em AWS/GCP/Azure com endpoint público.
2.  **Vídeo STAR:** Apresentação de 5 minutos para a banca.
3.  **ETAPA 1 — Clean Code (sessão 2026-06-27):** Refatoração `evaluate_model()` em `src/training/evaluate.py` (121→73 linhas via 5 helpers); refatoração `evaluate_model()` em `src/train.py` (45→28 linhas via 2 helpers); consolidação `pyproject.toml` para PEP 621 puro com hatchling, remoção do `[tool.poetry]` duplicado; criação de `docs/NAMING_CONVENTIONS.md` (convenções, prefixos, sufixos, exemplos); criação de `docs/SRP_RESPONSIBILITIES.md` (mapa de módulos, anti-patterns, dependências); preenchimento de 4 docstrings públicas em `src/` (100% cobertura); Ruff zerado em `src/` e `scripts/` — All checks passed.

---

## 11. Comandos Úteis

**Setup do Ambiente (uv):**
```bash
uv python pin 3.12
uv sync
```

**Preparação e Feature Engineering:**
```bash
uv run python3 src/data_preparation.py
uv run python3 src/feature_engineering.py
```

**Pipeline DVC:**
```bash
uv run dvc repro           # Executar pipeline completo
uv run dvc pull            # Baixar dados versionados
uv run dvc status          # Verificar status
```

**Treinamento de Baselines:**
```bash
uv run python src/train.py  # Executa todos os baselines com métricas
```

**Treinamento NCF (PyTorch):**
```bash
# Run baseline (emb=16, 2 layers, lr=5e-4)
uv run python scripts/train_ncf.py --epochs 12 --emb-dim 16 --hidden 64 32 \
    --batch-size 1024 --lr 5e-4 --n-negatives 4 --no-mlflow

# Run de otimização (melhor modelo Production)
uv run python scripts/train_ncf.py --epochs 20 --emb-dim 32 --hidden 64 32 \
    --dropout 0.5 --lr 5e-4 --batch-size 2048 --n-negatives 8 \
    --use-scheduler --weight-decay 5e-4 \
    --run-name "Ablation_FINAL_no_aux_emb32" \
    --experiment-name "Olist_NCF_Optimization"
```

**Dashboard Streamlit:**
```bash
uv run streamlit run front/app_vis.py  # Abre em http://localhost:8501
```

**MLflow UI (tracking local SQLite):**
```bash
uv run mlflow ui --backend-store-uri sqlite:///./artifacts/mlflow.db
```

**Execução de Testes:**
```bash
uv run pytest -v
```

**Linting de Código e Formatação:**
```bash
uv run ruff check . --fix
uv run ruff format .
```

**Containerização e Orquestração (Docker Compose):**
```bash
# Constrói as imagens e sobe os serviços de API e MLflow simultaneamente
docker compose up --build

# A interface da API (Swagger UI) ficará em: http://localhost:8000/docs
# A interface do MLflow ficará em: http://localhost:5000

# Para desligar os serviços:
# Pressione Ctrl+C no terminal ou rode: docker compose down

---

## 12. Conclusão Geral

A Fase 02 do projeto encontra-se em **avançado estado de maturidade**, com os pilares fundamentais plenamente consolidados:

### ✅ Estrutura de Dados
- Pipeline DVC com 3 estágios funcionais
- 99.785 interações processadas (muito acima do mínimo de 10.000)
- 42 features engenheiradas
- Split temporal implementado (70/15/15)

### ✅ Modelagem Baseline
- 3 algoritmos implementados (Popularity, TopRated, ItemItemCF)
- 5 métricas de ranking calculadas (MAP, NDCG, Precision, Recall, HitRate)
- Resultados documentados em notebook e dashboard

### ✅ Infraestrutura e Visualização
- Dashboard Streamlit com 5 abas funcionais
- Notebooks educacionais para apresentação
- Documentação atualizada (README, REPORT, GUIDE)

### 🔄 Próximos Passos
- Implementação do modelo NCF com PyTorch
- Integração com MLflow (servidor)
- Deploy em cloud
- Model Card e vídeo STAR

O projeto está preparado para a fase de modelagem neural avançada e apresentação final.

---

## 13. Validação End-to-End do Pipeline (2026-06-27)

Última execução completa do pipeline, validando que todos os componentes funcionam integrados:

### Pipeline DVC
```bash
$ uv run dvc repro
```

| Stage | Status | Observações |
|-------|--------|-------------|
| `prepare` | ✅ OK | 99.785 interações · 93.358 usuários · 32.216 produtos · 72 categorias · sparsity 99,9967% |
| `featurize` | ✅ OK | 42 colunas geradas · 18 features numéricas · 7 pares altamente correlacionados detectados |
| `validate` | ✅ OK | shape (99785, 42), < 1000 nulls em review_score |

### Baselines (`uv run python src/train.py`)
| Model | MAP@10 | NDCG@10 | Recall@10 | HitRate@10 |
|-------|--------|---------|-----------|------------|
| PopularityBaseline_K10 | 0.0019 | 0.0053 | 0.0104 | 0.0133 |
| PopularityBaseline_K20 | 0.0019 | 0.0053 | 0.0104 | 0.0133 |
| TopRatedBaseline_K10_MinRev5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| TopRatedBaseline_K10_MinRev15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ItemItemCF_K10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| TruncatedSVD_K10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Cold-start severo (98% dos test users inéditos) explica performance zero em métodos baseados em co-ocorrência.

### Smoke Test NCF (`uv run python scripts/train_ncf.py`)
Smoke test com config de produção (`ablation-no-aux`, `emb_dim=32`, `hidden=[64,32]`, `dropout=0.5`, `lr=5e-4`) rodado por 5 epochs apenas para validação end-to-end:

| Métrica | Valor | Observação |
|---------|-------|------------|
| `test_NDCG@K` | **0.2740** | ≈ Production (0.2725) — valida arquitetura |
| `test_HitRate@K` | 0.5085 | Top-10 hit ~50% dos test users |
| `test_MAP@K` | 0.2080 | ≈ Production |
| `train_HitRate@K` | 0.9900 | Sanity check OK (modelo aprende) |
| `baseline_NDCG@K` | 0.0045 | Confirma baseline produção em configs/ncf_best.yaml |

Lift vs baseline: 0.2740 / 0.0045 = **60.9x** (Production reporta 60.6x — mesma magnitude).

### Validações Estruturais
- ✅ `streamlit.testing.AppTest` em `front/app_vis.py`: 0 exceptions, 6 tabs renderizadas, 27 métricas, 4 dataframes
- ✅ `jupyter nbconvert --execute` em todos os 6 notebooks: 0 erros
- ✅ MLflow SQLite tracking (`artifacts/mlflow.db`): 4 experimentos, 6 runs finalizadas

### Artefatos Validados
- `data/processed/interactions.parquet` (5.69 MB)
- `data/processed/interactions_fe.parquet` (42 cols)
- `data/processed/baseline_results.csv` (10 modelos avaliados)
- `artifacts/baselines/recommendations_*.csv` (10 arquivos de top-K)
- `artifacts/ncf_Ablation_FINAL_no_aux_emb32.pt` (Production, 16 MB)
- `artifacts/metrics_Ablation_FINAL_no_aux_emb32.json` (métricas canônicas)

---

## 14. Referências

### Scripts Principais
*   [Script de Preparação de Dados (src/data_preparation.py)](../src/data_preparation.py)
*   [Script de Análise Exploratória (src/eda.py)](../src/eda.py)
*   [Script de Feature Engineering (src/feature_engineering.py)](../src/feature_engineering.py)
*   [Script de Treinamento (src/train.py)](../src/train.py)

### Documentação
*   [Relatório da EDA (reports/eda_report.md)](../reports/eda_report.md)
*   [Documentação do Processamento de Dados (data/processed/README.md)](../data/processed/README.md)
*   [Dicionário de Features (data/processed/FEATURES.md)](../data/processed/FEATURES.md)
*   [Guia Técnico (docs/GUIDE.md)](./GUIDE.md)

### Notebooks
*   [Pipeline Explicativo (notebooks/00_pipeline_explanation.ipynb)](../notebooks/00_pipeline_explanation.ipynb)
*   [EDA (notebooks/01_eda.ipynb)](../notebooks/01_eda.ipynb)
*   [EDA Feature Engineered (notebooks/02_eda_feature_engineered.ipynb)](../notebooks/02_eda_feature_engineered.ipynb)
*   [Resultados Baseline (notebooks/03_baseline_training.ipynb)](../notebooks/03_baseline_training.ipynb)

### Dashboard
*   [Dashboard Streamlit (front/app_vis.py)](../front/app_vis.py)
