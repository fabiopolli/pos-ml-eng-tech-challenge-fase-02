# Proposta de Plano de Action: Tech Challenge – Sistema de Recomendação

O escopo do projeto exige o desenvolvimento de um sistema de recomendação de produtos para e-commerce, fundamentado no histórico de navegação dos usuários. Para contemplar todos os requisitos técnicos (PyTorch, Docker, MLflow e DVC) e garantir o desenvolvimento em paralelo sem bloqueios entre os membros, proponho a divisão do projeto em cinco fases sequenciais e interdependentes.

## Divisão de Responsabilidades e Cronograma Macro

| Fase | Foco Estratégico | Tecnologias Principais |
| :--- | :--- | :--- |
| **Fase 1** | Arquitetura de Software e Ambiente | GitHub, Poetry, Ruff |
| **Fase 2** | Engenharia de Dados e Versionamento | Scikit-Learn, DVC |
| **Fase 3** | Ciência de Dados e Modelagem de IA | PyTorch, MLflow |
| **Fase 4** | DevOps e Infraestrutura em Nuvem | Docker, Nuvem (AWS/Azure/GCP) |
| **Fase 5** | Qualidade, Documentação e Entrega | Avaliação Geral |

---

## Detalhamento Técnico das Fases de Execução

### Fase 1: Arquitetura de Software e Parametrização do Ambiente
Estabelecimento da infraestrutura do repositório para garantir a padronização do código.
* Configurar o repositório base, assegurando a presença e o rigor dos arquivos `.gitignore`, `.dockerignore` e `.env.example`.
* Organizar a arquitetura de diretórios da aplicação, segmentando em: `src/`, `tests/`, `data/`, `models/` e `configs/`.
* Implementar a gestão de dependências via Poetry (ou uv) no arquivo `pyproject.toml`, separando rigorosamente as dependências de produção e desenvolvimento, além de versionar o arquivo de *lock*.
* Aplicar diretrizes de Clean Code implementando o linter Ruff e *pre-commit hooks*, garantindo conformidade com os princípios SOLID e padronização por *type hints*.

### Fase 2: Engenharia de Dados, Baselines e Versionamento
Estruturação do fluxo de dados que alimentará o sistema preditivo.
* Homologar um dataset de interações de e-commerce que atenda ao requisito mínimo estabelecido de 10.000 interações do tipo *user-item*.
* Desenvolver os pipelines de pré-processamento e os modelos preditivos base (*baselines*) utilizando o framework Scikit-Learn.
* Inicializar a ferramenta DVC para o controle de versão massivo de dados, configurando adequadamente o repositório remoto.
* Construir o pipeline de dados automatizado no arquivo `dvc.yaml`, contemplando no mínimo três estágios lógicos e sequenciais.

### Fase 3: Ciência de Dados, Treinamento e Rastreamento
Desenvolvimento e mensuração do modelo de rede neural central.
* Projetar a arquitetura da rede neural (MLP ou modelo baseado em *embeddings*) para o sistema de recomendação, utilizando a biblioteca PyTorch.
* Executar o treinamento com critério de parada antecipada (*early stopping*) e avaliar a performance do modelo contra os *baselines*, utilizando um mínimo de quatro métricas quantitativas.
* Integrar a plataforma MLflow ao código para garantir o rastreamento técnico (parâmetros, métricas e artefatos) de, ao menos, três ciclos de execução (*runs*).
* Cadastrar o modelo final validado no MLflow Model Registry e conduzir sua promoção formal para o estágio de Produção.

### Fase 4: DevOps, Containerização e Deploy (Nuvem)
Empacotamento da solução e disponibilização do ambiente em arquitetura escalável.
* Elaborar o arquivo Dockerfile adotando a abordagem *multi-stage*, visando a separação das camadas de construção (*builder*) e execução (*runtime*) para otimização da imagem final.
* Configurar o arquivo de orquestração `docker-compose.yml` para inicializar de forma simultânea o serviço de inferência/treinamento e o servidor do MLflow.
* Realizar o deploy do sistema em ambiente de nuvem (AWS, Azure ou GCP), contemplando a atividade bônus de infraestrutura.
* Assegurar e testar o acesso público ao container via URL.

### Fase 5: Qualidade, Documentação e Apresentação Executiva
Consolidação das entregas, validação de regras de negócios e estruturação da apresentação.
* Conduzir a auditoria do repositório,ax garantindo a utilização de um histórico de *commits* semântico.
* Redigir o *Model Card* do sistema, descrevendo detalhadamente a performance alcançada, as limitações do modelo e os eventuais vieses identificados nos dados.
* Revisar e finalizar a documentação no arquivo README, fornecendo instruções exaustivas para a reprodução do ambiente.
* Roteirizar e gravar a apresentação em vídeo (duração de 5 minutos), estruturando a narrativa obrigatoriamente através do método STAR: Situação (*Situation*), Tarefa (*Task*), Ação (*Action*) e Resultado (*Result*).
