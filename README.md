# Tech Challenge – Sistema de Recomendação de Produtos para E-commerce

Projeto desenvolvido como parte da Pós-Tech em Machine Learning Engineering da FIAP. O objetivo é criar um sistema preditivo baseado no histórico de navegação de usuários, utilizando ferramentas de nível de produção para o ciclo de vida de modelos de IA.

---

## 🚀 Divisão de Responsabilidades e Checklist de Acompanhamento

Abaixo está o cronograma macro e o painel de evolução do projeto. Os responsáveis devem marcar as tarefas concluídas alterando de `[ ]` para `[x]`.

### 📅 Fase 1: Arquitetura de Software e Ambiente (07/06 a 13/06)
**Responsável:** Fábio Polli
- [x] Configurar o repositório base, assegurando a presença e o rigor dos arquivos `.gitignore`, `.dockerignore` e `.env.example`.
- [x] Organizar a arquitetura de diretórios da aplicação: `src/`, `tests/`, `data/`, `models/` e `configs/`.
- [x] Implementar a gestão de dependências via Poetry no arquivo `pyproject.toml`, separando dependências de produção e desenvolvimento, versionando o arquivo de *lock*.
- [x] Aplicar diretrizes de Clean Code implementando o linter Ruff e *pre-commit hooks*, garantindo conformidade com os princípios SOLID e padronização por *type hints*.

### 📅 Fase 2: Engenharia de Dados e Versionamento (14/06 a 20/06)
**Responsáveis:** Denis & Romário
- [x] Homologar um dataset de interações de e-commerce que atenda ao requisito mínimo estabelecido de 10.000 interações do tipo *user-item*.
- [x] Desenvolver os pipelines de pré-processamento e os modelos preditivos base (*baselines*) utilizando o framework Scikit-Learn.
- [x] Inicializar a ferramenta DVC para o controle de versão massivo de dados, configurando adequadamente o repositório remoto.
- [x] Construir o pipeline de dados automatizado no arquivo `dvc.yaml`, contemplando no mínimo três estágios lógicos e sequenciais.

### 📅 Fase 3: Ciência de Dados e Modelagem de IA (21/06 a 27/06)
**Responsáveis:** A definir
- [ ] Projetar a arquitetura da rede neural (MLP ou modelo baseado em *embeddings*) para o sistema de recomendação utilizando PyTorch.
- [ ] Executar o treinamento com critério de parada antecipada (*early stopping*) e avaliar a performance contra os *baselines* usando ao menos 4 métricas quantitativas.
- [ ] Integrar a plataforma MLflow ao código para garantir o rastreamento técnico (parâmetros, métricas e artefatos) de, ao menos, três ciclos de execução (*runs*).
- [ ] Cadastrar o modelo final validado no MLflow Model Registry e conduzir sua promoção formal para o estágio de Produção.

### 📅 Fase 4: DevOps e Infraestrutura em Nuvem (04/07 a 10/07)
**Responsáveis:** Fábio Polli & Bill
- [ ] Elaborar o arquivo Dockerfile adotando a abordagem *multi-stage* (separação de *builder* e *runtime*).
- [ ] Configurar o arquivo de orquestração `docker-compose.yml` para inicializar de forma simultânea o serviço de inferência/treinamento e o servidor do MLflow.
- [ ] Realizar o deploy do sistema em ambiente de nuvem (AWS/Azure/GCP), contemplando a atividade bônus de infraestrutura.
- [ ] Assegurar e testar o acesso público ao container via URL.

### 📅 Fase 5: Qualidade, Documentação e Entrega (11/07 a 13/07)
**Responsáveis:** A definir (Entrega Final: 14/07)
- [ ] Conduzir a auditoria do repositório, garantindo a utilização de um histórico de *commits* semântico.
- [ ] Redigir o *Model Card* do sistema, descrevendo detalhadamente a performance alcançada, limitações do modelo e eventuais vieses identificados nos dados.
- [ ] Revisar e finalizar a documentação no arquivo README, fornecendo instruções exaustivas para a reprodução do ambiente.
- [ ] Roteirizar e gravar a apresentação em vídeo (duração de 5 minutos), estruturando a narrativa obrigatoriamente através do método STAR (Situação, Tarefa, Ação e Resultado).

---

## 🛠️ Como Executar o Projeto Localmente

Para garantir que todos os membros do time utilizem exatamente as mesmas versões de bibliotecas, utilizamos o **Poetry** como gerenciador de dependências e o **DVC** para versionamento de dados.

### Pré-requisitos
* Python 3.10 ou superior instalado.
* [Poetry](https://python-poetry.org/docs/#installation) instalado na máquina.

### Passo a Passo de Configuração

**1. Clonar o repositório:**
```bash
git clone [https://github.com/fabiopolli/pos-ml-eng-tech-challenge-fase-02.git](https://github.com/fabiopolli/pos-ml-eng-tech-challenge-fase-02.git)
cd pos-ml-eng-tech-challenge-fase-02
```

**2. Instalar as dependências do projeto:**
Este comando lerá o arquivo `poetry.lock` e criará um ambiente virtual isolado com as versões exatas do PyTorch, scikit-learn, MLflow, etc.
```bash
poetry install
```

**3. Ativar o ambiente virtual:**
*Nota: A partir da versão 2.0.0 do Poetry, o comando `shell` foi movido para um plugin. Se o comando abaixo der erro, instale o plugin primeiro com `poetry self add poetry-plugin-shell`.*
```bash
poetry shell
```

**4. Instalar os hooks de validação (Pre-commit):**
Essencial para garantir que o código passe pelo linter (Ruff) antes de qualquer commit na sua branch.
```bash
pre-commit install
```

**5. Configurar as variáveis de ambiente:**
Faça uma cópia do arquivo de template e preencha com as credenciais que serão definidas pelo time de infraestrutura.
* **Linux/Mac:** `cp .env.example .env`
* **Windows (PowerShell):** `Copy-Item .env.example -Destination .env`

---
**⚠️ Para as próximas fases (Dados e Modelos):**
Sempre que puxar atualizações da branch `main` que envolvam novos datasets ou modelos treinados, certifique-se de atualizar também os artefatos do DVC rodando:
```bash
dvc pull
```
