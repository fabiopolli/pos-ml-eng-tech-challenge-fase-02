# DVC Pipeline - Olist E-Commerce Review Score Prediction

## 📋 Visão Geral

Este projeto utiliza **DVC (Data Version Control)** para orquestrar um pipeline de machine learning reproduzível com 3 estágios sequenciais:

```
prepare (Dados Brutos)
    ↓
train (Modelos)
    ↓
evaluate (Artefatos)
```

## 🔧 Configuração Inicial

### Pré-requisitos
- Python 3.8+
- Git (para versionamento)
- DVC (já instalado e configurado)

### Inicializar DVC (primeira vez)
```bash
dvc init
dvc remote add -d local_storage C:\dvc_storage\pos-ml-eng-tech
```

## 📊 Pipeline: 3 Estágios Sequenciais

### Estágio 1: Preparação de Dados (`prepare`)
**Script:** `src/data_preparation.py`

**O que faz:**
- Carrega todos os 8 arquivos CSV do Olist
- Aplica pré-processamento e engenharia de features
- Divide em treino/teste
- Salva dados processados em `data/processed/`

**Dependências:** Arquivos CSV brutos em `data/`

**Saídas:**
- `data/processed/X_train.csv` - Features de treino
- `data/processed/X_test.csv` - Features de teste
- `data/processed/y_train.csv` - Alvo de treino
- `data/processed/y_test.csv` - Alvo de teste
- `data/processed/metadata.json` - Metadados

### Estágio 2: Treinamento de Modelos (`train`)
**Script:** `src/model_training.py`

**O que faz:**
- Carrega dados processados do Estágio 1
- Treina 4 modelos baseline:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Avalia com validação cruzada
- Salva modelos em `models/`

**Dependências:** Saídas do estágio `prepare`

**Saídas:**
- `models/[model_name]_model.pkl` - Modelo serializado
- `models/[model_name]_metrics.json` - Métricas de performance
- `models/model_comparison.csv` - Comparativa de modelos
- `models/best_model.txt` - Nome do melhor modelo

### Estágio 3: Avaliação e Artefatos (`evaluate`)
**Script:** `src/model_evaluation.py`

**O que faz:**
- Carrega dados e modelos dos estágios anteriores
- Gera visualizações do melhor modelo
- Cria matriz de confusão
- Gera gráfico de importância de features
- Produz relatório final

**Dependências:** Saídas dos estágios `prepare` e `train`

**Saídas:**
- `artifacts/confusion_matrix_[modelo].png` - Matriz de confusão
- `artifacts/feature_importance_[modelo].png` - Importância de features
- `artifacts/model_comparison.csv` - Tabela comparativa
- `artifacts/evaluation_report.json` - Relatório completo

## 🚀 Executando o Pipeline

### Executar todo o pipeline
```bash
dvc repro
```
Executa automaticamente os 3 estágios na ordem correta, pulando estágios cujas entradas não mudaram.

### Executar estágio específico
```bash
dvc repro --single-stage prepare
dvc repro --single-stage train
dvc repro --single-stage evaluate
```

### Forçar re-execução
```bash
dvc repro --force
```

### Ver status
```bash
dvc status
```

### Ver DAG (grafo) do pipeline
```bash
dvc dag
```

## 📦 Armazenamento Remoto

### Configuração Atual
- **Tipo:** Local filesystem
- **Caminho:** `C:\dvc_storage\pos-ml-eng-tech`
- **Padrão:** Sim (default remote)

### Usar S3 (Exemplo)
```bash
dvc remote modify local_storage url s3://my-bucket/pos-ml-eng-tech
dvc remote add -d aws_storage s3://my-bucket/dvc-storage
```

### Sincronizar com remoto
```bash
dvc push      # Enviar dados para remoto
dvc pull      # Baixar dados do remoto
dvc fetch     # Buscar atualizações
```

## 🔄 Rastreamento de Dados

### Arquivo de Lock
- **Arquivo:** `dvc.lock`
- **O que contém:** Hashes de todos os inputs/outputs para reproducibilidade
- **Atualizado:** Automaticamente após cada `dvc repro`

### .gitignore DVC
O DVC gerencia automaticamente o `.gitignore` para:
- `data/processed/`
- `models/`
- `artifacts/`
- Cache local (`.dvc/cache/`)

## 📈 Reprodutibilidade

Para garantir reproductibilidade entre máquinas:

1. **Clonar repositório**
   ```bash
   git clone <repo>
   cd pos-ml-eng-tech-challenge-fase-02
   ```

2. **Restaurar dados do remoto**
   ```bash
   dvc pull
   ```

3. **Reproduzir pipeline**
   ```bash
   dvc repro
   ```

Todos os dados, modelos e artefatos serão restaurados exatamente como foram gerados!

## 🛠️ Troubleshooting

### Erro: ".dvc is busy"
```bash
rm .dvc/tmp/rwlock
dvc repro
```

### Erro: Remote not configured
```bash
dvc remote list
dvc remote add -d local_storage C:\dvc_storage\pos-ml-eng-tech
```

### Limpar cache
```bash
dvc cache remove --not-in-remote
dvc cache prune
```

## 📊 Estrutura de Saídas

```
projeto/
├── data/
│   ├── raw/*.csv                  # Dados originais
│   └── processed/                 # DVC tracked
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── metadata.json
├── models/                        # DVC tracked
│   ├── gradient_boosting_model.pkl
│   ├── *_metrics.json
│   ├── model_comparison.csv
│   └── best_model.txt
├── artifacts/                     # DVC tracked
│   ├── confusion_matrix_*.png
│   ├── feature_importance_*.png
│   ├── model_comparison.csv
│   └── evaluation_report.json
├── src/
│   ├── data_preparation.py        # Stage 1
│   ├── model_training.py          # Stage 2
│   ├── model_evaluation.py        # Stage 3
│   ├── preprocessing_pipeline.py
│   └── baseline_models.py
├── dvc.yaml                       # Definição do pipeline
├── dvc.lock                       # Estado reproduzível
└── .dvc/
    └── config                     # Configuração DVC
```

## 🎯 Próximos Passos

1. **Integração com Git:**
   ```bash
   git add dvc.yaml dvc.lock .dvc/config .gitignore
   git commit -m "Add DVC pipeline"
   ```

2. **CI/CD:** Configure GitHub Actions para executar `dvc repro` automaticamente

3. **Versionamento de Modelos:** Use `git tag` com `dvc` para versionar releases

4. **Monitoramento:** Configure alertas para mudanças de performance

## 📚 Referências

- [DVC Documentation](https://dvc.org/doc)
- [DVC Pipelines](https://dvc.org/doc/user-guide/pipelines)
- [Remote Storage](https://dvc.org/doc/user-guide/remote)
