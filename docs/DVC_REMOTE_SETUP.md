# Configuração de Remote Storage para DVC

Este documento descreve como configurar o remote storage do DVC para o projeto Olist Recommender.

## Status Atual
- DVC inicializado e arquivos rastreados localmente
- Remote storage NÃO configurado (deve ser feito pelo time)
- 11 arquivos em data/raw/*.csv.dvc
- 2 arquivos em data/processed/*.parquet.dvc

## Opções de Remote Storage

### Opção 1: DagsHub (Recomendado para desenvolvimento)

**Vantagens**: Gratuito, integrado com Git + DVC + MLflow, ideal para projetos acadêmicos.

```bash
# 1. Criar conta em https://dagshub.com
# 2. Criar repositório no DagsHub
# 3. Configurar DVC:

dvc remote add origin https://dagshub.com/<usuario>/pos-ml-eng-tech-challenge-fase-02.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <seu_usuario_dagshub>
dvc remote modify origin --local password <seu_token_dagshub>

# 4. Push dos dados:
dvc push -r origin
```

### Opção 2: AWS S3 (Para produção)

```bash
# Configurar credenciais AWS
export AWS_ACCESS_KEY_ID=<sua_key>
export AWS_SECRET_ACCESS_KEY=<seu_secret>

# Configurar DVC
dvc remote add -d s3remote s3://<bucket-name>/dvc-store
dvc remote modify s3remote region us-east-1

# Push
dvc push -r s3remote
```

### Opção 3: Google Cloud Storage

```bash
# Configurar credenciais GCP
export GOOGLE_APPLICATION_CREDENTIALS=<path_to_json>

# Configurar DVC
dvc remote add -d gsremote gs://<bucket-name>/dvc-store

# Push
dvc push -r gsremote
```

## Comandos Úteis DVC

```bash
# Verificar status
dvc status

# Reproduzir pipeline completo
dvc repro

# Executar apenas um estágio
dvc repro prepare
dvc repro featurize
dvc repro validate

# Verificar integridade dos dados
dvc check

# Verificar métricas
dvc metrics show

# Verificar plots
dvc plots show

# Pull dados do remote
dvc pull

# Push dados para o remote
dvc push
```

## Pipeline Atual

O projeto possui 3 estágios no dvc.yaml:

1. **prepare**: Converte CSVs brutos em parquet limpo
2. **featurize**: Gera 42 features engenheiradas
3. **validate**: Valida shape e qualidade do dataset final
