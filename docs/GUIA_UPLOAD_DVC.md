# Guia de Upload de Dados no DVC (DagsHub)

> **Convenção do projeto:** todos os comandos DVC devem ser executados com `uv run dvc` dentro da pasta `pos-ml-eng-tech-challenge-fase-02`. Isso garante o uso das dependências declaradas em `pyproject.toml` e mantém o ambiente isolado.

---

## Visão Geral da Arquitetura

O projeto utiliza **DVC (Data Version Control)** com **DagsHub** como remote storage. A configuração fica centralizada em [`.dvc/config`](../.dvc/config):

```ini
[core]
    no_scm = true
    remote = origin
['remote "origin"']
    url = https://dagshub.com/deniscelclaro/projeto_fiap_modulo2.dvc
```

| Camada | Ferramenta | Conteúdo |
|--------|-----------|----------|
| **Metadados** | Git (GitHub) | Arquivos `*.csv.dvc` + `dvc.lock` |
| **Dados binários** | DagsHub (bucket S3-compatible) | Conteúdo bruto dos CSVs |

> **Sem credenciais locais.** O DagsHub autentica via token do usuário, dispensando Service Accounts, JSON keys e plugins (`dvc-gdrive`, etc.).

---

## Pré-requisitos

- `uv` instalado (≥ 0.4)
- Conta no **DagsHub** com acesso ao repositório `deniscelclaro/projeto_fiap_modulo2`
- **Token de autenticação** DagsHub (Settings → Access Tokens), com escopo de leitura/escrita no repositório

---

## 1. Configurar o remote (apenas primeira vez por máquina)

A configuração do remote já está versionada em `.dvc/config`. O que falta é **apenas o token local** (não vai para o Git):

```bash
# Linux / macOS / WSL
uv run dvc remote modify --local origin auth basic
uv run dvc remote modify --local origin user <seu_usuario_dagshub>
uv run dvc remote modify --local origin password <seu_token_dagshub>

# Alternativa: definir variáveis de ambiente (não persistem no disco)
export DAGSHUB_TOKEN="<seu_token>"
```

> **Dica:** prefira variáveis de ambiente em CI/CD ou Docker. Para uso local, `--local` salva em `.dvc/config.local` (já no `.gitignore`).

---

## 2. Rastrear arquivos no DVC

Coloque os CSVs em `data/` (ou onde o pipeline espera) e execute:

```bash
# Rastrear um arquivo específico
uv run dvc add data/nome_do_arquivo.csv

# Rastrear todos os CSVs de uma vez (bash)
for f in data/*.csv; do uv run dvc add "$f"; done
```

Isso gera:
- `data/<arquivo>.csv.dvc` — metadados (hash MD5, tamanho, path) — **versionado pelo Git**
- Entrada em `data/.gitignore` para ignorar o CSV bruto

**Faça commit dos metadados:**

```bash
git add data/*.dvc data/.gitignore
git commit -m "feat: rastreia datasets via DVC"
```

---

## 3. Enviar dados para o DagsHub

```bash
uv run dvc push
```

O DVC calcula os hashes, identifica os arquivos novos/modificados e faz upload apenas do delta para o bucket S3 do DagsHub.

---

## 4. Baixar dados em outra máquina (clone novo)

```bash
# Após clonar o repositório:
uv sync
uv run dvc pull
```

O `dvc pull` lê os `.dvc` do Git e baixa do DagsHub apenas o que está faltando no cache local.

---

## Fluxo Operacional Resumido

```
┌─────────────────────────────────────────────────────────┐
│ 1. dvc add data/foo.csv                                 │
│    └─> gera data/foo.csv.dvc (entra no Git)             │
├─────────────────────────────────────────────────────────┤
│ 2. git add data/foo.csv.dvc && git commit && git push   │
├─────────────────────────────────────────────────────────┤
│ 3. dvc push                                             │
│    └─> envia o conteúdo para DagsHub (S3)               │
└─────────────────────────────────────────────────────────┘

Em outra máquina:
┌─────────────────────────────────────────────────────────┐
│ 1. git pull                                             │
│ 2. dvc pull                                             │
│    └─> baixa do DagsHub conforme os .dvc do Git         │
└─────────────────────────────────────────────────────────┘
```

---

## Comandos Úteis do Dia-a-Dia

| Comando | Função |
|---------|--------|
| `uv run dvc repro` | Executa o pipeline completo (`dvc.yaml`) |
| `uv run dvc repro -s prepare` | Executa apenas o estágio `prepare` |
| `uv run dvc status` | Mostra o que está desatualizado em relação ao lock |
| `uv run dvc dag` | Imprime o grafo de dependências do pipeline |
| `uv run dvc metrics show` | Exibe métricas registradas pelos estágios |
| `uv run dvc push` | Sobe os dados para o DagsHub |
| `uv run dvc pull` | Baixa os dados do DagsHub |
| `uv run dvc remote list` | Lista os remotes configurados |

---

## Resolução de Problemas

| Erro | Causa | Solução |
|------|-------|---------|
| `HTTPError: 401 Unauthorized` | Token ausente ou inválido | Reconfigurar `--local origin password` |
| `HTTPError: 403 Forbidden` | Sem permissão no repo DagsHub | Solicitar acesso ao owner do projeto |
| `dvc: command not found` | DVC fora do PATH | Usar sempre `uv run dvc` |
| `ERROR: failed to upload 'X' - md5 mismatch` | Arquivo mudou entre `add` e `push` | Re-executar `dvc add` e refazer `commit + push` |
| `Cache 'XXXX' not found` | Cache local corrompido | `dvc pull -f` força re-download |

---

## Boas Práticas

1. **Nunca versione o CSV bruto** — apenas os `.dvc`. O `.gitignore` da pasta `data/` já garante isso.
2. **Sempre rode `dvc add` após alterar um CSV** — caso contrário o push não capturará a mudança.
3. **Mantenha `dvc.lock` versionado** — ele pina os hashes das saídas de cada estágio, garantindo reprodutibilidade.
4. **Documente o remote no `README`** — para que novos colaboradores saibam onde os dados vivem.
5. **Use `--no-scm` quando relevante** — `no_scm = true` no `.dvc/config` evita que o DVC tente versionar dados via Git.

---

## Estrutura Típica Gerada

```
data/
├── olist_orders_dataset.csv          ← ignorado pelo Git, presente no DagsHub
├── olist_orders_dataset.csv.dvc      ← versionado pelo Git
└── .gitignore                        ← ignora *.csv (criado/atualizado pelo dvc add)
```

---

## Migração do Pipeline Antigo (Google Drive → DagsHub)

O projeto utilizava Google Drive com Service Account como remote. A migração para DagsHub foi motivada por:

- Eliminação de credenciais locais (Service Account JSON)
- Remoção da dependência `dvc-gdrive`
- Resolução do erro `Service Accounts do not have storage quota` (limite do Google Drive)
- Integração nativa com o **MLflow Tracking** já hospedado no DagsHub
- Autenticação simplificada via token único

O remote antigo (`group_remote` apontando para Google Drive) foi removido do `.dvc/config` no commit `e1d3d7c`.