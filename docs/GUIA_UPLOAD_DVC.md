# Guia de Upload de Dados no DVC

> Todos os comandos DVC devem ser executados com `uv run dvc` dentro da pasta `pos-ml-eng-tech-challenge-fase-02`.

---

## Pré-requisitos

- `uv` instalado
- Arquivo de credencial da Service Account disponível em:  
  `../chave google cloud/dvc-projeto-fiap-be5daca720b5.json`
- A pasta de destino no Google Drive deve ser um **Shared Drive** com a Service Account adicionada como **Editor**

---

## 1. Configurar o remote (uma vez por máquina)

```powershell
# Adicionar o remote Google Drive como padrão
uv run dvc remote add --default group_remote gdrive://1mMcykBzLNdYPWavDNbf5o2UX9L5D0K6B

# Configurar autenticação por Service Account (salvo localmente, não vai pro Git)
uv run dvc remote modify --local group_remote gdrive_use_service_account true
uv run dvc remote modify --local group_remote gdrive_service_account_json_file_path "../chave google cloud/dvc-projeto-fiap-be5daca720b5.json"
```

---

## 2. Adicionar arquivos ao rastreamento DVC

Coloque os arquivos na pasta `data/` e execute:

```powershell
# Rastrear um arquivo específico
uv run dvc add data/nome_do_arquivo.csv

# Rastrear todos os CSVs (PowerShell)
Get-ChildItem data -Filter *.csv | ForEach-Object { uv run dvc add "data/$($_.Name)" }
```

Isso cria arquivos `.dvc` para cada dataset. **Adicione esses `.dvc` ao Git:**

```powershell
git add data/*.dvc .gitignore
git commit -m "feat: adiciona datasets via DVC"
```

---

## 3. Enviar dados para o Google Drive

```powershell
uv run dvc push
```

---

## 4. Baixar dados em outra máquina

```powershell
# Após clonar o repositório e configurar o remote (passo 1):
uv run dvc pull
```

---

## Resolução de problemas

| Erro | Causa | Solução |
|------|-------|---------|
| `No module named 'dvc_gdrive'` | Plugin não instalado | `uv add dvc-gdrive` |
| `Service Accounts do not have storage quota` | Pasta não é um Shared Drive | Criar Shared Drive e adicionar a Service Account como Editor |
| `dvc: command not found` | DVC não está no PATH global | Usar sempre `uv run dvc` |

---

## Estrutura de arquivos gerada

```
data/
├── olist_orders_dataset.csv          ← ignorado pelo Git
├── olist_orders_dataset.csv.dvc      ← versionado pelo Git
└── ...
```
