# ✅ SOLUÇÃO: Usar `python -m dvc` no PowerShell

## 🔴 O Erro Que Você Teve

```powershell
PS> dvc dag
dvc : O termo 'dvc' não é reconhecido...
```

## 🟢 A Solução

**Use `python -m dvc` em vez de apenas `dvc`:**

```powershell
python -m dvc dag
```

---

## 📊 Testes Que Agora Funcionam

### ✅ Teste 1: Versão
```powershell
python -m dvc --version
# Output: 3.67.1
```

### ✅ Teste 2: Remoto
```powershell
python -m dvc remote list
# Output: local_storage   C:\dvc_storage\pos-ml-eng-tech  (default)
```

### ✅ Teste 3: Pipeline
```powershell
python -m dvc dag
# Output: prepare → train → evaluate
```

### ✅ Teste 4: Status
```powershell
python -m dvc status
# Output: Mostra quais estágios precisam rodar
```

### ✅ Teste 5: Executar
```powershell
python -m dvc repro
# Executa todo o pipeline (7-13 minutos)
```

---

## 🎯 Comandos Rápidos Para Copiar

```powershell
# Versão
python -m dvc --version

# Status
python -m dvc status

# Pipeline
python -m dvc dag

# Remoto
python -m dvc remote list

# Executar
python -m dvc repro

# Um estágio
python -m dvc repro --single-stage train
```

---

## 💡 Por Quê Isso Acontece?

- **PowerShell** não consegue localizar `dvc` no PATH
- **DVC** é um módulo Python, não um executável do sistema
- **`python -m`** executa um módulo Python
- A solução é sempre usar `python -m dvc` no PowerShell

---

## ⚙️ Opcional: Criar Alias Permanente

Se quiser usar apenas `dvc` sem `python -m`, execute uma vez:

```powershell
# Como Admin
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Adicionar ao perfil
$ProfileDir = Split-Path $PROFILE
if (!(Test-Path $ProfileDir)) { mkdir $ProfileDir }
Add-Content $PROFILE 'Set-Alias -Name dvc -Value "python -m dvc"'

# Recarregar
. $PROFILE

# Agora funciona:
dvc dag
dvc repro
```

---

## ✅ Tudo Resolvido!

Use **`python -m dvc [comando]`** para todos os comandos DVC no PowerShell.
