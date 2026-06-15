# ⚡ Comandos DVC no PowerShell - Guia Corrigido

## 🔑 Diferença Importante

### ❌ No PowerShell NÃO funciona:
```powershell
dvc dag
dvc status
dvc remote list
```

### ✅ No PowerShell USE:
```powershell
python -m dvc dag
python -m dvc status
python -m dvc remote list
```

---

## 📋 Todos os Comandos DVC para PowerShell

| Comando | Versão PowerShell |
|---------|------------------|
| Ver versão | `python -m dvc --version` |
| Ver status | `python -m dvc status` |
| Ver pipeline | `python -m dvc dag` |
| Listar remoto | `python -m dvc remote list` |
| Executar pipeline | `python -m dvc repro` |
| Estágio específico | `python -m dvc repro --single-stage train` |
| Com logs | `python -m dvc repro -v` |
| Enviar para remoto | `python -m dvc push` |
| Baixar do remoto | `python -m dvc pull` |
| Config | `python -m dvc config -l` |

---

## 🚀 Atalho Rápido: Adicionar ao PowerShell Profile

Para não precisar digitar `python -m` toda vez, você pode criar um alias:

### Opção 1: Temporário (apenas nesta sessão)
```powershell
Set-Alias -Name dvc -Value "python -m dvc"

# Agora funciona:
dvc dag
dvc status
```

### Opção 2: Permanente (sempre funcionará)

1. Abra PowerShell como Admin
2. Digite:
```powershell
# Criar perfil se não existir
if (!(Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force }

# Adicionar alias
Add-Content $PROFILE 'Set-Alias -Name dvc -Value "python -m dvc"'

# Recarregar perfil
. $PROFILE
```

3. Pronto! Agora use:
```powershell
dvc dag
dvc status
dvc repro
```

---

## 🎯 Teste Agora

Execute no PowerShell:

```powershell
python -m dvc dag
```

Deve mostrar:
```
        +---------+      
        | prepare |      
        +---------+      
         *         **    
       **            *   
      *               ** 
+-------+               *
| train |             ** 
+-------+            *   
         *         **    
          **     **      
            *   *        
        +----------+     
        | evaluate |     
        +----------+     
```

---

## 📚 Resumo dos Comandos (Prontos para Copiar/Colar)

```powershell
# 1. Verificar versão
python -m dvc --version

# 2. Ver status
python -m dvc status

# 3. Ver pipeline
python -m dvc dag

# 4. Listar remoto
python -m dvc remote list

# 5. Executar pipeline
python -m dvc repro

# 6. Verificar status customizado
python check_dvc_status.py
```

---

## 🔧 Por Que Acontece Isso?

- **PowerShell** não reconhece `dvc` como comando executável
- **DVC** é instalado como módulo Python, não como executável do sistema
- A solução é usar `python -m` para rodar módulos Python
- Bash/Linux/Mac reconhecem `dvc` porque tem diferentes configurações de PATH

---

## ✅ Agora Funciona!

Use **`python -m dvc`** para todos os comandos no PowerShell.

Ou configure o alias permanente para economizar digitação.
