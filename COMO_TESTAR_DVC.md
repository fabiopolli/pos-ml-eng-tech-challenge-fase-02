# 🧪 5 Formas de Testar o DVC

**⚠️ IMPORTANTE:** No PowerShell, use `python -m dvc` em vez de apenas `dvc`

## 1️⃣ **Teste Rápido: Verificar Configuração**

```bash
# Ver versão
python -m dvc --version

# Ver status do repositório
python -m dvc status

# Listar remoto
python -m dvc remote list

# Ver configuração
python -m dvc config -l
```

**Saída esperada:**
```
DVC 3.67.1
...
data/processed: up to date
data/prepared: up to date
models: not created

local_storage  C:\dvc_storage\pos-ml-eng-tech (default)
```

---

## 2️⃣ **Teste Visual: Visualizar Pipeline**

```bash
# Ver DAG do pipeline
python -m dvc dag

# Ver dependências de um estágio
python -m dvc dag --target train
```

**Saída esperada:**
```
        +---------+      
        | prepare |      
        +---------+      
         *         **    
       **            *   
      *               ** 
    +-------+           *
    | train |         ** 
    +-------+        *   
         *         **    
          **     **      
            *   *        
        +----------+     
        | evaluate |     
        +----------+     
```

---

## 3️⃣ **Teste Funcional: Verificar Scripts**

```bash
# Listar arquivos dos estágios
dir src\*_*.py  # Windows
# ou
ls src/*_*.py   # Linux/Mac

# Verificar sintaxe dos scripts
python -m py_compile src/data_preparation.py
python -m py_compile src/model_training.py
python -m py_compile src/model_evaluation.py
```

**Saída esperada:**
```
(Sem erros = Scripts estão OK)
```

---

## 4️⃣ **Teste Completo: Executar Pipeline**

### Opção A: Um Estágio por Vez
```bash
# Rodar estágio 2
python -m dvc repro --single-stage train

# Rodar estágio 3
python -m dvc repro --single-stage evaluate
```

### Opção B: Todo o Pipeline
```bash
# Rodar tudo de uma vez
python -m dvc repro
```

### Opção C: Forçar Re-execução
```bash
# Re-executar mesmo se não mudou nada
python -m dvc repro --force
```

**Tempo esperado:** 7-13 minutos

**Saída esperada:**
```
Running stage 'train':
> python src/model_training.py
[SUCCESS] ...

Running stage 'evaluate':
> python src/model_evaluation.py
[SUCCESS] ...

Updating lock file 'dvc.lock'
```

---

## 5️⃣ **Teste Avançado: Sincronização com Remoto**

```bash
# Enviar dados para repositório remoto
python -m dvc push

# Baixar dados do remoto
python -m dvc pull

# Verificar o que vai ser enviado
python -m dvc status --cloud
```

**Saída esperada:**
```
Everything is up to date
(ou lista de arquivos a sincronizar)
```

---

## 🎯 **Teste Customizado: Script Python**

```bash
# Executar verificador de status
python check_dvc_status.py
```

**Saída esperada:**
```
✅ ESTÁGIO 1: PREPARAÇÃO
   ✓ X_train.csv
   ✓ X_test.csv
   ✓ metadata.json

⏳ ESTÁGIO 2: TREINAMENTO
   (aguardando execução)

⏳ ESTÁGIO 3: AVALIAÇÃO
   (aguardando execução)
```

---

## 🚀 **Teste Passo-a-Passo Recomendado**

### Fase 1: Validação (2 min)
```bash
# 1. Verificar versão
python -m dvc --version

# 2. Ver configuração
python -m dvc remote list

# 3. Visualizar pipeline
python -m dvc dag

# 4. Verificar status
python -m dvc status
```

### Fase 2: Execução (7-13 min)
```bash
# 5. Completar pipeline
python -m dvc repro
```

### Fase 3: Verificação (2 min)
```bash
# 6. Verificar status final
python check_dvc_status.py

# 7. Sincronizar com remoto
python -m dvc push

# 8. Verificar dvc.lock
cat dvc.lock | head -20
```

---

## 🔍 **Sinais de Sucesso**

### ✅ DVC Funcionando Corretamente
- [ ] `dvc --version` retorna versão (ex: 3.67.1)
- [ ] `dvc status` mostra "up to date"
- [ ] `dvc dag` mostra os 3 estágios conectados
- [ ] `dvc remote list` mostra repositório configurado
- [ ] `dvc repro` executa sem erros

### ✅ Pipeline Executado com Sucesso
- [ ] `data/processed/` contém 5 arquivos
- [ ] `models/` contém 8 arquivos (4 modelos + 4 metrics)
- [ ] `artifacts/` contém 4 arquivos (visualizações)
- [ ] `dvc.lock` foi atualizado
- [ ] Nenhuma mensagem de erro no terminal

---

## 🛠️ **Troubleshooting**

### Erro: "DVC not found"
```bash
pip install dvc
```

### Erro: ".dvc is busy"
```bash
# Remover lock antigo
rm .dvc/tmp/rwlock

# Tentar novamente
dvc repro
```

### Erro: "Remote not configured"
```bash
dvc remote list
dvc remote add -d local_storage C:\dvc_storage\pos-ml-eng-tech
```

### Ver Logs Detalhados
```bash
python -m dvc repro -v  # Verbose mode
python -m dvc repro -vv # Extra verbose
```

---

## 📊 **Verificação Final**

Após rodar o pipeline, você deve ter:

```
✅ Estágio 1: data/processed/ (5 arquivos)
✅ Estágio 2: models/ (8 arquivos)
✅ Estágio 3: artifacts/ (4 arquivos)
✅ dvc.lock: Atualizado com hashes
✅ .dvc/cache: Cache automático
```

---

**Começar pelo Teste 1, depois rodar o Teste 4 (Opção B) para testar completo! 🚀**

