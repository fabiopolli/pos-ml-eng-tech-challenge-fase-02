# ⚡ Testes DVC - Guia Rápido

## 🧪 4 Testes Principais (Copiar e Colar)

### **Teste 1: Versão**
```bash
dvc --version
# Saída esperada: 3.67.1
```

### **Teste 2: Status**
```bash
dvc status
# Saída esperada: Mostra quais estágios precisam rodar
```

### **Teste 3: Pipeline (DAG)**
```bash
dvc dag
# Saída esperada: prepare → train → evaluate
```

### **Teste 4: Remoto**
```bash
dvc remote list
# Saída esperada: local_storage  C:\dvc_storage\pos-ml-eng-tech (default)
```

### **Teste 5: Status Detalhado**
```bash
python check_dvc_status.py
# Saída esperada: Resumo dos 3 estágios
```

---

## 🚀 Teste Completo: Executar Pipeline

```bash
dvc repro
```

**Tempo:** 7-13 minutos  
**Resultado:** Modelos treinados + Visualizações

---

## ✅ Sinais de Sucesso

| Comando | ✅ Esperado |
|---------|-----------|
| `dvc --version` | 3.67.1 |
| `dvc status` | Mostra mudanças em train/evaluate |
| `dvc dag` | prepare → train → evaluate |
| `dvc remote list` | local_storage configurado |
| `python check_dvc_status.py` | Estágio 1 ✅, 2-3 ⏳ |

---

## 🔍 Testes Avançados

### Ver Logs Detalhados
```bash
dvc repro -v
dvc repro -vv
```

### Testar Um Estágio
```bash
dvc repro --single-stage train
dvc repro --single-stage evaluate
```

### Verificar Dependências
```bash
dvc dag --target train
dvc dag --target evaluate
```

### Sincronizar Remoto
```bash
dvc push
dvc pull
dvc status --cloud
```

---

## 🛠️ Troubleshooting

### Erro: ".dvc is busy"
```bash
rm .dvc/tmp/rwlock
dvc repro
```

### Erro: "Remote not configured"
```bash
dvc remote add -d local_storage C:\dvc_storage\pos-ml-eng-tech
```

---

## 📊 Verificação Pós-Pipeline

Após executar `dvc repro`, você deve ter:

```
✅ data/processed/       (5 arquivos)
✅ models/              (8 arquivos)
✅ artifacts/           (4 arquivos)
✅ dvc.lock             (atualizado)
```

---

**COMECE POR AQUI:** `dvc status` (2 seg) → `dvc dag` (1 seg) → `dvc repro` (7-13 min)
