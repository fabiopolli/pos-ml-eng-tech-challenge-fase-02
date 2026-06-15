# 🚀 Como Completar a Execução do Pipeline DVC

## Status Atual

✅ **Estágio 1 (Preparação):** COMPLETO
- 5 arquivos criados em `data/processed/`
- 94.516 amostras de treino com 32 features

⏳ **Estágio 2 (Treinamento):** PRONTO PARA EXECUTAR
⏳ **Estágio 3 (Avaliação):** PRONTO PARA EXECUTAR

---

## 🔧 Completar o Pipeline em 3 Passos

### **Opção 1: Executar Tudo Automaticamente (Recomendado)**

```bash
cd "c:\Users\denis\Desktop\Projeto modulo 2 FIAP\pos-ml-eng-tech-challenge-fase-02"
python -m dvc repro
```

**O que acontece:**
1. ✅ Estágio 1 (prepare): Pulado (já foi executado)
2. ⏳ Estágio 2 (train): Executado (~5-10 minutos)
3. ⏳ Estágio 3 (evaluate): Executado (~2-3 minutos)

**Tempo total esperado:** 7-13 minutos

---

### **Opção 2: Executar Estágio por Estágio**

```bash
# Estágio 2: Treinar modelos
python -m dvc repro --single-stage train

# Estágio 3: Avaliar e gerar artefatos
python -m dvc repro --single-stage evaluate
```

---

### **Opção 3: Forçar Re-execução de Tudo**

Se você quiser re-executar todos os estágios (incluindo o prepare):

```bash
python -m dvc repro --force
```

---

## 📊 Verificar Progresso

### Antes de Executar
```bash
python check_dvc_status.py
```

### Durante a Execução
O DVC mostrará mensagens como:
```
Running stage 'train':
> python src/model_training.py

============================================================
ESTÁGIO 2: TREINAMENTO DE MODELOS
============================================================

[INFO] Carregando dados processados de 'data/processed'...
[INFO] Iniciando treinamento de modelos...
Treinando Logistic Regression...
...
```

### Depois de Executar
```bash
python check_dvc_status.py
```

Você verá:
```
✅ MODELOS TREINADOS:
   ✓ logistic_regression_model.pkl
   ✓ decision_tree_model.pkl
   ✓ random_forest_model.pkl
   ✓ gradient_boosting_model.pkl

✅ ARTEFATOS FINAIS:
   ✓ confusion_matrix_gradient_boosting.png
   ✓ feature_importance_gradient_boosting.png
   ✓ model_comparison.csv
   ✓ evaluation_report.json
```

---

## 🔄 Se Algo Der Errado

### Erro: ".dvc is busy"
```bash
# Remover lock antigo
rm .dvc/tmp/rwlock

# Tentar novamente
python -m dvc repro
```

### Erro: "Remote not configured"
```bash
python -m dvc remote list
# Deve mostrar: local_storage  C:\dvc_storage\pos-ml-eng-tech
```

### Erro: "Permission denied"
Certifique-se que:
1. O diretório `C:\dvc_storage\pos-ml-eng-tech` existe
2. Você tem permissão de escrita
3. Nenhum arquivo está aberto em outro programa

### Ver Logs Detalhados
```bash
python -m dvc repro -v  # Verbose mode
```

---

## 📈 Exemplo de Saída Esperada

```
============================================================
ESTÁGIO 2: TREINAMENTO DE MODELOS
============================================================

[INFO] Carregando dados processados de 'data/processed'...
[INFO] Dados carregados. Shape treino: (94516, 32), Shape teste: (23630, 32)

[INFO] Iniciando treinamento de modelos...
Treinando Logistic Regression...
  Acurácia: 0.5885
  F1-Score (Weighted): 0.4654
  Acurácia CV: 0.5896 (+/- 0.0012)

Treinando Decision Tree...
  Acurácia: 0.6136
  F1-Score (Weighted): 0.5075
  Acurácia CV: 0.6109 (+/- 0.0056)

Treinando Random Forest...
  Acurácia: 0.6140
  F1-Score (Weighted): 0.4962
  Acurácia CV: 0.6147 (+/- 0.0021)

Treinando Gradient Boosting...
  Acurácia: 0.6309
  F1-Score (Weighted): 0.5296
  Acurácia CV: 0.6303 (+/- 0.0025)

[INFO] Sucesso! 4 modelos treinados

============================================================
ESTÁGIO 3: AVALIAÇÃO E GERAÇÃO DE ARTEFATOS
============================================================

[INFO] Gerando matriz de confusão...
[SUCCESS] Matriz de confusão salva em 'artifacts/confusion_matrix_gradient_boosting.png'

[INFO] Gerando gráfico de importância de features...
[SUCCESS] Gráfico de features salvo em 'artifacts/feature_importance_gradient_boosting.png'

[INFO] Gerando relatório de avaliação...
[SUCCESS] Relatório de avaliação salvo em 'artifacts/evaluation_report.json'

[SUCCESS] Estágio de avaliação concluído!
```

---

## ✅ Após a Execução

### Arquivos Criados
```
models/
├── logistic_regression_model.pkl
├── logistic_regression_metrics.json
├── decision_tree_model.pkl
├── decision_tree_metrics.json
├── random_forest_model.pkl
├── random_forest_metrics.json
├── gradient_boosting_model.pkl
├── gradient_boosting_metrics.json
├── model_comparison.csv
└── best_model.txt

artifacts/
├── confusion_matrix_gradient_boosting.png
├── feature_importance_gradient_boosting.png
├── model_comparison.csv
└── evaluation_report.json
```

### Verificar Resultado
```bash
python check_dvc_status.py
```

### Salvar no Git
```bash
git add dvc.lock
git commit -m "Complete DVC pipeline execution (train + evaluate stages)"
git push
```

### Sincronizar com Remoto DVC
```bash
dvc push
```

---

## 📊 Próximos Passos

1. **Verificar resultados:**
   - Abrir `artifacts/confusion_matrix_gradient_boosting.png`
   - Abrir `artifacts/feature_importance_gradient_boosting.png`
   - Ler `artifacts/evaluation_report.json`

2. **Integrar com versionamento:**
   ```bash
   git tag -a v1.0-pipeline-complete -m "Pipeline DVC completo"
   ```

3. **Monitorar performance:**
   ```bash
   dvc plots show
   ```

4. **Automatizar com CI/CD:**
   - Configure GitHub Actions para rodar `dvc repro` automaticamente
   - Envie alertas se performance cair

---

## 💡 Dicas Úteis

### Ver DAG do Pipeline
```bash
python -m dvc dag
```

### Ver Status de Cada Estágio
```bash
python -m dvc status
```

### Forçar Atualização do Cache
```bash
dvc cache prune
```

### Listar Dependências de um Estágio
```bash
python -m dvc dag --target train
```

---

**Tempo estimado para conclusão:** 7-13 minutos

**Recomendação:** Use a Opção 1 para executar tudo automaticamente
