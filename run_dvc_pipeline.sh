#!/bin/bash
# Script para executar o pipeline DVC completo
# Limpa locks antigos e executa todo o pipeline

echo "=========================================="
echo "DVC Pipeline Complete Execution"
echo "=========================================="

# 1. Limpar locks antigos (se houver)
echo ""
echo "[1/3] Limpando locks antigos..."
if [ -f ".dvc/tmp/rwlock" ]; then
    rm -f .dvc/tmp/rwlock
    echo "✓ Lock removido"
fi

# 2. Verificar status
echo ""
echo "[2/3] Verificando status do pipeline..."
python -m dvc status
echo ""

# 3. Executar pipeline
echo "[3/3] Executando pipeline DVC (prepare → train → evaluate)..."
python -m dvc repro

# 4. Resumo final
echo ""
echo "=========================================="
echo "Pipeline Execution Summary"
echo "=========================================="
python check_dvc_status.py

echo ""
echo "Pipeline execution completed!"
