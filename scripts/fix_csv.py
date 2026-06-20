#!/usr/bin/env python
"""Script para corrigir problemas de encoding nos arquivos CSV."""

import pandas as pd
import os

def fix_order_payments_csv():
    """Corrige o arquivo olist_order_payments_dataset.csv."""
    filepath = 'data/olist_order_payments_dataset.csv'
    
    print(f"Corrigindo {filepath}...")
    
    # Ler o arquivo
    df = pd.read_csv(filepath)
    
    # Mostrar colunas antes
    print(f"Colunas antes: {df.columns.tolist()}")
    
    # Remover caracteres estranhos e aspas
    cols_fixed = [col.replace('Reload Window', '').strip('"').strip() for col in df.columns]
    
    # Mostrar colunas depois
    print(f"Colunas depois: {cols_fixed}")
    
    # Atribuir novas colunas
    df.columns = cols_fixed
    
    # Salvar
    df.to_csv(filepath, index=False)
    print(f"✓ {filepath} corrigido com sucesso!")

if __name__ == "__main__":
    fix_order_payments_csv()
