"""Teste simples para ver o que está acontecendo."""

print("Starting simple test...")

try:
    import pandas as pd
    print("✓ Pandas imported")
except Exception as e:
    print(f"✗ Pandas import failed: {e}")
    exit(1)

# Teste 2: Verifique se o arquivo existe
import os
file_path = "dataset/olist_orders_dataset.csv"
if os.path.exists(file_path):
    print(f"✓ File exists: {file_path}")
else:
    print(f"✗ File does not exist: {file_path}")
    exit(1)

# Teste 3: Tenta ler o arquivo
try:
    df = pd.read_csv(file_path)
    print(f"✓ File read successfully: {df.shape}")
except Exception as e:
    print(f"✗ File read failed: {e}")
    exit(1)

print("Simple test completed successfully!")