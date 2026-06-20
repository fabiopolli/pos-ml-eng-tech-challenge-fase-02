"""Teste o pré-processamento passo a passo."""

import sys
import os
sys.path.insert(0, os.getcwd())

print("Testing preprocessing step by step...")

# Teste 1: Importe o módulo
try:
    import preprocessing_pipeline
    print("✓ Preprocessing pipeline module imported")
except Exception as e:
    print(f"✗ Failed to import preprocessing pipeline: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Teste 2: Criar OlistDataLoader
try:
    loader = preprocessing_pipeline.OlistDataLoader("dataset")
    print("✓ OlistDataLoader created")
except Exception as e:
    print(f"✗ Failed to create OlistDataLoader: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Teste 3: Carregue todos os dados
try:
    df = loader.load_all_data()
    print(f"✓ All data loaded: {df.shape}")
except Exception as e:
    print(f"✗ Failed to load all data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Teste 4: Verifique se review_score existe
try:
    if 'review_score' in df.columns:
        print("✓ review_score column found")
        print(f"  Unique values: {sorted(df['review_score'].unique())}")
    else:
        print("✗ review_score column NOT found")
        print(f"  Available columns: {list(df.columns)}")
except Exception as e:
    print(f"✗ Error checking review_score: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("All tests passed!")