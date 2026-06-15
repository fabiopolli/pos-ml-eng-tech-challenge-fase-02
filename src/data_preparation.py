"""
Estágio 1: Preparação de Dados
Carrega, limpa e pré-processa os dados do Olist.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_pipeline import load_and_preprocess_data


def prepare_data(dataset_path: str = "data", output_dir: str = "data/processed") -> None:
    """Executa o estágio de preparação de dados.
    
    Args:
        dataset_path: Caminho para os dados brutos.
        output_dir: Diretório para salvar dados processados.
    """
    print("\n" + "="*60)
    print("ESTÁGIO 1: PREPARAÇÃO DE DADOS")
    print("="*60)
    
    # Criar diretório de saída
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Carregando e pré-processando dados de '{dataset_path}'...")
    X_train, X_test, y_train, y_test, feature_names, preprocessor = load_and_preprocess_data(
        dataset_path=dataset_path
    )
    
    print(f"[INFO] Dados carregados e pré-processados com sucesso")
    print(f"[INFO] Shape treino: {X_train.shape}")
    print(f"[INFO] Shape teste: {X_test.shape}")
    print(f"[INFO] Features: {len(feature_names)}")
    
    # Salvar dados processados
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False, header=True)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False, header=True)
    
    # Salvar metadata
    metadata = {
        "n_samples_train": int(X_train.shape[0]),
        "n_samples_test": int(X_test.shape[0]),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "target_distribution_train": y_train.value_counts().to_dict()
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[INFO] Dados salvos em '{output_dir}/'")
    print("[SUCCESS] Estágio de preparação concluído!")
    

if __name__ == "__main__":
    prepare_data()
