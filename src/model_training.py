"""
Estágio 2: Treinamento de Modelos
Treina todos os modelos baseline e salva os artefatos.
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from baseline_models import train_and_evaluate_all_models, compare_models


def train_models(data_dir: str = "data/processed", output_dir: str = "models") -> None:
    """Executa o estágio de treinamento de modelos.
    
    Args:
        data_dir: Diretório contendo dados pré-processados.
        output_dir: Diretório para salvar modelos treinados.
    """
    print("\n" + "="*60)
    print("ESTÁGIO 2: TREINAMENTO DE MODELOS")
    print("="*60)
    
    # Criar diretório de saída
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Carregar dados processados
    print(f"\n[INFO] Carregando dados processados de '{data_dir}'...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").iloc[:, 0]
    
    print(f"[INFO] Dados carregados. Shape treino: {X_train.shape}, Shape teste: {X_test.shape}")
    
    # Treinar todos os modelos
    print("\n[INFO] Iniciando treinamento de modelos...")
    models_results = train_and_evaluate_all_models(X_train, X_test, y_train, y_test)
    
    print(f"[SUCCESS] {len(models_results)} modelos treinados com sucesso!")
    
    # Salvar artefatos de treinamento
    print(f"\n[INFO] Salvando artefatos em '{output_dir}'...")
    
    # Salvar modelos e métricas
    for model_name, result in models_results.items():
        model_filename = f"{output_dir}/{model_name.replace(' ', '_').lower()}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(result['model'], f)
        
        metrics_filename = f"{output_dir}/{model_name.replace(' ', '_').lower()}_metrics.json"
        with open(metrics_filename, 'w') as f:
            metrics = {
                'accuracy': float(result['metrics']['accuracy']),
                'precision_weighted': float(result['metrics']['precision_weighted']),
                'recall_weighted': float(result['metrics']['recall_weighted']),
                'f1_weighted': float(result['metrics']['f1_weighted']),
                'precision_macro': float(result['metrics']['precision_macro']),
                'recall_macro': float(result['metrics']['recall_macro']),
                'f1_macro': float(result['metrics']['f1_macro']),
                'cv_accuracy_mean': float(result['cv_results']['cv_mean']),
                'cv_accuracy_std': float(result['cv_results']['cv_std'])
            }
            json.dump(metrics, f, indent=2)
    
    # Comparar modelos e salvar resultado
    comparison_df = compare_models(models_results)
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    
    print("\n[INFO] Comparativa de modelos:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    print(f"\n[INFO] Comparativa salva em '{output_dir}/model_comparison.csv'")
    
    # Salvar nome do melhor modelo
    best_model_name = comparison_df.iloc[0]['Model']
    with open(f"{output_dir}/best_model.txt", 'w') as f:
        f.write(best_model_name)
    
    print(f"\n[SUCCESS] Melhor modelo identificado: {best_model_name}")
    print("[SUCCESS] Estágio de treinamento concluído!")
    

if __name__ == "__main__":
    train_models()
