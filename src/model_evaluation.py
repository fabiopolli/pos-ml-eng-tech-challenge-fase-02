"""
Estágio 3: Avaliação e Geração de Artefatos
Gera visualizações e artefatos finais do melhor modelo.
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from baseline_models import plot_confusion_matrix, plot_feature_importance, train_and_evaluate_all_models


def evaluate_models(data_dir: str = "data/processed", models_dir: str = "models", output_dir: str = "artifacts") -> None:
    """Executa o estágio de avaliação e geração de artefatos.
    
    Args:
        data_dir: Diretório contendo dados pré-processados.
        models_dir: Diretório contendo modelos treinados.
        output_dir: Diretório para salvar artefatos de avaliação.
    """
    print("\n" + "="*60)
    print("ESTÁGIO 3: AVALIAÇÃO E GERAÇÃO DE ARTEFATOS")
    print("="*60)
    
    # Criar diretório de saída
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Carregar dados
    print(f"\n[INFO] Carregando dados processados...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").iloc[:, 0]
    
    # Carregar metadados
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    
    # Identificar melhor modelo
    print(f"[INFO] Identificando melhor modelo...")
    comparison_df = pd.read_csv(f"{models_dir}/model_comparison.csv")
    best_model_name = comparison_df.iloc[0]['Model']
    
    with open(f"{models_dir}/best_model.txt", 'r') as f:
        best_model_name = f.read().strip()
    
    print(f"[INFO] Melhor modelo: {best_model_name}")
    
    # Carregar melhor modelo
    model_path = f"{models_dir}/{best_model_name.replace(' ', '_').lower()}_model.pkl"
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
    
    # Gerar matriz de confusão
    print(f"\n[INFO] Gerando matriz de confusão...")
    cm_result = best_model.predict(X_test)
    from sklearn.metrics import confusion_matrix as cm_func
    cm = cm_func(y_test, cm_result)
    
    fig_cm = plot_confusion_matrix(cm, model_name=best_model_name)
    if fig_cm:
        cm_filename = f"{output_dir}/confusion_matrix_{best_model_name.replace(' ', '_').lower()}.png"
        fig_cm.savefig(cm_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_cm)
        print(f"[SUCCESS] Matriz de confusão salva em '{cm_filename}'")
    
    # Gerar gráfico de importância de features
    print(f"[INFO] Gerando gráfico de importância de features...")
    fig_fi = plot_feature_importance(
        best_model,
        feature_names,
        model_name=best_model_name,
        top_n=10
    )
    if fig_fi:
        fi_filename = f"{output_dir}/feature_importance_{best_model_name.replace(' ', '_').lower()}.png"
        fig_fi.savefig(fi_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_fi)
        print(f"[SUCCESS] Gráfico de features salvo em '{fi_filename}'")
    
    # Salvar relatório de avaliação
    print(f"\n[INFO] Gerando relatório de avaliação...")
    report = {
        "best_model": best_model_name,
        "training_data": {
            "n_samples": metadata['n_samples_train'],
            "n_features": metadata['n_features']
        },
        "test_data": {
            "n_samples": metadata['n_samples_test']
        },
        "model_comparison": comparison_df.to_dict('records')
    }
    
    report_filename = f"{output_dir}/evaluation_report.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[SUCCESS] Relatório de avaliação salvo em '{report_filename}'")
    
    # Copiar arquivo de comparação
    import shutil
    shutil.copy(f"{models_dir}/model_comparison.csv", f"{output_dir}/model_comparison.csv")
    
    print(f"\n[SUCCESS] Estágio de avaliação concluído!")
    print(f"[INFO] Artefatos disponíveis em '{output_dir}/'")
    

if __name__ == "__main__":
    evaluate_models()
