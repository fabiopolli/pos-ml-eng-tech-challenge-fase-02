"""
Script Principal para Previsão de Notas de Avaliações (Review Score) do E-Commerce Olist.

Orquestra o carregamento de dados, pré-processamento, treinamento de modelos e avaliação.
"""

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

from src.preprocessing_pipeline import load_and_preprocess_data
from src.baseline_models import (
    train_and_evaluate_all_models,
    compare_models,
    plot_feature_importance,
    plot_confusion_matrix
)


def main() -> None:
    """Função principal para executar todo o pipeline de ponta a ponta."""
    print("=" * 60)
    print("PREVISÃO DE NOTAS DE AVALIAÇÃO (REVIEW SCORE) - OLIST E-COMMERCE")
    print("=" * 60)

    # Passo 1: Carregar e pré-processar os dados
    print("\n1. Carregando e pré-processando os dados...")
    dataset_path = "data"

    try:
        X_train, X_test, y_train, y_test, feature_names, preprocessor = load_and_preprocess_data(dataset_path)
        print("   [INFO] Dados carregados e tratados com sucesso")
        print(f"   [INFO] Shape do treino: {X_train.shape}")
        print(f"   [INFO] Shape do teste: {X_test.shape}")
        print(f"   [INFO] Total de colunas/features: {len(feature_names)}")
        print(f"   [INFO] Distribuição do alvo:\n{y_train.value_counts().sort_index()}")
    except Exception as e:
        print(f"   [ERRO] Falha ao carregar dados: {e}")
        return

    # Passo 2: Treinar e avaliar modelos base (baselines)
    print("\n2. Treinando e avaliando os modelos de baseline...")
    try:
        models_results = train_and_evaluate_all_models(X_train, X_test, y_train, y_test)
        print(f"   [INFO] Sucesso! {len(models_results)} modelos treinados")
    except Exception as e:
        print(f"   [ERRO] Falha ao treinar os modelos: {e}")
        return

    # Passo 3: Comparar métricas
    print("\n3. Comparando o desempenho dos modelos...")
    try:
        comparison_df = compare_models(models_results)
        print("   Tabela Comparativa (ordenada por F1-Score Ponderado):")
        print(comparison_df.to_string(index=False, float_format='%.4f'))

        comparison_df.to_csv('model_comparison.csv', index=False)
        print("   [INFO] Comparativo salvo como 'model_comparison.csv'")
    except Exception as e:
        print(f"   [ERRO] Falha ao comparar modelos: {e}")
        return

    # Passo 4: Gerar artefatos visuais para o melhor classificador
    print("\n4. Gerando visualizações do melhor modelo...")
    try:
        best_model_name = comparison_df.iloc[0]['Model']
        best_model_result = models_results[best_model_name]

        print(f"   [INFO] Melhor classificador encontrado: {best_model_name}")

        cm = best_model_result['metrics']['confusion_matrix']
        fig_cm = plot_confusion_matrix(cm, model_name=best_model_name)
        if fig_cm:
            nome_arquivo_cm = f'confusion_matrix_{best_model_name.replace(" ", "_").lower()}.png'
            fig_cm.savefig(nome_arquivo_cm, dpi=300, bbox_inches='tight')
            plt.close(fig_cm)
            print(f"   [INFO] Matriz de Confusão salva em '{nome_arquivo_cm}'")

        fig_fi = plot_feature_importance(
            best_model_result['model'],
            feature_names,
            model_name=best_model_name,
            top_n=10
        )
        if fig_fi:
            nome_arquivo_fi = f'feature_importance_{best_model_name.replace(" ", "_").lower()}.png'
            fig_fi.savefig(nome_arquivo_fi, dpi=300, bbox_inches='tight')
            plt.close(fig_fi)
            print(f"   [INFO] Gráfico de importância das variáveis salvo em '{nome_arquivo_fi}'")

    except Exception as e:
        print(f"   [ERRO] Falha ao gerar gráficos: {e}")

    # Passo 5: Exportar o melhor modelo treinado
    print("\n5. Salvando o melhor modelo...")
    try:
        import joblib
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = models_results[best_model_name]['model']
        nome_pickle = f'best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
        joblib.dump(best_model, nome_pickle)
        print(f"   [INFO] Melhor modelo persistido em '{nome_pickle}'")
    except Exception as e:
        print(f"   [ERRO] Falha ao salvar o modelo: {e}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTADO COM SUCESSO!")
    print("=" * 60)
    print("\nArquivos gerados:")
    print("- model_comparison.csv: Comparação tabulada de performance")
    print("- confusion_matrix_[nome_modelo].png: Matriz de Confusão em heatmap")
    print("- feature_importance_[nome_modelo].png: Variáveis mais importantes")
    print("- best_model_[nome_modelo].pkl: Objeto serializado do melhor modelo")


if __name__ == "__main__":
    main()
