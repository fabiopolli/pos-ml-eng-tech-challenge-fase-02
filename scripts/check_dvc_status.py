#!/usr/bin/env python3
"""
Quick Status Checker para o Pipeline DVC
Verifica o status da execução do pipeline
"""

from pathlib import Path


def check_dvc_status():
    """Verifica o status da execução do pipeline DVC"""

    project_root = Path(".")

    print("\n" + "=" * 70)
    print("DVC PIPELINE STATUS CHECK")
    print("=" * 70)

    # 1. Verificar estrutura
    print("\nESTRUTURA DO PIPELINE:")
    print("   - dvc.yaml (Definição do pipeline)")
    print("   - dvc.lock (Estado reproduzível)")
    print("   - .dvc/config (Configuração remota)")

    if (project_root / "dvc.yaml").exists():
        print("   [OK] dvc.yaml encontrado")
    if (project_root / "dvc.lock").exists():
        print("   [OK] dvc.lock encontrado")
    if (project_root / ".dvc" / "config").exists():
        print("   [OK] .dvc/config encontrado")

    # 2. Verificar estágios implementados
    print("\nESTAGIOS IMPLEMENTADOS (DVC Pipeline):")
    print("   1. prepare   (src/data_preparation.py)")
    print("      - Entradas: 8 arquivos CSV brutos")
    print("      - Saidas: data/processed/interactions.parquet")
    if (project_root / "data" / "processed" / "interactions.parquet").exists():
        size_kb = (project_root / "data" / "processed" / "interactions.parquet").stat().st_size / 1024
        print(f"      [OK] interactions.parquet existe ({size_kb:.1f} KB)")
    else:
        print("      [PENDENTE] Execute: dvc repro --single-stage prepare")

    print("\n   2. featurize (src/feature_engineering.py)")
    print("      - Entradas: data/processed/interactions.parquet")
    print("      - Saidas: data/processed/interactions_fe.parquet + id_mappings.json")
    if (project_root / "data" / "processed" / "interactions_fe.parquet").exists():
        size_kb = (project_root / "data" / "processed" / "interactions_fe.parquet").stat().st_size / 1024
        print(f"      [OK] interactions_fe.parquet existe ({size_kb:.1f} KB)")
    else:
        print("      [PENDENTE] Execute: dvc repro --single-stage featurize")

    print("\n   3. validate  (validacao de shape)")
    print("      - Entradas: data/processed/interactions_fe.parquet")
    print("      - Validacao: shape == (99785, 42)")
    print("      [OK] Stage inline no dvc.yaml")

    # 3. Verificar dados processados
    print("\nDADOS PROCESSADOS:")
    processed_dir = project_root / "data" / "processed"
    if processed_dir.exists():
        files = list(processed_dir.glob("*"))
        for file in sorted(files):
            size_kb = file.stat().st_size / 1024
            print(f"   {file.name:35} ({size_kb:.1f} KB)")

    # 4. Treinamento (train.py standalone)
    print("\nTREINAMENTO (scripts independentes do DVC):")
    print("   - src/train.py: Baselines de recomendacao (Popularity, Top-Rated, Item-CF)")
    if (project_root / "src" / "train.py").exists():
        print("   [OK] train.py existe")
        print("   [PENDENTE] Execute: uv run python src/train.py")
    else:
        print("   [ERRO] train.py nao encontrado")

    # 5. Instruções para completar
    print("\nPROXIMOS PASSOS:")
    print("   1. Executar pipeline DVC:")
    print("      -> dvc repro")
    print("   2. Executar treinamento de modelos:")
    print("      -> uv run python src/train.py")
    print("   3. Ver status DVC:")
    print("      -> dvc status")
    print("      -> dvc dag")
    print("   4. MLflow tracking (inicie em outro terminal):")
    print("      -> mlflow ui --host 127.0.0.1 --port 5000")

    print("\n" + "=" * 70)
    print("Pipeline DVC: prepare -> featurize -> validate")
    print("Pipeline de Modelos: src/train.py (standalone)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    check_dvc_status()
