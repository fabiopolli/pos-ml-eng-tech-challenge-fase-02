#!/usr/bin/env python3
"""
Quick Status Checker para o Pipeline DVC
Verifica o status da execução do pipeline
"""

import json
import os
from pathlib import Path

def check_dvc_status():
    """Verifica o status da execução do pipeline DVC"""
    
    project_root = Path(".")
    
    print("\n" + "="*70)
    print("📊 DVC PIPELINE STATUS CHECK")
    print("="*70)
    
    # 1. Verificar estrutura
    print("\n✅ ESTRUTURA DO PIPELINE:")
    print("   ├── dvc.yaml (Definição do pipeline)")
    print("   ├── dvc.lock (Estado reproduzível)")
    print("   └── .dvc/config (Configuração remota)")
    
    if (project_root / "dvc.yaml").exists():
        print("      ✓ dvc.yaml encontrado")
    if (project_root / "dvc.lock").exists():
        print("      ✓ dvc.lock encontrado")
    if (project_root / ".dvc" / "config").exists():
        print("      ✓ .dvc/config encontrado")
    
    # 2. Verificar estágios implementados
    print("\n✅ ESTÁGIOS IMPLEMENTADOS:")
    print("   1. prepare   (src/data_preparation.py)")
    print("      └─ Entradas: 8 arquivos CSV brutos")
    print("      └─ Saídas: data/processed/ (5 arquivos)")
    if (project_root / "data" / "processed").exists():
        files = list((project_root / "data" / "processed").glob("*"))
        print(f"      └─ ✓ EXECUTADO ({len(files)} arquivos criados)")
    
    print("\n   2. train     (src/model_training.py)")
    print("      └─ Entradas: data/processed/*")
    print("      └─ Saídas: models/ (modelos + métricas)")
    if (project_root / "models").exists() and list((project_root / "models").glob("*")):
        files = list((project_root / "models").glob("*"))
        print(f"      └─ ✓ EXECUTADO ({len(files)} arquivos criados)")
    else:
        print("      └─ ⏳ PENDENTE (execute: dvc repro --single-stage train)")
    
    print("\n   3. evaluate  (src/model_evaluation.py)")
    print("      └─ Entradas: data/processed/* + models/*")
    print("      └─ Saídas: artifacts/ (visualizações + relatório)")
    if (project_root / "artifacts").exists() and list((project_root / "artifacts").glob("*")):
        files = list((project_root / "artifacts").glob("*"))
        print(f"      └─ ✓ EXECUTADO ({len(files)} arquivos criados)")
    else:
        print("      └─ ⏳ PENDENTE (execute: dvc repro --single-stage evaluate)")
    
    # 3. Verificar dados processados
    print("\n✅ DADOS PROCESSADOS:")
    processed_dir = project_root / "data" / "processed"
    if processed_dir.exists():
        files = list(processed_dir.glob("*"))
        for file in sorted(files):
            size_kb = file.stat().st_size / 1024
            print(f"   ✓ {file.name:25} ({size_kb:.1f} KB)")
        
        # Carregar metadata
        metadata_file = processed_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                print(f"\n   Dataset Summary:")
                print(f"   ├─ Treino: {metadata.get('n_samples_train', 'N/A')} amostras × {metadata.get('n_features', 'N/A')} features")
                print(f"   ├─ Teste:  {metadata.get('n_samples_test', 'N/A')} amostras")
                print(f"   └─ Features: {metadata.get('n_features', 'N/A')}")
    
    # 4. Verificar modelos
    print("\n✅ MODELOS TREINADOS:")
    models_dir = project_root / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*_model.pkl"))
        if model_files:
            for model_file in sorted(model_files):
                print(f"   ✓ {model_file.name}")
        else:
            print("   ⏳ Nenhum modelo encontrado")
    else:
        print("   📁 Diretório 'models/' não existe ainda")
    
    # 5. Verificar artefatos
    print("\n✅ ARTEFATOS FINAIS:")
    artifacts_dir = project_root / "artifacts"
    if artifacts_dir.exists():
        files = list(artifacts_dir.glob("*"))
        for file in sorted(files):
            print(f"   ✓ {file.name}")
    else:
        print("   📁 Diretório 'artifacts/' não existe ainda")
    
    # 6. Configuração remota
    print("\n✅ CONFIGURAÇÃO REMOTA:")
    dvc_config = project_root / ".dvc" / "config"
    if dvc_config.exists():
        with open(dvc_config) as f:
            content = f.read()
            if "local_storage" in content:
                print("   ✓ Remote storage configurado: C:\\dvc_storage\\pos-ml-eng-tech")
                print("   ✓ Status: ATIVO (padrão)")
    
    # 7. Instruções para completar
    print("\n📋 PRÓXIMOS PASSOS:")
    print("   1. Continuar/Completar pipeline:")
    print("      → dvc repro")
    print("\n   2. Ou executar estágio específico:")
    print("      → dvc repro --single-stage train")
    print("      → dvc repro --single-stage evaluate")
    print("\n   3. Ver status:")
    print("      → dvc status")
    print("      → dvc dag")
    print("\n   4. Sincronizar com remoto:")
    print("      → dvc push")
    print("      → dvc pull")
    
    print("\n" + "="*70)
    print("Para mais informações, veja: DVC_PIPELINE.md")
    print("="*70 + "\n")

if __name__ == "__main__":
    check_dvc_status()
