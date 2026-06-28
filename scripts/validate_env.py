#!/usr/bin/env python3
"""Valida que o ambiente tem todas as dependências, configurações e arquivos necessários.

Usage:
    uv run python scripts/validate_env.py

Exit codes:
    0 - Ambiente válido
    1 - Falha em uma ou mais checagens
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Limites mínimos de versão
MIN_PYTHON = (3, 12)

# Dependências obrigatórias do projeto
REQUIRED_PACKAGES = [
    "torch",
    "sklearn",
    "mlflow",
    "pandas",
    "numpy",
    "scipy",
    "pydantic_settings",
    "loguru",
    "streamlit",
]

# Diretórios esperados
REQUIRED_DIRS = [
    "data",
    "data/processed",
    "artifacts",
    "configs",
    "scripts",
    "src",
]

# Arquivos esperados
REQUIRED_FILES = [
    "pyproject.toml",
    "dvc.yaml",
    "README.md",
    "src/data/splits.py",
    "src/train.py",
    "configs/selected_features.yaml",
]


def check_python_version() -> bool:
    """Verifica versão do Python >= 3.12."""
    actual = sys.version_info[:2]
    if actual < MIN_PYTHON:
        print(
            f"[ERRO] Python {'.'.join(map(str, MIN_PYTHON))}+ necessario, "
            f"{'.'.join(map(str, actual))} encontrado"
        )
        return False
    print(f"[OK]   Python {'.'.join(map(str, actual))}")
    return True


def check_dependencies() -> bool:
    """Verifica que pacotes obrigatórios estão importáveis."""
    missing: list[str] = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"[OK]   {pkg}")
        except ImportError:
            print(f"[ERRO] {pkg} nao instalado")
            missing.append(pkg)
    return len(missing) == 0


def check_directories() -> bool:
    """Verifica que diretórios essenciais existem."""
    cwd = Path.cwd()
    all_ok = True
    for d in REQUIRED_DIRS:
        p = cwd / d
        if p.exists() and p.is_dir():
            print(f"[OK]   {d}/")
        else:
            print(f"[ERRO] {d}/ nao existe")
            all_ok = False
    return all_ok


def check_files() -> bool:
    """Verifica que arquivos essenciais existem."""
    cwd = Path.cwd()
    all_ok = True
    for f in REQUIRED_FILES:
        p = cwd / f
        if p.exists() and p.is_file():
            print(f"[OK]   {f}")
        else:
            print(f"[ERRO] {f} nao existe")
            all_ok = False
    return all_ok


def check_dvc() -> bool:
    """Verifica que DVC está inicializado no projeto."""
    cwd = Path.cwd()
    dvc_dir = cwd / ".dvc"
    dvc_config = dvc_dir / "config"
    if dvc_dir.exists() and dvc_config.exists():
        print("[OK]   DVC inicializado (.dvc/config)")
        return True
    print("[ERRO] DVC nao inicializado (execute: uv run dvc init)")
    return False


def check_processed_data() -> bool:
    """Verifica que artefatos de dados processados existem."""
    cwd = Path.cwd()
    candidates = [
        "data/processed/interactions.parquet",
        "data/processed/interactions_fe.parquet",
    ]
    found = [c for c in candidates if (cwd / c).exists()]
    if found:
        for c in found:
            print(f"[OK]   {c}")
        return True
    print(
        "[AVISO] Nenhum artefato de dados encontrado em data/processed/. "
        "Execute: uv run python src/data_preparation.py"
    )
    return False


def main() -> int:
    """Executa todas as checagens e retorna exit code apropriado."""
    print("=" * 60)
    print("Validacao de Ambiente - Olist Recommender System")
    print("=" * 60)

    results = {
        "Python version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Directories": check_directories(),
        "Files": check_files(),
        "DVC": check_dvc(),
        "Processed data": check_processed_data(),
    }

    print("=" * 60)
    print("Resumo:")
    for name, ok in results.items():
        status = "[OK]   " if ok else "[ERRO] "
        print(f"  {status}{name}")

    if all(results.values()):
        print("\n[OK] Ambiente valido!")
        return 0

    failed = [name for name, ok in results.items() if not ok]
    print(f"\n[ERRO] Ambiente invalido - corrija: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
