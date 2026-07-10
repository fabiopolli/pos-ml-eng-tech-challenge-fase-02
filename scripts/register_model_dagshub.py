"""Registra o modelo NCF Production (ncf_production.pt) no MLflow Tracking do DagsHub.

Este script implementa a Rota B do plano de versionamento:
carrega o binário já versionado via DVC (em models/) e o registra
como Model Version no MLflow do DagsHub, promovendo-o para o
stage Production. O scaler associado é anexado como artifact.

Pré-requisitos:
    - Token DagsHub configurado em .dvc/config.local (mesmo usado pelo DVC)
    - Arquivo models/ncf_production.pt presente (versão DVC já sincronizada)

Uso:
    uv run python scripts/register_model_dagshub.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ncf import NCFHybrid  # noqa: E402

# Configuração DagsHub (lida de .dvc/config.local para evitar hardcode)
DAGSHUB_REPO = "deniscelclaro/projeto_fiap_modulo2"
DAGSHUB_MLFLOW_URI = f"https://dagshub.com/{DAGSHUB_REPO}.mlflow"

REGISTERED_MODEL_NAME = "olist-ncf-recommender"
EXPERIMENT_NAME = "Olist_NCF_Production_Transfer"
RUN_NAME = "register_ncf_production_to_dagshub_2026-07-10"


def load_dvc_credentials() -> tuple[str, str]:
    """Lê usuário e token DagsHub de .dvc/config.local.

    Returns:
        Tupla (username, token).

    Raises:
        FileNotFoundError: Se .dvc/config.local não existir.
        ValueError: Se user/password não estiverem configurados.
    """
    config_local = PROJECT_ROOT / ".dvc" / "config.local"
    if not config_local.exists():
        raise FileNotFoundError(
            f"Arquivo {config_local} não encontrado. "
            "Configure o token DagsHub antes de executar este script."
        )

    creds: dict[str, str] = {}
    in_origin_block = False
    with open(config_local) as f:
        for line in f:
            line = line.strip()
            if line == '[\'remote "origin"\']':
                in_origin_block = True
                continue
            if in_origin_block and line.startswith("["):
                break
            if in_origin_block and "=" in line:
                key, value = line.split("=", 1)
                creds[key.strip()] = value.strip()

    user = creds.get("user")
    password = creds.get("password")
    if not user or not password:
        raise ValueError(
            "user/password não encontrados em .dvc/config.local. "
            "Execute: uv run dvc remote modify --local origin auth basic"
        )
    return user, password


def load_production_model() -> tuple[NCFHybrid, dict[str, Any]]:
    """Carrega o modelo Production a partir de models/.

    Returns:
        Tupla (modelo_ncf, metadata_dict).
    """
    pt_path = PROJECT_ROOT / "models" / "ncf_production.pt"
    cfg_path = PROJECT_ROOT / "configs" / "ncf_best.yaml"
    mappings_path = PROJECT_ROOT / "data" / "processed" / "id_mappings.json"

    if not pt_path.exists():
        raise FileNotFoundError(f"Modelo Production não encontrado: {pt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config não encontrado: {cfg_path}")
    if not mappings_path.exists():
        raise FileNotFoundError(f"id_mappings não encontrado: {mappings_path}")

    with open(mappings_path) as f:
        mappings = json.load(f)

    n_users = len(mappings["user_id_mapping"])
    n_items = len(mappings["product_id_mapping"])
    n_categories = len(mappings["category_id_mapping"])

    # Hiperparâmetros canônicos do modelo Production (configs/ncf_best.yaml)
    emb_dim = 32
    cat_emb_dim = 8
    hidden = [64, 32]
    dropout = 0.5
    # IMPORTANTE: o checkpoint foi treinado com 20 features numéricas (ablation
    # no-aux = zerar os valores durante o forward, mas o shape permanece 20
    # porque a MLP aprende a ignorar via pesos próximos de zero).
    # input_dim esperado = emb_dim*2 + cat_emb_dim + n_aux_features = 64 + 8 + 20 = 92
    n_aux_features = 20

    model = NCFHybrid(
        n_users=n_users,
        n_items=n_items,
        n_categories=n_categories,
        n_aux_features=n_aux_features,
        emb_dim=emb_dim,
        cat_emb_dim=cat_emb_dim,
        hidden=hidden,
        dropout=dropout,
    )
    state_dict = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "n_users": n_users,
        "n_items": n_items,
        "n_categories": n_categories,
        "emb_dim": emb_dim,
        "cat_emb_dim": cat_emb_dim,
        "hidden": hidden,
        "dropout": dropout,
        "ablation": "no_aux",
        "model_class": "NCFHybrid",
        "loss": "BPR",
    }
    return model, metadata


def main() -> None:
    """Executa o registro completo do modelo no DagsHub MLflow."""
    import mlflow
    import mlflow.pytorch

    # 1. Configurar credenciais e tracking URI
    user, token = load_dvc_credentials()
    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    os.environ["MLFLOW_TRACKING_URI"] = DAGSHUB_MLFLOW_URI
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    print(f"[1/5] Tracking URI: {DAGSHUB_MLFLOW_URI}")
    print(f"      Usuário: {user}")

    # 2. Carregar modelo
    print("[2/5] Carregando modelo Production...")
    model, model_meta = load_production_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Arquitetura: NCFHybrid(emb={model_meta['emb_dim']}, "
          f"hidden={model_meta['hidden']}, dropout={model_meta['dropout']})")
    print(f"      Parâmetros totais: {total_params:,}")

    # 3. Criar/buscar experimento
    print(f"[3/5] Configurando experimento '{EXPERIMENT_NAME}'...")
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"      Experiment ID: {experiment.experiment_id}")

    # 4. Iniciar run e logar modelo + scaler + métricas
    print("[4/5] Registrando modelo no Model Registry...")
    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"      Run ID: {run_id}")

        # Loga hiperparâmetros
        mlflow.log_params(
            {
                "n_users": model_meta["n_users"],
                "n_items": model_meta["n_items"],
                "n_categories": model_meta["n_categories"],
                "emb_dim": model_meta["emb_dim"],
                "cat_emb_dim": model_meta["cat_emb_dim"],
                "hidden_layers": str(model_meta["hidden"]),
                "dropout": model_meta["dropout"],
                "ablation": model_meta["ablation"],
                "loss": model_meta["loss"],
                "model_class": model_meta["model_class"],
                "total_parameters": total_params,
            }
        )

        # Loga métricas canônicas do test set (de configs/ncf_best.yaml)
        mlflow.log_metrics(
            {
                "test_NDCG_at_10": 0.2725,
                "test_HitRate_at_10": 0.4949,
                "test_Recall_at_10": 0.4868,
                "test_Precision_at_10": 0.0509,
                "test_MAP_at_10": 0.2081,
                "baseline_popularity_NDCG_at_10": 0.0045,
                "lift_vs_baseline": 60.6,
            }
        )

        # Anexa o scaler como artifact
        scaler_path = PROJECT_ROOT / "models" / "scaler_production.pkl"
        if scaler_path.exists():
            mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")
            print(f"      Artifact anexado: {scaler_path.name}")

        # Anexa o config YAML
        config_yaml = PROJECT_ROOT / "configs" / "ncf_best.yaml"
        if config_yaml.exists():
            mlflow.log_artifact(str(config_yaml), artifact_path="config")
            print(f"      Artifact anexado: {config_yaml.name}")

        # Loga o modelo PyTorch e registra no Model Registry.
        # Usa serialization_format='pickle' (serializa o objeto nn.Module
        # via pickle) para evitar o requisito de input_example do
        # formato 'pt2' (traced graph, exigiria tensor de exemplo).
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )
        print(f"      Model Version registrada: {model_info.model_uri}")

        # 5. Promover para Production
        print("[5/5] Promovendo para stage 'Production'...")
        client = mlflow.tracking.MlflowClient()
        # Pega a versão mais recente do modelo registrado
        latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
        if not latest_versions:
            raise RuntimeError(f"Nenhuma versão encontrada para {REGISTERED_MODEL_NAME}")
        new_version = latest_versions[0].version
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=new_version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"      Versão {new_version} promovida para Production")

    print("\n✓ Modelo registrado com sucesso no DagsHub MLflow!")
    print(f"  URI do tracking: {DAGSHUB_MLFLOW_URI}")
    print(f"  Model Registry: {REGISTERED_MODEL_NAME}")
    print(f"  Run ID: {run_id}")
    print(f"  UI: {DAGSHUB_MLFLOW_URI.replace('.mlflow', '')}/#/models/{REGISTERED_MODEL_NAME}")


if __name__ == "__main__":
    main()
