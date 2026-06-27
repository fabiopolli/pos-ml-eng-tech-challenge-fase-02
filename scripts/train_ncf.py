"""Script principal de treinamento do NCF Híbrido com MLflow tracking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Imports do projeto
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ImplicitFeedbackDataset, build_user_items_map  # noqa: E402
from src.data.preprocessing import fit_scaler, save_scaler_stats, transform_features  # noqa: E402
from src.data.splits import temporal_split  # noqa: E402
from src.models.losses import bpr_loss  # noqa: E402
from src.models.ncf import NCFHybrid  # noqa: E402
from src.training.evaluate import calculate_metrics_at_k, evaluate_model  # noqa: E402


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: NCFHybrid,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Treina uma época com BPR loss.

    Suporta n_negatives >= 1. Para cada par (user, pos_item),
    calcula pos_score e n_neg_scores, depois aplica BPR pairwise.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        user = batch["user"].to(device)
        pos_item = batch["pos_item"].to(device)
        pos_cat = batch["pos_category"].to(device)
        neg_items = batch["neg_item"].to(device)  # (B, n_neg)
        aux = batch["aux_features"].to(device)

        batch_size = user.size(0)
        n_neg = neg_items.size(1)

        # Expandir user e aux para corresponder a n_neg
        user_exp = user.unsqueeze(1).expand(-1, n_neg).reshape(-1)
        aux_exp = aux.unsqueeze(1).expand(-1, n_neg, -1).reshape(batch_size * n_neg, -1)
        neg_flat = neg_items.reshape(-1)
        neg_cat_flat = torch.zeros_like(neg_flat)

        # Positivos: shape (B,)
        pos_scores = model(user, pos_item, pos_cat, aux)

        # Negativos: shape (B * n_neg,)
        neg_scores = model(user_exp, neg_flat, neg_cat_flat, aux_exp)

        # Reshape para comparar cada pos com seus n_neg
        neg_scores = neg_scores.view(batch_size, n_neg)

        # BPR pairwise: pos (B,1) vs neg (B, n_neg)
        loss = -torch.nn.functional.logsigmoid(
            pos_scores.unsqueeze(1) - neg_scores
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar NCF Híbrido")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "selected_features.yaml",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-negatives", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="sqlite:///./artifacts/mlflow.db",
        help="URI do MLflow tracking (SQLite)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Olist_NCF_Recommender",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Desabilita MLflow")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-scheduler", action="store_true", help="Usar ReduceLROnPlateau")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--ablation-no-aux",
        action="store_true",
        help="Ablation: desabilitar features auxiliares (só embeddings)",
    )
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # MLflow setup
    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)

    # Carregar dados
    print("\n[1/5] Carregando dados...")
    config = load_config(args.config)
    numeric_cols: list[str] = config["numeric_features"]

    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "interactions_fe.parquet")
    print(f"  Total: {len(df):,} interações")

    # Split temporal (usa days_since_reference como proxy temporal)
    print("\n[2/5] Split temporal...")
    train_df, val_df, test_df = temporal_split(df, time_col="days_since_reference", descending=False)

    # Scaler
    print("\n[3/5] Ajustando StandardScaler...")
    scaler = fit_scaler(train_df, numeric_cols)
    train_aux = transform_features(train_df, scaler, numeric_cols)
    val_aux = transform_features(val_df, scaler, numeric_cols)
    test_aux = transform_features(test_df, scaler, numeric_cols)

    # Ablation: zerar aux features se solicitado
    if args.ablation_no_aux:
        print("  [ABLAÇÃO] Features auxiliares zeradas")
        train_aux = np.zeros_like(train_aux)
        val_aux = np.zeros_like(val_aux)
        test_aux = np.zeros_like(test_aux)
        # Manter n_aux_features para o modelo (será ignorado pois zeramos os valores)
        n_aux_features_eff = len(numeric_cols)
    else:
        n_aux_features_eff = len(numeric_cols)

    # Salvar scaler
    scaler_path = PROJECT_ROOT / "artifacts" / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    save_scaler_stats(scaler, numeric_cols, PROJECT_ROOT / "artifacts" / "feature_stats.json")
    print(f"  Scaler salvo em {scaler_path}")

    # Mapas
    all_item_ids = df["product_id_idx"].unique()
    user_items_map = build_user_items_map(train_df)
    # Adicionar val ao mapa para evitar recomendar itens já vistos
    for uid, items in build_user_items_map(val_df).items():
        user_items_map.setdefault(uid, set()).update(items)

    # Datasets
    train_dataset = ImplicitFeedbackDataset(
        interactions_df=train_df,
        all_item_ids=all_item_ids,
        user_items_map=user_items_map,
        numeric_features=train_aux,
        category_ids=train_df["category_id"].values,
        user_ids=train_df["user_id"].values,
        item_ids=train_df["product_id_idx"].values,
        n_negatives=args.n_negatives,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Modelo
    print("\n[4/5] Inicializando modelo...")
    model = NCFHybrid(
        n_users=config["n_users"],
        n_items=config["n_items"],
        n_categories=config["n_categories"],
        n_aux_features=n_aux_features_eff,
        emb_dim=args.emb_dim,
        cat_emb_dim=8,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros: {n_params:,}")

    # Treinamento
    print("\n[5/5] Treinando...")
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"NCF_emb{args.emb_dim}_h{'-'.join(map(str, args.hidden))}_d{args.dropout}"
    best_ndcg = -1.0
    patience_counter = 0

    mlflow_ctx = (
        __import__("mlflow").start_run(run_name=run_name)
        if not args.no_mlflow
        else _NullContext()
    )
    with mlflow_ctx as run:
        if not args.no_mlflow:
            import mlflow
            mlflow.log_params({
                "emb_dim": args.emb_dim,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "n_negatives": args.n_negatives,
                "epochs": args.epochs,
                "n_params": n_params,
                "n_aux_features": len(numeric_cols),
            })

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            print(f"  Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f}")

            # Avaliação no val a cada 2 épocas (economizar tempo)
            if epoch % 2 == 0 or epoch == args.epochs:
                val_metrics = evaluate_model(
                    model, val_df, val_aux, user_items_map, all_item_ids,
                    k=10, n_neg_eval=99, device=device,
                )
                print(f"    val: " + " | ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                ))

                if not args.no_mlflow:
                    import mlflow
                    safe_metrics = {
                        f"val_{k.replace('@', '_at_')}": v
                        for k, v in val_metrics.items()
                    }
                    mlflow.log_metrics(safe_metrics, step=epoch)
                    mlflow.log_metric("train_loss", train_loss, step=epoch)

                # LR scheduling baseado em validação
                if scheduler is not None:
                    scheduler.step(val_metrics["NDCG@K"])

                # Early stopping monitorando NDCG@10
                if val_metrics["NDCG@K"] > best_ndcg:
                    best_ndcg = val_metrics["NDCG@K"]
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"    Early stopping na epoch {epoch}")
                        break

        # Restaurar melhor modelo
        if "best_state" in dir():
            model.load_state_dict(best_state)

        # Avaliação final no teste
        print("\n=== AVALIAÇÃO FINAL NO TESTE ===")
        test_metrics = evaluate_model(
            model, test_df, test_aux, user_items_map, all_item_ids,
            k=10, n_neg_eval=99, device=device,
        )
        for k, v in test_metrics.items():
            print(f"  test_{k} = {v:.4f}")

        # Baseline: Top-K popularidade (sanity check de limite inferior)
        print("\n=== BASELINE POPULARIDADE ===")
        popular_items = (
            train_df["product_id_idx"].value_counts().head(10).index.tolist()
        )
        pop_metrics = {}
        for k_metric in ["HitRate@K", "Recall@K", "Precision@K", "NDCG@K", "MAP@K"]:
            pop_metrics[k_metric] = 0.0
        n_eval = 0
        for uid in test_df["user_id"].unique():
            true_items = set(
                test_df[test_df["user_id"] == uid]["product_id_idx"].tolist()
            )
            if not true_items:
                continue
            metrics = calculate_metrics_at_k(
                np.array(popular_items), true_items, k=10
            )
            for k_metric, v in metrics.items():
                pop_metrics[k_metric] += v
            n_eval += 1
        if n_eval > 0:
            for k_metric in pop_metrics:
                pop_metrics[k_metric] /= n_eval
                print(f"  baseline_{k_metric} = {pop_metrics[k_metric]:.4f}")

        # Sanity check: avaliar também em uma amostra do treino
        # para validar que o modelo aprendeu a ranquear corretamente.
        print("\n=== SANITY CHECK (TREINO) ===")
        train_sample = train_df.sample(min(2000, len(train_df)), random_state=42)
        train_sample_aux = transform_features(train_sample, scaler, numeric_cols)
        train_metrics = evaluate_model(
            model, train_sample, train_sample_aux, {uid: set() for uid in train_sample["user_id"].unique()},
            all_item_ids, k=10, n_neg_eval=99, device=device, filter_cold_start=False,
        )
        for k, v in train_metrics.items():
            print(f"  train_{k} = {v:.4f}")

        if not args.no_mlflow:
            import mlflow
            safe_test = {f"test_{k.replace('@', '_at_')}": v for k, v in test_metrics.items()}
            safe_train = {f"train_{k.replace('@', '_at_')}": v for k, v in train_metrics.items()}
            safe_baseline = {f"baseline_{k.replace('@', '_at_')}": v for k, v in pop_metrics.items()}
            mlflow.log_metrics(safe_test)
            mlflow.log_metrics(safe_train)
            mlflow.log_metrics(safe_baseline)

            # MLflow precisa de input_example para o formato pt2 (traced graph)
            sample_users = torch.zeros(2, dtype=torch.long, device=device)
            sample_items = torch.zeros(2, dtype=torch.long, device=device)
            sample_cats = torch.zeros(2, dtype=torch.long, device=device)
            sample_aux = torch.zeros(2, len(numeric_cols), dtype=torch.float32, device=device)
            input_example = (sample_users, sample_items, sample_cats, sample_aux)
            mlflow.pytorch.log_model(
                model, "model",
                input_example=input_example,
                serialization_format="pickle",
            )

        # Salvar artefatos
        model_path = PROJECT_ROOT / "artifacts" / f"ncf_{run_name}.pt"
        torch.save(model.state_dict(), model_path)
        metrics_path = PROJECT_ROOT / "artifacts" / f"metrics_{run_name}.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "test": test_metrics,
                "train_sanity": train_metrics,
                "baseline": pop_metrics,
                "best_val_ndcg": best_ndcg,
                "run_name": run_name,
            }, f, indent=2)

        print(f"\n[OK] Modelo salvo em {model_path}")
        print(f"[OK] Métricas salvas em {metrics_path}")


class _NullContext:
    """Contexto nulo para substituir MLflow quando desabilitado."""
    def __enter__(self): return self
    def __exit__(self, *args): return False


if __name__ == "__main__":
    main()