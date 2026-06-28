"""Gera as figuras do notebook para uso no dashboard Streamlit."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.dataset import build_user_items_map  # noqa: E402
from src.data.preprocessing import fit_scaler, transform_features  # noqa: E402
from src.data.splits import temporal_split  # noqa: E402
from src.models.ncf import NCFHybrid  # noqa: E402
from src.training.evaluate import calculate_metrics_at_k, evaluate_model  # noqa: E402

plt.style.use('seaborn-v0_8-whitegrid')

CONFIGS = PROJECT_ROOT / 'configs'
ARTIFACTS = PROJECT_ROOT / 'artifacts'
FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIGS / 'selected_features.yaml') as f:
    config = yaml.safe_load(f)
numeric_cols = config['numeric_features']

# Carregar dados
df = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'interactions_fe.parquet')
train_df, val_df, test_df = temporal_split(df, time_col='days_since_reference', descending=False)
scaler = fit_scaler(train_df, numeric_cols)
test_aux = transform_features(test_df, scaler, numeric_cols)
val_aux = transform_features(val_df, scaler, numeric_cols)
train_aux = transform_features(train_df, scaler, numeric_cols)

user_items_map = build_user_items_map(train_df)
for uid, items in build_user_items_map(val_df).items():
    user_items_map.setdefault(uid, set()).update(items)

all_item_ids = df['product_id_idx'].unique()

# Carregar modelo
model = NCFHybrid(
    n_users=config['n_users'], n_items=config['n_items'],
    n_categories=config['n_categories'], n_aux_features=len(numeric_cols),
    emb_dim=16, cat_emb_dim=8, hidden=[64, 32], dropout=0.3,
)
model.load_state_dict(torch.load(ARTIFACTS / 'ncf_emb16.pt', weights_only=True))
model.eval()

print('=== Calculando métricas ===')
ncf_test = evaluate_model(model, test_df, test_aux, user_items_map, all_item_ids, k=10, n_neg_eval=99, device='cpu')

train_sample = train_df.sample(min(2000, len(train_df)), random_state=42)
train_sample_aux = transform_features(train_sample, scaler, numeric_cols)
ncf_train = evaluate_model(
    model, train_sample, train_sample_aux,
    {uid: set() for uid in train_sample['user_id'].unique()},
    all_item_ids, k=10, n_neg_eval=99, device='cpu', filter_cold_start=False,
)
ncf_val = evaluate_model(model, val_df, val_aux, user_items_map, all_item_ids, k=10, n_neg_eval=99, device='cpu')

# Baseline de popularidade
popular_items = train_df['product_id_idx'].value_counts().head(10).index.to_numpy()
baseline_test = dict.fromkeys(['HitRate@K', 'Recall@K', 'Precision@K', 'NDCG@K', 'MAP@K'], 0.0)
n_eval = 0
for uid in test_df['user_id'].unique():
    true_items = set(test_df[test_df['user_id'] == uid]['product_id_idx'].tolist())
    if not true_items:
        continue
    bm = calculate_metrics_at_k(popular_items, true_items, k=10)
    for m, v in bm.items():
        baseline_test[m] += v
    n_eval += 1
for m in baseline_test:
    baseline_test[m] /= n_eval

print('Métricas calculadas.')

# --- Figura 1: NCF vs Baseline ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
metrics_to_plot = ['HitRate@K', 'Recall@K', 'NDCG@K', 'MAP@K']
x = np.arange(len(metrics_to_plot))
width = 0.35

ax = axes[0]
ax.bar(x - width/2, [ncf_test[m] for m in metrics_to_plot], width, label='NCF Híbrido', color='#2E86AB')
ax.bar(x + width/2, [baseline_test[m] for m in metrics_to_plot], width, label='Baseline (Popularidade)', color='#A23B72')
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylabel('Score')
ax.set_title('Comparação de Métricas @K=10 (Test Set)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
lifts = [ncf_test[m] / max(baseline_test[m], 1e-6) for m in metrics_to_plot]
bars = ax.bar(metrics_to_plot, lifts, color='#F18F01')
ax.set_ylabel('Lift (x vezes)')
ax.set_title('NCF vs Baseline: Quantas vezes melhor?')
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline = 1.0')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar, lift in zip(bars, lifts, strict=False):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{lift:.1f}x', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ncf_vs_baseline.png', dpi=100, bbox_inches='tight')
plt.close()
print('[OK] ncf_vs_baseline.png')

# --- Figura 2: Train/Val/Test ---
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics_to_plot))
width = 0.25
ax.bar(x - width, [ncf_train[m] for m in metrics_to_plot], width, label='Train (sanity)', color='#06A77D')
ax.bar(x, [ncf_val[m] for m in metrics_to_plot], width, label='Validation', color='#2E86AB')
ax.bar(x + width, [ncf_test[m] for m in metrics_to_plot], width, label='Test', color='#D62246')
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylabel('Score')
ax.set_title('Train vs Validation vs Test — Análise de Generalização')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ncf_train_val_test.png', dpi=100, bbox_inches='tight')
plt.close()
print('[OK] ncf_train_val_test.png')

# --- Figura 3: Cold-start analysis ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
user_counts = df.groupby('user_id').size()
test_users = set(test_df['user_id'].unique())
train_users = set(train_df['user_id'].unique())
cold_start = test_users - train_users
warm_start = test_users & train_users

ax = axes[0]
counts_clipped = user_counts[user_counts <= 5]
ax.hist(counts_clipped, bins=5, edgecolor='black', color='#2E86AB')
ax.set_xlabel('Número de compras')
ax.set_ylabel('Número de usuários')
ax.set_title('Distribuição de compras por usuário (clipped em 5)')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
labels = ['Cold-start\n(não visto no treino)', 'Warm-start\n(visto no treino)']
sizes = [len(cold_start), len(warm_start)]
colors_pie = ['#D62246', '#06A77D']
explode = (0.05, 0)
ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
ax.set_title(f'Cold-start no Test Set\n({len(test_users):,} usuários)')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'coldstart_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print('[OK] coldstart_analysis.png')

# --- Figura 4: Embedding norms ---
fig, ax = plt.subplots(figsize=(10, 5))
user_emb_norm = model.user_emb.weight.norm(dim=1).detach().numpy()
item_emb_norm = model.item_emb.weight.norm(dim=1).detach().numpy()
cat_emb_norm = model.cat_emb.weight.norm(dim=1).detach().numpy()

ax.hist(user_emb_norm, bins=50, alpha=0.5, label=f'Users (μ={user_emb_norm.mean():.2f})', color='#2E86AB')
ax.hist(item_emb_norm, bins=50, alpha=0.5, label=f'Items (μ={item_emb_norm.mean():.2f})', color='#F18F01')
ax.hist(cat_emb_norm, bins=20, alpha=0.7, label=f'Categories (μ={cat_emb_norm.mean():.2f})', color='#06A77D')
ax.set_xlabel('Norma L2 do embedding')
ax.set_ylabel('Frequência')
ax.set_title('Distribuição das normas L2 dos embeddings aprendidos')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'embedding_norms.png', dpi=100, bbox_inches='tight')
plt.close()
print('[OK] embedding_norms.png')

print('\nTodas as figuras geradas em', FIGURES_DIR)
