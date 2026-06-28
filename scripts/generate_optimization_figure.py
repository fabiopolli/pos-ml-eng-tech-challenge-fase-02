"""Gera figura de comparação de runs do NCF (Etapa 4)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path('.') / 'artifacts'
FIGURES = Path('.') / 'reports' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

# Coletar métricas de todos os runs
runs_data = []
for f in sorted(ARTIFACTS.glob('metrics_*.json')):
    with open(f) as fp:
        m = json.load(fp)
    run_name = m.get('run_name', f.stem.replace('metrics_', ''))
    runs_data.append({
        'name': run_name,
        'ndcg': m['test']['NDCG@K'],
        'map': m['test']['MAP@K'],
        'hr': m['test']['HitRate@K'],
        'recall': m['test']['Recall@K'],
    })

# Ordenar por NDCG
runs_data.sort(key=lambda x: x['ndcg'], reverse=True)

print('Runs encontradas:')
for r in runs_data:
    print(f"  {r['name']:55s} | NDCG={r['ndcg']:.4f}")

# Adicionar baseline de popularidade
baseline = {'name': 'Baseline (Popularidade)', 'ndcg': 0.0045, 'map': 0.0031, 'hr': 0.0099, 'recall': 0.0096}

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Subplot 1: NDCG, MAP, HitRate por run
ax = axes[0]
names = [r['name'][:30] + '...' if len(r['name']) > 30 else r['name'] for r in runs_data]
names.append(baseline['name'])
ndcgs = [r['ndcg'] for r in runs_data] + [baseline['ndcg']]
maps = [r['map'] for r in runs_data] + [baseline['map']]
hrs = [r['hr'] for r in runs_data] + [baseline['hr']]

x = np.arange(len(names))
width = 0.25

bars1 = ax.bar(x - width, ndcgs, width, label='NDCG@10', color='#2E86AB')
bars2 = ax.bar(x, maps, width, label='MAP@10', color='#06A77D')
bars3 = ax.bar(x + width, hrs, width, label='HitRate@10', color='#F18F01')

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Score')
ax.set_title('Comparação de Runs — Métricas de Test Set', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Destacar o melhor com cor
best_idx = 0
bars1[best_idx].set_color('#D62246')
bars2[best_idx].set_color('#D62246')
bars3[best_idx].set_color('#D62246')

# Subplot 2: Lift vs baseline
ax = axes[1]
ndcg_lifts = [r['ndcg'] / baseline['ndcg'] for r in runs_data]
colors = ['#D62246' if i == 0 else '#2E86AB' for i in range(len(runs_data))]
bars = ax.bar(range(len(runs_data)), ndcg_lifts, color=colors)
ax.set_xticks(range(len(runs_data)))
ax.set_xticklabels([r['name'][:25] + '...' if len(r['name']) > 25 else r['name'] for r in runs_data],
                    rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Lift vs Baseline (x vezes)')
ax.set_title('NCF vs Baseline Popularidade — NDCG@10', fontsize=12, fontweight='bold')
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

for bar, lift in zip(bars, ndcg_lifts, strict=False):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{lift:.1f}x', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES / 'ncf_optimization_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print(f'\n[OK] Figura salva em {FIGURES / "ncf_optimization_comparison.png"}')
