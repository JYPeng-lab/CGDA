# -*- coding: utf-8 -*-
"""Per-pair difficulty, sample size and method contrast for panel selection."""

import pathlib
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / 'results' / 'osr_pairs15_k2'
KEEP = ['base', 'cal_sup', 'cg_sup']

rows = []
for f in SRC.rglob('tasks/*/metrics.csv'):
    try:
        rows.append(pd.read_csv(f))
    except Exception:
        continue
df = pd.concat(rows, ignore_index=True)
df = df[df.method.isin(KEEP)]

g = df.groupby(['setting', 'method']).agg(
    auroc=('auroc_k2', 'mean'),
    oscr=('oscr_k2', 'mean'),
    n_unknown=('n_unknown', 'first'),
    n_known=('n_known', 'first'),
).reset_index()

p = g.pivot(index='setting', columns='method')
out = pd.DataFrame({
    'n_unknown': p[('n_unknown', 'base')],
    'n_known': p[('n_known', 'base')],
    'auroc_cal': p[('auroc', 'cal_sup')],
    'auroc_cg': p[('auroc', 'cg_sup')],
    'oscr_cal': p[('oscr', 'cal_sup')],
    'oscr_cg': p[('oscr', 'cg_sup')],
})
out['d_auroc'] = out.auroc_cg - out.auroc_cal
out['d_oscr'] = out.oscr_cg - out.oscr_cal
out['difficulty'] = out.auroc_cal + out.auroc_cg
out = out.sort_values('difficulty', ascending=False)

print(out.round(4).to_string())
print('\nwins: %d / 15' % (out.d_auroc > 0).sum())
print('min unknown n:', int(out.n_unknown.min()), 'at', out.n_unknown.idxmin())

out.round(4).to_csv(REPO / 'results' / 'osr_pair_selection.csv', encoding='utf-8-sig')
