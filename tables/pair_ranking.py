# -*- coding: utf-8 -*-
"""Per-pair spread of open-set ranking metrics."""

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

piv = df.pivot_table(index='setting', columns='method', values='auroc_k2', aggfunc='mean').round(4)
piv = piv[KEEP].sort_values('cg_sup', ascending=False)
piv['rank'] = range(1, len(piv) + 1)
print(piv.to_string())
print('\ncg_sup range: %.4f - %.4f, mean %.4f' % (piv.cg_sup.min(), piv.cg_sup.max(), piv.cg_sup.mean()))
print('cal_sup range: %.4f - %.4f, mean %.4f' % (piv.cal_sup.min(), piv.cal_sup.max(), piv.cal_sup.mean()))
piv.to_csv(REPO / 'results' / 'osr_pair_ranking.csv', encoding='utf-8-sig')
