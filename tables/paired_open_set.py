# -*- coding: utf-8 -*-
"""Paired contrasts for the open-set summary table."""

import pathlib
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / 'results' / 'osr_pairs15_k2'

rows = []
for f in SRC.rglob('tasks/*/metrics.csv'):
    try:
        rows.append(pd.read_csv(f).iloc[0])
    except Exception:
        continue
df = pd.DataFrame(rows)
df = df[df.method.isin(['base', 'cal_sup', 'cg_sup'])]

print('=== mean +- sd over %d runs ===' % len(df))
g = df.groupby('method').agg(AUROC=('auroc_k2', 'mean'), AUROC_sd=('auroc_k2', 'std'),
                             OSCR=('oscr_k2', 'mean'), OSCR_sd=('oscr_k2', 'std'),
                             KAcc=('known_accuracy', 'mean'), KAcc_sd=('known_accuracy', 'std'),
                             KF1=('known_macro_f1', 'mean'), KF1_sd=('known_macro_f1', 'std'))
print(g.round(4).to_string())

# ---- paired by (pair, seed, fold)
w = df.pivot_table(index=['setting', 'seed', 'fold'], columns='method',
                   values=['auroc_k2', 'oscr_k2', 'known_accuracy'])
print('\n=== paired contrasts (cg_sup - other) ===')
rng = np.random.default_rng(0)
for metric, label in [('auroc_k2', 'AUROC'), ('oscr_k2', 'OSCR'), ('known_accuracy', 'KnownAcc')]:
    for other in ['base', 'cal_sup']:
        d = (w[(metric, 'cg_sup')] - w[(metric, other)]).dropna().values
        n = len(d)
        mean = d.mean()
        se = d.std(ddof=1) / np.sqrt(n)
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        boot = rng.choice(d, size=(2000, n), replace=True).mean(axis=1)
        lo_b, hi_b = np.percentile(boot, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        print('%s CGDA-%s: %+.4f  95%%CI [%+.4f, %+.4f]  boot[%+.4f,%+.4f]  n=%d  win=%d'
              % (label, other, mean, lo, hi, lo_b, hi_b, n, int((d > 0).sum())))
