# -*- coding: utf-8 -*-
"""Ship compact open-set result tables for the public repository."""

import pathlib
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / 'results' / 'osr_pairs15_k2'
DST = REPO / 'results' / 'open_set'
DST.mkdir(parents=True, exist_ok=True)

LABEL = {'base': 'source_only', 'cal_sup': 'cal_sup', 'cg_sup': 'cgda'}

rows = []
for f in SRC.rglob('tasks/*/metrics.csv'):
    try:
        rows.append(pd.read_csv(f).iloc[0])
    except Exception:
        continue
df = pd.DataFrame(rows)
df = df[df.method.isin(LABEL)].copy()
df['method'] = df.method.map(LABEL)

per_run = df[['setting', 'method', 'budget', 'seed', 'fold', 'auroc_k2', 'oscr_k2',
              'known_accuracy', 'known_macro_f1', 'n_test', 'n_known', 'n_unknown',
              'n_source_known', 'n_calibration']].rename(columns={
    'setting': 'unknown_pair', 'auroc_k2': 'auroc', 'oscr_k2': 'oscr'})
per_run = per_run.sort_values(['unknown_pair', 'method', 'seed', 'fold'])
per_run.to_csv(DST / 'osr_pairs15_per_run.csv', index=False)
print('per_run', per_run.shape)

by_pair = (df.groupby(['setting', 'method'])
           .agg(auroc_mean=('auroc_k2', 'mean'), auroc_sd=('auroc_k2', 'std'),
                oscr_mean=('oscr_k2', 'mean'), oscr_sd=('oscr_k2', 'std'),
                known_accuracy_mean=('known_accuracy', 'mean'),
                known_accuracy_sd=('known_accuracy', 'std'),
                macro_f1_mean=('known_macro_f1', 'mean'),
                macro_f1_sd=('known_macro_f1', 'std'),
                n_runs=('auroc_k2', 'size'),
                n_known=('n_known', 'first'),
                n_unknown=('n_unknown', 'first'))
           .reset_index().rename(columns={'setting': 'unknown_pair'}))
by_pair = by_pair.round(4).sort_values(['unknown_pair', 'method'])
by_pair.to_csv(DST / 'osr_pairs15_by_pair.csv', index=False)
print('by_pair', by_pair.shape)

print(by_pair.to_string(index=False))
