# -*- coding: utf-8 -*-
"""Consolidate all 15 unknown-class pairs into one workbook."""

import pathlib
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / 'results' / 'osr_pairs15_k2'
OUT = REPO / 'main_results'
KEEP = ['base', 'cal_sup', 'cg_sup']

LABEL = {'base': 'Source only', 'cal_sup': 'Cal-Sup', 'cg_sup': 'CGDA'}


def load_runs():
    rows = []
    for f in SRC.rglob('tasks/*/metrics.csv'):
        try:
            d = pd.read_csv(f).iloc[0]
        except Exception:
            continue
        if d.method in KEEP:
            rows.append(d)
    return pd.DataFrame(rows)


def main():
    df = load_runs()
    df = df[df.method.isin(KEEP)].copy()
    print('runs', len(df))

    per_run = df[['setting', 'method', 'budget', 'seed', 'fold',
                  'auroc_k2', 'oscr_k2', 'known_accuracy', 'known_macro_f1',
                  'n_test', 'n_known', 'n_unknown', 'n_source_known',
                  'n_calibration']].rename(columns={
        'setting': 'unknown_pair', 'auroc_k2': 'AUROC', 'oscr_k2': 'OSCR',
        'known_accuracy': 'known_accuracy', 'known_macro_f1': 'known_macro_f1'})
    per_run['method'] = per_run.method.map(LABEL)
    per_run.sort_values(['unknown_pair', 'method', 'seed', 'fold'], inplace=True)

    agg = (df.groupby(['setting', 'method'])
           .agg(AUROC_mean=('auroc_k2', 'mean'), AUROC_sd=('auroc_k2', 'std'),
                OSCR_mean=('oscr_k2', 'mean'), OSCR_sd=('oscr_k2', 'std'),
                KnownAcc_mean=('known_accuracy', 'mean'), KnownAcc_sd=('known_accuracy', 'std'),
                KnownF1_mean=('known_macro_f1', 'mean'),
                n_runs=('auroc_k2', 'size'),
                n_known=('n_known', 'first'), n_unknown=('n_unknown', 'first'))
           .reset_index())

    wide = agg.pivot(index='setting', columns='method')
    summary = pd.DataFrame({
        'unknown_pair': wide.index,
        'n_known': wide[('n_known', 'base')].values,
        'n_unknown': wide[('n_unknown', 'base')].values,
        'n_runs_per_method': wide[('n_runs', 'base')].values,
        'AUROC_source_only': wide[('AUROC_mean', 'base')].values.round(4),
        'AUROC_cal_sup': wide[('AUROC_mean', 'cal_sup')].values.round(4),
        'AUROC_CGDA': wide[('AUROC_mean', 'cg_sup')].values.round(4),
        'OSCR_source_only': wide[('OSCR_mean', 'base')].values.round(4),
        'OSCR_cal_sup': wide[('OSCR_mean', 'cal_sup')].values.round(4),
        'OSCR_CGDA': wide[('OSCR_mean', 'cg_sup')].values.round(4),
        'KnownAcc_CGDA': wide[('KnownAcc_mean', 'cg_sup')].values.round(4),
    })
    summary['delta_AUROC_CGDA_minus_cal'] = (summary.AUROC_CGDA - summary.AUROC_cal_sup).round(4)
    summary['delta_OSCR_CGDA_minus_cal'] = (summary.OSCR_CGDA - summary.OSCR_cal_sup).round(4)
    summary['CGDA_better'] = summary.delta_AUROC_CGDA_minus_cal > 0
    summary['difficulty_AUROC'] = (summary.AUROC_CGDA + summary.AUROC_cal_sup).round(4)
    summary.sort_values('AUROC_CGDA', ascending=False, inplace=True)
    summary.insert(0, 'rank_by_CGDA_AUROC', range(1, len(summary) + 1))

    means = summary[['AUROC_source_only', 'AUROC_cal_sup', 'AUROC_CGDA',
                     'OSCR_source_only', 'OSCR_cal_sup', 'OSCR_CGDA']].mean().round(4)
    pooled = (df.groupby('method').agg(AUROC=('auroc_k2', 'mean'),
                                       OSCR=('oscr_k2', 'mean'),
                                       KnownAcc=('known_accuracy', 'mean'))
              .reset_index())
    pooled['method'] = pooled.method.map(LABEL)
    pooled['n_runs'] = [len(df[df.method == m]) for m in ['base', 'cal_sup', 'cg_sup']]

    xlsx = OUT / 'table_44_open_set_all_pairs.xlsx'
    with pd.ExcelWriter(xlsx, engine='openpyxl') as w:
        summary.to_excel(w, sheet_name='Summary_15pairs', index=False)
        per_run.to_excel(w, sheet_name='PerRun_2250', index=False)
        pooled.to_excel(w, sheet_name='Pooled_mean', index=False)
        pd.DataFrame([means]).to_excel(w, sheet_name='Column_mean_check', index=False)

    summary.to_csv(OUT / 'table_44_open_set_all_pairs.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
    print('\npooled:')
    print(pooled.to_string(index=False))
    print('\nwins: %d/15' % summary.CGDA_better.sum())
    print('written', xlsx)


if __name__ == '__main__':
    main()
