# -*- coding: utf-8 -*-
"""Compare single-centroid and two-prototype cosine scoring on the 15-pair sweep."""

import pathlib
import numpy as np
import pandas as pd
from scipy import stats

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / 'results' / 'osr_pairs15_k2'
OUT = REPO / 'results'


def main():
    rows = []
    for f in SRC.rglob('tasks/*/metrics.csv'):
        try:
            rows.append(pd.read_csv(f))
        except Exception:
            continue
    df = pd.concat(rows, ignore_index=True)

    core = df[df.method.isin(['base', 'cal_sup', 'cg_sup'])]
    table = core.groupby('method').agg(
        n=('auroc_k2', 'size'),
        auroc_k1=('auroc', 'mean'), oscr_k1=('oscr', 'mean'),
        auroc_k2=('auroc_k2', 'mean'), oscr_k2=('oscr_k2', 'mean'),
        known_acc=('known_accuracy', 'mean'),
    ).round(4)
    print(table.to_string())
    table.to_csv(OUT / 'osr_k2_by_method.csv', encoding='utf-8-sig')

    print('\nall methods')
    allm = df.groupby('method').agg(
        auroc_k1=('auroc', 'mean'), oscr_k1=('oscr', 'mean'),
        auroc_k2=('auroc_k2', 'mean'), oscr_k2=('oscr_k2', 'mean'),
        known_acc=('known_accuracy', 'mean'),
    ).round(4)
    print(allm.to_string())
    allm.to_csv(OUT / 'osr_k2_all_methods.csv', encoding='utf-8-sig')

    for score in ['oscr', 'oscr_k2']:
        piv = df[df.method.isin(['cg_sup', 'cal_sup'])].pivot_table(
            index=['setting', 'seed', 'fold'], columns='method', values=score)
        diff = piv['cg_sup'] - piv['cal_sup']
        t = stats.ttest_rel(piv['cg_sup'], piv['cal_sup'])
        w = stats.wilcoxon(piv['cg_sup'], piv['cal_sup'])
        print('\n%s: cg_sup - cal_sup mean=%+.4f  CI95=[%+.4f,%+.4f]  t_p=%.5f  wilcoxon_p=%.5f  W/L/T=%d/%d/%d'
              % (score, diff.mean(),
                 diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)),
                 diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)),
                 t.pvalue, w.pvalue, int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())))


if __name__ == '__main__':
    main()
