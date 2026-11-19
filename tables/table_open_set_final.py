# -*- coding: utf-8 -*-
"""Final open-set table: three-method comparison with paired contrasts."""

import pathlib
import numpy as np
import pandas as pd
from scipy import stats

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / 'results' / 'osr_pairs15_k2'
KEEP = ['base', 'cal_sup', 'cg_sup']
LABEL = {'base': 'Source only', 'cal_sup': 'Cal-Sup', 'cg_sup': 'CGDA (cg_sup)'}


def load():
    rows = []
    for f in SRC.rglob('tasks/*/metrics.csv'):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.iloc[0]['method'] in KEEP:
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    df = load()
    rec = []
    for m in KEEP:
        z = df[df.method == m]
        rec.append({
            'Method': LABEL[m],
            'AUROC_mean': round(z.auroc_k2.mean(), 4),
            'AUROC_sd': round(z.auroc_k2.std(), 4),
            'OSCR_mean': round(z.oscr_k2.mean(), 4),
            'OSCR_sd': round(z.oscr_k2.std(), 4),
            'KnownAcc_mean': round(z.known_accuracy.mean(), 4),
            'KnownAcc_sd': round(z.known_accuracy.std(), 4),
            'n_units': len(z),
        })
    table = pd.DataFrame(rec)
    print(table.to_string(index=False))
    table.to_csv(REPO / 'results' / 'table_open_set_final.csv', index=False, encoding='utf-8-sig')

    print('\npaired contrasts on OSCR')
    for a, bname in [('cal_sup', 'cg_sup'), ('cg_sup', 'base')]:
        piv = df[df.method.isin([a, bname])].pivot_table(
            index=['setting', 'seed', 'fold'], columns='method', values='oscr_k2')
        diff = piv[a] - piv[bname]
        lo = diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
        hi = diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
        print('%s - %s = %+.4f  CI95=[%+.4f, %+.4f]  p=%.6f'
              % (a, bname, diff.mean(), lo, hi, stats.ttest_rel(piv[a], piv[bname]).pvalue))


if __name__ == '__main__':
    main()
