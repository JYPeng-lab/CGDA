# -*- coding: utf-8 -*-
"""Aggregate the 15-pair open-set runs and decompose variance components.

Reads results/osr_pairs15/<method>/b0.10/tasks/*/metrics.csv and writes:
  results/osr_pairs15_summary.csv    per-setting and per-method means
  results/osr_pairs15_variance.csv   random-effects variance components
  results/osr_pairs15_paired.csv     paired cg_sup vs cal_sup per unknown pair
"""

import pathlib
import numpy as np
import pandas as pd
from scipy import stats

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / 'results' / 'osr_pairs15'
OUT = REPO / 'results'
METRICS = ['auroc', 'oscr', 'known_accuracy']


def load_all():
    rows = []
    for f in SRC.rglob('tasks/*/metrics.csv'):
        try:
            rows.append(pd.read_csv(f))
        except Exception:
            continue
    if not rows:
        raise SystemExit('no metrics found under %s' % SRC)
    return pd.concat(rows, ignore_index=True)


def variance_components(df, metric):
    """Balanced two-way random-effects ANOVA (unknown pair x method)."""
    piv = df.pivot_table(index=['setting', 'method'], values=metric, aggfunc='mean')
    counts = df.groupby(['setting', 'method'])[metric].count()
    n_rep = int(counts.min())
    u = df.setting.nunique()
    m = df.method.nunique()
    grand = df[metric].mean()
    cell = df.groupby(['setting', 'method'])[metric].mean()
    mu = df.groupby('setting')[metric].mean()
    mm = df.groupby('method')[metric].mean()

    ss_u = n_rep * m * ((mu - grand) ** 2).sum()
    ss_m = n_rep * u * ((mm - grand) ** 2).sum()
    ss_um = n_rep * ((cell - mu.reindex(cell.index.get_level_values(0)).values -
                      mm.reindex(cell.index.get_level_values(1)).values + grand) ** 2).sum()
    ss_e = ((df[metric] - cell.reindex(list(zip(df.setting, df.method))).values) ** 2).sum()

    df_u, df_m, df_um, df_e = u - 1, m - 1, (u - 1) * (m - 1), u * m * (n_rep - 1)
    ms_u, ms_m, ms_um, ms_e = ss_u / df_u, ss_m / df_m, ss_um / df_um, ss_e / df_e

    s2_um = max(0.0, (ms_um - ms_e) / n_rep)
    s2_u = max(0.0, (ms_u - ms_um) / (n_rep * m))
    s2_m = max(0.0, (ms_m - ms_um) / (n_rep * u))
    s2_e = ms_e
    total = s2_u + s2_m + s2_um + s2_e

    return [
        {'metric': metric, 'source': 'unknown_pair', 'df': df_u, 'MS': ms_u, 'sigma2': s2_u,
         'pct': 100 * s2_u / total},
        {'metric': metric, 'source': 'method', 'df': df_m, 'MS': ms_m, 'sigma2': s2_m,
         'pct': 100 * s2_m / total},
        {'metric': metric, 'source': 'pair_x_method', 'df': df_um, 'MS': ms_um, 'sigma2': s2_um,
         'pct': 100 * s2_um / total},
        {'metric': metric, 'source': 'residual', 'df': df_e, 'MS': ms_e, 'sigma2': s2_e,
         'pct': 100 * s2_e / total},
    ]


def main():
    df = load_all()
    print('rows', len(df), 'pairs', df.setting.nunique(), 'methods', sorted(df.method.unique()),
          'reps/cell', df.groupby(['setting', 'method']).size().min())

    summary = (df.groupby(['setting', 'method'])[METRICS]
               .agg(['mean', 'std', 'count']).round(4))
    summary.to_csv(OUT / 'osr_pairs15_summary.csv', encoding='utf-8-sig')
    by_method = df.groupby('method')[METRICS].agg(['mean', 'std', 'count']).round(4)
    by_method.to_csv(OUT / 'osr_pairs15_by_method.csv', encoding='utf-8-sig')
    print(by_method.to_string())

    var_rows = []
    for met in METRICS:
        var_rows.extend(variance_components(df, met))
    var = pd.DataFrame(var_rows).round(4)
    var.to_csv(OUT / 'osr_pairs15_variance.csv', index=False, encoding='utf-8-sig')
    print('\nvariance components')
    print(var.to_string())

    if {'cg_sup', 'cal_sup'} <= set(df.method.unique()):
        piv = df[df.method.isin(['cg_sup', 'cal_sup'])].pivot_table(
            index=['setting', 'seed', 'fold'], columns='method', values='oscr')
        pivot = piv.reset_index()
        diff = piv['cg_sup'] - piv['cal_sup']
        t = stats.ttest_rel(piv['cg_sup'], piv['cal_sup'])
        w = stats.wilcoxon(piv['cg_sup'], piv['cal_sup'])
        summary_stats = {
            'n_paired_units': len(piv),
            'mean_diff_oscr': round(float(diff.mean()), 5),
            'ci95_low': round(float(diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))), 5),
            'ci95_high': round(float(diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))), 5),
            'paired_t_p': round(float(t.pvalue), 6),
            'wilcoxon_p': round(float(w.pvalue), 6),
            'wins': int((diff > 0).sum()), 'losses': int((diff < 0).sum()), 'ties': int((diff == 0).sum()),
        }
        pd.DataFrame([summary_stats]).to_csv(OUT / 'osr_pairs15_paired.csv', index=False,
                                             encoding='utf-8-sig')
        print('\npaired cg_sup vs cal_sup (OSCR)')
        print(summary_stats)


if __name__ == '__main__':
    main()
