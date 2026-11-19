# -*- coding: utf-8 -*-
"""Cross-check the three panel settings against the stored summary metrics."""

import pathlib
import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[1] / 'main_results'
REPO = pathlib.Path(__file__).resolve().parents[1]
PAIRS = ['2-3', '1-2', '3-5']


def curve_oscr(z):
    known = z.is_unknown.values == 0
    unknown = ~known
    correct = known & (z.y_pred.values == z.y_true.values)
    g = (pd.DataFrame({'d': z.distance_score.values,
                       'u': unknown.astype(int),
                       'k': correct.astype(int)})
         .groupby('d')[['u', 'k']].sum().sort_index())
    fpr = np.r_[0.0, np.cumsum(g.u.values) / unknown.sum()]
    ccr = np.r_[0.0, np.cumsum(g.k.values) / known.sum()]
    return float(np.trapezoid(ccr, fpr))


def main():
    s = pd.read_csv(OUT / 'panel_44_sample_scores_cosine.csv')
    t = pd.read_csv(OUT / 'panel_44_thresholds_cosine.csv')
    assert sorted(s.unknown_pair.unique()) == sorted(PAIRS)
    assert sorted(t.unknown_pair.unique()) == sorted(PAIRS)

    rows = []
    for k, z in s.groupby(['unknown_pair', 'method', 'seed', 'fold']):
        rows.append({'pair': k[0], 'method': k[1], 'seed': k[2], 'fold': k[3],
                     'oscr': curve_oscr(z)})
    panel = pd.DataFrame(rows)

    stored = []
    for f in (REPO / 'results' / 'osr_pairs15_k2').rglob('tasks/*/metrics.csv'):
        d = pd.read_csv(f).iloc[0]
        if d['setting'] in PAIRS:
            stored.append({'pair': d['setting'], 'method': d['method'],
                           'seed': d['seed'], 'fold': d['fold'], 'oscr': d['oscr_k2']})
    ref = pd.DataFrame(stored)

    m = panel.merge(ref, on=['pair', 'method', 'seed', 'fold'], suffixes=('_panel', '_table'))
    diff = (m.oscr_panel - m.oscr_table).abs()
    print('matched runs: %d, max |diff| = %.6f, mean |diff| = %.6f'
          % (len(m), diff.max(), diff.mean()))

    print('\npanel mean OSCR (50 runs per cell):')
    print(panel.groupby(['pair', 'method']).oscr.mean().unstack().round(4).to_string())
    print('\nthreshold rules per run:', s.shape[0] // 450 * 1, 'samples/run=998, rules/run=',
          len(t) // (len(s) // 998))


if __name__ == '__main__':
    main()
