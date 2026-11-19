# -*- coding: utf-8 -*-
"""Aggregate Fig. 6 panels for the settings used in figure_6_a_c.R."""

import pathlib
import shutil

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / 'results' / 'fig6_panels_k2'
OUT = REPO / 'main_results'
ARCH = OUT / '_archive' / '2026-11-19_fig6_rerun'

PAIRS = ['2-3', '1-2', '3-5']
KEYS = ['unknown_pair', 'method', 'seed', 'fold']


def archive():
    ARCH.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in OUT.glob('panel_44*.csv'):
        dest = ARCH / (p.stem + '.prev_run' + p.suffix)
        shutil.move(str(p), str(dest))
        moved += 1
    print('archived %d previous panel files' % moved)


def collect(pattern):
    frames = []
    for f in SRC.rglob(pattern):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_curves(s):
    rows = []
    for k, z in s.groupby(KEYS, sort=False):
        known = z.is_unknown.values == 0
        unknown = ~known
        correct = known & (z.y_pred.values == z.y_true.values)
        g = (pd.DataFrame({'d': z.distance_score.values,
                           'u': unknown.astype(int),
                           'k': correct.astype(int)})
             .groupby('d')[['u', 'k']].sum().sort_index())
        rows.append(pd.DataFrame({
            'setting': k[0], 'method': k[1], 'seed': k[2], 'fold': k[3],
            'unknown_fpr': np.r_[0.0, np.cumsum(g.u.values) / unknown.sum()],
            'known_ccr': np.r_[0.0, np.cumsum(g.k.values) / known.sum()],
        }))
    return pd.concat(rows, ignore_index=True)


def build_contamination(s, t):
    rows = []
    for k, z in s.groupby(KEYS, sort=False):
        q = t[(t.unknown_pair == k[0]) & (t.method == k[1]) &
              (t.seed == k[2]) & (t.fold == k[3])]
        known = z.is_unknown.values == 0
        unknown = ~known
        for _, r in q.iterrows():
            if r.threshold_method == 'known_reference_q90':
                continue
            strategy = 'full' if '_full_' in r.threshold_method else 'tail'
            tau = r.threshold
            rows.append({
                'setting': k[0], 'method': k[1], 'seed': k[2], 'fold': k[3],
                'strategy': strategy, 'rho': r.contamination_ratio,
                'known_correct_acceptance': float(
                    ((z.distance_score.values[known] <= tau) &
                     (z.y_pred.values[known] == z.y_true.values[known])).mean()),
                'unknown_rejection': float((z.distance_score.values[unknown] > tau).mean()),
            })
    return pd.DataFrame(rows)


def main():
    s = collect('samples.csv')
    t = collect('thresholds.csv')
    s = s[s.unknown_pair.isin(PAIRS)].reset_index(drop=True)
    t = t[t.unknown_pair.isin(PAIRS)].reset_index(drop=True)

    print('samples', s.shape, 'thresholds', t.shape)
    print('pairs:', sorted(s.unknown_pair.unique()))
    required = {'protocol', 'direction', 'budget', 'seed', 'fold', 'method',
                'unknown_pair', 'sample_id', 'is_unknown', 'distance_score', 'y_true', 'y_pred'}
    assert required.issubset(s.columns), list(s.columns)
    assert np.isfinite(s.distance_score.values).all()
    assert s.is_unknown.isin([0, 1]).all()
    assert not s.duplicated(subset=KEYS + ['sample_id']).any()

    sizes = s.groupby(KEYS).size()
    per_pair = s.groupby(['unknown_pair', 'seed', 'fold']).size().groupby('unknown_pair').nunique()
    assert (per_pair == 1).all(), per_pair
    print('runs=%d, n per run=%s' % (len(sizes), sorted(sizes.unique())))

    archive()

    s.to_csv(OUT / 'panel_44_sample_scores_cosine.csv', index=False)
    t.to_csv(OUT / 'panel_44_thresholds_cosine.csv', index=False)

    a = build_curves(s)
    print('curves', a.shape)
    a.to_csv(OUT / 'panel_44a_oscr_cosine.csv', index=False)

    b = build_contamination(s, t)
    print('contamination', b.shape)
    b.to_csv(OUT / 'panel_44b_contamination_cosine.csv', index=False)

    s[['unknown_pair', 'method', 'seed', 'fold', 'distance_score', 'is_unknown']].to_csv(
        OUT / 'panel_44c_distribution_cosine.csv', index=False)
    print('written to', OUT)


if __name__ == '__main__':
    main()
