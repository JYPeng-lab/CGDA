# -*- coding: utf-8 -*-
"""Export sample-level cosine scores and contamination thresholds for Fig. 6.

Split rule is identical to run_osr_pairs15.py: the target pool is partitioned
2:8 with a seed derived from (seed, fold), so every method under one
(seed, fold) sees the same calibration/test split.  The calibration-excluded
remainder of the pool serves as the threshold reference pool, which keeps
Fig. 6 numerically consistent with the summary table.

Usage:
    python export_fig6_panels.py --pairs 1-2,3-5,4-6 --workers 6
"""

import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parents[2]
for _d in ("core", "experiments", "experiments/protocols", "experiments/cross_domain",
           "experiments/open_set", "experiments/event_detection", "experiments/efficiency",
           "overhead", "tables", "data", "benchmark"):
    _p = _REPO / _d
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.append(str(_p))

import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import argparse, math, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJECT = _REPO
_sys.path.insert(0, str(PROJECT / 'core'))
_sys.path.insert(0, str(PROJECT / 'experiments'))

from revision_core import (StandardScaler, stratified_parts, train_network,
                           network_output, check_disjoint)
from run_revision import load_drift, shuffled_pool

CLASSES = list(range(1, 7))
SEEDS = [7, 42, 123, 2024, 2025]
FOLDS = list(range(1, 11))
BUDGET = 0.1
EPOCHS = 50
RHO = [0.0, 0.25, 0.5, 0.75]


def split_seed_for(seed, fold):
    return int(seed) * 100 + int(fold)


def known_calibration(pool, target, y, fraction, known_classes):
    out = []
    for c in known_classes:
        z = pool[y[pool] == c]
        if len(z) == 0:
            continue
        n = min(len(z), max(1, int(fraction * np.sum(y[target] == c))))
        out.extend(z[:n])
    return np.array(sorted(out), dtype=int)


def kproto_distances(f_src, y_src, f_test, k, seed):
    """Cosine distance to the nearest of k per-class prototypes."""
    d = np.full(len(f_test), np.inf, dtype=float)
    for c in np.unique(y_src):
        z = f_src[y_src == c].astype(float)
        if k == 1 or len(z) < 4 * k:
            mus = z.mean(0, keepdims=True)
        else:
            mus = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z).cluster_centers_
        for mu in mus:
            cos = np.dot(f_test, mu) / (np.linalg.norm(f_test, axis=1) * np.linalg.norm(mu) + 1e-8)
            d = np.minimum(d, 1 - cos)
    return d


def cosine_to_nearest(f_test, f_src, y_src, known):
    """Cosine distance to the nearest known-class source prototype."""
    d = np.full(len(f_test), np.inf, dtype=float)
    for c in known:
        m = y_src == c
        if not m.any():
            continue
        mu = f_src[m].mean(0)
        cos = np.dot(f_test, mu) / (np.linalg.norm(f_test, axis=1) * np.linalg.norm(mu) + 1e-8)
        d = np.minimum(d, 1 - cos)
    return d


def q90(v):
    return float(np.quantile(v, 0.9))


def threshold_rows(dp, kp, seed, fold):
    """Contamination rules mirroring run_revision.run_osr."""
    rows = []
    if kp.any():
        rows.append({'contamination_ratio': 0.0, 'threshold_method': 'known_reference_q90',
                     'threshold': q90(dp[kp]), 'n_reference': int(kp.sum())})
    n = (min(100, int(kp.sum()), int((~kp).sum() / .75)) // 4) * 4
    if n >= 4:
        rng = np.random.RandomState(seed + int(fold) * 17)
        ki = rng.permutation(np.flatnonzero(kp))
        ui = rng.permutation(np.flatnonzero(~kp))
        for rho in RHO:
            nu = int(n * rho)
            sel = np.r_[ki[:n - nu], ui[:nu]]
            mix = dp[sel]
            tail = mix[mix <= np.median(mix)]
            tag = '%02d' % int(rho * 100)
            rows.append({'contamination_ratio': rho,
                         'threshold_method': 'mixed%s_full_q90' % tag,
                         'threshold': q90(mix), 'n_reference': n})
            rows.append({'contamination_ratio': rho,
                         'threshold_method': 'mixed%s_tail_q90' % tag,
                         'threshold': q90(tail), 'n_reference': n})
    return rows


def run_task(task):
    method, pair, seed, fold = task['method'], task['pair'], task['seed'], task['fold']
    out_dir = PROJECT / 'results' / 'fig6_panels_k2' / ('%s_%s_s%d_f%d' % (pair, method, seed, fold))
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'completion.csv').exists():
        return task['task_id'], 'cached', 0.0

    t0 = time.time()
    try:
        unknown = [int(t) for t in pair.split('-')]
        known = [c for c in CLASSES if c not in unknown]

        x, y, b = load_drift(str(PROJECT))
        source = np.flatnonzero((b == 1) & np.isin(y, known))
        target_all = np.flatnonzero(b == 2)
        ss = split_seed_for(seed, fold)

        pool, test = stratified_parts(target_all, y, [.2, .8], ss)
        pool = shuffled_pool(pool, ss)
        cal = known_calibration(pool, target_all, y, BUDGET, known)
        assert len(cal) > 0, 'empty calibration set'
        thpool = np.setdiff1d(pool, cal)
        check_disjoint(source=source, calibration=cal, test=test)

        sc = StandardScaler().fit(x[source])
        xs = sc.transform(x[source]).astype(np.float32)
        xc = sc.transform(x[cal]).astype(np.float32)
        xt = sc.transform(x[test]).astype(np.float32)
        xp = sc.transform(x[thpool]).astype(np.float32)
        steps = math.ceil((len(source) + len(pool) + 60 * len(known)) / 128)

        model, _ = train_network(xs, y[source], xc, y[cal], method, seed, known, EPOCHS, steps)

        pred, _, f_src = network_output(model, xs, known)
        pred_t, _, f_test = network_output(model, xt, known)
        _, _, f_pool = network_output(model, xp, known)

        d_test = kproto_distances(f_src, y[source], f_test, 2, seed)
        d_th = kproto_distances(f_src, y[source], f_pool, 2, seed)

        is_unknown = np.isin(y[test], unknown).astype(int)
        kp = np.isin(y[thpool], known)

        samples = pd.DataFrame({
            'protocol': 'OSR', 'direction': pair, 'budget': BUDGET,
            'seed': seed, 'fold': fold, 'method': method, 'unknown_pair': pair,
            'sample_id': np.arange(len(xt)),
            'is_unknown': is_unknown,
            'distance_score': d_test,
            'y_true': y[test],
            'y_pred': pred_t,
        })
        samples.to_csv(out_dir / 'samples.csv', index=False)

        th = pd.DataFrame(threshold_rows(d_th, kp, seed, fold))
        th.insert(0, 'method', method)
        th.insert(0, 'unknown_pair', pair)
        th.insert(0, 'fold', fold)
        th.insert(0, 'seed', seed)
        th.insert(0, 'budget', BUDGET)
        th.insert(0, 'direction', pair)
        th.insert(0, 'protocol', 'OSR')
        th.to_csv(out_dir / 'thresholds.csv', index=False)

        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out_dir / 'completion.csv', index=False)
        return task['task_id'], 'complete', time.time() - t0
    except Exception as exc:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(exc)}]).to_csv(out_dir / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def build_tasks(methods, pairs):
    tasks = []
    for method in methods:
        for pair in pairs:
            for seed in SEEDS:
                for fold in FOLDS:
                    tasks.append(dict(method=method, pair=pair, seed=seed, fold=fold,
                                      task_id='fig6_%s_%s_s%d_f%d' % (pair, method, seed, fold)))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default='base,cal_sup,cg_sup')
    ap.add_argument('--pairs', default='1-2,3-5,4-6')
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    pairs = [p.strip() for p in args.pairs.split(',') if p.strip()]
    tasks = build_tasks(methods, pairs)

    print('PLAN methods=%s pairs=%s tasks=%d workers=%d' %
          (methods, pairs, len(tasks), args.workers), flush=True)
    t0 = time.time()
    done = fail = cached = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_task, t): t for t in tasks}
            for i, f in enumerate(as_completed(futs), 1):
                tid, st, _ = f.result()
                done += st == 'complete'; fail += st == 'failed'; cached += st == 'cached'
                if i % 50 == 0 or i == len(tasks):
                    print('[%d/%d] done=%d fail=%d cached=%d %.0fs' %
                          (i, len(tasks), done, fail, cached, time.time() - t0), flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            tid, st, _ = run_task(t)
            done += st == 'complete'; fail += st == 'failed'; cached += st == 'cached'
            if i % 50 == 0 or i == len(tasks):
                print('[%d/%d] done=%d fail=%d cached=%d %.0fs' %
                      (i, len(tasks), done, fail, cached, time.time() - t0), flush=True)
    print('FINISHED complete=%d failed=%d cached=%d wall=%.1fs' %
          (done, fail, cached, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
