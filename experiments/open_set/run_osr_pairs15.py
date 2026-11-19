# -*- coding: utf-8 -*-
"""Open-set evaluation over all C(6,2)=15 unknown-class pairs.

Each task trains on known classes only, calibrates from known-class samples of the
target pool, and scores softmax-rejected testing samples by cosine distance to the
nearest known-class prototype.

Usage:
    python run_osr_pairs15.py --methods cg_sup --limit 2
    python run_osr_pairs15.py --methods base,cal_sup,mixup_std,cg_sup,cg_shuffle --workers 6
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

import argparse, itertools, math, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score

PROJECT = _REPO
_sys.path.insert(0, str(PROJECT / 'core'))
_sys.path.insert(0, str(PROJECT / 'experiments'))

from revision_core import (StandardScaler, stratified_parts, train_network,
                           network_output, oscr, check_disjoint,
                           distance_model, distances)
from run_revision import load_drift, shuffled_pool

CLASSES = list(range(1, 7))
PAIRS = ['%d-%d' % p for p in itertools.combinations(CLASSES, 2)]
SEEDS = [7, 42, 123, 2024, 2025]
FOLDS = list(range(1, 11))
BUDGET = 0.1
EPOCHS = 50
METHODS = ['base', 'cal_sup', 'mixup_std', 'cg_sup', 'cg_shuffle']


def split_seed_for(seed, fold):
    """Distinct partition per (seed, fold); training seed stays the shared one."""
    return int(seed) * 100 + int(fold)


def known_calibration(pool, target, y, fraction, known_classes):
    """Per-class truncated calibration drawn from known classes only."""
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


def run_task(task):
    method, pair, seed, fold = task['method'], task['pair'], task['seed'], task['fold']
    tag = task.get('tag', '')
    out_dir = PROJECT / 'results' / ('osr_pairs15' + tag) / method / ('b%.2f' % BUDGET) / 'tasks' / task['task_id']
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
        check_disjoint(source=source, calibration=cal, test=test)

        sc = StandardScaler().fit(x[source])
        xs = sc.transform(x[source]).astype(np.float32)
        xc = sc.transform(x[cal]).astype(np.float32)
        xt = sc.transform(x[test]).astype(np.float32)
        steps = math.ceil((len(source) + len(pool) + 60 * len(known)) / 128)

        model, _ = train_network(xs, y[source], xc, y[cal], method, seed, known, EPOCHS, steps)

        pred, _, f_src = network_output(model, xs, known)
        pred_t, _, f_test = network_output(model, xt, known)

        d_cos = np.full(len(f_test), np.inf)
        for c in known:
            m = y[source] == c
            if not m.any():
                continue
            mu = f_src[m].mean(0)
            cos = np.dot(f_test, mu) / (np.linalg.norm(f_test, axis=1) * np.linalg.norm(mu) + 1e-8)
            d_cos = np.minimum(d_cos, 1 - cos)

        params = distance_model(f_src, y[source])
        d_maha = distances(f_test, params)
        d_k2 = kproto_distances(f_src, y[source], f_test, 2, seed)

        is_unknown = np.isin(y[test], unknown)
        n1, n0 = int(is_unknown.sum()), int((~is_unknown).sum())

        auroc = float(roc_auc_score(is_unknown.astype(int), d_cos)) if n0 and n1 else np.nan
        oscr_val = float(oscr(y[test], pred_t, d_cos, ~is_unknown)[0])
        auroc_maha = float(roc_auc_score(is_unknown.astype(int), d_maha)) if n0 and n1 else np.nan
        oscr_maha = float(oscr(y[test], pred_t, d_maha, ~is_unknown)[0])
        auroc_k2 = float(roc_auc_score(is_unknown.astype(int), d_k2)) if n0 and n1 else np.nan
        oscr_k2 = float(oscr(y[test], pred_t, d_k2, ~is_unknown)[0])
        known_acc = float((pred_t[~is_unknown] == y[test][~is_unknown]).mean()) if n0 else np.nan
        known_f1 = float(f1_score(y[test][~is_unknown], pred_t[~is_unknown], average='macro',
                                  zero_division=0)) if n0 else np.nan

        pd.DataFrame([{
            'family': 'osr', 'setting': pair, 'budget': BUDGET, 'seed': seed, 'fold': fold,
            'split_seed': ss, 'method': method, 'auroc': auroc, 'oscr': oscr_val,
            'auroc_maha': auroc_maha, 'oscr_maha': oscr_maha,
            'auroc_k2': auroc_k2, 'oscr_k2': oscr_k2,
            'known_accuracy': known_acc, 'known_macro_f1': known_f1,
            'n_test': len(xt), 'n_known': n0, 'n_unknown': n1,
            'n_source_known': len(source), 'n_calibration': len(cal),
        }]).to_csv(out_dir / 'metrics.csv', index=False)
        pd.DataFrame([{'status': 'complete', 'seconds': time.time() - t0}]).to_csv(
            out_dir / 'completion.csv', index=False)
        return task['task_id'], 'complete', time.time() - t0
    except Exception as exc:
        traceback.print_exc()
        pd.DataFrame([{'status': 'failed', 'error': repr(exc)}]).to_csv(out_dir / 'failure.csv', index=False)
        return task['task_id'], 'failed', time.time() - t0


def build_tasks(methods, pairs, limit=None, tag=''):
    tasks = []
    for method in methods:
        for pair in pairs:
            for seed in SEEDS:
                for fold in FOLDS:
                    if limit and len(tasks) >= limit:
                        return tasks
                    tasks.append(dict(method=method, pair=pair, budget=BUDGET, seed=seed, fold=fold,
                                      tag=tag,
                                      task_id='osr15_%s_b%.2f_s%d_f%d' % (pair, BUDGET, seed, fold)))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--pairs', default='all')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    pairs = PAIRS if args.pairs == 'all' else [p.strip() for p in args.pairs.split(',')]
    tasks = build_tasks(methods, pairs, args.limit or None, args.tag)

    print('PLAN methods=%s pairs=%d tasks=%d workers=%d' % (methods, len(pairs), len(tasks), args.workers), flush=True)
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
    print('FINISHED complete=%d failed=%d cached=%d wall=%.1fs' % (done, fail, cached, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
