# -*- coding: utf-8 -*-
"""Compute CGDA network params/FLOPs via thop (same caliber as lit models)."""


# --- repository layout bootstrap: makes shared modules importable ---
import sys as _sys, pathlib as _pl
_REPO = _pl.Path(__file__).resolve().parents[1]
for _d in ("core", "experiments", "experiments/protocols", "experiments/cross_domain",
           "experiments/open_set", "experiments/event_detection", "experiments/efficiency",
           "overhead", "tables", "data", "benchmark"):
    _p = _REPO / _d
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.append(str(_p))

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import sys
sys.path.insert(0, str(REPO / 'core'))
sys.path.insert(0, str(REPO / 'experiments'))
import torch
import thop
from revision_core import Network

m = Network(6, 6)
macs, params = thop.profile(m, inputs=(torch.zeros(1, 1, 6),), verbose=False)
print(f'params={params} macs={macs} flops={macs*2} flops_M={(macs*2)/1e6:.4f} params_M={params/1e6:.4f}')
