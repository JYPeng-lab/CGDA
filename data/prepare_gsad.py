# -*- coding: utf-8 -*-
"""Download the public benchmarks and build the processed CSV files."""

import sys
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'benchmark'))

from gasbench.data import ensure_data

if __name__ == '__main__':
    out = ensure_data(REPO / 'data')
    print('processed data ->', out)
