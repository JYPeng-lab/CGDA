# -*- coding: utf-8 -*-
"""Convert MENS sensor-drift dataset (.mat) to CSV files.

Paper: Lei Zhang, Yan Liu, Pingling Deng, "Odor Recognition in Multiple E-nose
Systems via Cross-domain Discriminative Subspace Learning",
IEEE Trans. Instrum. Meas., 66(7):1679-1692, 2017.

source: master device; target: slave1 / slave2 (different devices).
Files produced under REPO/data/MENS_csv/:
  mens_master_slave1.csv   (source + target, 6 features, label 1..6)
  mens_master_slave2.csv
  mens_all.csv             (data.mat: master 389x8, slave1 607x8, slave2 576x8)
"""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root
import os
import numpy as np
import pandas as pd
import scipy.io as sio

IN = REPO / 'data' / 'MENS' / 'MENS_paper_code' / 'codes_other_methods' / 'data' / 'data_setting_1'
OUT = REPO / 'data' / 'MENS_csv'
os.makedirs(OUT, exist_ok=True)

GAS_NAMES = {1: 'ammonia', 2: 'benzene', 3: 'carbon monoxide',
             4: 'formaldehyde', 5: 'nitrogen dioxide', 6: 'toluene'}


def dump(fname, matfile, src_label='master'):
    m = sio.loadmat(os.path.join(IN, matfile))
    Xs, Xt = m['source'], m['target']
    ys, yt = m['slabel'].ravel(), m['tlabel'].ravel()
    df = pd.DataFrame(np.vstack([Xs, Xt]), columns=[f'f{i}' for i in range(1, Xs.shape[1] + 1)])
    df['label'] = np.concatenate([ys, yt])
    df['domain'] = [src_label] * len(Xs) + [matfile.replace('origin', 'slave')] * len(Xt)
    df['gas'] = df['label'].map(GAS_NAMES)
    df.to_csv(os.path.join(OUT, fname), index=False)
    print(f'{fname}: {len(df)} rows x {df.shape[1]} cols | '
          f'src={len(Xs)} tgt={len(Xt)} | classes={sorted(df.label.unique())}')


if __name__ == '__main__':
    dump('mens_master_slave1.csv', 'origin1.mat')
    dump('mens_master_slave2.csv', 'origin2.mat')

    # full version from data.mat (8 features)
    m = sio.loadmat(REPO / 'data' / 'MENS' / 'MENS_paper_code' / 'MENS' / 'data.mat')
    parts = []
    for dom in ['master', 'slave1', 'slave2']:
        d = pd.DataFrame(m[dom], columns=[f'f{i}' for i in range(1, m[dom].shape[1] + 1)])
        d['domain'] = dom
        parts.append(d)
    allc = pd.concat(parts, ignore_index=True)
    allc.to_csv(os.path.join(OUT, 'mens_all_8feat.csv'), index=False)
    print(f'mens_all_8feat.csv: {len(allc)} rows x {allc.shape[1]} cols '
          f'| domains: {allc.domain.value_counts().to_dict()}')

    # save a README
    readme = """MENS sensor-drift dataset (Zhang et al., IEEE TIM 2017)
============================================================
Source: Lei Zhang, Yan Liu, Pingling Deng, "Odor Recognition in Multiple
E-nose Systems via Cross-domain Discriminative Subspace Learning",
IEEE Transactions on Instrumentation and Measurement, 66(7):1679-1692, 2017.
Download: https://www.leizhang.tk/resources/CDSL_paper_code.rar

Structure
- mens_master_slave1.csv : origin1.mat -> master (420) as source, slave1 (608) as target, 6 features
- mens_master_slave2.csv : origin2.mat -> master (420) as source, slave2 (576) as target, 6 features
- mens_all_8feat.csv     : data.mat raw arrays (master 389, slave1 607, slave2 576, 8 features)

Columns (6-feat files): f1..f6, label (1..6), domain (master/slave1/slave2), gas (name)
Gases: 1=ammonia, 2=benzene, 3=carbon monoxide, 4=formaldehyde,
       5=nitrogen dioxide, 6=toluene
Drift source: master collected ~5 years earlier than slave1/slave2 (device difference)
"""
    with open(os.path.join(OUT, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(readme)
    print('README.txt written')
