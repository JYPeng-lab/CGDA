# Data preparation

The four public benchmarks are not redistributed here. Place the raw files where
the scripts expect them, then run the converters once.

| Paper name | Source | Target location |
|---|---|---|
| GSAD | UCI Machine Learning Repository, *Gas Sensor Array Drift* (ID 224), `batch1.dat`-`batch10.dat` | `data/raw/` |
| MENS | Zhang, Liu & Deng, *Odor Recognition in Multiple E-nose Systems with Cross-domain Discriminative Subspace Learning*, IEEE TIM 66(7), 2017 (author package `CDSL_paper_code.rar`) | `data/MENS/MENS_paper_code/` |
| GSA-LC | UCI Machine Learning Repository, *Gas Sensor Array Low-concentration* (ID 1081) | `data/raw/gsalc.csv` |
| AQ | UCI Machine Learning Repository, *Air Quality* (ID 360) | `data/raw/AirQualityUCI.csv` |

## GSAD

```bash
python data/prepare_gsad.py
```

This downloads the archives, verifies them by SHA-256 and writes
`data/processed/drift_clean.csv` together with the GSA-LC and AQ CSV files.
`data/download_drift.py` is a lighter fallback that only fetches the ten batch
files into `data/raw/`.

## MENS

1. Download `CDSL_paper_code.rar` from the link given in the original publication.
2. Extract it and rename the folder to `MENS_paper_code`, so that the tree is
   `data/MENS/MENS_paper_code/codes_other_methods/data/data_setting_1/`.
3. Run:

```bash
python data/convert_mens_mat.py
```

This writes `data/MENS_csv/mens_master_slave1.csv` and `mens_master_slave2.csv`
(master as source, slave1 / slave2 as target, six features, labels 1-6).

## GSA-LC and AQ

Download the archives from the two UCI pages and keep the CSV files at
`data/raw/gsalc.csv` and `data/raw/AirQualityUCI.csv` (the AQ file keeps the
original semicolon separator and comma decimal mark).

## Notes

* All splits, calibration pools and test sets are rebuilt from the raw files by
  the scripts, so no split file needs to be shipped.
* Every script resolves paths relative to the repository root; there are no
  hard-coded absolute paths in the released code.
