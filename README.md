# CGDA: centroid-guided data augmentation for electronic-nose drift compensation

Reference implementation of **CGDA**, a calibration-driven augmentation scheme for
domain adaptation in electronic-nose drift compensation. Class-level target
centroids estimated from a small labelled calibration set anchor the interpolation
of source samples, and the same frozen feature extractor supports closed-set
recognition, unknown-gas rejection and environmental event detection through a
cosine prototype distance.

Datasets used in the paper: **GSAD** (UCI gas sensor drift), **MENS** (multiple
e-nose systems, Zhang et al., IEEE TIM 2017), **GSA-LC** (UCI low-concentration)
and **AQ** (UCI air quality).

## Repository layout

```
core/          shared numerical core: feature extractor, training loop,
               mixup/CGDA samplers, cosine distance model, OSCR / ECE metrics
data/          download and format-conversion utilities (+ data/README.md)
experiments/
  run_revision.py        shared entry point (closed set, GSA, AQ, open set)
  protocols/             Section 4.2-4.3  partition protocols and their controls
  cross_domain/          Section 4.4      cross-domain recognition and ablations
  open_set/              Section 4.5      unknown-gas rejection and thresholds
  event_detection/       Section 4.6      AQ environmental event detection
  efficiency/            Section 4.7      quantisation and heavy-model cost
overhead/      Section 4.7  parameter / FLOPs / peak-memory / latency measurement
tables/        aggregation scripts that turn run outputs into the paper tables
results/       frozen tables consumed by the figure scripts
figures/       the 13 R scripts that draw the published figures
```

Run outputs from a fresh execution are written to `main_results/` (ignored by git),
whereas `results/` holds the frozen tables behind the published figures.

## Environment

Python 3.9+ with `numpy pandas scipy scikit-learn torch torchvision matplotlib thop psutil`:

```bash
pip install -r requirements.txt
```

R 4.x with `ggplot2`, `patchwork`, `dplyr`, `tidyr`, `scales`, `readr` for the figures.

`benchmark/` ships the GasBench helper package (data download and the deployment
cost models used in Section 4.7). It is imported straight from the repository,
so no install step is needed.

## Quick start

```bash
# 0. data: see data/README.md, then
python data/prepare_gsad.py       # download GSAD / GSA-LC / AQ, build data/processed

# 1. reference closed-set results on the four datasets (Section 4.4)
python experiments/cross_domain/run_single_source.py
python experiments/cross_domain/run_forward_budget_scan.py
python experiments/cross_domain/run_mens_full.py
python experiments/cross_domain/run_gsa_loco.py

# 2. protocol controls (Sections 4.2-4.3)
python experiments/protocols/run_lobo_target_first.py
python experiments/protocols/run_correct_protocols.py
python experiments/protocols/run_mixed_crossval.py

# 3. centroid-guidance ablation (Section 4.4.3)
python experiments/cross_domain/run_ablation_supplement.py
python experiments/cross_domain/run_protocols_extra.py
python experiments/cross_domain/run_forward_mixup.py
python experiments/cross_domain/run_feature_metrics.py

# 4. open-set rejection (Section 4.5)
python experiments/open_set/run_osr_cosine.py
python experiments/open_set/run_osr_cosine_multi.py --method cg_sup

# 5. environmental event detection (Section 4.6)
python experiments/event_detection/run_aq_cosine.py

# 6. computational cost (Section 4.7)
python experiments/efficiency/run_heavy_models.py
python experiments/efficiency/run_deploy_quant.py
python overhead/measure_reproduced_overhead.py
python overhead/measure_lit_peakmem.py
python overhead/bench_lit_time.py

# 7. aggregate run outputs into paper tables
python tables/aggregate_single_source.py
python tables/aggregate_protocol_overview.py
python tables/aggregate_osr_cosine.py
python tables/aggregate_event_panels.py
python tables/build_event_summary.py
python tables/merge_final_overhead.py
python tables/build_deployment_light.py
```

Each script is self-contained and takes its paths from the repository root;
`experiments/run_revision.py` is the only script with a CLI
(`--project --out --family --workers --aggregate --preflight`).

## Figures

`figures/*.R` read only from `results/`. Run from anywhere, e.g.

```bash
Rscript figures/figure_1_a.R
```

| Script | Panel | Tables consumed |
|---|---|---|
| `figure_1_a.R` | Fig. 1a GSAD batch structure | `data/processed/drift_clean.csv` |
| `figure_1_b.R` | Fig. 1b MENS device PCA | `data/MENS_csv/mens_master_slave{1,2}.csv` |
| `figure_2_a.R` | Fig. 2a GSA-LC concentration | `data/raw/gsalc.csv` |
| `figure_2_b-c.R` | Fig. 2b,c AQ environmental blocks | `data/raw/AirQualityUCI.csv` |
| `figure_4_c-e.R` | Fig. 4c-e protocol comparison | `panel_a_data.csv`, `metrics_by_fold_seed.csv`, `panel_single_source_long.csv`, `data/processed/drift_clean.csv` |
| `figure_5_a_c.R` | Fig. 5a,c budget and stability | `panel_a_long.csv`, `panel_correct_protocols_long.csv` |
| `figure_5_b.R` | Fig. 5b budget sensitivity | `panel_correct_protocols_long.csv` |
| `figure_5_d.R` | Fig. 5d calibration size | `sample_counts.csv` |
| `figure_5_e.R` | Fig. 5e centroid geometry | `panel_centroids.csv` |
| `figure_6_a_c.R` | Fig. 6a,c open-set performance | `panel_44a_oscr_cosine.csv`, `panel_44b_contamination_cosine.csv`, `panel_44_sample_scores_cosine.csv`, `panel_44_thresholds_cosine.csv` |
| `figure_7_a-d.R` | Fig. 7a-d AQ event detection | `panel_45{a,b,c,d}_*_cosine.csv` |
| `figure_7_e.R` | Fig. 7e cost comparison | `table_46_reproduced_overhead.csv`, `table_mens_lit_overhead.csv` |
| `figure_7_f.R` | Fig. 7f accuracy-latency trade-off | `table_46_deployment_light.csv`, `table_mens_deployment_light.csv` |

## Notes on reproducibility

* Splits, calibration pools, threshold references and test sets are rebuilt from
  the raw data with fixed seeds; nothing is sampled at figure time.
* Parameters and FLOPs are structural (computed from module shapes) and are
  therefore dataset independent; latency and peak memory were measured on a
  single desktop CPU and should be read as relative, not as field deployment
  guarantees.
* Feature standardisation uses source-domain statistics only; target statistics
  are never used to fit the scaler.

## Citation

The accompanying manuscript is currently under review. Until it is accepted,
please cite this software release:

```bibtex
@misc{peng2026cgda,
  author       = {Peng, Jingyi},
  title        = {{CGDA}: centroid-guided data augmentation for domain adaptation
                  in electronic-nose drift compensation},
  year         = {2026},
  howpublished = {\url{https://github.com/JYPeng-lab/CGDA}},
  note         = {Software release accompanying a manuscript under review}
}
```

This entry will be replaced by the journal article once it is accepted.

## License

MIT, see `LICENSE`.
