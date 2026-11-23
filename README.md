# Closed-Loop Olive Knot Management in an Orchard Digital Twin

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Data: ERA5 CC BY 4.0](https://img.shields.io/badge/Weather-ERA5%20CC%20BY%204.0-orange.svg)](https://open-meteo.com/)

Official companion repository for the manuscript:

> **Closed-Loop Olive Knot Management in an Orchard Digital Twin: Hierarchical Reinforcement Learning over Agronomic Skills, Benchmarked on Real Reanalysis Weather**  
> Ali Chouikhi, Sawsan Selmi, Mourad Mars, Mohammed J. AlGhamdi, Khalil Jouili, and Hedi Sakli  
> *Submitted to MDPI Technologies (2026)*

## Overview

Olive knot (*Pseudomonas savastanoi* pv. *savastanoi*) spreads with splashing rain and infects olive trees through harvest, pruning, frost, and hail wounds. This repository provides a self-contained, CPU-only Python/NumPy implementation of:

1. **Orchard Digital Twin (`src/twin.py`)**: A six-sector, 600-tree, 243-day (1 October – 31 May) partially observable Markov decision process (POMDP) driven by 96 seasons of real **ECMWF ERA5** weather across four Tunisian olive regions (**Sfax**, **Medenine**, **Kairouan**, **Beja**; 2000–2023).
2. **Deterministic Front End & Safety Shield (`src/skills_shield.py`)**: Weather-and-wound infection-risk estimator, optional UAV multispectral belief tracker, rule-based agronomic skills (`protect`, `sanitize`, `scout`), and reactive safety projection enforcing rain-block, spray/pruning interval, cap, and weekly flight constraints.
3. **Agronomic Baselines (`src/baselines.py`)**: No-action, Fixed Calendar (`C`), and Grid-Searched Rain-Threshold Rule (`R`, tuned strictly on training-pool seeds `0–63`).
4. **From-Scratch NumPy PPO (`src/ppo_numpy.py`)**: Two-hidden-layer tanh MLP (8,519 parameters for `H`, 8,324 for `H-NU`, 14,044 for flat `F`/`F+S`) with manual backpropagation, GAE, clipped surrogate loss, and Adam—no GPU or deep learning framework required.

## Repository Structure

```text
Olive-Knot-Digital-Twin/
├── config/
│   └── twin_params.json          # All epidemic, sensing, rule, cost, and PPO parameters (Table 5 & Table 7)
├── src/
│   ├── weather.py                # ERA5 loader + strict train (2000-2014) / test (2015-2023) split
│   ├── twin.py                   # Six-sector olive knot POMDP simulator (Equations 1-5)
│   ├── skills_shield.py          # Risk estimator, UAV belief update, agronomic skills, and safety shield
│   ├── baselines.py              # No-action, Fixed Calendar (C), and Tuned Rain Rule (R)
│   └── ppo_numpy.py              # Self-contained NumPy PPO actor-critic trainer
├── scripts/
│   ├── download_era5.py          # Fetches Open-Meteo ERA5 weather for 4 Tunisian regions (2000-2023)
│   ├── tune_rain_rule.py         # Grid search for R on seeds 0..63 (training pool)
│   ├── train_ppo.py              # Trains H, H-NU, F, F+S, H-LN, H-NS, H+R, H-NU+R
│   ├── evaluate.py               # Evaluates frozen policies on 200 held-out seasons (seeds 100000..100199)
│   ├── plot_paper_figures.py     # Reproduces summary plots from results CSVs
│   └── run_all.sh                # Full end-to-end reproduction script
├── data/                         # ERA5 weather archive destination
├── checkpoints/                  # Trained NumPy PPO checkpoints (*.npz)
├── results/
│   └── paper_summary_metrics.csv # Published summary metrics (Table 9)
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download real ERA5 weather records (2000-2023)
python scripts/download_era5.py

# 3. Tune rain-threshold baseline R on training-pool seeds 0..63
python scripts/tune_rain_rule.py

# 4. Train a single seed of the hierarchical orchestrator (H) or UAV-free ablation (H-NU)
python scripts/train_ppo.py --method H    --seed 0 --iters 500 --envs 64
python scripts/train_ppo.py --method H-NU --seed 0 --iters 500 --envs 64

# 5. Evaluate all frozen policies on the 200 held-out test seasons (seeds 100000..100199)
python scripts/evaluate.py --seasons 200 --out results/raw_test_seasons.csv

# 6. Reproduce summary figures
python scripts/plot_paper_figures.py
```

To run the full multi-seed benchmark (10 seeds for `H` and `H-NU`, 3 seeds for `F`, `F+S`, `H-LN`, and `H-NS`), run `bash scripts/run_all.sh`.

## Headline Results (200 Held-Out ERA5 Seasons, 2015–2023)

| Method | Burden (pp) | Copper (sprays/sector) | UAV Flights | Latency (days) | Violations | Composite Cost *J* [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| No action | 13.55 | 0.00 | 0.0 | — | 0.00 | 13.55 |
| **C** (Fixed calendar) | 6.25 | 7.27 | 0.0 | 176.1 | 0.00 | 12.79 [12.24, 13.38] |
| **R** (Tuned rain rule) | 6.78 | 5.22 | 0.0 | 171.0 | 0.00 | 11.47 [10.81, 12.14] |
| **F** (Flat PPO) | 13.29 | 0.20 | 0.6 | 165.7 | 0.52 | 13.76 [12.65, 14.88] |
| **F+S** (Flat PPO + Shield) | 11.73 | 1.41 | 79.9 | 140.6 | 0.00 | 18.31 [17.31, 19.42] |
| **H** (Hierarchical PPO) | 8.92 | 2.40 | 4.7 | 149.8 | 0.00 | 11.35 [10.58, 12.13] |
| **H-NU** (No-UAV ablation) | 7.90 | 2.43 | 0.0 | 171.1 | 0.00 | **10.57 [9.85, 11.27]** |

## Citation

If you use this codebase, please cite the accompanying manuscript (see `CITATION.cff`).

## License

Code is released under the [MIT License](LICENSE). ERA5 weather records retrieved via Open-Meteo are licensed under CC BY 4.0.
