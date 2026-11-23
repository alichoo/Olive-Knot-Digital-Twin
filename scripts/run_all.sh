#!/usr/bin/env bash
set -euo pipefail

# 1. Download real ERA5 weather data (2000-2023) for Sfax, Medenine, Kairouan, and Beja
python scripts/download_era5.py

# 2. Tune rain-threshold baseline R on training-pool seeds 0..63
python scripts/tune_rain_rule.py

# 3. Train 10 seeds of H and H-NU, and 3 seeds of flat baselines F and F+S
for s in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/train_ppo.py --method H    --seed "$s" --iters 500 --envs 64
  python scripts/train_ppo.py --method H-NU --seed "$s" --iters 500 --envs 64
done
for s in 0 1 2; do
  python scripts/train_ppo.py --method F    --seed "$s" --iters 500 --envs 64
  python scripts/train_ppo.py --method F+S  --seed "$s" --iters 500 --envs 64
  python scripts/train_ppo.py --method H-LN --seed "$s" --iters 500 --envs 64
  python scripts/train_ppo.py --method H-NS --seed "$s" --iters 500 --envs 64
done

# 4. Evaluate all frozen policies on 200 held-out test seasons (seeds 100000..100199)
python scripts/evaluate.py --seasons 200 --out results/raw_test_seasons.csv

# 5. Generate summary figures
python scripts/plot_paper_figures.py
