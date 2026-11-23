"""ERA5 weather loader with strict train/test pool separation (2000-2014 vs. 2015-2023)."""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

REGIONS = ["Sfax", "Medenine", "Kairouan", "Beja"]


@dataclass
class WeatherTrace:
    region: str
    season_start_year: int
    rain_mm: np.ndarray        # shape (243,)
    wetness_h: np.ndarray      # shape (243,)
    tmin_c: np.ndarray         # shape (243,)


class WeatherPool:
    """Manages real ERA5 traces split into training (60 traces) and test (36 traces) pools."""

    def __init__(self, data_dir: str = "data/era5", allow_synthetic_fallback: bool = True):
        self.data_dir = data_dir
        self.train_traces: List[WeatherTrace] = []
        self.test_traces: List[WeatherTrace] = []
        self._load_or_fallback(allow_synthetic_fallback)

    def _load_or_fallback(self, allow_fallback: bool) -> None:
        npz_path = os.path.join(self.data_dir, "era5_tunisia_2000_2023.npz")
        if os.path.exists(npz_path):
            arr = np.load(npz_path, allow_pickle=True)
            regions = arr["regions"]
            years = arr["years"]
            rain = arr["rain_mm"]
            wet = arr["wetness_h"]
            tmin = arr["tmin_c"]
            for i in range(len(years)):
                tr = WeatherTrace(
                    region=str(regions[i]),
                    season_start_year=int(years[i]),
                    rain_mm=rain[i].astype(np.float64),
                    wetness_h=wet[i].astype(np.float64),
                    tmin_c=tmin[i].astype(np.float64),
                )
                if tr.season_start_year <= 2014:
                    self.train_traces.append(tr)
                else:
                    self.test_traces.append(tr)
            return

        if not allow_fallback:
            raise FileNotFoundError(
                f"ERA5 dataset not found at {npz_path}. Run `python scripts/download_era5.py` first."
            )

        # Deterministic offline fallback matching Table 3 regional rainfall/wetness statistics
        # (used only for offline smoke testing when ERA5 has not yet been downloaded).
        rng = np.random.default_rng(2026)
        profiles = {
            "Sfax":     (145.0, 27.5, 3.1, 0.0),
            "Medenine": (108.0, 19.5, 2.6, 0.0),
            "Kairouan": (213.0, 36.0, 4.0, 0.6),
            "Beja":     (420.0, 72.0, 9.2, 1.5),
        }
        for yr in range(2000, 2024):
            for reg in REGIONS:
                mean_rain, rainy_days, mean_wet, frost_days = profiles[reg]
                p_rain = np.clip(rainy_days / 243.0, 0.02, 0.6)
                is_wet = rng.random(243) < p_rain
                rain_mm = np.where(is_wet, rng.exponential(mean_rain / max(rainy_days, 1.0), 243), 0.0)
                wet_h = np.clip(rng.normal(mean_wet, 2.5, 243) + (rain_mm > 0.1) * 4.0, 0.0, 24.0)
                tmin_c = rng.normal(9.5, 4.0, 243)
                if frost_days > 0:
                    f_idx = rng.choice(np.arange(70, 131), size=max(1, int(rng.poisson(frost_days))), replace=False)
                    tmin_c[f_idx] = -1.0
                tr = WeatherTrace(reg, yr, rain_mm, wet_h, tmin_c)
                if yr <= 2014:
                    self.train_traces.append(tr)
                else:
                    self.test_traces.append(tr)

    def select_trace(self, seed: int) -> WeatherTrace:
        """Strict seed-to-pool mapping: 100000..199999 -> test pool (mod 36); others -> train pool (mod 60)."""
        if 100000 <= seed <= 199999:
            return self.test_traces[seed % len(self.test_traces)]
        return self.train_traces[seed % len(self.train_traces)]
