# ERA5 Weather Data

Historical hourly precipitation, hourly relative humidity, daily precipitation sums, and daily minimum temperatures for four Tunisian olive regions (**Sfax**, **Medenine**, **Kairouan**, and **Beja**; 1 October to 31 May, 2000–2023; 96 region-seasons) are sourced from the public **Open-Meteo Historical Weather API** (ECMWF ERA5 reanalysis, CC BY 4.0).

To download and package the dataset locally as `data/era5/era5_tunisia_2000_2023.npz`, run:

```bash
python scripts/download_era5.py
```

- **Training / tuning pool:** 2000–2014 (60 traces; 4 regions × 15 seasons)
- **Held-out test pool:** 2015–2023 (36 traces; 4 regions × 9 seasons, mapped deterministically to test seeds `100000–100199`)
