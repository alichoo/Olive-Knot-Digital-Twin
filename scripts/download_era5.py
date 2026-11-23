"""Download ERA5 hourly/daily weather records (2000-2023) for 4 Tunisian olive regions via Open-Meteo."""

import os, json, time
import requests
import numpy as np

REGIONS = {
    "Sfax": (34.74, 10.76),
    "Medenine": (33.35, 10.50),
    "Kairouan": (35.68, 10.10),
    "Beja": (36.73, 9.18),
}


def fetch_season(lat: float, lon: float, start_year: int):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{start_year}-10-01",
        "end_date": f"{start_year + 1}-05-31",
        "hourly": "precipitation,relative_humidity_2m",
        "daily": "precipitation_sum,temperature_2m_min",
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    js = resp.json()
    rain_d = np.array(js["daily"]["precipitation_sum"][:243], dtype=np.float64)
    tmin_d = np.array(js["daily"]["temperature_2m_min"][:243], dtype=np.float64)
    pr_h = np.array(js["hourly"]["precipitation"][: 243 * 24], dtype=np.float64).reshape(243, 24)
    rh_h = np.array(js["hourly"]["relative_humidity_2m"][: 243 * 24], dtype=np.float64).reshape(243, 24)
    wet_d = ((rh_h >= 90.0) | (pr_h >= 0.1)).sum(axis=1).astype(np.float64)
    return rain_d, wet_d, tmin_d


def main():
    os.makedirs("data/era5", exist_ok=True)
    regions_out, years_out, rains, wets, tmins = [], [], [], [], []
    for yr in range(2000, 2024):
        for name, (lat, lon) in REGIONS.items():
            print(f"Downloading {name} {yr}-{yr+1}...")
            r, w, tm = fetch_season(lat, lon, yr)
            regions_out.append(name)
            years_out.append(yr)
            rains.append(r)
            wets.append(w)
            tmins.append(tm)
            time.sleep(0.4)

    out_path = "data/era5/era5_tunisia_2000_2023.npz"
    np.savez_compressed(
        out_path,
        regions=np.array(regions_out),
        years=np.array(years_out, dtype=np.int32),
        rain_mm=np.stack(rains),
        wetness_h=np.stack(wets),
        tmin_c=np.stack(tmins),
    )
    print(f"Saved 96 ERA5 region-seasons to {out_path}")


if __name__ == "__main__":
    main()
