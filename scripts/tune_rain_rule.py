"""Grid-search tuning for the rain-threshold baseline R on seeds 0..63 (training pool)."""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.weather import WeatherPool
from src.twin import OliveKnotTwin
from src.baselines import rain_rule_policy


def main():
    with open("config/twin_params.json") as f:
        params = json.load(f)
    pool = WeatherPool("data/era5", allow_synthetic_fallback=True)
    env = OliveKnotTwin(params, pool)

    best = (float("inf"), None, None)
    for theta in [2.0, 5.0, 10.0, 20.0]:
        for w_min in [0.05, 0.10, 0.20, 0.30]:
            costs = []
            for seed in range(64):
                env.reset(seed)
                done = False
                while not done:
                    c, u = rain_rule_policy(env, theta_mm=theta, w_min=w_min)
                    _, _, done, info = env.step(c, u, use_shield=True)
                costs.append(info["cost_J"])
            mean_j = float(np.mean(costs))
            print(f"theta={theta:4.1f} mm, w_min={w_min:.2f} -> mean J = {mean_j:.3f}")
            if mean_j < best[0]:
                best = (mean_j, theta, w_min)
    print(f"Winner: theta={best[1]} mm, w_min={best[2]} (J={best[0]:.3f})")


if __name__ == "__main__":
    main()
