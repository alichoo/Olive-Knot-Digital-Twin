"""Evaluate frozen baselines and checkpoints on 200 held-out test seasons (seeds 100000-100199)."""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src.weather import WeatherPool
from src.twin import OliveKnotTwin
from src.baselines import no_action_policy, calendar_policy, rain_rule_policy
from src.skills_shield import apply_skills
from src.ppo_numpy import NumPyPPOAgent


def bootstrap_ci(values: np.ndarray, n_resamples: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(vals), size=(n_resamples, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def eval_method(method: str, params: dict, pool: WeatherPool, n_seasons: int = 200, ckpt_paths=None):
    env = OliveKnotTwin(params, pool)
    agents = [NumPyPPOAgent.load(p) for p in ckpt_paths] if ckpt_paths else [None]
    is_flat = method in ("F", "F+S")
    use_shield = method not in ("F", "H-NS")
    use_uav = method not in ("H-NU", "H-NU+R")
    use_rain_skill = method in ("H+R", "H-NU+R")

    rows = []
    for k in range(n_seasons):
        seed = 100000 + k
        seed_metrics = []
        for ag in agents:
            ob = env.reset(seed)
            done = False
            while not done:
                if method == "None":
                    c, u = no_action_policy(env)
                elif method == "C":
                    c, u = calendar_policy(env)
                elif method == "R":
                    c, u = rain_rule_policy(env, theta_mm=2.0, w_min=0.10)
                else:
                    a, _, _ = ag.act(ob, greedy=True)
                    if is_flat:
                        n_u = 2 * env.S + 1
                        c, u = a // n_u, a % n_u
                    else:
                        rr = rain_rule_policy(env) if use_rain_skill else None
                        c, u = apply_skills(
                            a, env.risk_est, env.belief, env.belief_age, env.days_since_prune,
                            env.A, env.feas_spray, env.feas_prune, env.params, use_uav=use_uav, rain_rule_action=rr
                        )
                ob, _, done, info = env.step(c, u, use_shield=use_shield)
            seed_metrics.append(info)
        rows.append({
            "method": method,
            "season_seed": seed,
            "region": seed_metrics[0]["region"],
            "burden_pp": float(np.mean([m["burden_pp"] for m in seed_metrics])),
            "copper_per_sector": float(np.mean([m["copper_per_sector"] for m in seed_metrics])),
            "flights": float(np.mean([m["flights"] for m in seed_metrics])),
            "latency": float(np.nanmean([m["latency"] for m in seed_metrics])),
            "violations": float(np.mean([m["violations"] for m in seed_metrics])),
            "interventions": float(np.mean([m["interventions"] for m in seed_metrics])),
            "cost_J": float(np.mean([m["cost_J"] for m in seed_metrics])),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, default=200)
    ap.add_argument("--out", default="results/raw_test_seasons.csv")
    args = ap.parse_args()

    with open("config/twin_params.json") as f:
        params = json.load(f)
    pool = WeatherPool("data/era5", allow_synthetic_fallback=True)

    dfs = [eval_method(m, params, pool, n_seasons=args.seasons) for m in ["None", "C", "R"]]
    for m in ["H", "H-NU", "F", "F+S"]:
        ckpts = sorted([
            os.path.join("checkpoints", fn) for fn in os.listdir("checkpoints")
            if fn.startswith(f"{m}_seed") and fn.endswith(".npz")
        ]) if os.path.exists("checkpoints") else []
        if ckpts:
            dfs.append(eval_method(m, params, pool, n_seasons=args.seasons, ckpt_paths=ckpts))

    all_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_df.to_csv(args.out, index=False)

    print(f"Saved per-season results to {args.out}\n")
    for m, grp in all_df.groupby("method", sort=False):
        mu, lo, hi = bootstrap_ci(grp["cost_J"].to_numpy())
        print(f"  {m:5s}: J = {mu:6.2f} [{lo:6.2f}, {hi:6.2f}] | D = {grp['burden_pp'].mean():5.2f} | C = {grp['copper_per_sector'].mean():4.2f}")


if __name__ == "__main__":
    main()
