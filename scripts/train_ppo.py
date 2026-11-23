"""Train hierarchical (H, H-NU, H-LN, H-NS, H+R, H-NU+R) or flat (F, F+S) PPO policies."""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.weather import WeatherPool
from src.twin import OliveKnotTwin
from src.skills_shield import apply_skills
from src.baselines import rain_rule_policy
from src.ppo_numpy import NumPyPPOAgent, compute_gae


def decode_flat_action(act_idx: int, S: int):
    # |A| = (S + 1) * (2S + 1)
    n_u = 2 * S + 1
    return act_idx // n_u, act_idx % n_u


def train_one(method: str, seed: int, n_iters: int, n_envs: int, overrides: dict | None = None):
    with open("config/twin_params.json") as f:
        params = json.load(f)
    pool = WeatherPool("data/era5", allow_synthetic_fallback=True)
    env = OliveKnotTwin(params, pool, overrides=overrides)

    is_flat = method in ("F", "F+S")
    use_shield = method not in ("F", "H-NS")
    use_uav = method not in ("H-NU", "H-NU+R")
    use_rain_skill = method in ("H+R", "H-NU+R")

    obs_dim = 9 * env.S + 6
    if is_flat:
        act_dim = (env.S + 1) * (2 * env.S + 1)
    elif use_uav:
        act_dim = 7 if use_rain_skill else 6
    else:
        act_dim = 4 if use_rain_skill else 3

    agent = NumPyPPOAgent(obs_dim, act_dim, hidden_dim=params["ppo"]["hidden_dim"], seed=seed)
    rng = np.random.default_rng(seed + 1000)
    seed_cursor = params["splits"]["train_seed_offset"] + seed * 100000

    for it in range(1, n_iters + 1):
        obs_buf, act_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], []
        ep_costs = []
        for _ in range(n_envs):
            seed_cursor += 1
            ob = env.reset(seed_cursor)
            ep_obs, ep_acts, ep_logp, ep_rews, ep_vals = [], [], [], [], []
            done = False
            while not done:
                a, lp, val = agent.act(ob, rng=rng, greedy=False)
                if is_flat:
                    c, u = decode_flat_action(a, env.S)
                else:
                    rr = rain_rule_policy(env) if use_rain_skill else None
                    c, u = apply_skills(
                        a, env.risk_est, env.belief, env.belief_age, env.days_since_prune,
                        env.A, env.feas_spray, env.feas_prune, env.params, use_uav=use_uav, rain_rule_action=rr
                    )
                next_ob, rew, done, info = env.step(c, u, use_shield=use_shield)
                ep_obs.append(ob); ep_acts.append(a); ep_logp.append(lp); ep_rews.append(rew); ep_vals.append(val)
                ob = next_ob
            ep_costs.append(info["cost_J"])
            adv, ret = compute_gae(np.array(ep_rews), np.array(ep_vals),
                                   gamma=params["ppo"]["gamma"], lam=params["ppo"]["gae_lambda"])
            obs_buf.append(np.array(ep_obs)); act_buf.append(np.array(ep_acts))
            logp_buf.append(np.array(ep_logp)); adv_buf.append(adv); ret_buf.append(ret)

        agent.update(
            np.concatenate(obs_buf), np.concatenate(act_buf), np.concatenate(logp_buf),
            np.concatenate(adv_buf), np.concatenate(ret_buf),
            clip_eps=params["ppo"]["clip_eps"], c_v=params["ppo"]["c_v"], c_e=params["ppo"]["c_e"],
            lr=params["ppo"]["lr"], epochs=params["ppo"]["epochs"],
            minibatch_size=params["ppo"]["minibatch_size"], rng=rng
        )
        if it == 1 or it % 50 == 0 or it == n_iters:
            print(f"[{method} seed={seed}] iter {it:3d}/{n_iters} | mean train J = {np.mean(ep_costs):.2f}")

    os.makedirs("checkpoints", exist_ok=True)
    out = f"checkpoints/{method}_seed{seed}.npz"
    agent.save(out)
    print(f"Saved checkpoint: {out} ({agent.param_count():,} params)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="H", choices=["H", "H-NU", "F", "F+S", "H-LN", "H-NS", "H+R", "H-NU+R"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--envs", type=int, default=64)
    args = ap.parse_args()
    ov = {"sigma_n": 0.005} if args.method == "H-LN" else None
    train_one(args.method, args.seed, args.iters, args.envs, overrides=ov)
