"""Six-sector Olive Knot Orchard Digital Twin (POMDP Equations 1-5, Algorithm 1)."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from .weather import WeatherPool
from .skills_shield import build_grid_adjacency, feasible_masks, apply_shield


class OliveKnotTwin:
    """Single-season reproducible digital twin for olive knot management."""

    def __init__(self, params: Dict, weather_pool: WeatherPool, overrides: Optional[Dict] = None):
        self.params = json.loads(json.dumps(params)) if "json" in globals() else _deep_copy(params)
        if overrides:
            for k, v in overrides.items():
                for grp in self.params.values():
                    if isinstance(grp, dict) and k in grp:
                        grp[k] = v
        self.pool = weather_pool
        self.S = int(self.params["orchard"]["S"])
        self.n = int(self.params["orchard"]["n_trees"])
        self.T = int(self.params["orchard"]["T"])
        self.A = build_grid_adjacency(self.params["orchard"]["rows"], self.params["orchard"]["cols"])
        self.reset(seed=100000)

    def reset(self, seed: int) -> np.ndarray:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.trace = self.pool.select_trace(self.seed)
        self.t = 0

        ep = self.params["epidemic"]
        # Initial uneven infection across sectors (m hot sectors)
        hot_sectors = self.rng.choice(self.S, size=ep["m_hot"], replace=False)
        rho = np.full(self.S, ep["rho_0"], dtype=np.float64)
        rho[hot_sectors] = self.rng.uniform(ep["rho_lo"], ep["rho_hi"], size=ep["m_hot"])
        self.V = self.rng.binomial(self.n, rho).astype(np.int64)
        self.B0 = float(self.V.sum()) / (self.S * self.n)

        # Latent infection ring buffer of length Lambda
        self.Lambda = int(ep["Lambda"])
        self.latent_hist = np.zeros((self.T + self.Lambda + 2, self.S), dtype=np.int64)
        self.p_true = np.zeros(self.S, dtype=np.float64)

        # Pre-generate daily wound events (leaf scars + harvest + frost + hail)
        self.wounds_true = np.full((self.T, self.S), ep["lambda_leaf"], dtype=np.float64)
        for s in range(self.S):
            h_day = int(self.rng.integers(30, 91))
            self.wounds_true[h_day, s] += ep["harvest_wound"]
        for d in range(70, min(131, self.T)):
            if self.trace.tmin_c[d] <= 0.0:
                self.wounds_true[d, :] += self.rng.uniform(0.05, ep["frost_wound_max"])
        hail_day = int(self.rng.integers(40, 200))
        hail_secs = self.rng.choice(self.S, size=2, replace=False)
        self.wounds_true[hail_day, hail_secs] += ep["hail_wound"]

        # Deterministic front-end trackers
        self.belief = np.zeros(self.S, dtype=np.float64)
        self.belief_age = np.full(self.S, 30.0, dtype=np.float64)
        self.p_track = np.zeros(self.S, dtype=np.float64)
        self.days_since_spray = np.full(self.S, 30, dtype=np.int64)
        self.spray_count = np.zeros(self.S, dtype=np.int64)
        self.days_since_prune = np.full(self.S, 21, dtype=np.int64)
        self.prune_count = np.zeros(self.S, dtype=np.int64)
        self.total_prunes = 0
        self.flight_hist = np.zeros(self.T, dtype=np.int64)
        self.rain_obs_hist = np.zeros(self.T, dtype=np.float64)

        # Detection latency trackers
        self.onset_day = np.full(self.S, -1, dtype=np.int64)
        self.detect_day = np.full(self.S, -1, dtype=np.int64)
        self._update_onset()

        # Running season metrics
        self.total_sprays = 0
        self.total_flights = 0
        self.total_violations = 0
        self.total_interventions = 0
        self.decision_log: List[Dict] = []

        self._prepare_day_obs()
        return self.obs.copy()

    def _update_onset(self) -> None:
        theta_det = self.params["perception"]["theta_det"]
        x = self.V / float(self.n)
        newly = (self.onset_day < 0) & (x >= theta_det)
        self.onset_day[newly] = self.t

    def _prepare_day_obs(self) -> None:
        pc = self.params["perception"]
        ep = self.params["epidemic"]
        r_true = float(self.trace.rain_mm[self.t])
        self.r_obs = r_true * float(np.exp(self.rng.normal(0.0, pc["sigma_r"])))
        self.rain_obs_hist[self.t] = self.r_obs

        def _noisy_fc(offset: int) -> float:
            d = min(self.t + offset, self.T - 1)
            wet_ind = 1.0 if (self.trace.rain_mm[d] >= 1.0 and self.trace.wetness_h[d] >= ep["l_min"]) else 0.0
            return float(np.clip(0.2 + 0.55 * wet_ind + self.rng.normal(0.0, pc["sigma_f"]), 0.0, 1.0))

        self.f1 = _noisy_fc(1)
        self.f2 = _noisy_fc(2)

        W0 = int(ep["W0_nominal"])
        start = max(0, self.t - W0 + 1)
        O_nom = np.minimum(1.0, self.wounds_true[start : self.t + 1].sum(axis=0))
        self.O_logged = np.clip(O_nom * (1.0 + self.rng.normal(0.0, pc["sigma_O"], size=self.S)), 0.0, 1.0)

        self.omega_eff = max(
            1.0 - np.exp(-self.r_obs / ep["r_ref"]),
            pc["phi_f"] * max(self.f1, self.f2),
        )
        inoc_est = self.belief + ep["nu"] * (self.A @ self.belief)
        self.risk_est = 1.0 - np.exp(
            -ep["beta"] * self.omega_eff * self.O_logged * (1.0 + ep["gamma_c"] * inoc_est) * (1.0 - self.p_track)
        )

        fl_start = max(0, self.t - 7)
        self.flights_7d = int(self.flight_hist[fl_start : self.t].sum())

        self.feas_fly, self.feas_spray, self.feas_prune = feasible_masks(
            self.r_obs, self.f1, self.f2,
            self.days_since_spray, self.spray_count,
            self.days_since_prune, self.prune_count,
            self.total_prunes, self.flights_7d, self.params["rules"],
        )

        # Build 60-D observation vector (9S + 6)
        per_sec = np.stack([
            np.clip(self.risk_est * 20.0, 0.0, 1.0),
            np.clip(self.belief * 10.0, 0.0, 1.0),
            np.clip(self.belief_age / 30.0, 0.0, 1.0),
            self.p_track,
            np.clip(self.days_since_spray / 30.0, 0.0, 1.0),
            self.spray_count / float(self.params["rules"]["K_s"]),
            self.O_logged,
            np.clip(self.days_since_prune / 21.0, 0.0, 1.0),
            self.prune_count / float(self.params["rules"]["K_p"]),
        ], axis=1).reshape(-1)

        glob = np.array([
            min(self.r_obs / 10.0, 1.0),
            self.f1,
            self.f2,
            self.t / float(self.T),
            self.flights_7d / float(self.params["rules"]["F_w"]),
            self.omega_eff,
        ], dtype=np.float64)
        self.obs = np.concatenate([per_sec, glob])

    def _current_burden(self) -> float:
        L = self.latent_hist[max(0, self.t - self.Lambda + 1) : self.t + 1].sum(axis=0)
        return float((L + self.V).sum()) / (self.S * self.n)

    def step(self, c_prop: int, u_prop: int, use_shield: bool = True) -> Tuple[np.ndarray, float, bool, Dict]:
        ep = self.params["epidemic"]
        cp = self.params["copper"]
        pc = self.params["perception"]
        pr = self.params["pruning"]
        w = self.params["weights"]

        B_before = self._current_burden()

        # Check feasibility / apply safety shield
        if use_shield:
            c_exe, u_exe, inter = apply_shield(
                c_prop, u_prop, self.S, self.feas_fly, self.feas_spray, self.feas_prune
            )
            viol = 0
        else:
            _, _, viol = apply_shield(
                c_prop, u_prop, self.S, self.feas_fly, self.feas_spray, self.feas_prune
            )
            c_exe, u_exe, inter = c_prop, u_prop, 0

        self.total_interventions += inter
        self.total_violations += viol

        # Execute treatment (spray or sanitation prune)
        did_spray = 0
        did_prune = 0
        C_removed = np.zeros(self.S, dtype=np.int64)
        r_true = float(self.trace.rain_mm[self.t])

        if 1 <= u_exe <= self.S:
            s = u_exe - 1
            did_spray = 1
            phi = cp["phi_washoff"] if r_true > self.params["rules"]["r_blk"] else 1.0
            self.p_true[s] = max(self.p_true[s], cp["p_max"] * phi)
            self.p_track[s] = max(self.p_track[s], cp["p_max"] * (1.0 if self.r_obs <= self.params["rules"]["r_blk"] else cp["phi_washoff"]))
            self.days_since_spray[s] = 0
            self.spray_count[s] += 1
            self.total_sprays += 1
        elif self.S + 1 <= u_exe <= 2 * self.S:
            s = u_exe - self.S - 1
            did_prune = 1
            C_removed[s] = int(self.rng.binomial(self.V[s], pr["eta"]))
            self.wounds_true[self.t, s] += pr["w_p"]
            self.days_since_prune[s] = 0
            self.prune_count[s] += 1
            self.total_prunes += 1

        # Epidemic transition (Equations 1-4)
        W = int(ep["W"])
        O_true = np.minimum(1.0, self.wounds_true[max(0, self.t - W + 1) : self.t + 1].sum(axis=0))
        omega_true = (1.0 - np.exp(-r_true / ep["r_ref"])) * (1.0 if self.trace.wetness_h[self.t] >= ep["l_min"] else 0.0)
        x = self.V / float(self.n)
        inoc = x + ep["nu"] * (self.A @ x)
        h = ep["beta"] * omega_true * O_true * (1.0 + ep["gamma_c"] * inoc) * (1.0 - self.p_true)

        L_curr = self.latent_hist[max(0, self.t - self.Lambda + 1) : self.t + 1].sum(axis=0)
        H_trees = np.maximum(0, self.n - L_curr - self.V)
        N_new = self.rng.binomial(H_trees, np.clip(1.0 - np.exp(-h), 0.0, 1.0))
        self.latent_hist[self.t] = N_new

        emerging = self.latent_hist[self.t - self.Lambda + 1] if (self.t - self.Lambda + 1) >= 0 else 0
        self.V = np.clip(self.V + emerging - C_removed, 0, self.n)

        # Copper decay
        self.p_true *= np.exp(-(cp["delta_T"] + cp["delta_R"] * r_true))
        self.p_track *= np.exp(-(cp["delta_T"] + cp["delta_R"] * self.r_obs))

        # UAV flight perception update (available for day t+1)
        did_flight = 0
        if 1 <= c_exe <= self.S:
            did_flight = 1
            sc = c_exe - 1
            self.flight_hist[self.t] = 1
            self.total_flights += 1
            y = pc["kappa"] * (self.V[sc] / float(self.n)) + self.rng.normal(0.0, pc["sigma_n"])
            self.belief[sc] = float(np.clip(y / pc["kappa"], 0.0, 1.0))
            self.belief_age[sc] = 0.0
            if self.onset_day[sc] >= 0 and self.detect_day[sc] < 0 and self.belief[sc] >= pc["theta_det"]:
                self.detect_day[sc] = self.t
            for s in range(self.S):
                if s != sc:
                    self.belief_age[s] += 1.0
        else:
            self.belief_age += 1.0

        # Age spray/prune counters for non-acted sectors
        for s in range(self.S):
            if u_exe != 1 + s:
                self.days_since_spray[s] += 1
            if u_exe != 1 + self.S + s:
                self.days_since_prune[s] += 1

        self.t += 1
        self._update_onset()
        B_after = self._current_burden()

        reward = -(
            w["w_D"] * 100.0 * (B_after - B_before)
            + w["w_C"] * did_spray
            + w["w_E"] * did_flight
            + w["w_P"] * did_prune
            + w["w_V"] * viol
        )

        done = self.t >= self.T
        if not done:
            self._prepare_day_obs()

        info = {
            "done": done,
            "burden_pp": 100.0 * (B_after - self.B0),
            "copper_per_sector": self.total_sprays / float(self.S),
            "flights": self.total_flights,
            "prunes": self.total_prunes,
            "violations": self.total_violations,
            "interventions": self.total_interventions,
            "region": self.trace.region,
        }
        if done:
            on_mask = self.onset_day >= 0
            if np.any(on_mask):
                det = np.where(self.detect_day[on_mask] >= 0, self.detect_day[on_mask], self.T)
                info["latency"] = float(np.mean(det - self.onset_day[on_mask]))
            else:
                info["latency"] = float("nan")
            info["cost_J"] = (
                w["w_D"] * info["burden_pp"]
                + w["w_C"] * self.total_sprays
                + w["w_E"] * self.total_flights
                + w["w_P"] * self.total_prunes
                + w["w_V"] * self.total_violations
            )
        return self.obs.copy(), float(reward), done, info


def _deep_copy(obj):
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj
