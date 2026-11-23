"""Deterministic risk estimator, UAV belief tracker, agronomic skills, and reactive safety shield."""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def build_grid_adjacency(rows: int = 2, cols: int = 3) -> np.ndarray:
    """Row-normalized 4-neighbor adjacency matrix A for an S = rows * cols orchard grid."""
    S = rows * cols
    A = np.zeros((S, S), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    A[i, nr * cols + nc] = 1.0
    row_sums = A.sum(axis=1, keepdims=True)
    return np.divide(A, np.maximum(row_sums, 1.0))


def feasible_masks(
    r_obs: float,
    f1: float,
    f2: float,
    days_since_spray: np.ndarray,
    spray_count: np.ndarray,
    days_since_prune: np.ndarray,
    prune_count: np.ndarray,
    total_prunes: int,
    flights_7d: int,
    rules: Dict,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Evaluate feasibility predicates for flights, per-sector sprays, and per-sector prunings."""
    feas_fly = (r_obs <= 0.1) and (flights_7d < rules["F_w"])
    feas_spray = (
        (r_obs <= rules["r_blk"])
        & (days_since_spray >= rules["g_s"])
        & (spray_count < rules["K_s"])
    )
    dry_forecast = (r_obs <= 0.1) and (f1 < 0.5) and (f2 < 0.5)
    feas_prune = (
        dry_forecast
        & (days_since_prune >= rules["g_p"])
        & (prune_count < rules["K_p"])
        & (total_prunes < rules["K_tot"])
    )
    return bool(feas_fly), feas_spray, feas_prune


def apply_skills(
    macro_action: int,
    risks: np.ndarray,
    beliefs: np.ndarray,
    belief_ages: np.ndarray,
    days_since_prune: np.ndarray,
    A: np.ndarray,
    feas_spray: np.ndarray,
    feas_prune: np.ndarray,
    params: Dict,
    use_uav: bool = True,
    rain_rule_action: Tuple[int, int] | None = None,
) -> Tuple[int, int]:
    """
    Map macro-action m_t to proposed (c_t, u_t).
    Macro-actions (use_uav=True):
      0: idle, 1: protect, 2: scout, 3: sanitize, 4: protect+scout, 5: sanitize+scout, [6: rain_skill]
    Macro-actions (use_uav=False):
      0: idle, 1: protect, 2: sanitize, [3: rain_skill]
    Treatment encoding u_t:
      0 = none, 1..S = spray sector (u_t - 1), S+1..2S = prune sector (u_t - S - 1).
    """
    S = len(risks)
    sk = params["skills"]
    nu = params["epidemic"]["nu"]

    if not use_uav:
        if macro_action == 3 and rain_rule_action is not None:
            return rain_rule_action
        inc_protect = (macro_action == 1)
        inc_sanitize = (macro_action == 2)
        inc_scout = False
    else:
        if macro_action == 6 and rain_rule_action is not None:
            return rain_rule_action
        inc_protect = macro_action in (1, 4)
        inc_sanitize = macro_action in (3, 5)
        inc_scout = macro_action in (2, 4, 5)

    c_t = 0
    u_t = 0

    if inc_protect and np.any(feas_spray):
        masked_risk = np.where(feas_spray, risks, -1.0)
        sp = int(np.argmax(masked_risk))
        if risks[sp] > sk["eps_r"]:
            u_t = 1 + sp
    elif inc_sanitize and np.any(feas_prune):
        if use_uav:
            masked_b = np.where(feas_prune, beliefs, -1.0)
            sz = int(np.argmax(masked_b))
            if beliefs[sz] >= sk["theta_b"]:
                u_t = 1 + S + sz
        else:
            # Blind round-robin over feasible sectors longest unpruned (H-NU)
            masked_days = np.where(feas_prune, days_since_prune, -1.0)
            sz = int(np.argmax(masked_days))
            u_t = 1 + S + sz

    if inc_scout:
        inoc = beliefs + nu * (A @ beliefs)
        prio = np.minimum(belief_ages / float(sk["tau_max"]), 1.0) * (sk["eta_0"] + inoc)
        c_t = 1 + int(np.argmax(prio))

    return c_t, u_t


def apply_shield(
    c_t: int,
    u_t: int,
    S: int,
    feas_fly: bool,
    feas_spray: np.ndarray,
    feas_prune: np.ndarray,
) -> Tuple[int, int, int]:
    """Project proposed (c_t, u_t) onto feasible set A_feas(o_t); return executed action and interventions."""
    c_exe, u_exe, inter = c_t, u_t, 0
    if c_t > 0 and not feas_fly:
        c_exe = 0
        inter += 1
    if 1 <= u_t <= S:
        if not feas_spray[u_t - 1]:
            u_exe = 0
            inter += 1
    elif S + 1 <= u_t <= 2 * S:
        if not feas_prune[u_t - S - 1]:
            u_exe = 0
            inter += 1
    return c_exe, u_exe, inter
