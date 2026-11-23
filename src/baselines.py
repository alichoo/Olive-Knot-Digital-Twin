"""Deterministic baselines: No-Action, Fixed Calendar (C), and Tuned Rain-Threshold Rule (R)."""

from __future__ import annotations
from typing import Tuple
import numpy as np
from .twin import OliveKnotTwin


def no_action_policy(env: OliveKnotTwin) -> Tuple[int, int]:
    return 0, 0


def calendar_policy(env: OliveKnotTwin) -> Tuple[int, int]:
    """Fixed calendar (C): from day 30 onward, spray feasible sector with >=30 d since last spray."""
    if env.t < 30:
        return 0, 0
    eligible = env.feas_spray & (env.days_since_spray >= 30)
    if not np.any(eligible):
        return 0, 0
    s = int(np.argmax(np.where(eligible, env.days_since_spray, -1)))
    return 0, 1 + s


def rain_rule_policy(env: OliveKnotTwin, theta_mm: float = 2.0, w_min: float = 0.10) -> Tuple[int, int]:
    """Tuned rain-threshold rule (R): spray feasible sector maximizing O_s*(1 - p_s) when 3-day rain > theta."""
    r3 = float(env.rain_obs_hist[max(0, env.t - 2) : env.t + 1].sum())
    if r3 <= theta_mm or not np.any(env.feas_spray):
        return 0, 0
    scores = env.O_logged * (1.0 - env.p_track)
    masked = np.where(env.feas_spray, scores, -1.0)
    s = int(np.argmax(masked))
    if scores[s] > w_min:
        return 0, 1 + s
    return 0, 0
