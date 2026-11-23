"""Self-contained NumPy PPO with 2-layer tanh MLP, manual backprop, GAE, and Adam."""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class NumPyPPOAgent:
    """Two-hidden-layer tanh MLP actor-critic (8,519 params for 60->64->64->6+1)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64, seed: int = 0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        def glorot(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, size=(fan_in, fan_out))

        self.params = {
            "W1": glorot(obs_dim, hidden_dim), "b1": np.zeros(hidden_dim),
            "W2": glorot(hidden_dim, hidden_dim), "b2": np.zeros(hidden_dim),
            "Wp": glorot(hidden_dim, act_dim), "bp": np.zeros(act_dim),
            "Wv": glorot(hidden_dim, 1), "bv": np.zeros(1),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.adam_t = 0

    def param_count(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        h1 = np.tanh(obs @ self.params["W1"] + self.params["b1"])
        h2 = np.tanh(h1 @ self.params["W2"] + self.params["b2"])
        logits = h2 @ self.params["Wp"] + self.params["bp"]
        probs = _softmax(logits)
        value = (h2 @ self.params["Wv"] + self.params["bv"]).squeeze(-1)
        return probs, value, {"obs": obs, "h1": h1, "h2": h2, "probs": probs}

    def act(self, obs: np.ndarray, rng: np.random.Generator | None = None, greedy: bool = False) -> Tuple[int, float, float]:
        probs, val, _ = self.forward(obs[None, :])
        p = probs[0]
        if greedy or rng is None:
            a = int(np.argmax(p))
        else:
            a = int(rng.choice(self.act_dim, p=p))
        logp = float(np.log(max(p[a], 1e-12)))
        return a, logp, float(val[0])

    def update(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        old_logp: np.ndarray,
        adv: np.ndarray,
        ret: np.ndarray,
        clip_eps: float = 0.2,
        c_v: float = 0.5,
        c_e: float = 0.01,
        lr: float = 3e-4,
        epochs: int = 4,
        minibatch_size: int = 512,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, float]:
        N = len(obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if rng is None:
            rng = np.random.default_rng(0)

        last_loss = 0.0
        for _ in range(epochs):
            perm = rng.permutation(N)
            for start in range(0, N, minibatch_size):
                idx = perm[start : start + minibatch_size]
                ob, ac, olp, ad, rt = obs[idx], acts[idx], old_logp[idx], adv[idx], ret[idx]
                B = len(idx)

                probs, vals, cache = self.forward(ob)
                p_a = np.maximum(probs[np.arange(B), ac], 1e-12)
                logp = np.log(p_a)
                ratio = np.exp(logp - olp)
                unclipped = ratio * ad
                clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * ad
                surr = np.minimum(unclipped, clipped)

                # Gradient of -surr wrt logits
                use_unclipped = (unclipped <= clipped).astype(np.float64)
                d_logp = -(use_unclipped * ratio * ad) / B
                # Entropy bonus gradient: maximize -sum p log p -> minimize sum p log p
                log_probs = np.log(np.maximum(probs, 1e-12))
                ent = -np.sum(probs * log_probs, axis=1)
                d_ent_logits = (probs * (log_probs + ent[:, None])) / B

                d_logits = np.zeros_like(probs)
                d_logits[np.arange(B), ac] = d_logp
                d_logits = d_logits - probs * d_logp[:, None] + c_e * d_ent_logits

                # Value loss gradient: c_v * mean((vals - rt)^2)
                d_val = (2.0 * c_v * (vals - rt) / B)[:, None]

                # Backprop through heads
                h2 = cache["h2"]
                h1 = cache["h1"]
                grads = {}
                grads["Wp"] = h2.T @ d_logits
                grads["bp"] = d_logits.sum(axis=0)
                grads["Wv"] = h2.T @ d_val
                grads["bv"] = d_val.sum(axis=0)

                dh2 = d_logits @ self.params["Wp"].T + d_val @ self.params["Wv"].T
                dz2 = dh2 * (1.0 - h2 ** 2)
                grads["W2"] = h1.T @ dz2
                grads["b2"] = dz2.sum(axis=0)

                dh1 = dz2 @ self.params["W2"].T
                dz1 = dh1 * (1.0 - h1 ** 2)
                grads["W1"] = ob.T @ dz1
                grads["b1"] = dz1.sum(axis=0)

                # Global norm clip (0.5) + Adam update
                gnorm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
                scale = min(1.0, 0.5 / (gnorm + 1e-8))
                self.adam_t += 1
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                for k, g in grads.items():
                    g = g * scale
                    self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * g
                    self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (g ** 2)
                    m_hat = self.m[k] / (1.0 - beta1 ** self.adam_t)
                    v_hat = self.v[k] / (1.0 - beta2 ** self.adam_t)
                    self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

                last_loss = float(-surr.mean() + c_v * np.mean((vals - rt) ** 2) - c_e * ent.mean())
        return {"loss": last_loss}

    def save(self, path: str) -> None:
        np.savez(path, obs_dim=self.obs_dim, act_dim=self.act_dim, hidden_dim=self.hidden_dim, **self.params)

    @classmethod
    def load(cls, path: str) -> "NumPyPPOAgent":
        data = np.load(path)
        agent = cls(int(data["obs_dim"]), int(data["act_dim"]), int(data["hidden_dim"]))
        for k in agent.params:
            agent.params[k] = data[k]
        return agent


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.999, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + values
