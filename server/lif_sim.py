"""
A plain-numpy leaky integrate-and-fire (LIF) simulation running on top of
the real connectome subgraph. No compiled backend (no Brian2/NEST) needed --
this keeps the project dependency-light and fully portable, at the cost of
being a simplified model of what's actually a much richer biological system.

Model, per neuron i, per timestep:
    v[i] += (-v[i] / tau) * dt + synaptic input + stimulus + noise
    if v[i] >= threshold: spike, v[i] = reset, refractory for a few steps

Synaptic input: last step's spikes, propagated along the real connection
weights. Each neuron's incoming weights are normalised to sum to 1, so a
neuron whose *entire* input population fired last step receives exactly
`synaptic_gain`. That keeps hub neurons with thousands of synapses from
saturating the network while still letting weakly-connected neurons
respond. Presynaptic neurons predicted to be GABAergic / glutamatergic
(see NT_SIGN in scripts/02_build_subgraph.py) contribute negative input.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import sparse

# Neurons whose class/type text matches one of these are treated as the
# "sensory" population that external stimulus (webcam / slider) drives.
SENSORY_KEYWORDS = ("visual", "optic", "sensory", "photoreceptor")


class FlyBrainSim:
    def __init__(self, subgraph_dir: pathlib.Path, dt: float = 1.0, seed: int | None = 0):
        W_raw: sparse.csr_matrix = sparse.load_npz(subgraph_dir / "adjacency.npz").tocsr()
        with open(subgraph_dir / "neurons.json") as f:
            self.neurons: list[dict] = json.load(f)

        self.n = W_raw.shape[0]
        if len(self.neurons) != self.n:
            raise ValueError(f"neurons.json has {len(self.neurons)} entries but adjacency is {self.n}x{self.n}")
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # ---- tunables ----------------------------------------------------
        self.tau = 20.0            # membrane time constant (steps)
        self.threshold = 1.0
        self.reset_v = 0.0
        self.refractory_steps = 3
        self.synaptic_gain = 4.0   # input when 100% of a neuron's inputs fired last step
        self.stimulus_gain = 0.25  # per-step drive into sensory neurons at stimulus=1
        self.noise_std = 0.1       # per-step Gaussian background noise (rare spontaneous spikes)

        # ---- weights: excitatory/inhibitory sign, then input-normalise -----
        self.sign = np.array([n.get("sign", 1) for n in self.neurons], dtype=np.float32)
        self.n_inhibitory = int((self.sign < 0).sum())
        self.W_raw = W_raw

        in_total = np.asarray(W_raw.sum(axis=0)).ravel()          # per postsynaptic neuron
        in_scale = np.where(in_total > 0, 1.0 / np.maximum(in_total, 1e-9), 0.0).astype(np.float32)
        signed = sparse.diags(self.sign) @ W_raw                  # rows = presynaptic
        self.W: sparse.csr_matrix = (signed @ sparse.diags(in_scale)).tocsr()
        self.W.data = self.W.data.astype(np.float32)

        # ---- state ---------------------------------------------------------
        self.v = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.spikes = np.zeros(self.n, dtype=bool)
        self.step_count = 0

        # ---- sensory population ---------------------------------------------
        self.sensory_idx = self._pick_sensory()
        self.n_sensory = len(self.sensory_idx)
        self.is_sensory = np.zeros(self.n, dtype=bool)
        self.is_sensory[self.sensory_idx] = True

    def _pick_sensory(self) -> np.ndarray:
        """Up to 10% of the network, preferring neurons whose class/type looks sensory.

        Rows are ordered by connection weight, so taking matches in row order
        keeps the best-connected sensory cells. Falls back to the first 10% of
        rows when the metadata carries no usable class labels.
        """
        cap = max(1, self.n // 10)
        matches = [
            i for i, m in enumerate(self.neurons)
            if any(k in f"{m.get('class', '')} {m.get('type', '')}".lower() for k in SENSORY_KEYWORDS)
        ]
        if len(matches) >= 5:
            return np.array(matches[:cap], dtype=np.int64)
        return np.arange(cap, dtype=np.int64)

    # ---- API -------------------------------------------------------------
    def inject_stimulus(self, intensity: float) -> None:
        """intensity in [0, 1] -- e.g. webcam motion, audio level, a slider."""
        self.v[self.sensory_idx] += float(intensity) * self.stimulus_gain

    def step(self) -> dict:
        # Leak toward rest.
        self.v += (-self.v / self.tau) * self.dt

        # Propagate last step's spikes as (signed, normalised) synaptic current.
        if self.spikes.any():
            incoming = self.spikes.astype(np.float32) @ self.W
            self.v += np.asarray(incoming).ravel() * self.synaptic_gain

        # Background noise keeps the network from being perfectly silent.
        if self.noise_std > 0:
            self.v += self.rng.normal(0.0, self.noise_std, self.n).astype(np.float32)

        # Refractory neurons are clamped and can't build voltage.
        refractory_mask = self.refractory > 0
        self.v[refractory_mask] = self.reset_v
        self.refractory[refractory_mask] -= 1

        # Voltage can't go (much) below rest -- inhibition saturates.
        np.maximum(self.v, -self.threshold, out=self.v)

        # Spike + reset.
        self.spikes = (self.v >= self.threshold) & ~refractory_mask
        self.v[self.spikes] = self.reset_v
        self.refractory[self.spikes] = self.refractory_steps
        self.step_count += 1

        return {
            "spikes": np.nonzero(self.spikes)[0].tolist(),
            "mean_v": float(self.v.mean()),
        }

    def reset(self) -> None:
        self.v[:] = 0.0
        self.refractory[:] = 0
        self.spikes[:] = False
        self.step_count = 0
