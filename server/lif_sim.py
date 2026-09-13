"""
A plain-numpy leaky integrate-and-fire (LIF) simulation running on top of
the real connectome subgraph. No compiled backend (no Brian2/NEST) needed --
this keeps the project dependency-light and fully portable, at the cost of
being a simplified model of what's actually a much richer biological system.

Model, per neuron i, per timestep:
    v[i] += (-v[i] / tau + input_current[i]) * dt
    if v[i] >= threshold: spike, v[i] = reset, refractory for a few steps
    spikes propagate next step as extra input current, scaled by the
    real synaptic weight and the presynaptic neuron's neurotransmitter
    sign (excitatory/inhibitory) if you choose to encode one -- this
    starter treats all connections as excitatory for simplicity.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import sparse


class FlyBrainSim:
    def __init__(self, subgraph_dir: pathlib.Path, dt: float = 1.0):
        self.W: sparse.csr_matrix = sparse.load_npz(subgraph_dir / "adjacency.npz")
        with open(subgraph_dir / "neurons.json") as f:
            self.neurons: list[dict] = json.load(f)

        self.n = self.W.shape[0]
        self.dt = dt

        # Normalize weights so a handful of very heavy hub connections
        # don't instantly saturate everything -- keeps the sim visually
        # readable rather than one big constant firestorm.
        max_w = self.W.data.max() if self.W.nnz else 1.0
        self.W = self.W.multiply(1.0 / max_w).tocsr()

        self.tau = 20.0          # membrane time constant (ms)
        self.threshold = 1.0
        self.reset_v = 0.0
        self.refractory_steps = 3
        self.synaptic_gain = 0.6  # how strongly incoming spikes drive current

        self.v = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.spikes = np.zeros(self.n, dtype=bool)

        # Which neurons count as "sensory input" -- for the starter, just
        # the first 10% of the subgraph (by weight rank, since 02_build
        # sorted candidates by total connection weight). Swap this for a
        # real visual-system cell-type filter once you're ready.
        self.n_sensory = max(1, self.n // 10)
        self.sensory_idx = np.arange(self.n_sensory)

    def inject_stimulus(self, intensity: float) -> None:
        """intensity in [0, 1] -- e.g. webcam brightness/motion, audio level, etc."""
        self.v[self.sensory_idx] += intensity * 0.8

    def step(self) -> dict:
        # Leak
        self.v += (-self.v / self.tau) * self.dt

        # Propagate last step's spikes as synaptic current.
        if self.spikes.any():
            incoming = self.spikes.astype(np.float32) @ self.W
            self.v += np.asarray(incoming).ravel() * self.synaptic_gain

        # Refractory neurons can't build voltage.
        refractory_mask = self.refractory > 0
        self.v[refractory_mask] = self.reset_v
        self.refractory[refractory_mask] -= 1

        # Spike + reset.
        self.spikes = (self.v >= self.threshold) & ~refractory_mask
        self.v[self.spikes] = self.reset_v
        self.refractory[self.spikes] = self.refractory_steps

        spike_ids = np.nonzero(self.spikes)[0].tolist()
        return {
            "spikes": spike_ids,
            "v_sample": self.v[:: max(1, self.n // 200)].round(3).tolist(),
        }
