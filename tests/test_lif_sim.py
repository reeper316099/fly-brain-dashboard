import json

import numpy as np
import pytest
from scipy import sparse

from lif_sim import FlyBrainSim


def make_subgraph(tmp_path, W, meta):
    sparse.save_npz(tmp_path / "adjacency.npz", sparse.csr_matrix(np.asarray(W, dtype=np.float32)))
    with open(tmp_path / "neurons.json", "w") as f:
        json.dump(meta, f)
    return tmp_path


def quiet(sim):
    sim.noise_std = 0.0
    return sim


def test_input_normalisation_and_signs(tmp_path):
    # neuron 0 (excitatory) and 1 (inhibitory) both project to 2; 3 is isolated.
    W = [[0, 0, 300, 0], [0, 0, 100, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    meta = [{"class": "visual", "sign": 1}, {"class": "central", "sign": -1}, {"class": "x"}, {"class": "x"}]
    sim = FlyBrainSim(make_subgraph(tmp_path, W, meta))
    col = sim.W[:, 2].toarray().ravel()
    assert col[0] == pytest.approx(0.75) and col[1] == pytest.approx(-0.25)
    assert sim.W[:, 3].nnz == 0
    assert sim.n_inhibitory == 1


def test_stimulus_drives_sensory_neurons_only(tmp_path):
    n = 20
    meta = [{"class": "visual" if i < 4 else "central", "sign": 1} for i in range(n)]
    sim = quiet(FlyBrainSim(make_subgraph(tmp_path, np.zeros((n, n)), meta)))
    assert sorted(sim.sensory_idx.tolist()) == [0, 1]  # capped at 10% of the network
    fired = set()
    for _ in range(50):
        sim.inject_stimulus(1.0)
        fired.update(sim.step()["spikes"])
    assert fired == {0, 1}


def test_sensory_fallback_when_no_labels(tmp_path):
    n = 30
    sim = FlyBrainSim(make_subgraph(tmp_path, np.zeros((n, n)), [{} for _ in range(n)]))
    assert sim.sensory_idx.tolist() == [0, 1, 2]


def run(sim, steps, stimulus=1.0):
    fired = set()
    for _ in range(steps):
        sim.inject_stimulus(stimulus)
        fired.update(sim.step()["spikes"])
    return fired


def test_excitation_propagates_and_inhibition_blocks(tmp_path):
    # chain 0 -> 1 -> 2 ; neuron 3 strongly inhibits 2 and is driven directly.
    W = np.zeros((4, 4))
    W[0, 1] = 5
    W[1, 2] = 5
    W[3, 2] = 20
    meta = [{"class": "visual", "sign": 1}, {"sign": 1}, {"sign": 1}, {"class": "x", "sign": -1}]
    sim = quiet(FlyBrainSim(make_subgraph(tmp_path, W, meta)))
    sim.sensory_idx = np.array([0])
    assert run(sim, 60) == {0, 1, 2}

    # Now drive the inhibitory neuron too: it keeps neuron 2 below threshold.
    sim.reset()
    sim.sensory_idx = np.array([0, 3])
    assert run(sim, 60) == {0, 1, 3}


def test_refractory_period(tmp_path):
    n = 5
    meta = [{"class": "visual"} for _ in range(n)]
    sim = quiet(FlyBrainSim(make_subgraph(tmp_path, np.zeros((n, n)), meta)))
    sim.stimulus_gain = 5.0  # would fire every step without a refractory period
    last = -10
    for t in range(30):
        sim.inject_stimulus(1.0)
        if 0 in sim.step()["spikes"]:
            assert t - last > sim.refractory_steps
            last = t
    assert last >= 0


def test_mismatched_metadata_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        FlyBrainSim(make_subgraph(tmp_path, np.zeros((3, 3)), [{}]))
