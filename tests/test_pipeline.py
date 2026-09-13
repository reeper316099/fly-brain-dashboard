"""End-to-end check of scripts/02_build_subgraph.py on a tiny synthetic
dataset that mirrors the real MaleCNS v1.0 column names and dtypes."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

build_subgraph = importlib.import_module("02_build_subgraph")


@pytest.fixture
def synthetic_data(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    ids = np.arange(10_000, 10_000 + n).astype("int64")
    annot = pd.DataFrame({
        "bodyId": ids,
        "type": ["T%d" % (i % 7) for i in range(n)],
        "class": rng.choice(["visual projection", "descending", "central", None], size=n),
        "somaSide": rng.choice(["L", "R"], size=n),
        "superclass": "x",
    })
    annot.to_feather(tmp_path / build_subgraph.ANNOT_FILE)

    e = 1500
    pre = rng.choice(ids, size=e)
    post = rng.choice(ids, size=e)
    pre[:100] = 99_999_999  # an unannotated fragment with lots of synapses
    pd.DataFrame({
        "body_pre": pre, "body_post": post,
        "weight": rng.integers(1, 40, size=e).astype("int64"),
        "type_pre": "a", "type_post": "b",
    }).to_feather(tmp_path / build_subgraph.WEIGHTS_CANDIDATES[0])

    pd.DataFrame({
        "body": ids,
        "consensus_nt": rng.choice(["acetylcholine", "gaba", "glutamate", None], size=n),
    }).to_feather(tmp_path / build_subgraph.NT_FILE)
    return tmp_path


def test_build_produces_consistent_outputs(synthetic_data):
    out = synthetic_data / "sub"
    stats = build_subgraph.build(synthetic_data, out, n_neurons=25, preferred_classes=["visual", "descending"])
    W = sparse.load_npz(out / "adjacency.npz")
    meta = json.load(open(out / "neurons.json"))

    assert stats["n"] == 25 and W.shape == (25, 25) and len(meta) == 25
    assert W.nnz > 0 and (W.diagonal() == 0).all(), "autapses must be dropped"
    assert all(m["bodyId"] != 99_999_999 for m in meta), "unannotated fragments must not be chosen"
    assert all(m["side"] in ("L", "R") for m in meta), "somaSide should be picked up"
    assert not any(v in ("nan", "None") for m in meta for v in (m["type"], m["class"], m["side"]))
    assert all(m["sign"] in (1, -1) for m in meta)
    assert any(m["sign"] == -1 for m in meta), "GABA/glutamate neurons should be inhibitory"
    assert all(m["sign"] == 1 for m in meta if m["nt"] == "")
    # preferred classes rank first, then hubs by weight within each block
    pref = [("visual" in m["class"] or "descending" in m["class"]) for m in meta]
    assert pref == sorted(pref, reverse=True)


def test_build_without_class_bias_orders_by_weight(synthetic_data):
    out = synthetic_data / "sub2"
    build_subgraph.build(synthetic_data, out, n_neurons=10, preferred_classes=[])
    meta = json.load(open(out / "neurons.json"))
    weights = [m["totalWeight"] for m in meta]
    assert weights == sorted(weights, reverse=True)


def test_cli_parses_flags(synthetic_data, monkeypatch):
    out = synthetic_data / "cli"
    build_subgraph.main(["--n-neurons", "12", "--classes", "", "--data-dir", str(synthetic_data), "--out", str(out)])
    assert len(json.load(open(out / "neurons.json"))) == 12
