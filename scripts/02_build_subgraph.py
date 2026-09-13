"""
Loads the raw connectome files and cuts them down to a subgraph small
enough to simulate live on a laptop CPU.

The full graph is ~166,700 neurons / ~125M synapses -- way more than you
need for a dashboard, and too much for a real-time numpy sim. This script:

  1. Loads neuron annotations (cell type, class, side).
  2. Loads the full weighted connectivity table.
  3. Picks a subset of neurons -- by default, the N most-connected neurons
     (by total synapse weight), optionally restricted to specific cell
     classes (e.g. visual system + descending neurons -- the closest thing
     to a "sensory in, motor out" pathway).
  4. Builds a sparse adjacency matrix over just that subset.
  5. Saves everything the simulator/dashboard need into ./data/subgraph/.

Run this after 01_download_data.py. This step is also fully offline.

NOTE ON COLUMN NAMES: Janelia's exact column names shift slightly between
dataset releases. This script prints the columns it finds and tries a few
likely candidates. If it can't guess right, it'll tell you exactly what
to edit.
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "subgraph"
OUT_DIR.mkdir(exist_ok=True)

ANNOT_FILE = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILE = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# How many neurons to keep in the local subgraph. 500-2000 runs smoothly
# in real time on a normal laptop. Push higher on a beefier desktop.
N_NEURONS = 800

# If you want to bias toward a "sensory in -> motor out" feel rather than
# pure hub neurons, list class/type substrings to prioritize here (case
# insensitive, matched against whatever class/type column exists).
# Leave empty to just take the most-connected neurons overall.
PREFERRED_CLASSES = ["visual", "optic", "descending", "motor"]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def main() -> None:
    if not ANNOT_FILE.exists() or not WEIGHTS_FILE.exists():
        raise SystemExit("Missing data files -- run 01_download_data.py first.")

    print("Loading annotations...")
    annot = pd.read_feather(ANNOT_FILE)
    print(f"  {len(annot):,} annotated neurons. Columns: {list(annot.columns)}")

    id_col = find_col(annot, ["bodyid", "body_id", "bodyid_", "id"])
    type_col = find_col(annot, ["type", "celltype", "cell_type"])
    class_col = find_col(annot, ["class", "superclass", "cellclass"])
    side_col = find_col(annot, ["side"])

    if id_col is None:
        raise SystemExit(
            f"Couldn't find a neuron-ID column in annotations. "
            f"Columns present: {list(annot.columns)}. Edit find_col() candidates above."
        )
    print(f"  Using: id={id_col!r}, type={type_col!r}, class={class_col!r}, side={side_col!r}")

    print("Loading full connectivity table (this is the big one, ~1.1 GB)...")
    weights = pd.read_feather(WEIGHTS_FILE)
    print(f"  {len(weights):,} weighted edges. Columns: {list(weights.columns)}")

    pre_col = find_col(weights, ["bodyid_pre", "body_pre", "pre", "presynaptic_id"])
    post_col = find_col(weights, ["bodyid_post", "body_post", "post", "postsynaptic_id"])
    w_col = find_col(weights, ["weight", "count", "n_syn", "syn_count", "synapse_count"])

    if pre_col is None or post_col is None or w_col is None:
        raise SystemExit(
            f"Couldn't confidently find pre/post/weight columns. "
            f"Columns present: {list(weights.columns)}. Edit find_col() candidates above."
        )
    print(f"  Using: pre={pre_col!r}, post={post_col!r}, weight={w_col!r}")

    # Rank neurons by total synaptic weight touching them (in + out).
    print("Ranking neurons by total connection weight...")
    deg = pd.concat(
        [
            weights.groupby(pre_col)[w_col].sum(),
            weights.groupby(post_col)[w_col].sum(),
        ],
        axis=1,
    ).fillna(0)
    deg.columns = ["out", "in"]
    deg["total"] = deg["out"] + deg["in"]

    candidate_ids = deg.index

    if PREFERRED_CLASSES and (class_col or type_col):
        text_col = class_col or type_col
        mask_series = annot[text_col].astype(str).str.lower()
        pattern = "|".join(PREFERRED_CLASSES)
        preferred_ids = set(annot.loc[mask_series.str.contains(pattern, na=False), id_col])
        preferred = deg.loc[deg.index.isin(preferred_ids)].sort_values("total", ascending=False)
        rest = deg.loc[~deg.index.isin(preferred_ids)].sort_values("total", ascending=False)
        chosen = pd.concat([preferred, rest]).head(N_NEURONS)
        print(f"  {len(preferred)} candidates matched preferred classes {PREFERRED_CLASSES}; "
              f"took {min(len(preferred), N_NEURONS)} of them plus top hubs to fill out {N_NEURONS}.")
    else:
        chosen = deg.sort_values("total", ascending=False).head(N_NEURONS)

    chosen_ids = list(chosen.index)
    id_to_idx = {bid: i for i, bid in enumerate(chosen_ids)}
    n = len(chosen_ids)
    print(f"Subgraph size: {n} neurons")

    print("Filtering edges to the chosen subgraph...")
    sub_edges = weights[weights[pre_col].isin(id_to_idx) & weights[post_col].isin(id_to_idx)]
    rows = sub_edges[pre_col].map(id_to_idx).to_numpy()
    cols = sub_edges[post_col].map(id_to_idx).to_numpy()
    vals = sub_edges[w_col].to_numpy(dtype=np.float32)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    print(f"  {W.nnz:,} edges kept inside the subgraph.")

    # Neuron metadata for labeling in the dashboard.
    annot_indexed = annot.set_index(id_col)
    meta = []
    for bid in chosen_ids:
        row = annot_indexed.loc[bid] if bid in annot_indexed.index else {}
        meta.append({
            "bodyId": int(bid),
            "type": str(row.get(type_col, "")) if type_col else "",
            "class": str(row.get(class_col, "")) if class_col else "",
            "side": str(row.get(side_col, "")) if side_col else "",
            "totalWeight": float(chosen.loc[bid, "total"]),
        })

    sparse.save_npz(OUT_DIR / "adjacency.npz", W)
    with open(OUT_DIR / "neurons.json", "w") as f:
        json.dump(meta, f)

    print(f"\nSaved subgraph to {OUT_DIR}")
    print("Next: run server/sim_server.py")


if __name__ == "__main__":
    main()
