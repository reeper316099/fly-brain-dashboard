"""
Loads the raw connectome files and cuts them down to a subgraph small
enough to simulate live on a laptop CPU.

The full graph is ~166,700 neurons / tens of millions of body-to-body edges
-- far more than you need for a dashboard, and too much for a real-time
numpy sim. This script:

  1. Loads neuron annotations (cell type, class, side).
  2. Loads the weighted connectivity table (only the three columns we need).
  3. Picks a subset of neurons -- by default, the N most-connected annotated
     neurons (by total synapse weight), with neurons whose class matches
     --classes ranked first (e.g. visual + descending neurons: the closest
     thing to a "sensory in, motor out" pathway).
  4. Builds a sparse adjacency matrix over just that subset.
  5. Attaches each neuron's predicted neurotransmitter (if the file is
     present) so the simulator can make GABA/glutamate synapses inhibitory.
  6. Saves everything the simulator/dashboard need into ./data/subgraph/.

Run this after 01_download_data.py. This step is also fully offline.

    python scripts/02_build_subgraph.py                 # defaults: 800 neurons
    python scripts/02_build_subgraph.py --n-neurons 1500
    python scripts/02_build_subgraph.py --classes ""    # pure hubs, no class bias

NOTE ON COLUMN NAMES: this is set up for the MaleCNS v1.0 column names
(verified against the published files). Janelia's names can shift between
releases, so the script prints the columns it finds and tries a few likely
candidates. If it can't guess right, it tells you exactly what to edit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy import sparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

ANNOT_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
# Whichever of these exists is used (first match wins).
WEIGHTS_CANDIDATES = [
    "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]

# How many neurons to keep in the local subgraph. 500-2000 runs smoothly
# in real time on a normal laptop. Push higher on a beefier desktop.
N_NEURONS = 800

# Class/type substrings to rank first (case insensitive), to bias the
# subgraph toward a "sensory in -> motor out" feel instead of pure hubs.
PREFERRED_CLASSES = ["visual", "optic", "descending", "motor"]

# Sign convention for the simulator, keyed on the dataset's consensus_nt
# label. In Drosophila, acetylcholine is the main excitatory transmitter;
# GABA and glutamate (via GluCl) are inhibitory; histamine (photoreceptors)
# is inhibitory; the monoamines are modulatory and treated as mildly
# excitatory here. Edit freely -- this is a simplification.
NT_SIGN = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 1,
    "serotonin": 1,
    "octopamine": 1,
}


def find_col(columns, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def schema_columns(path: pathlib.Path) -> list[str]:
    """Column names of a feather file without loading any data."""
    with ipc.open_file(str(path)) as reader:
        return reader.schema.names


def load_columns(path: pathlib.Path, columns: list[str]) -> pd.DataFrame:
    return feather.read_table(str(path), columns=columns).to_pandas()


def build(data_dir: pathlib.Path, out_dir: pathlib.Path, n_neurons: int,
          preferred_classes: list[str]) -> dict:
    annot_path = data_dir / ANNOT_FILE
    weights_path = next((data_dir / c for c in WEIGHTS_CANDIDATES if (data_dir / c).exists()), None)
    if not annot_path.exists() or weights_path is None:
        raise SystemExit(f"Missing data files in {data_dir} -- run scripts/01_download_data.py first.")

    # ---- annotations -----------------------------------------------------
    print("Loading annotations...")
    annot_cols = schema_columns(annot_path)
    print(f"  Columns: {annot_cols}")
    id_col = find_col(annot_cols, ["bodyid", "body_id", "body", "id"])
    type_col = find_col(annot_cols, ["type", "celltype", "cell_type"])
    class_col = find_col(annot_cols, ["class", "superclass", "cellclass"])
    side_col = find_col(annot_cols, ["somaside", "rootside", "side"])
    if id_col is None:
        raise SystemExit(
            f"Couldn't find a neuron-ID column in annotations. Columns present: {annot_cols}. "
            f"Edit find_col() candidates in {__file__}."
        )
    print(f"  Using: id={id_col!r}, type={type_col!r}, class={class_col!r}, side={side_col!r}")
    annot = load_columns(annot_path, [c for c in (id_col, type_col, class_col, side_col) if c])
    annot = annot.dropna(subset=[id_col]).drop_duplicates(subset=id_col)
    annot[id_col] = annot[id_col].astype("int64")
    print(f"  {len(annot):,} annotated neurons.")

    # ---- weights ---------------------------------------------------------
    print(f"Loading connectivity table {weights_path.name} ({weights_path.stat().st_size / 1e6:,.0f} MB)...")
    w_cols = schema_columns(weights_path)
    print(f"  Columns: {w_cols}")
    pre_col = find_col(w_cols, ["body_pre", "bodyid_pre", "pre", "presynaptic_id"])
    post_col = find_col(w_cols, ["body_post", "bodyid_post", "post", "postsynaptic_id"])
    w_col = find_col(w_cols, ["weight", "count", "n_syn", "syn_count", "synapse_count"])
    if pre_col is None or post_col is None or w_col is None:
        raise SystemExit(
            f"Couldn't confidently find pre/post/weight columns. Columns present: {w_cols}. "
            f"Edit find_col() candidates in {__file__}."
        )
    print(f"  Using: pre={pre_col!r}, post={post_col!r}, weight={w_col!r}")
    weights = load_columns(weights_path, [pre_col, post_col, w_col])
    weights = weights[weights[pre_col] != weights[post_col]]  # drop autapses
    print(f"  {len(weights):,} weighted edges.")

    # ---- rank neurons ----------------------------------------------------
    print("Ranking neurons by total connection weight...")
    deg = pd.concat(
        [weights.groupby(pre_col)[w_col].sum(), weights.groupby(post_col)[w_col].sum()],
        axis=1,
    ).fillna(0)
    deg.columns = ["out", "in"]
    deg["total"] = deg["out"] + deg["in"]

    # Keep only bodies that are annotated neurons, so the "top hubs" are real
    # cells rather than unlabelled fragments that happen to carry synapses.
    deg = deg.loc[deg.index.isin(set(annot[id_col]))]
    print(f"  {len(deg):,} annotated neurons carry synapses.")

    text_col = class_col or type_col
    if preferred_classes and text_col:
        text = annot[text_col].astype(str).str.lower()
        pattern = "|".join(preferred_classes)
        preferred_ids = set(annot.loc[text.str.contains(pattern, na=False), id_col])
        preferred = deg.loc[deg.index.isin(preferred_ids)].sort_values("total", ascending=False)
        rest = deg.loc[~deg.index.isin(preferred_ids)].sort_values("total", ascending=False)
        chosen = pd.concat([preferred, rest]).head(n_neurons)
        print(f"  {len(preferred):,} neurons matched preferred classes {preferred_classes}; "
              f"took {min(len(preferred), n_neurons)} of them plus top hubs to fill out {n_neurons}.")
    else:
        chosen = deg.sort_values("total", ascending=False).head(n_neurons)

    chosen_ids = list(chosen.index)
    id_to_idx = {bid: i for i, bid in enumerate(chosen_ids)}
    n = len(chosen_ids)
    print(f"Subgraph size: {n} neurons")

    # ---- adjacency -------------------------------------------------------
    print("Filtering edges to the chosen subgraph...")
    sub = weights[weights[pre_col].isin(id_to_idx) & weights[post_col].isin(id_to_idx)]
    rows = sub[pre_col].map(id_to_idx).to_numpy()
    cols = sub[post_col].map(id_to_idx).to_numpy()
    vals = sub[w_col].to_numpy(dtype=np.float32)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W.sum_duplicates()
    print(f"  {W.nnz:,} edges kept inside the subgraph "
          f"(mean {W.nnz / n:.1f} per neuron, density {W.nnz / (n * n):.3%}).")

    # ---- neurotransmitters -----------------------------------------------
    nt_by_body: dict[int, str] = {}
    nt_path = data_dir / NT_FILE
    if nt_path.exists():
        nt_cols = schema_columns(nt_path)
        nt_id = find_col(nt_cols, ["body", "bodyid", "body_id"])
        nt_lab = find_col(nt_cols, ["consensus_nt", "predicted_nt", "celltype_predicted_nt", "nt"])
        if nt_id and nt_lab:
            nt = load_columns(nt_path, [nt_id, nt_lab]).dropna()
            nt = nt[nt[nt_id].isin(id_to_idx)]
            nt_by_body = dict(zip(nt[nt_id].astype("int64"), nt[nt_lab].astype(str).str.lower()))
            print(f"Neurotransmitter labels found for {len(nt_by_body)}/{n} neurons "
                  f"(column {nt_lab!r}).")
        else:
            print(f"Neurotransmitter file present but columns unrecognised: {nt_cols}. Skipping.")
    else:
        print("No neurotransmitter file -- every connection will be treated as excitatory.")

    # ---- metadata --------------------------------------------------------
    annot_indexed = annot.set_index(id_col)

    def label(row, col: str | None) -> str:
        if not col:
            return ""
        val = row.get(col, "")
        return "" if pd.isna(val) else str(val)

    meta = []
    for bid in chosen_ids:
        row = annot_indexed.loc[bid] if bid in annot_indexed.index else {}
        nt_label = nt_by_body.get(int(bid), "")
        meta.append({
            "bodyId": int(bid),
            "type": label(row, type_col),
            "class": label(row, class_col),
            "side": label(row, side_col),
            "nt": nt_label,
            "sign": NT_SIGN.get(nt_label, 1),
            "totalWeight": float(chosen.loc[bid, "total"]),
        })

    n_inhib = sum(1 for m in meta if m["sign"] < 0)
    print(f"  {n_inhib} neurons treated as inhibitory, {n - n_inhib} as excitatory.")

    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "adjacency.npz", W)
    with open(out_dir / "neurons.json", "w") as f:
        json.dump(meta, f)

    print(f"\nSaved subgraph to {out_dir}")
    print("Next: python server/sim_server.py")
    return {"n": n, "nnz": int(W.nnz), "n_inhibitory": n_inhib}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cut the MaleCNS graph down to a simulable subgraph.")
    parser.add_argument("--n-neurons", type=int, default=N_NEURONS,
                        help=f"neurons to keep (default {N_NEURONS})")
    parser.add_argument("--classes", default=",".join(PREFERRED_CLASSES),
                        help="comma-separated class/type substrings to rank first; "
                             f"pass '' for pure hubs (default: {','.join(PREFERRED_CLASSES)})")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="output directory (default: <data-dir>/subgraph)")
    args = parser.parse_args(argv)

    classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
    build(args.data_dir, args.out or (args.data_dir / "subgraph"), args.n_neurons, classes)


if __name__ == "__main__":
    main(sys.argv[1:])
