# Fly Brain — Live Connectome Dashboard

A local, offline, spiking-neural-network simulation running on top of a
**real subgraph** of the **MaleCNS v1.0** connectome — the complete
166,700-neuron male fruit fly central nervous system, released by HHMI
Janelia FlyEM, University of Cambridge, MRC LMB, and Google Research in
September 2026. Visualized live in your browser as a scrolling spike
raster, optionally driven by your webcam.

**Everything runs on your own machine.** The only network call in this
entire project is the one-time file download in Step 2 below. After
that, the data lives on disk and the simulation + dashboard never touch
the internet again — no API key, no neuPrint account, no cloud calls,
nothing.

---

## Table of contents

- [How it works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Setup — pip / venv](#setup--pip--venv)
- [Setup — uv](#setup--uv)
- [Running it](#running-it)
- [What you'll see](#what-youll-see)
- [Project structure](#project-structure)
- [Putting this on GitHub](#putting-this-on-github)
- [Troubleshooting](#troubleshooting)
- [Where to take it next](#where-to-take-it-next)
- [Performance notes](#performance-notes)
- [Data license](#data-license)

---

## How it works

1. **Download** — pull two files straight from Google's public storage
   bucket for the dataset over plain HTTPS: neuron annotations (cell
   types/classes, 13 MB) and the full synaptic connection-weight graph
   (1.1 GB). CC-BY licensed, no auth required.
2. **Build subgraph** — the full graph is too big to simulate live on a
   laptop, so a script picks the ~800 most-connected neurons and saves a
   small adjacency matrix + metadata.
3. **Simulate** — a lightweight, dependency-free (no compiled backend)
   leaky integrate-and-fire (LIF) model runs over that real
   connectivity, in plain numpy, at 30 steps/sec.
4. **Serve + visualize** — a local `aiohttp` server broadcasts live spike
   data over WebSocket to a browser dashboard that draws a scrolling
   raster plot, and optionally reads webcam motion as a live sensory
   input signal.

---

## Prerequisites

- **Python 3.10+**
- **~2 GB free disk space** (1.1 GB download + working room)
- A **webcam** if you want to use motion as input (optional — there's a
  manual slider fallback)
- **git** if you're pushing this to GitHub (see below)

Check your Python version:

```bash
python3 --version
```

---

## Setup — pip / venv

```bash
# 1. Enter the project folder
cd fly-brain-dashboard

# 2. Create and activate a virtual environment
python3 -m venv venv

# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd.exe):
venv\Scripts\activate.bat

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Setup — uv

If you use [uv](https://docs.astral.sh/uv/) instead, it handles the
virtual environment and lockfile for you — no manual `venv` step needed.

```bash
# 1. Enter the project folder
cd fly-brain-dashboard

# 2. Install uv itself, if you don't already have it
#    macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Create the environment and install dependencies from pyproject.toml
uv sync

# 4. Run any command inside that environment with `uv run`, e.g.:
uv run python scripts/01_download_data.py
```

With uv, prefix every Python command below with `uv run` instead of
activating a venv manually (e.g. `uv run python server/sim_server.py`).
uv will also generate a `uv.lock` file the first time you run `uv sync`
— commit that to your repo so anyone cloning it gets the exact same
dependency versions.

---

## Running it

### 1. Download the connectome data (one time, ~1.1 GB)

pip/venv:
```bash
python scripts/01_download_data.py
```
uv:
```bash
uv run python scripts/01_download_data.py
```

This downloads two files into `data/`:
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` — cell type/class/side annotations (13 MB)
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` — the full weighted connection graph (1.1 GB)

It shows a progress bar and skips files that are already downloaded, so
it's safe to re-run if it gets interrupted partway through.

### 2. Build a local subgraph

pip/venv:
```bash
python scripts/02_build_subgraph.py
```
uv:
```bash
uv run python scripts/02_build_subgraph.py
```

This loads the full ~166,700-neuron graph, picks the ~800 most-connected
neurons (configurable via `N_NEURONS` at the top of the script), and
saves a small adjacency matrix (`data/subgraph/adjacency.npz`) and
neuron metadata (`data/subgraph/neurons.json`).

**Column-name heads up:** Janelia's exact column names can shift
slightly between dataset releases. This script prints every column it
finds in both files before doing anything else. If it can't confidently
guess which columns are the neuron ID / cell type / connection weight,
it stops and tells you exactly what columns exist so you can add the
right name to the `find_col()` candidate lists near the top of the file
— it's a one-line fix, not a rewrite.

### 3. Run the simulation server

pip/venv:
```bash
python server/sim_server.py
```
uv:
```bash
uv run python server/sim_server.py
```

Then open **http://localhost:8765/** in your browser.

Stop it any time with `Ctrl+C`.

---

## What you'll see

- A live scrolling **raster plot** — each dot is a real neuron from the
  connectome firing, rows ordered by connection weight (top = biggest
  hub neurons).
- An **"Enable Webcam Input"** button — frame-to-frame motion in your
  webcam feed drives a stimulus current into the most-connected ~80
  neurons, and you watch that ripple through the network along real
  synaptic weights.
- A **manual slider** if you'd rather not use the webcam, or want a
  stable, repeatable stimulus level for testing.
- Live stats: neuron count, spikes per frame, mean firing rate, and the
  simulation's actual frame rate.

---

## Project structure

```
fly-brain-dashboard/
├── README.md
├── DATA_LICENSE.md
├── requirements.txt          # pip/venv dependency list
├── pyproject.toml            # uv dependency list
├── .gitignore                # excludes data/ and venv/ from git
├── data/                     # created at runtime, gitignored
│   ├── *.feather              # downloaded connectome files
│   └── subgraph/               # built adjacency matrix + metadata
├── scripts/
│   ├── 01_download_data.py   # one-time HTTPS download, no API
│   └── 02_build_subgraph.py  # cuts the graph down to a runnable size
├── server/
│   ├── lif_sim.py             # numpy leaky integrate-and-fire model
│   └── sim_server.py          # aiohttp server: static files + WebSocket
└── dashboard/
    └── index.html             # raster plot + webcam input, vanilla JS
```

---

## Putting this on GitHub

The `.gitignore` already excludes `data/` and any virtual environment,
so the repo itself stays small (just code) even though the dataset is
gigabytes — anyone who clones it re-downloads the data themselves in
Step 1.

```bash
cd fly-brain-dashboard
git init
git add .
git commit -m "Initial commit: local fly connectome dashboard"

# Create the repo on GitHub first (via github.com or gh CLI), then:
git remote add origin https://github.com/reeper316099/fly-brain-dashboard.git
git branch -M main
git push -u origin main
```

If you used `gh` (GitHub CLI) instead of creating the repo on the
website first:

```bash
gh repo create fly-brain-dashboard --public --source=. --remote=origin --push
```

If you're using uv, make sure `uv.lock` (generated by `uv sync`) gets
committed too — it's not in `.gitignore` — so anyone else running
`uv sync` gets identical dependency versions to what you tested with.

Consider adding a short note at the top of your GitHub repo's About
section linking back to the MaleCNS v1.0 dataset and its CC-BY license
(see `DATA_LICENSE.md`) — good practice since the code depends entirely
on someone else's licensed dataset.

---

## Troubleshooting

**`02_build_subgraph.py` exits with "Couldn't confidently find pre/post/weight columns"**
Read the printed column list, find the actual names for neuron ID,
presynaptic ID, postsynaptic ID, and weight/synapse count, and add them
as strings to the relevant `find_col([...])` call near the top of the
script.

**Download seems to hang or is very slow**
The `connectome-weights` file is 1.1 GB — on a slow connection this can
take a while. The script shows live progress; if it actually stalls,
Ctrl+C and re-run — it'll skip any file that already fully downloaded
and only retry ones that didn't finish (delete the `.part` file for
that entry first if it got corrupted mid-download).

**Webcam button does nothing / permission denied**
Browsers only allow webcam access on `localhost` or HTTPS — since this
serves on `localhost:8765`, that should just work. If your browser
still blocks it, check your OS-level camera permissions for the
browser app itself, or just use the manual slider instead.

**Simulation feels too "flat" (barely any spikes) or too "loud" (constant firing)**
Tune `tau`, `threshold`, and `synaptic_gain` in `server/lif_sim.py` —
these are simplified starting values, not biologically calibrated ones.
Lower `threshold` or raise `synaptic_gain` for more activity; the
reverse for less.

**Port 8765 already in use**
Change the `port=8765` argument in `main()` in `server/sim_server.py`.

---

## Where to take it next

- **Bigger/different subgraph** — raise `N_NEURONS` in
  `02_build_subgraph.py`, or tune `PREFERRED_CLASSES` toward specific
  circuits (olfactory, a specific descending neuron type, etc.) once
  you've looked at what class labels actually exist in the annotation
  file.
- **Excitatory/inhibitory signs** — every connection is currently
  treated as excitatory. Download the `body-neurotransmitters` file
  (add its URL to `01_download_data.py`) to get predicted
  neurotransmitter per neuron, and use it in `lif_sim.py` to flip
  inhibitory connections negative.
- **3D fly body** — pair this with `NeuroMechFly` and the skeleton SWC
  files from the dataset to animate an actual fly body instead of, or
  alongside, the raster plot.
- **Real motor output / a game** — take the firing rate of a chosen
  "descending neuron" subset and map it to actual actions (steer a
  sprite, drive a Minecraft bot via `mineflayer`, control a driving-sim
  car) — this dashboard is the visualization layer that pairs naturally
  with that next step.

---

## Performance notes

- 800 neurons at 30 Hz runs comfortably on basically any laptop from the
  last several years — this is deliberately lightweight pure-numpy code,
  no GPU required.
- Pushing `N_NEURONS` well past a few thousand will start to bottleneck
  on the per-step sparse matrix multiply in `lif_sim.py`. At that point,
  consider a GPU-backed array library (`cupy`) or a compiled spiking
  simulator backend (`brian2`, `nest`) instead of the numpy loop here.

---

## Data license

See `DATA_LICENSE.md`. Short version: the connectome data is CC-BY
4.0 — free to use with attribution, not covered by whatever license you
put on your own code in this repo.
