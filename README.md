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

1. **Download** — pull three files straight from Google's public storage
   bucket for the dataset over plain HTTPS: neuron annotations (cell
   types/classes, 14 MB), predicted neurotransmitter per neuron (43 MB),
   and the synaptic connection-weight graph (500 MB for the
   "significant connections only" table, or 1.05 GB for the full one).
   CC-BY licensed, no auth required.
2. **Build subgraph** — the full graph is too big to simulate live on a
   laptop, so a script picks the ~800 most-connected neurons and saves a
   small adjacency matrix + metadata, including whether each neuron is
   predicted to be excitatory (acetylcholine) or inhibitory (GABA,
   glutamate).
3. **Simulate** — a lightweight, dependency-free (no compiled backend)
   leaky integrate-and-fire (LIF) model runs over that real, signed
   connectivity, in plain numpy, at 30 steps/sec.
4. **Serve + visualize** — a local `aiohttp` server broadcasts live spike
   data over WebSocket to a browser dashboard that draws a scrolling
   raster plot, and optionally reads webcam motion as a live sensory
   input signal.

---

## Prerequisites

- **Python 3.10+**
- **~1.5 GB free disk space** (~560 MB download by default, ~2 GB
  with `--full-weights`, plus working room)
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

### 1. Download the connectome data (one time, ~560 MB)

pip/venv:
```bash
python scripts/01_download_data.py
```
uv:
```bash
uv run python scripts/01_download_data.py
```

This downloads three files into `data/`:
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` — cell type/class/side annotations (14 MB)
- `body-neurotransmitters-male-cns-v1.0.feather` — predicted neurotransmitter per neuron (43 MB)
- `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` — weighted connection graph, strong edges only (500 MB)

Add `--full-weights` to fetch the complete 1.05 GB weight table instead
(it also includes every 1–2 synapse edge; for a hub subgraph it makes no
visible difference). Downloads show a progress bar, **resume** from a
partial `.part` file if interrupted, retry on network errors, and skip
files that are already complete — so it's always safe to re-run.

### 2. Build a local subgraph

pip/venv:
```bash
python scripts/02_build_subgraph.py
```
uv:
```bash
uv run python scripts/02_build_subgraph.py
```

This loads the full ~166,700-neuron graph (only the three columns it
needs, to keep memory down), picks the ~800 most-connected annotated
neurons, and saves a small adjacency matrix
(`data/subgraph/adjacency.npz`) plus neuron metadata
(`data/subgraph/neurons.json`: body ID, type, class, side, predicted
neurotransmitter and excitatory/inhibitory sign).

Options:

```bash
python scripts/02_build_subgraph.py --n-neurons 1500        # bigger subgraph
python scripts/02_build_subgraph.py --classes "visual,descending"
python scripts/02_build_subgraph.py --classes ""            # pure hubs, no class bias
python scripts/02_build_subgraph.py --help
```

**Column-name heads up:** the script is set up for the MaleCNS v1.0
column names (`bodyId` / `type` / `class` / `somaSide` in the
annotations, `body_pre` / `body_post` / `weight` in the connection
table). Janelia's names can shift between dataset releases, so it
prints every column it finds in both files before doing anything else.
If it can't confidently guess which columns are the neuron ID / cell
type / connection weight, it stops and tells you exactly what columns
exist so you can add the right name to the `find_col()` candidate lists
near the top of the file — it's a one-line fix, not a rewrite.

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

Stop it any time with `Ctrl+C`. Flags: `--port 9000`, `--host 0.0.0.0`
(to reach it from another device on your LAN — note webcam access then
needs HTTPS or the manual slider), `--hz 60` for a faster simulation.

---

## What you'll see

- A live scrolling **raster plot** — each dot is a real neuron from the
  connectome firing, coloured by cell class, rows ordered by connection
  weight (top = biggest hub neurons). The pale band in the left gutter
  marks the sensory neurons that receive external stimulus.
- **Hover** any row to see that neuron's body ID, type, class, side,
  predicted neurotransmitter, and recent spike count. **Click** a class
  in the legend to hide or show it.
- An **"Enable Webcam Input"** button — frame-to-frame motion in your
  webcam feed drives a stimulus current into up to ~80 of the
  best-connected visual/sensory neurons, and you watch that ripple
  through the network along real synaptic weights (with GABA and
  glutamate connections pulling activity down).
- A **manual slider** if you'd rather not use the webcam, or want a
  stable, repeatable stimulus level for testing, and a **reset** button
  that clears the network's membrane state.
- Live stats: neuron / synapse / sensory / inhibitory counts, spikes per
  frame, mean firing rate per neuron, and the simulation's actual step
  rate.

---

## Project structure

```
fly-brain-dashboard/
├── README.md
├── DATA_LICENSE.md
├── requirements.txt          # pip/venv dependency list
├── pyproject.toml            # uv dependency list (+ pytest dev group)
├── .gitignore                # excludes data/ and venv/ from git
├── data/                     # created at runtime, gitignored
│   ├── *.feather              # downloaded connectome files
│   └── subgraph/               # built adjacency matrix + metadata
├── scripts/
│   ├── 01_download_data.py   # one-time resumable HTTPS download, no API
│   └── 02_build_subgraph.py  # cuts the graph down to a runnable size
├── server/
│   ├── lif_sim.py             # numpy leaky integrate-and-fire model
│   └── sim_server.py          # aiohttp server: static files + WebSocket
├── dashboard/
│   └── index.html             # raster plot + webcam input, vanilla JS
└── tests/                     # pytest suite (runs on tiny synthetic data,
                               # no download needed)
```

Run the tests with:

```bash
uv run pytest            # uv
python -m pytest         # pip/venv, after `pip install pytest`
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
Tune the values in the "tunables" block near the top of
`server/lif_sim.py` — these are simplified starting values, not
biologically calibrated ones. `synaptic_gain` is the input a neuron
receives when *all* of its presynaptic partners fired on the previous
step (each neuron's incoming weights are normalised to sum to 1, so hub
neurons don't dominate); raise it for more cascading activity, lower it
for less. `noise_std` sets how often neurons fire spontaneously with no
stimulus, and `stimulus_gain` how hard the webcam/slider drives the
sensory neurons.

**Port 8765 already in use**
Run with `--port 9000` (or any free port).

**Can't reach the dashboard from another device (phone, another PC) — "connection timed out"**
By default the server binds to `localhost`, which only accepts
connections from the same machine — that's why `http://<your-pc-ip>:8765`
times out from elsewhere on your network, even though `http://localhost:8765`
works fine locally. Two things to fix:

1. Start the server with `--host 0.0.0.0` so it listens on all network
   interfaces, not just loopback:
   ```bash
   python server/sim_server.py --host 0.0.0.0
   ```
   The startup banner then prints the LAN URL to use from other devices
   (e.g. `http://192.168.1.23:8765/`) — use that, not `localhost`, on the
   other device.
2. **Your OS firewall is almost certainly still blocking it** even after
   that — a timeout (rather than an immediate refusal) is the classic
   sign of a firewall silently dropping the packets. Allow inbound
   connections on the port:
   - **Windows:** Settings → Network & security → Windows Security →
     Firewall & network protection → Allow an app through firewall, and
     allow `python.exe` (or `pythonw.exe`) for both Private and Public
     networks. Or from an elevated PowerShell/cmd:
     ```
     netsh advfirewall firewall add rule name="Fly Brain Dashboard" dir=in action=allow protocol=TCP localport=8765
     ```
   - **macOS:** System Settings → Network → Firewall → Options, allow
     incoming connections for `python3`.
   - **Linux:** `sudo ufw allow 8765/tcp`

   Also confirm both machines are actually on the same network (not one
   on Wi-Fi guest/isolated network, or a VPN routing traffic elsewhere),
   and that you're using the PC's actual LAN IP (`ipconfig` on Windows,
   look for the `IPv4 Address` under your active adapter) — not a VPN or
   Hyper-V virtual adapter's IP if you have several listed.

   Note the webcam button needs `localhost` or HTTPS to work in most
   browsers — over plain `http://<lan-ip>:8765/` from another device,
   use the manual slider instead.

**`uv sync` fails with "Multiple top-level packages discovered in a flat-layout"**
You have an older copy of `pyproject.toml` that still declares a
`[build-system]`. This project is an app, not an installable package —
make sure `pyproject.toml` has `[tool.uv] package = false` and no
`[build-system]` table, then re-run `uv sync`.

---

## Where to take it next

- **Bigger/different subgraph** — `--n-neurons 2000`, or `--classes`
  aimed at specific circuits (olfactory, a specific descending neuron
  type, etc.) once you've looked at what class labels actually exist in
  the annotation file.
- **Better excitatory/inhibitory model** — connections are signed by the
  presynaptic neuron's predicted neurotransmitter via the `NT_SIGN`
  table in `02_build_subgraph.py` (ACh +, GABA/glutamate/histamine −,
  monoamines treated as mildly +). Real synapses depend on the receptor
  too; the `body-neurotransmitters` file also carries per-neuron
  confidence values you could use to weight uncertain predictions.
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
- Pushing `--n-neurons` well past a few thousand will start to bottleneck
  on the per-step sparse matrix multiply in `lif_sim.py`. At that point,
  consider a GPU-backed array library (`cupy`) or a compiled spiking
  simulator backend (`brian2`, `nest`) instead of the numpy loop here.

---

## Data license

See `DATA_LICENSE.md`. Short version: the connectome data is CC-BY
4.0 — free to use with attribution, not covered by whatever license you
put on your own code in this repo.
