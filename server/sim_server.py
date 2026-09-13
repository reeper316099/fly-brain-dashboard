"""
Runs the fly-brain simulation locally and serves:
  - the static dashboard (dashboard/index.html) at http://localhost:8765/
  - neuron metadata as JSON at            http://localhost:8765/neurons
  - a live WebSocket feed of spikes at    ws://localhost:8765/ws

Everything here runs on your machine. The only network traffic is your
own browser talking to your own localhost server -- no cloud, no API.

Run:
    python server/sim_server.py                 # http://localhost:8765/
    python server/sim_server.py --port 9000
    python server/sim_server.py --hz 60         # faster simulation

WebSocket protocol (JSON):
    client -> server   {"stimulus": 0.0..1.0}     set external drive
                       {"reset": true}            reset membrane state
    server -> client   {"spikes": [...], "t": unix_time, "mean_v": float}
                       one message per simulation step
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

from aiohttp import web, WSMsgType

# Make `lif_sim` importable whether this is run as `python server/sim_server.py`
# or from some other working directory / launcher.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lif_sim import FlyBrainSim  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUBGRAPH_DIR = ROOT / "data" / "subgraph"
DASHBOARD_DIR = ROOT / "dashboard"

SIM = web.AppKey("sim", FlyBrainSim)
SIM_TASK = web.AppKey("sim_task", asyncio.Task)
CLIENTS = web.AppKey("clients", set)
STIMULUS = web.AppKey("stimulus", dict)
STEP_HZ = web.AppKey("step_hz", float)

IDLE_STIMULUS = 0.05  # small spontaneous drive when nobody is connected


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(DASHBOARD_DIR / "index.html")


async def neurons_meta(request: web.Request) -> web.Response:
    sim = request.app[SIM]
    meta = [dict(m, sensory=bool(sim.is_sensory[i])) for i, m in enumerate(sim.neurons)]
    return web.json_response({
        "neurons": meta,
        "n": sim.n,
        "n_sensory": sim.n_sensory,
        "n_inhibitory": sim.n_inhibitory,
        "n_edges": int(sim.W.nnz),
        "step_hz": request.app[STEP_HZ],
    })


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    clients = request.app[CLIENTS]
    clients.add(ws)
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if "stimulus" in data:
                try:
                    request.app[STIMULUS]["value"] = max(0.0, min(1.0, float(data["stimulus"])))
                except (TypeError, ValueError):
                    pass
            if data.get("reset"):
                request.app[SIM].reset()
    finally:
        clients.discard(ws)
    return ws


async def sim_loop(app: web.Application) -> None:
    sim, clients, stimulus = app[SIM], app[CLIENTS], app[STIMULUS]
    period = 1.0 / app[STEP_HZ]
    next_tick = time.perf_counter()
    while True:
        sim.inject_stimulus(stimulus["value"])
        result = sim.step()
        result["t"] = time.time()
        payload = json.dumps(result)

        for ws in list(clients):
            if ws.closed:
                clients.discard(ws)
                continue
            try:
                await ws.send_str(payload)
            except Exception:  # client went away mid-send; drop it
                clients.discard(ws)

        # Fixed-rate scheduling that doesn't drift when a step runs long.
        next_tick += period
        delay = next_tick - time.perf_counter()
        if delay < -period:  # fell way behind (e.g. laptop slept): resync
            next_tick = time.perf_counter()
            delay = 0.0
        await asyncio.sleep(max(0.0, delay))


async def on_startup(app: web.Application) -> None:
    sim = FlyBrainSim(SUBGRAPH_DIR)
    app[SIM] = sim
    print(f"Loaded subgraph: {sim.n} neurons, {sim.W.nnz} edges, "
          f"{sim.n_sensory} sensory, {sim.n_inhibitory} inhibitory.")
    app[SIM_TASK] = asyncio.create_task(sim_loop(app))


async def on_cleanup(app: web.Application) -> None:
    task = app.get(SIM_TASK)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def make_app(step_hz: float) -> web.Application:
    app = web.Application()
    app[CLIENTS] = set()
    app[STIMULUS] = {"value": IDLE_STIMULUS}
    app[STEP_HZ] = float(step_hz)
    app.router.add_get("/", index)
    app.router.add_get("/neurons", neurons_meta)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", DASHBOARD_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the fly-brain dashboard locally.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hz", type=float, default=30.0, help="simulation steps per second (default 30)")
    args = parser.parse_args(argv)

    if not (SUBGRAPH_DIR / "adjacency.npz").exists() or not (SUBGRAPH_DIR / "neurons.json").exists():
        raise SystemExit(
            f"No subgraph found in {SUBGRAPH_DIR}.\n"
            "Run scripts/01_download_data.py and then scripts/02_build_subgraph.py "
            "before starting the server."
        )

    print(f"Dashboard: http://{args.host}:{args.port}/   (Ctrl+C to stop)")
    web.run_app(make_app(args.hz), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
