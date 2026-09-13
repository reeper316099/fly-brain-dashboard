"""
Runs the fly-brain simulation locally and serves:
  - the static dashboard (dashboard/index.html) at http://localhost:8765/
  - a live WebSocket feed of spikes at ws://localhost:8765/ws

Everything here runs on your machine. The only network traffic is your
own browser talking to your own localhost server -- no cloud, no API.

Run:
    python server/sim_server.py
Then open:
    http://localhost:8765/
"""
from __future__ import annotations

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

HOST = "localhost"
PORT = 8765
STEP_HZ = 30  # simulation + broadcast rate

SIM_TASK = web.AppKey("sim_task", asyncio.Task)

sim: FlyBrainSim | None = None
clients: set[web.WebSocketResponse] = set()
current_stimulus = {"value": 0.05}  # small idle/spontaneous drive


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(DASHBOARD_DIR / "index.html")


async def neurons_meta(request: web.Request) -> web.Response:
    return web.json_response(sim.neurons)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if "stimulus" in data:
                        current_stimulus["value"] = max(0.0, min(1.0, float(data["stimulus"])))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
    finally:
        clients.discard(ws)
    return ws


async def sim_loop() -> None:
    period = 1.0 / STEP_HZ
    while True:
        start = time.perf_counter()

        sim.inject_stimulus(current_stimulus["value"])
        result = sim.step()
        result["t"] = time.time()
        payload = json.dumps(result)

        dead = []
        for ws in list(clients):
            if ws.closed:
                dead.append(ws)
                continue
            try:
                await ws.send_str(payload)
            except Exception:  # client went away mid-send; drop it
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

        elapsed = time.perf_counter() - start
        await asyncio.sleep(max(0.0, period - elapsed))


async def on_startup(app: web.Application) -> None:
    global sim
    sim = FlyBrainSim(SUBGRAPH_DIR)
    print(f"Loaded subgraph with {sim.n} neurons, {sim.W.nnz} edges.")
    app[SIM_TASK] = asyncio.create_task(sim_loop())


async def on_cleanup(app: web.Application) -> None:
    task = app.get(SIM_TASK)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def main() -> None:
    if not (SUBGRAPH_DIR / "adjacency.npz").exists() or not (SUBGRAPH_DIR / "neurons.json").exists():
        raise SystemExit(
            f"No subgraph found in {SUBGRAPH_DIR}.\n"
            "Run scripts/01_download_data.py and then scripts/02_build_subgraph.py "
            "before starting the server."
        )

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/neurons", neurons_meta)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", DASHBOARD_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    print(f"Dashboard: http://{HOST}:{PORT}/   (Ctrl+C to stop)")
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
