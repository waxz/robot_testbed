"""Viser web UI for any test-bed config: MuJoCo scene, command controls, controller status, timing."""
from __future__ import annotations

import json
import time

import numpy as np


def run_viewer(tb, port: int = 8090) -> None:
    import viser
    from mjviser import ViserMujocoScene

    server = viser.ViserServer(host="0.0.0.0", port=port)
    scene = ViserMujocoScene(server, tb.sim.m, 1)
    ui = dict(paused=False, reset=False, speed=1.0)
    with server.gui.add_folder("Test bed"):
        server.gui.add_markdown(f"**{tb.cfg['name']}**  \ncontroller `{tb.cfg['controller']['type']}`, "
                                f"{len(tb.robot.joint_names)} joints, control {1 / tb.sim.control_dt:.0f} Hz, "
                                f"physics {1 / tb.sim.m.opt.timestep:.0f} Hz")
        pause = server.gui.add_checkbox("Pause", initial_value=False)
        pause.on_update(lambda e: ui.__setitem__("paused", e.target.value))
        speed = server.gui.add_slider("Speed (x real time)", min=0.1, max=4.0, step=0.1, initial_value=1.0)
        speed.on_update(lambda e: ui.__setitem__("speed", e.target.value))
        rb = server.gui.add_button("Reset")
        rb.on_click(lambda _: ui.__setitem__("reset", True))
        status = server.gui.add_text("Controller", initial_value="", disabled=True)
        timing = server.gui.add_text("Timing ms/step", initial_value="", disabled=True)
        episode = server.gui.add_text("Episode", initial_value="", disabled=True)
    if tb.command is not None:
        tb.command.gui(server, scene)
    scene.create_visualization_gui()
    print(f"[TESTBED] {tb.cfg['name']}: http://localhost:{server.get_port()}", flush=True)
    next_t = time.perf_counter()
    while True:
        if ui["reset"]:
            ui["reset"] = False
            tb.reset()
        if not ui["paused"]:
            tb.step()
            scene.update_from_mjdata(tb.sim.d)
            if tb.steps % 10 == 0:
                st = {k: v for k, v in tb.controller.status().items()}
                status.value = json.dumps(st, default=lambda x: np.round(x, 3).tolist() if isinstance(x, np.ndarray) else str(x))[:300]
                t = tb.timing_ms()
                timing.value = f"controller {t['controller']} | physics {t['physics']} | sensors {t['sensors']} | total {t['total']} ({t['realtime_factor']}x)"
                episode.value = f"t={tb.sim.time - tb.episode_t0:.1f}s  falls={tb.falls}"
        next_t += tb.sim.control_dt / ui["speed"]
        if tb.sim.realtime:
            time.sleep(max(0.0, next_t - time.perf_counter()))
        if time.perf_counter() - next_t > 1.0:  # fell behind (slow controller): don't try to catch up
            next_t = time.perf_counter()
