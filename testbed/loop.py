"""The test-bed loop: config -> components -> (sensors -> command -> controller -> actuators -> physics)*."""
from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

import testbed.commands  # noqa: F401  (registers command sources)
import testbed.controllers  # noqa: F401  (registers controllers)
import testbed.sensors  # noqa: F401  (registers sensors)
from testbed.registry import build
from testbed.robot import Robot
from testbed.sim import Sim


class TestBed:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        s = cfg["sim"]
        self.sim = Sim(model=s["model"], control_dt=s.get("control_dt", 0.02), physics_dt=s.get("physics_dt"),
                       keyframe=s.get("keyframe"), realtime=s.get("realtime", True))
        rc = cfg.get("robot", {})
        self.robot = Robot(self.sim, root_body=rc.get("root_body"), joints=rc.get("joints", "auto"),
                           extra_ctrl=rc.get("extra_ctrl"))
        ctx = dict(sim=self.sim, robot=self.robot)
        self.sensors = [build("sensor", spec, **ctx) for spec in cfg.get("sensors", [])]
        self.command = build("command", cfg["command"], **ctx) if cfg.get("command") else None
        self.controller = build("controller", cfg["controller"], **ctx)
        self.max_episode_s = cfg.get("episode_s")
        self.timing = {"sensors": 0.0, "controller": 0.0, "physics": 0.0, "n": 0}
        self.falls = 0
        self.steps = 0
        self.log = [] if cfg.get("log", {}).get("enabled") else None
        self.reset()

    # ---- episode ----------------------------------------------------------------------------

    def read_state(self) -> dict:
        return {s.name: s.read() for s in self.sensors}

    def reset(self) -> None:
        self.sim.reset()
        if self.controller.init_robot():
            mujoco.mj_forward(self.sim.m, self.sim.d)
        for s in self.sensors:
            s.reset()
        if self.command is not None:
            self.command.reset()
        self.episode_t0 = self.sim.time
        self.state = self.read_state()
        self.controller.reset(self.state)

    def step(self) -> bool:
        """One control period. Returns True when the episode ended (and the test bed was reset)."""
        t = self.sim.time
        cmd = {}
        if self.command is not None:
            self.command.apply_script(t)
            cmd = self.command.values()
        t0 = time.perf_counter()
        actions = self.controller.step(t, self.state, cmd)
        t1 = time.perf_counter()
        self.robot.apply(actions)
        self.sim.step()
        t2 = time.perf_counter()
        self.state = self.read_state()
        t3 = time.perf_counter()
        T = self.timing
        T["controller"] += t1 - t0
        T["physics"] += t2 - t1
        T["sensors"] += t3 - t2
        T["n"] += 1
        self.steps += 1
        if self.log is not None:
            self.log.append(dict(t=t, cmd={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cmd.items()},
                                 status=self.controller.status(),
                                 state={n: {k: np.asarray(v).tolist() for k, v in r.items()} for n, r in self.state.items()}))
        ended = self.controller.terminated(self.state)
        if ended:
            self.falls += 1
        if self.max_episode_s and self.sim.time - self.episode_t0 >= self.max_episode_s:
            ended = True
        if ended:
            self.reset()
        return ended

    def timing_ms(self) -> dict:
        n = max(self.timing["n"], 1)
        out = {k: round(v / n * 1000, 3) for k, v in self.timing.items() if k != "n"}
        out["total"] = round(sum(out.values()), 3)
        out["realtime_factor"] = round(self.sim.control_dt * 1000 / out["total"], 1) if out["total"] else None
        return out

    # ---- runs -------------------------------------------------------------------------------

    def run_headless(self, seconds: float | None = None, out: str | None = None) -> dict:
        if not seconds:
            scripted = self.command is not None and self.command.script
            seconds = self.command.script_end() + 2.0 if scripted else self.cfg.get("headless_s", 30.0)
        n = int(seconds / self.sim.control_dt)
        trace = []
        for _ in range(n):
            self.step()
            if self.steps % max(1, int(0.5 / self.sim.control_dt)) == 0:
                trace.append(dict(t=round(self.sim.time, 2), **{k: v for k, v in self.controller.status().items()
                                                              if isinstance(v, (int, float, str, bool, list))}))
        small = {name: {k: np.round(np.asarray(v, float), 3).tolist() for k, v in r.items()}
                 for name, r in self.state.items() if sum(np.size(v) for v in r.values()) <= 16}
        res = dict(config=self.cfg["name"], sim_seconds=round(n * self.sim.control_dt, 2), falls=self.falls,
                   timing_ms=self.timing_ms(), final_status=self.controller.status(), final_state=small, trace=trace)
        if out:
            Path(out).expanduser().parent.mkdir(parents=True, exist_ok=True)
            Path(out).expanduser().write_text(json.dumps(res, indent=1, default=str))
            if self.log is not None:
                Path(out).expanduser().with_suffix(".log.json").write_text(json.dumps(self.log, default=str))
        return res
