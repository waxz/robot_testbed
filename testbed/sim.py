"""MuJoCo simulator wrapper (CPU): model loading, timing, reset, stepping."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


def load_model(spec: dict) -> mujoco.MjModel:
    """model: {xml: path} | {mjb: path} | {menagerie: name, entry: scene}"""
    if "xml" in spec:
        return mujoco.MjModel.from_xml_path(str(Path(spec["xml"]).expanduser()))
    if "mjb" in spec:
        return mujoco.MjModel.from_binary_path(str(Path(spec["mjb"]).expanduser()))
    if "menagerie" in spec:
        import mujoco_menagerie
        return mujoco_menagerie.load(spec["menagerie"], spec.get("entry"))
    raise ValueError(f"model spec needs xml, mjb or menagerie: {spec}")


class Sim:
    def __init__(self, model: dict, control_dt: float = 0.02, physics_dt: float | None = None,
                 keyframe: str | int | None = None, realtime: bool = True):
        self.m = load_model(model)
        if physics_dt:
            self.m.opt.timestep = physics_dt
        self.d = mujoco.MjData(self.m)
        self.control_dt = float(control_dt)
        self.decimation = max(1, int(round(self.control_dt / self.m.opt.timestep)))
        self.keyframe = keyframe
        self.realtime = realtime

    def reset(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        if self.keyframe is not None and self.m.nkey > 0:
            k = self.keyframe if isinstance(self.keyframe, int) else mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, self.keyframe)
            if k < 0:
                raise ValueError(f"keyframe '{self.keyframe}' not in model")
            mujoco.mj_resetDataKeyframe(self.m, self.d, k)
        mujoco.mj_forward(self.m, self.d)

    def step(self) -> None:
        for _ in range(self.decimation):
            mujoco.mj_step(self.m, self.d)

    @property
    def time(self) -> float:
        return float(self.d.time)

    def ray_down(self, point, geomgroup=(0,), max_distance: float = 5.0):
        """Distance straight down from point to geoms in geomgroup (None on a miss)."""
        gg = np.zeros(6, dtype=np.uint8)
        for g in geomgroup:
            gg[g] = 1
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.m, self.d, np.asarray(point, dtype=np.float64), np.array([0.0, 0.0, -1.0]), gg, 1, -1, gid)
        return None if dist < 0 or dist > max_distance else float(dist)
