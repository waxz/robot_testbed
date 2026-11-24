"""Command sources: what the operator (or a script) asks for, independent of how a controller achieves it.

Each source returns a dict once per control step. In the viewer it builds its own controls; headless runs replay
an optional `script` of [time_s, {field: value}] entries.
"""
from __future__ import annotations

import numpy as np

from testbed.registry import register


class CommandSource:
    def __init__(self, sim, robot, script: list | None = None, **_):
        self.sim, self.robot = sim, robot
        self.script = sorted(script or [], key=lambda e: e[0])
        self._next = 0

    def reset(self):
        self._next = 0

    def apply_script(self, t: float) -> None:
        while self._next < len(self.script) and self.script[self._next][0] <= t:
            self.set(self.script[self._next][1])
            self._next += 1

    def set(self, values: dict) -> None:
        raise NotImplementedError

    def values(self) -> dict:
        raise NotImplementedError

    def gui(self, server, scene=None) -> None:
        pass

    def script_end(self) -> float:
        return self.script[-1][0] if self.script else 0.0


@register("command", "joystick")
class Joystick(CommandSource):
    """Named axes with ranges, e.g. vx/vy/wz for legged robots, v/w + lift for a forklift."""
    def __init__(self, axes: dict, presets: dict | None = None, **kw):
        super().__init__(**kw)
        self.axes = {k: (float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 0.0) for k, v in axes.items()}
        self.presets = presets or {}
        self.state = {k: v[2] for k, v in self.axes.items()}

    def reset(self):
        super().reset()
        self.state = {k: v[2] for k, v in self.axes.items()}

    def set(self, values):
        for k, v in values.items():
            self.state[k] = float(v)

    def values(self):
        return dict(self.state)

    def gui(self, server, scene=None):
        sliders = {}
        with server.gui.add_folder("Command (joystick)"):
            for k, (lo, hi, init) in self.axes.items():
                sliders[k] = server.gui.add_slider(k, min=lo, max=hi, step=(hi - lo) / 100, initial_value=init)
                sliders[k].on_update(lambda e, k=k: self.state.__setitem__(k, float(e.target.value)))
            zero = server.gui.add_button("Zero")

            @zero.on_click
            def _(_):
                for k, (lo, hi, init) in self.axes.items():
                    sliders[k].value = init if k.startswith("lift") else 0.0
            if self.presets:
                group = server.gui.add_button_group("Presets", options=tuple(self.presets))

                @group.on_click
                def _(e):
                    for k, v in self.presets[e.target.value].items():
                        sliders[k].value = v


@register("command", "waypoints")
class Waypoints(CommandSource):
    """A path of 2-D points plus a cruise speed; drawn in the viewer."""
    def __init__(self, points: list, speed: float = 0.5, loop: bool = True, **kw):
        super().__init__(**kw)
        self.points = np.asarray(points, dtype=float)
        self.speed, self.loop, self.enabled = float(speed), bool(loop), True

    def set(self, values):
        self.speed = float(values.get("speed", self.speed))
        self.enabled = bool(values.get("enabled", self.enabled))

    def values(self):
        return {"points": self.points, "speed": self.speed if self.enabled else 0.0, "loop": self.loop}

    def gui(self, server, scene=None):
        pts = np.c_[self.points, np.full(len(self.points), 0.02)]
        if self.loop:
            pts = np.vstack([pts, pts[:1]])
        server.scene.add_spline_catmull_rom("/testbed/path", positions=pts, tension=0.0, line_width=3.0, color=(255, 140, 0))
        for i, p in enumerate(self.points):
            server.scene.add_icosphere(f"/testbed/wp{i}", radius=0.05, color=(255, 140, 0), position=(p[0], p[1], 0.05))
        with server.gui.add_folder("Command (waypoints)"):
            sp = server.gui.add_slider("speed (m/s)", min=0.0, max=2.0, step=0.05, initial_value=self.speed)
            sp.on_update(lambda e: self.set({"speed": e.target.value}))
            en = server.gui.add_checkbox("follow path", initial_value=True)
            en.on_update(lambda e: self.set({"enabled": e.target.value}))


@register("command", "pose_target")
class PoseTarget(CommandSource):
    """A 3-D target pose (end effector), draggable in the viewer with a transform gizmo."""
    def __init__(self, position, quat=(1.0, 0.0, 0.0, 0.0), orientation: bool = False, **kw):
        super().__init__(**kw)
        self.init = (np.asarray(position, float), np.asarray(quat, float))
        self.pos, self.quat = self.init[0].copy(), self.init[1].copy()
        self.use_orientation = orientation
        self._gizmo = None

    def reset(self):
        super().reset()
        self.pos, self.quat = self.init[0].copy(), self.init[1].copy()
        if self._gizmo is not None:
            self._gizmo.position, self._gizmo.wxyz = tuple(self.pos), tuple(self.quat)

    def set(self, values):
        if "position" in values:
            self.pos = np.asarray(values["position"], float)
        if "quat" in values:
            self.quat = np.asarray(values["quat"], float)
        if self._gizmo is not None:
            self._gizmo.position, self._gizmo.wxyz = tuple(self.pos), tuple(self.quat)

    def values(self):
        return {"position": self.pos.copy(), "quat": self.quat.copy(), "orientation": self.use_orientation}

    def gui(self, server, scene=None):
        g = server.scene.add_transform_controls("/testbed/target", scale=0.15, position=tuple(self.pos), wxyz=tuple(self.quat))
        self._gizmo = g

        @g.on_update
        def _(_):
            self.pos, self.quat = np.array(g.position), np.array(g.wxyz)
        with server.gui.add_folder("Command (pose target)"):
            server.gui.add_markdown("Drag the gizmo to move the end-effector target.")
            r = server.gui.add_button("Reset target")
            r.on_click(lambda _: self.set({"position": self.init[0], "quat": self.init[1]}))
