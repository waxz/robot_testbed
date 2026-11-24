"""Hold a joint posture: the simplest controller, and a sanity check for any new robot config.

Targets come from `targets: {joint: value}`, else from the joint positions at reset (e.g. a keyframe).
Position actuators get the target as ctrl; torque actuators get kp * (q* - q) - kd * dq.
"""
from __future__ import annotations

import numpy as np

from testbed.controllers.base import Command, Controller
from testbed.registry import register


@register("controller", "pd_hold")
class PdHold(Controller):
    def __init__(self, targets: dict | None = None, kp: float = 50.0, kd: float = 2.0, joint_sensor: str = "joint_state", **kw):
        super().__init__(**kw)
        self.targets_cfg, self.kp, self.kd, self.js = targets, kp, kd, joint_sensor
        r = self.robot
        self.kinds = [r.actuator_kind(a) for a in r.ctrl_for(r.joint_names)]
        self.ctrl = r.ctrl_for(r.joint_names)
        self.q_star = None

    def reset(self, state):
        q = state[self.js]["q"].copy()
        if self.targets_cfg:
            for n, v in self.targets_cfg.items():
                q[self.robot.joint_names.index(n)] = v
        self.q_star = q

    def step(self, t, state, command):
        q, dq = state[self.js]["q"], state[self.js]["dq"]
        out = np.where(np.array(self.kinds) == "position", self.q_star, self.kp * (self.q_star - q) - self.kd * dq)
        return [Command(self.ctrl, out)]

    def status(self):
        return {"actuators": dict(zip(*np.unique(self.kinds, return_counts=True)))}
