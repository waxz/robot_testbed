"""Differential-drive controller (AGV, forklift): pure pursuit on a waypoint path, or direct v/w teleop.

Pure pursuit: find the closest point on the path, walk one lookahead distance L forward along the path, and
steer along the arc through that point:
    curvature k = 2 * y_r / L^2   (y_r = lateral offset of the lookahead point in the robot frame)
    w = v * k;  wheel rates  wl = (v - w * b / 2) / r,  wr = (v + w * b / 2) / r
An optional lift (prismatic mast) follows the `lift` command axis.
"""
from __future__ import annotations

import math

import numpy as np

from testbed.controllers.base import Command, Controller
from testbed.registry import register


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@register("controller", "diff_drive")
class DiffDrive(Controller):
    def __init__(self, left_joint: str, right_joint: str, wheel_radius: float, track_width: float,
                 pose_sensor: str = "base_pose", lookahead: float = 0.6, max_speed: float = 1.5,
                 max_yaw_rate: float = 1.5, max_accel: float = 1.0, lift_joint: str | None = None, **kw):
        super().__init__(**kw)
        self.r, self.b = wheel_radius, track_width
        self.wheels = self.robot.ctrl_for([left_joint, right_joint], "velocity")
        self.lift = self.robot.ctrl_for([lift_joint], "position") if lift_joint else None
        self.pose_sensor, self.L = pose_sensor, lookahead
        self.vmax, self.wmax, self.amax = max_speed, max_yaw_rate, max_accel
        self.v = self.w = 0.0
        self.seg = 0
        self.info = {}

    def reset(self, state):
        self.v = self.w = 0.0
        self.seg = 0

    def _segments(self, pts, loop):
        n = len(pts)
        return [(i, (i + 1) % n) for i in range(n if loop else n - 1)]

    def _pursuit(self, pos, yaw, cmd):
        pts, loop = cmd["points"], cmd["loop"]
        segs = self._segments(pts, loop)
        # 1. closest point on the path, searched from the current segment a few segments ahead
        best = None
        for j in range(min(3, len(segs))):
            k = (self.seg + j) % len(segs) if loop else min(self.seg + j, len(segs) - 1)
            a, b = pts[segs[k][0]], pts[segs[k][1]]
            ab = b - a
            s = float(np.clip(np.dot(pos - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
            dist = float(np.linalg.norm(pos - (a + s * ab)))
            if best is None or dist < best[0] - 1e-6:
                best = (dist, k, s)
        xte, self.seg, s = best
        # 2. walk one lookahead distance forward along the path from that point
        k, rem = self.seg, self.L
        a, b = pts[segs[k][0]], pts[segs[k][1]]
        p = a + s * (b - a)
        end_reached = False
        while True:
            seg_left = float(np.linalg.norm(b - p))
            if rem <= seg_left or (not loop and k == len(segs) - 1):
                target = p + (b - p) * (min(rem, seg_left) / max(seg_left, 1e-9))
                end_reached = not loop and k == len(segs) - 1 and rem > seg_left
                break
            rem -= seg_left
            k = (k + 1) % len(segs)
            a, b = pts[segs[k][0]], pts[segs[k][1]]
            p = a
        d = target - pos
        dist = float(np.linalg.norm(d))
        y_r = -math.sin(yaw) * d[0] + math.cos(yaw) * d[1]
        x_r = math.cos(yaw) * d[0] + math.sin(yaw) * d[1]
        v = cmd["speed"]
        if end_reached:
            v = min(v, 0.8 * dist)  # slow down onto the last point
        if x_r < 0.1 * self.L:  # target beside/behind (sharp corner): turn in place towards it
            return 0.0, math.copysign(self.wmax * 0.5, y_r), self._set_info(xte, dist)
        k_curv = 2.0 * y_r / max(dist, 1e-3) ** 2  # pure pursuit: arc through the lookahead point
        return v, v * k_curv, self._set_info(xte, dist)

    def _set_info(self, xte, dist):
        self.info = {"segment": self.seg, "cross_track_m": round(xte, 3), "dist_to_lookahead": round(dist, 2)}
        return None

    def step(self, t, state, command):
        pose = state[self.pose_sensor]
        pos, yaw = np.asarray(pose["pos"])[:2], float(pose["yaw"])
        if "points" in command:
            v_des, w_des, _ = self._pursuit(pos, yaw, command)
        else:
            v_des, w_des = command.get("v", 0.0), command.get("w", 0.0)
        v_des = float(np.clip(v_des, -self.vmax, self.vmax))
        w_des = float(np.clip(w_des, -self.wmax, self.wmax))
        dv = self.amax * self.sim.control_dt  # acceleration limit
        self.v += float(np.clip(v_des - self.v, -dv, dv))
        self.w += float(np.clip(w_des - self.w, -3 * dv, 3 * dv))
        wl = (self.v - self.w * self.b / 2) / self.r
        wr = (self.v + self.w * self.b / 2) / self.r
        out = [Command(self.wheels, np.array([wl, wr]))]
        if self.lift is not None:
            out.append(Command(self.lift, np.array([command.get("lift", 0.0)])))
        return out

    def status(self):
        return {"v": round(self.v, 2), "w": round(self.w, 2), **self.info}
