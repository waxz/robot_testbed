"""Learned motion-tracking policy (exported SuperMimic G1 tracker, optionally + terrain residual) driven by the
joystick motion-matching streamer.

Needs an export directory from mimic_joystick/export_onnx.py (policy.onnx + meta.json) and the mimic_joystick and
SuperMimic source trees on `python_paths`. Inputs are built from test-bed sensors only:
    joint_state (encoders), imu on the pelvis (gyro, gravity direction), base_pose (pelvis pose for the
    streamer's anchoring), torso pose (reference-orientation term), height_scan (residual policies; its centre
    ray also gives the ground height under the pelvis).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from testbed.controllers.base import Command, Controller
from testbed.registry import register
from testbed.sensors import quat_to_mat

TERMS = (29, 29, 3, 3, 29)  # joint_pos, joint_vel, base_ang_vel, projected_gravity, last action


@register("controller", "rl_tracking")
class RLTracking(Controller):
    def __init__(self, export_dir: str, python_paths: list[str], backend: str = "ort-cpu", streamer_device: str = "cpu",
                 sensors: dict | None = None, **kw):
        super().__init__(**kw)
        for p in python_paths:
            p = str(Path(p).expanduser())
            if p not in sys.path:
                sys.path.insert(0, p)
        from joystick_mm import MMConfig
        from onnx_runner import Policy
        from standalone_streamer import StandaloneStreamer

        self.s = {"joints": "joint_state", "imu": "imu", "base": "base_pose", "torso": "torso_pose", "scan": "height_scan",
                  **(sensors or {})}
        d = Path(export_dir).expanduser()
        self.meta = meta = json.loads((d / "meta.json").read_text())
        self.policy = Policy(d, meta, backend)
        self.use_scan = "height_scan" in meta["policy_inputs"]
        r = self.robot
        self.obs_idx = r.joint_index(meta["obs_joint_names"])           # robot joint order -> policy order
        self.default = np.asarray(meta["default_joint_pos"])
        self.ctrl = r.ctrl_for(meta["action_joint_names"], "position")
        self.scale, self.offset = np.asarray(meta["action_scale"]), np.asarray(meta["action_offset"])
        lib_names = [str(n) for n in np.load(meta["motion_clips"][0])["joint_names"]]
        self.lib_to_obs = np.asarray([lib_names.index(n) for n in meta["obs_joint_names"]])
        self.streamer = StandaloneStreamer(meta["motion_clips"], self.default[np.argsort(self.lib_to_obs)],
                                           device=streamer_device, mm=MMConfig(log_costs=False))
        self.H = meta["history"]
        self.hist = np.zeros((self.H, sum(TERMS)))
        self.action = np.zeros(len(self.ctrl))
        self.stick = None
        self.first = True

    # ---- reset ------------------------------------------------------------------------------

    def init_robot(self) -> bool:
        """Start on the streamer's standing reference pose (above the ground at the robot's spawn position)."""
        s, r = self.streamer, self.robot
        d = self.sim.d
        origin = d.xpos[r.root_body].copy()
        g = self.sim.ray_down(origin + np.array([0, 0, 2.0]))
        origin[2] = (origin[2] + 2.0 - g) if g is not None else 0.0
        s.set_origin(origin)
        s.terrain_z.zero_()
        s.reset(0.0)
        pos, quat, jp, lin, ang, jv = s.ref_root_world()
        r.set_base_state(pos + np.array([0, 0, 0.03]), quat, lin, quat_to_mat(quat).T @ ang)
        r.set_joint_state(self.meta["obs_joint_names"], jp[self.lib_to_obs], jv[self.lib_to_obs])
        return True

    def _frame(self, state):
        js, imu = state[self.s["joints"]], state[self.s["imu"]]
        q, dq = js["q"][self.obs_idx], js["dq"][self.obs_idx]
        return np.concatenate([q - self.default, dq, imu["gyro"], imu["gravity_b"], self.action])

    def reset(self, state):
        self.action[:] = 0.0
        self.hist[:] = self._frame(state)
        self.first = True

    # ---- step -------------------------------------------------------------------------------

    def step(self, t, state, command):
        st = self.streamer
        stick = (command.get("vx", 0.0), command.get("vy", 0.0), command.get("wz", 0.0))
        if stick != self.stick:
            st.set_joystick(*stick)
            self.stick = stick
        if not self.first:  # the streamer and history advance with each completed physics period
            b = state[self.s["base"]]
            st.set_robot_state(b["pos"], b["quat"], b["lin_vel_w"], b["ang_vel_w"])
            if self.s["scan"] in state:
                gz = state[self.s["scan"]]["ground_z"]
                st.set_ground_height(None if np.isnan(gz) else float(gz))
            st.step()
            self.hist = np.roll(self.hist, -1, axis=0)
            self.hist[-1] = self._frame(state)
        self.first = False
        tp, tq = st.ref_torso_world()
        torso = state[self.s["torso"]]
        R = quat_to_mat(torso["quat"])
        anchor = np.concatenate([R.T @ (tp - torso["pos"]), (R.T @ quat_to_mat(tq))[:, :2].reshape(-1)])
        goal = np.concatenate([st.goal_frames(), anchor]).astype(np.float32)
        obs, off = [], 0
        for w in TERMS:  # term-major history, oldest -> newest
            obs.append(self.hist[:, off:off + w].reshape(-1))
            off += w
        inputs = [np.concatenate(obs).astype(np.float32), goal]
        if self.use_scan:
            inputs.append(np.asarray(state[self.s["scan"]]["heights"], np.float32))
        self.action = self.policy(*inputs).astype(np.float64)
        return [Command(self.ctrl, self.offset + self.scale * self.action)]

    def terminated(self, state) -> bool:
        b = state[self.s["base"]]
        pos, quat, *_ = self.streamer.ref_root_world()
        g = np.array([0.0, 0.0, -1.0])
        t = self.meta["terminations"]
        return (abs(pos[2] - b["pos"][2]) > t["root_height"]
                or abs((quat_to_mat(quat).T @ g)[2] - (quat_to_mat(b["quat"]).T @ g)[2]) > t["root_tilt"])

    def status(self):
        s = self.streamer.status()
        return {"clip": s["clip"], "hold": s["hold"], "stick": s["stick"], "backend": self.policy.provider}
