"""Resolved-rate (differential) inverse kinematics for an arm with joint position servos.

    v = Kp * e (position, optionally orientation), clipped
    dq = J^T (J J^T + lambda^2 I)^-1 v  +  (I - J^+ J) * k_null * (q_home - q)      (damped least squares + posture)
    q_des <- q_des + dq * dt   -> joint position actuators

The Jacobian comes from the controller's own model copy, filled from the joint encoder readings.
"""
from __future__ import annotations

import mujoco
import numpy as np

from testbed.controllers.base import Command, Controller
from testbed.registry import register


@register("controller", "diff_ik")
class DiffIK(Controller):
    def __init__(self, joints: list[str], site: str | None = None, body: str | None = None, kp_pos: float = 4.0,
                 kp_rot: float = 2.0, damping: float = 0.05, max_lin: float = 0.5, max_ang: float = 1.5,
                 k_null: float = 1.0, joint_sensor: str = "joint_state", **kw):
        super().__init__(**kw)
        r = self.robot
        self.names, self.js = joints, joint_sensor
        self.idx = r.joint_index(joints)
        self.ctrl = r.ctrl_for(joints, "position")
        self.site = r.site_id(site) if site else None
        self.body = r.body_id(body) if body else None
        assert (self.site is None) != (self.body is None), "diff_ik needs exactly one of site / body"
        self.kp_pos, self.kp_rot, self.lam = kp_pos, kp_rot, damping
        self.max_lin, self.max_ang, self.k_null = max_lin, max_ang, k_null
        self.md = self.model_data()
        m = self.sim.m
        self.dofs = np.array([m.jnt_dofadr[r.joint_id(n)] for n in joints])
        self.qadr = np.array([m.jnt_qposadr[r.joint_id(n)] for n in joints])
        self.q_des = self.q_home = None
        self.err = (0.0, 0.0)

    def _kin(self, q):
        m, d = self.sim.m, self.md
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        if self.site is not None:
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            pos, mat = d.site_xpos[self.site].copy(), d.site_xmat[self.site].copy()
        else:
            mujoco.mj_jacBody(m, d, jp, jr, self.body)
            pos, mat = d.xpos[self.body].copy(), d.xmat[self.body].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return pos, quat, jp[:, self.dofs], jr[:, self.dofs]

    def reset(self, state):
        q = state[self.js]["q"][self.idx]
        self.q_des, self.q_home = q.copy(), q.copy()

    def step(self, t, state, command):
        q = state[self.js]["q"][self.idx]
        pos, quat, Jp, Jr = self._kin(q)
        e_p = command["position"] - pos
        v = self.kp_pos * e_p
        n = np.linalg.norm(v)
        if n > self.max_lin:
            v *= self.max_lin / n
        if command.get("orientation"):
            e_r = np.zeros(3)
            mujoco.mju_subQuat(e_r, np.asarray(command["quat"], float), quat)
            w = self.kp_rot * e_r
            n = np.linalg.norm(w)
            if n > self.max_ang:
                w *= self.max_ang / n
            J, x = np.vstack([Jp, Jr]), np.r_[v, w]
            self.err = (float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r)))
        else:
            J, x = Jp, v
            self.err = (float(np.linalg.norm(e_p)), 0.0)
        JJt = J @ J.T + self.lam ** 2 * np.eye(J.shape[0])
        J_pinv = J.T @ np.linalg.inv(JJt)
        null = (np.eye(len(q)) - J_pinv @ J) @ (self.k_null * (self.q_home - self.q_des))
        dq = J_pinv @ x + null
        self.q_des = self.q_des + dq * self.sim.control_dt
        return [Command(self.ctrl, self.q_des.copy())]

    def status(self):
        return {"pos_err_cm": round(self.err[0] * 100, 2), "rot_err_deg": round(np.degrees(self.err[1]), 1)}
