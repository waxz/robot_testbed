"""Robot description on top of a MuJoCo model: joints, floating base, actuators, name lookup.

Names are matched ignoring prefixes ("robot/left_knee_joint" matches "left_knee_joint"), so models exported from
mjlab, Menagerie models and hand-written MJCF work the same way.

Actuators are classified from their MuJoCo parameters:
  position  (servo: bias = -kp * q ...)  -> ctrl is a joint position target
  velocity  (bias = -kv * qdot only)     -> ctrl is a joint velocity target
  torque    (no bias)                    -> ctrl is a force / torque
Controllers resolve ctrl indices once (ctrl_for) and return Commands (ctrl index array, values).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def short(name: str | None) -> str:
    return (name or "").split("/")[-1]


@dataclass
class Command:
    """Values for a set of MuJoCo ctrl slots."""
    ctrl_idx: np.ndarray
    values: np.ndarray


class Robot:
    def __init__(self, sim, root_body: str | None = None, joints: str | list[str] = "auto",
                 extra_ctrl: dict[str, float] | None = None):
        self.sim = sim
        m = sim.m
        J = mujoco.mjtJoint
        free = [j for j in range(m.njnt) if m.jnt_type[j] == J.mjJNT_FREE]
        self.has_floating_base = bool(free) and root_body is not None
        self.root_body = self.body_id(root_body) if root_body else None
        if self.has_floating_base:
            fj = [j for j in free if m.jnt_bodyid[j] == self.root_body]
            fj = fj[0] if fj else free[0]
            self.root_qadr, self.root_vadr = int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
        if joints == "auto":  # every hinge/slide joint that has an actuator, in model order
            acted = {int(m.actuator_trnid[a, 0]) for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
            jids = [j for j in range(m.njnt) if m.jnt_type[j] in (J.mjJNT_HINGE, J.mjJNT_SLIDE) and j in acted]
        else:
            jids = [self.joint_id(n) for n in joints]
        self.joint_ids = jids
        self.joint_names = [short(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) for j in jids]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids], dtype=int)
        self.vadr = np.array([m.jnt_dofadr[j] for j in jids], dtype=int)
        self.extra_ctrl = {self.actuator_id(k): float(v) for k, v in (extra_ctrl or {}).items()}

    # ---- lookup ---------------------------------------------------------------------------

    def _find(self, obj, name: str) -> int:
        m = self.sim.m
        n = {mujoco.mjtObj.mjOBJ_BODY: m.nbody, mujoco.mjtObj.mjOBJ_JOINT: m.njnt, mujoco.mjtObj.mjOBJ_SITE: m.nsite,
             mujoco.mjtObj.mjOBJ_ACTUATOR: m.nu, mujoco.mjtObj.mjOBJ_GEOM: m.ngeom}[obj]
        hits = [i for i in range(n) if short(mujoco.mj_id2name(m, obj, i)) == name]
        if len(hits) != 1:
            raise KeyError(f"{obj.name} '{name}': {len(hits)} matches")
        return hits[0]

    def body_id(self, name): return self._find(mujoco.mjtObj.mjOBJ_BODY, name)
    def joint_id(self, name): return self._find(mujoco.mjtObj.mjOBJ_JOINT, name)
    def site_id(self, name): return self._find(mujoco.mjtObj.mjOBJ_SITE, name)
    def actuator_id(self, name): return self._find(mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def actuator_kind(self, a: int) -> str:
        b = self.sim.m.actuator_biasprm[a]
        if abs(b[1]) > 0:
            return "position"
        if abs(b[2]) > 0:
            return "velocity"
        return "torque"

    def ctrl_for(self, joint_names: list[str], kind: str | None = None) -> np.ndarray:
        """ctrl indices of the actuators driving these joints (optionally asserting their kind)."""
        m = self.sim.m
        out = []
        for n in joint_names:
            j = self.joint_id(n)
            hits = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == j]
            if not hits:
                raise KeyError(f"no actuator on joint '{n}'")
            if kind is not None:
                hits = [a for a in hits if self.actuator_kind(a) == kind] or hits
                if self.actuator_kind(hits[0]) != kind:
                    raise ValueError(f"joint '{n}' has a {self.actuator_kind(hits[0])} actuator, controller wants {kind}")
            out.append(hits[0])
        return np.array(out, dtype=int)

    def joint_index(self, names: list[str]) -> np.ndarray:
        """Positions of these joints in self.joint_names (for reordering joint_state readings)."""
        return np.array([self.joint_names.index(n) for n in names], dtype=int)

    # ---- state access (for initialisation; controllers read sensors, not this) -------------

    def set_joint_state(self, names, q, dq=None) -> None:
        d, m = self.sim.d, self.sim.m
        for i, n in enumerate(names):
            j = self.joint_id(n)
            d.qpos[m.jnt_qposadr[j]] = q[i]
            if dq is not None:
                d.qvel[m.jnt_dofadr[j]] = dq[i]

    def set_base_state(self, pos, quat, lin_vel_w=None, ang_vel_b=None) -> None:
        d, qa, va = self.sim.d, self.root_qadr, self.root_vadr
        d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = pos, quat
        if lin_vel_w is not None:
            d.qvel[va:va + 3] = lin_vel_w
        if ang_vel_b is not None:
            d.qvel[va + 3:va + 6] = ang_vel_b

    # ---- actuation ---------------------------------------------------------------------------

    def apply(self, commands: list[Command]) -> None:
        ctrl = self.sim.d.ctrl
        for a, v in self.extra_ctrl.items():
            ctrl[a] = v
        for c in commands:
            ctrl[c.ctrl_idx] = c.values
