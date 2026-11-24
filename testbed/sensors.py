"""Sensors: each reads the simulator once per control step and returns a dict of numpy arrays.

The state handed to a controller is {sensor_name: {field: array}}. Ground-truth sensors are marked as such;
swap them for estimators (e.g. wheel_odometry instead of base_pose) to test a controller under realistic
information. Optional Gaussian noise per sensor: `noise: {field: std}`.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from testbed.registry import register


def quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def yaw_of(q) -> float:
    w, x, y, z = q
    return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Sensor:
    ground_truth = False

    def __init__(self, sim, robot, name: str | None = None, noise: dict | None = None, seed: int = 0):
        self.sim, self.robot = sim, robot
        self.name = name or self.type_name
        self.noise = noise or {}
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def read(self) -> dict:
        out = self._read()
        for k, std in self.noise.items():
            out[k] = out[k] + self.rng.normal(0.0, std, np.shape(out[k]))
        return out

    def _read(self) -> dict:
        raise NotImplementedError


@register("sensor", "joint_state")
class JointState(Sensor):
    """Encoders: position and velocity of the robot's joints (robot.joint_names order)."""
    def _read(self):
        d = self.sim.d
        return {"q": d.qpos[self.robot.qadr].copy(), "dq": d.qvel[self.robot.vadr].copy()}


@register("sensor", "imu")
class Imu(Sensor):
    """Orientation, angular velocity (body frame), gravity direction in the body frame, linear acceleration."""
    def __init__(self, body: str, **kw):
        super().__init__(**kw)
        self.b = self.robot.body_id(body)

    def _read(self):
        d, m = self.sim.d, self.sim.m
        quat = d.xquat[self.b].copy()
        R = quat_to_mat(quat)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.b, vel, 1)  # local frame: [ang, lin]
        acc = np.zeros(6)
        mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_BODY, self.b, acc, 1)
        return {"quat": quat, "gyro": vel[:3], "gravity_b": R.T @ np.array([0.0, 0.0, -1.0]), "acc": acc[3:]}


@register("sensor", "base_pose")
class BasePose(Sensor):
    """Ground truth pose and velocity of a body (stand-in for a state estimator / SLAM)."""
    ground_truth = True

    def __init__(self, body: str, **kw):
        super().__init__(**kw)
        self.b = self.robot.body_id(body)

    def _read(self):
        d, m = self.sim.d, self.sim.m
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.b, vel, 0)  # world frame: [ang, lin]
        q = d.xquat[self.b].copy()
        return {"pos": d.xpos[self.b].copy(), "quat": q, "yaw": np.array(yaw_of(q)), "ang_vel_w": vel[:3], "lin_vel_w": vel[3:]}


@register("sensor", "body_pose")
class BodyPose(Sensor):
    """Ground truth pose of any body (e.g. a humanoid torso; on hardware: IMU + forward kinematics)."""
    ground_truth = True

    def __init__(self, body: str, **kw):
        super().__init__(**kw)
        self.b = self.robot.body_id(body)

    def _read(self):
        d = self.sim.d
        return {"pos": d.xpos[self.b].copy(), "quat": d.xquat[self.b].copy()}


@register("sensor", "site_pose")
class SitePose(Sensor):
    """Pose of a site, e.g. an arm's end effector."""
    ground_truth = True

    def __init__(self, site: str, **kw):
        super().__init__(**kw)
        self.s = self.robot.site_id(site)

    def _read(self):
        d = self.sim.d
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, d.site_xmat[self.s])
        return {"pos": d.site_xpos[self.s].copy(), "quat": quat}


@register("sensor", "height_scan")
class HeightScan(Sensor):
    """Grid of downward rays around a body, yaw-aligned: height of the body above the terrain hit per ray."""
    def __init__(self, body: str, size=(1.0, 1.0), resolution: float = 0.1, max_distance: float = 5.0,
                 geom_groups=(0,), **kw):
        super().__init__(**kw)
        self.b = self.robot.body_id(body)
        x = np.arange(-size[0] / 2, size[0] / 2 + resolution * 0.5, resolution)
        y = np.arange(-size[1] / 2, size[1] / 2 + resolution * 0.5, resolution)
        gx, gy = np.meshgrid(x, y, indexing="xy")  # same ray order as mjlab's GridPatternCfg
        self.offsets = np.stack([gx.ravel(), gy.ravel()], -1)
        self.max_distance = max_distance
        self.gg = np.zeros(6, dtype=np.uint8)
        for g in geom_groups:
            self.gg[g] = 1
        self.gid = np.zeros(1, dtype=np.int32)

    def _read(self):
        d, m = self.sim.d, self.sim.m
        p = d.xpos[self.b]
        yaw = yaw_of(d.xquat[self.b])
        c, s = math.cos(yaw), math.sin(yaw)
        down = np.array([0.0, 0.0, -1.0])
        h = np.empty(len(self.offsets), dtype=np.float32)
        for i, (ox, oy) in enumerate(self.offsets):
            o = np.array([p[0] + c * ox - s * oy, p[1] + s * ox + c * oy, p[2]])
            dist = mujoco.mj_ray(m, d, o, down, self.gg, 1, -1, self.gid)
            h[i] = dist if 0 <= dist <= self.max_distance else self.max_distance
        mid = mujoco.mj_ray(m, d, p.copy(), down, self.gg, 1, -1, self.gid)
        return {"heights": h, "ground_z": np.array(p[2] - mid if mid >= 0 else np.nan)}


@register("sensor", "wheel_odometry")
class WheelOdometry(Sensor):
    """Dead reckoning for a differential drive from wheel encoder rates (drifts, like the real thing)."""
    def __init__(self, left_joint: str, right_joint: str, wheel_radius: float, track_width: float, **kw):
        super().__init__(**kw)
        m = self.sim.m
        self.vl = m.jnt_dofadr[self.robot.joint_id(left_joint)]
        self.vr = m.jnt_dofadr[self.robot.joint_id(right_joint)]
        self.r, self.w = wheel_radius, track_width
        self.reset()

    def reset(self):
        self.x = self.y = 0.0
        self.yaw = 0.0
        self._init = False

    def _read(self):
        d = self.sim.d
        if not self._init and self.robot.root_body is not None:  # start from the true pose, then integrate
            self.x, self.y = d.xpos[self.robot.root_body][:2]
            self.yaw = yaw_of(d.xquat[self.robot.root_body])
            self._init = True
        wl, wr = d.qvel[self.vl], d.qvel[self.vr]
        v, w = self.r * (wl + wr) / 2, self.r * (wr - wl) / self.w
        dt = self.sim.control_dt
        self.x += v * math.cos(self.yaw + 0.5 * w * dt) * dt
        self.y += v * math.sin(self.yaw + 0.5 * w * dt) * dt
        self.yaw += w * dt
        return {"pos": np.array([self.x, self.y]), "yaw": np.array(self.yaw), "v": np.array(v), "w": np.array(w)}
