"""Controller interface.

A controller sees only sensor readings and the operator command, and returns actuator Commands:

    init_robot()                       optional: place the robot at reset (e.g. on a reference pose)
    reset(state)                       after the reset, with the first sensor readings
    step(t, state, command) -> [Command]   once per control period, before physics
    terminated(state) -> bool          optional failure check (fall, collision, ...)
    status() -> dict                   shown in the viewer and stored in logs

Model-based controllers (IK, MPC, WBC) may build their OWN model copy (self.model_data()) for kinematics and
dynamics; they never read the simulator's MjData.
"""
from __future__ import annotations

import mujoco

from testbed.robot import Command  # noqa: F401  (re-export for controllers)


class Controller:
    def __init__(self, sim, robot, **_):
        self.sim, self.robot = sim, robot

    def model_data(self) -> mujoco.MjData:
        """A private MjData on the same model, for the controller's own kinematics / dynamics."""
        return mujoco.MjData(self.sim.m)

    def init_robot(self) -> bool:
        return False

    def reset(self, state: dict) -> None:
        pass

    def step(self, t: float, state: dict, command: dict) -> list:
        raise NotImplementedError

    def terminated(self, state: dict) -> bool:
        return False

    def status(self) -> dict:
        return {}
