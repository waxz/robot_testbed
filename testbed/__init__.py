"""Standalone robot test bed: config-driven MuJoCo loop for any robot and any controller.

    config (YAML) -> Sim (MuJoCo) -> Robot (joint/actuator map) -> Sensors -> Command source -> Controller
                  -> actuator commands -> physics -> ... ; viser web UI or headless metrics

See README.md.
"""
from testbed.registry import REGISTRY, build, register  # noqa: F401
