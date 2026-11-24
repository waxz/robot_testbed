# Standalone robot test bed

A config-driven MuJoCo test bed for **any robot** (humanoid, legged, arm, AGV / forklift) and **any controller**
(learned policies, classic control, MPC). One YAML file describes an experiment; the glue code runs the loop:

```
config.yaml ──> Sim (MuJoCo, CPU) ──> Robot (joints, floating base, actuators by name)
                  │
   every control period (default 20 ms):
   Sensors ──> state {sensor: {field: array}} ──┐
   Command source (joystick / waypoints / pose target / script) ──> command ──┤
                                                         Controller.step(t, state, command) ──> [Command(ctrl idx, values)]
                                                                         ──> MuJoCo actuators ──> physics (decimation)
   viser web UI (scene, command controls, controller status, timing)  |  headless run -> JSON results
```

Controllers see **only sensor readings and the command**. Model-based controllers build their own model copy for
kinematics and dynamics; none of them read the simulator's internal state. That keeps every controller one step
away from hardware: replace the sensors with real drivers and the actuator writes with real motor commands.

## Quick start (WSL)

```bash
source ~/dev/robot_world/wsl_gpu_env.sh
cd ~/dev/robot_world/testbed
python -m testbed.run configs/forklift_pure_pursuit.yaml               # web UI: http://localhost:8090
python -m testbed.run configs/panda_diff_ik.yaml --headless            # scripted run -> RESULT=...
python -m testbed.run configs/g1_rl_joystick.yaml --set controller.backend=ort-cuda
```

`--headless` runs the command source's script (or `headless_s`, default 30 s) and prints timing, falls, the final
controller status and small sensor readings; `--out file.json` stores them with a status trace every 0.5 s.
`--set key.path=value` overrides any config value (YAML syntax).

## Examples

| Config | Robot | Controller | Command | Measured (headless, CPU) |
|---|---|---|---|---|
| `g1_rl_joystick` | Unitree G1 29-DoF humanoid (scene from the SuperMimic export) | `rl_tracking`: ONNX SuperMimic tracker + joystick motion-matching streamer | joystick vx/vy/wz | 21 s scripted course, 0 falls, 3.7 ms/step (5.4x real time) |
| `go2_pd_stand` | Unitree Go2 quadruped (Menagerie, torque motors) | `pd_hold` | none | holds the home stance, 0 falls, 60x real time |
| `panda_diff_ik` | Franka Panda arm (Menagerie, position servos) | `diff_ik`: damped least-squares resolved rate + null-space posture | draggable pose target | target steps settle to < 0.4 cm in ~1 s |
| `forklift_pure_pursuit` | Custom forklift AGV (`robots/forklift`) | `diff_drive`: pure pursuit | waypoint loop | cross-track error mean 0.25 m (peaks at 90 deg corners); wheel odometry drifts ~0.6 m in 30 s |
| `forklift_teleop` | same | `diff_drive`: v/w + lift | joystick v/w/lift | script drives under the pallet, lifts it 0.28 m level and carries it back 1.2 m |

## Regression checks

Each config can declare pass/fail `checks:` on its headless result. `python -m testbed.check` runs every config
that has checks (each in its own process), prints a table, writes `outputs/check_report.json`, and exits 0 only if
everything passes, so it can gate commits or CI. Run it before and after changing the core.

```yaml
checks:
  - {metric: falls, max: 0}
  - {metric: final_state.pallet.pos.2, min: 0.2}                     # dotted path into the result JSON
  - {metric: trace.cross_track_m, agg: mean, after: 8, max: 0.35}    # over the 0.5 s status trace
  - {metric: trace.pos_err_cm, agg: max, at: [2.5, 4.5], max: 0.6}   # trace samples nearest these times
  - {metric: timing_ms.realtime_factor, min: 1.0}
```

`agg` is `mean`, `max`, `min` or `last` (default); `after` / `before` limit the time window; a metric that does not
exist counts as a failure. Current status (2026-11-24): 5/5 configs pass in about 23 s.

The package also has a `pyproject.toml` (`pip install -e .` gives `testbed` and `testbed-check` commands); until
then, run the modules from this folder.

## Config reference

```yaml
name: my_experiment
paths: {any: values}                 # free-form; reference with "${self.paths.any}"
sim:
  model: {xml: path} | {mjb: path} | {menagerie: unitree_go2, entry: scene}
  control_dt: 0.02                   # controller period; physics steps = control_dt / model timestep
  physics_dt: 0.002                  # optional override of the model timestep
  keyframe: home                     # optional reset keyframe
  realtime: true                     # viewer pacing
robot:
  root_body: base                    # floating base (omit for fixed-base arms)
  joints: auto | [names]             # auto = every actuated hinge/slide joint
  extra_ctrl: {actuator_name: value} # constant actuator values (e.g. an open gripper)
sensors:                             # list; each entry {type, name?, noise?: {field: std}, ...}
  - {type: joint_state}
  - {type: imu, body: pelvis}
command: {type: joystick | waypoints | pose_target, ..., script: [[t, {field: value}], ...]}
controller: {type: ..., ...}
episode_s: 30                        # optional episode length (reset afterwards)
headless_s: 30                       # headless duration when the command has no script
viewer: {port: 8090}
log: {enabled: false}                # true: full per-step state/command/status log next to --out
```

`${config_dir}` is the config file's folder; `~` and `${ENV_VAR}` are expanded. Names of bodies, joints, sites
and actuators match without model prefixes (`robot/left_knee_joint` matches `left_knee_joint`).

### Sensors

| Type | Fields | Notes |
|---|---|---|
| `joint_state` | `q`, `dq` | robot joint order |
| `imu` (`body`) | `quat`, `gyro` (body frame), `gravity_b`, `acc` | |
| `base_pose` (`body`) | `pos`, `quat`, `yaw`, `lin_vel_w`, `ang_vel_w` | ground truth: stand-in for an estimator / SLAM |
| `body_pose` (`body`) / `site_pose` (`site`) | `pos`, `quat` | ground truth |
| `height_scan` (`body`, `size`, `resolution`) | `heights`, `ground_z` | yaw-aligned ray grid, same layout as mjlab |
| `wheel_odometry` (wheel joints, radius, track) | `pos`, `yaw`, `v`, `w` | dead reckoning: drifts like the real thing |

### Command sources

`joystick` (named axes with ranges and presets), `waypoints` (2-D path + speed, drawn in the viewer),
`pose_target` (3-D target with a draggable gizmo). All replay a `script` in headless runs.

### Controllers

| Type | For | Actuators |
|---|---|---|
| `pd_hold` | any robot (sanity check, stance) | position servos, or torque via kp/kd |
| `diff_drive` | AGV / forklift: pure pursuit or v/w teleop, optional lift | velocity (wheels), position (lift) |
| `diff_ik` | arms: resolved-rate IK to a pose target | position servos |
| `rl_tracking` | G1: exported tracker + motion-matching streamer (`mimic_joystick/`) | position servos |

## Extending

- **New robot:** an MJCF / Menagerie name in `sim.model`, `root_body` if it floats, a sensor list. Run it with
  `pd_hold` first.
- **New sensor / command / controller:** subclass `Sensor` / `CommandSource` / `Controller` in `testbed/`, decorate
  with `@register("sensor" | "command" | "controller", "my_type")`, import it in the package `__init__`/`loop.py`,
  and use `type: my_type` in a config. A controller implements `step(t, state, command) -> [Command]`
  (plus optional `init_robot`, `reset`, `terminated`, `status`) and resolves ctrl indices once with
  `robot.ctrl_for(joint_names, kind)`.

## MPC roadmap ("old school" methods)

[qm_door](https://github.com/danisotelo/qm_door) runs, at its core, a nonlinear MPC on a centroidal model (OCS2),
a whole-body QP controller turning the MPC plan into joint torques, and a linear Kalman filter estimating the
base, for an AlienGo quadruped with a Z1 arm, on ROS Noetic / Gazebo. ROS Noetic targets Ubuntu 20.04, so the
original runs here only in Docker. The plan is to rebuild the same ideas inside this test bed, from simple to full:

| Step | Controller | Robot / config | Concepts | Needs |
|---|---|---|---|---|
| M1 | Linear MPC path tracking (linearised unicycle, QP over a horizon) | forklift, same path as pure pursuit | horizon, cost weights, input/rate constraints, warm start; compare corner error with pure pursuit | `osqp` |
| M2 | Joint-space or task-space MPC for the arm | Panda | linearised dynamics, joint limits as constraints | `osqp` |
| M3 | Convex MPC (single rigid body, ground reaction forces, friction cones; MIT Cheetah 3 style) + whole-body QP to torques | Go2 (torque motors) | gait schedule, contact forces, WBC, state estimation (swap `base_pose` for a Kalman filter) | `osqp`, Pinocchio (installed) |
| M4 | Nonlinear MPC on the centroidal / whole-body model (OCS2-like) | Go2 (+ arm), G1 | multiple shooting, SQP / RTI, constraint handling | `casadi` (or acados), Pinocchio |
| ref | qm_door itself | AlienGo + Z1 in Gazebo | the original stack, for comparison | Docker: ROS Noetic + OCS2 build |

Each MPC controller is a regular test-bed controller, so it runs against the same configs, metrics and web UI
as the pure-pursuit, IK and RL baselines.

Solvers installed in `~/.venv` (2026-11-24): `osqp` 1.1.3, `casadi` 3.8.2 (IPOPT, qpOASES, OSQP, FATROP, SQP
plugins). Pinocchio 4.1 is the PyPI `pin` wheel, built **without** CasADi bindings (`pinocchio.casadi` is not
available; PyPI has no build with them). So M4 uses either SQP with Pinocchio's analytical derivatives and OSQP
(how OCS2 / Crocoddyl work), or CasADi with a symbolic model written in the test bed; a conda-forge Pinocchio
(which has the bindings) would need a separate environment.
