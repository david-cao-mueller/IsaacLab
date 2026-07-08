# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: does runtime joint position-limit randomization reach Kamino?

Drives the real DR term :class:`isaaclab.envs.mdp.events.randomize_joint_parameters`
(position limits, ``operation="abs"``), writing ``joint_limit_lower`` /
``joint_limit_upper`` (``JOINT_DOF_PROPERTIES``). After PR #3386 Kamino re-copies these
into ``joints.q_j_min`` / ``q_j_max`` -> ROUTED. A constant torque drives the
horizontal spinner into the upper limit (gravity off); it should stop there.

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_joint_limits.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime joint position-limit changes route to Kamino.")
args_cli, simulation_app = common.launch(parser)

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_joint_parameters
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics.newton_manager import NewtonManager

# re-center from rest this often so the limit stop can be re-observed
RECENTER_STEPS = 150


class State:
    def __init__(self) -> None:
        self.lower = -0.4
        self.upper = 0.4
        self.torque = 0.35
        self.dirty = False
        self.lines: list[str] = []


def main() -> None:
    dt = max(args_cli.dt, common.SPIN_DT)  # honor a larger user --dt, otherwise use the bigger default
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=dt, gravity=(0.0, 0.0, 0.0), physics=common.make_physics_cfg(args_cli.physics)
    )
    sim = SimulationContext(sim_cfg)
    scene = common.build_scene(sim, common.spinner_scene_cfg())
    robot = scene["robot"]
    common.set_spinner_camera(sim)
    common.settle(sim, robot, args_cli.settle_steps)

    common.warn_if_not_kamino(sim)
    kamino = "kamino" in sim.physics_manager.__name__.lower()
    model = NewtonManager.get_model()
    joint_id = common.spinner_joint_id(robot)
    common.set_joint_angle(robot, sim, joint_id, 0.0, 0.0)
    state = State()
    # Drive toward the upper limit; reset from rest on a step cap only (not each revolution).
    drive = common.HorizontalSpinnerDrive(robot, sim, joint_id, max_steps=RECENTER_STEPS, revolutions=1.0e9)

    # Drive the real Isaac Lab DR term instead of hand-calling the asset API.
    asset_cfg = SceneEntityCfg("robot", joint_ids=[joint_id])
    joint_term, env_shim = common.make_event_term(
        randomize_joint_parameters, scene, params={"asset_cfg": asset_cfg, "operation": "abs"}
    )

    def apply() -> None:
        joint_term(
            env_shim,
            None,
            asset_cfg=asset_cfg,
            lower_limit_distribution_params=(state.lower, state.lower),
            upper_limit_distribution_params=(state.upper, state.upper),
            operation="abs",
        )

    def step() -> None:
        drive.torque = state.torque
        drive.step()

    def measure() -> None:
        lo = float(common.as_flat_tensor(model.joint_limit_lower)[joint_id].item())
        up = float(common.as_flat_tensor(model.joint_limit_upper)[joint_id].item())
        angle = float(robot.data.joint_pos.torch[0, joint_id].item())
        lines = [f"limits: newton=[{lo:+.3f}, {up:+.3f}]"]
        if kamino:
            kam_lo = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.joints.q_j_min)[joint_id].item())
            kam_up = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.joints.q_j_max)[joint_id].item())
            routed = abs(lo - kam_lo) <= common.ROUTING_TOL and abs(up - kam_up) <= common.ROUTING_TOL
            lines.append(f"        kamino=[{kam_lo:+.3f}, {kam_up:+.3f}]")
            lines.append(f"  routing: {common.short_verdict(routed, expected_routed=True)}")
        exceeds = angle > up + 1e-2 or angle < lo - 1e-2
        lines.append(f"  angle={angle:+.3f} rad   {'EXCEEDS limit (ignored!)' if exceeds else 'within limit'}")
        state.lines = lines

    panel = common.PropertyPanel(
        "Joint position limits (horizontal spinner)",
        state,
        [("lower (rad)", "lower", -0.8, 0.0), ("upper (rad)", "upper", 0.0, 0.8), ("torque (N*m)", "torque", 0.0, 1.0)],
        lambda: state.lines,
    )
    common.run_check(
        sim=sim,
        simulation_app=simulation_app,
        state=state,
        panel=panel,
        apply_fn=apply,
        step_fn=step,
        measure_fn=measure,
        backend=args_cli.physics,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
