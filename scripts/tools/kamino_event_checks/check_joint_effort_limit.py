# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: does runtime joint effort-limit randomization reach Kamino?

No MDP event term covers effort limits (``randomize_joint_parameters`` handles only
friction, armature and position limits), so this drives the asset write API
``Articulation.write_joint_effort_limit_to_sim_index`` directly (writes
``model.joint_effort_limit``, raises ``JOINT_DOF_PROPERTIES``). After PR #3386 Kamino
re-copies it into ``joints.tau_j_max`` -> ROUTED. A huge commanded effort clips at the
limit, so ``alpha = effort_limit / I_joint``: raise the limit for a faster spin-up.

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_joint_effort_limit.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime joint effort-limit changes route to Kamino.")
args_cli, simulation_app = common.launch(parser)

import isaaclab.sim as sim_utils
import torch
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics.newton_manager import NewtonManager


class State:
    def __init__(self) -> None:
        self.effort_limit = 1.5
        self.command = 6.0
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
    state = State()
    n_env = robot.num_instances
    drive = common.HorizontalSpinnerDrive(robot, sim, joint_id)

    def apply() -> None:
        robot.write_joint_effort_limit_to_sim_index(
            limits=torch.full((n_env, 1), state.effort_limit, device=sim.device, dtype=torch.float32),
            joint_ids=[joint_id],
        )

    def step() -> None:
        drive.torque = state.command  # deliberately larger than the effort limit
        drive.step()

    def measure() -> None:
        lim = float(common.as_flat_tensor(model.joint_effort_limit)[joint_id].item())
        lines = [f"effort limit: newton={lim:.4f}"]
        if kamino:
            kam = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.joints.tau_j_max)[joint_id].item())
            routed = abs(lim - kam) <= common.ROUTING_TOL
            lines.append(f"              kamino.tau_j_max={kam:.4f}   routing: {common.short_verdict(routed, expected_routed=True)}")
        lines.append(f"  commanded effort={state.command:.0f}   speed={drive.speed:+.2f} rad/s   initial alpha={drive.alpha0:+.2f}")
        lines.append("  torque clips at the effort limit -> higher limit means faster spin-up")
        state.lines = lines

    panel = common.PropertyPanel(
        "Joint effort limit (horizontal spinner)",
        state,
        [("effort limit (N*m)", "effort_limit", 0.1, 5.0), ("commanded effort (N*m)", "command", 0.0, 12.0)],
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
