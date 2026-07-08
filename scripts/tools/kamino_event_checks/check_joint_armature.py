# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: does runtime joint-armature randomization reach Kamino?

Drives the real DR term :class:`isaaclab.envs.mdp.events.randomize_joint_parameters`
(armature, ``operation="abs"``), writing ``model.joint_armature``
(``JOINT_DOF_PROPERTIES``). Kamino aliases the array (``joints.a_j``) -> ROUTED, and
armature enters the effective joint inertia directly (no stale inverse):
``alpha = tau / (I_joint + armature)``. A constant torque spins the horizontal spinner
(gravity off); higher armature visibly slows the spin-up.

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_joint_armature.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime joint-armature changes route to Kamino dynamics.")
args_cli, simulation_app = common.launch(parser)

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_joint_parameters
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics.newton_manager import NewtonManager


SPIN_REVOLUTIONS = 6.0  # let the bar spin several turns before resetting so the effect is easy to watch


class State:
    def __init__(self) -> None:
        self.armature = 0.1
        self.torque = 0.5
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
    drive = common.HorizontalSpinnerDrive(robot, sim, joint_id, revolutions=SPIN_REVOLUTIONS)

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
            armature_distribution_params=(state.armature, state.armature),
            operation="abs",
        )

    def step() -> None:
        drive.torque = state.torque
        drive.step()

    def measure() -> None:
        arm = float(common.as_flat_tensor(model.joint_armature)[joint_id].item())
        lines = [f"armature: newton={arm:.4f}"]
        if kamino:
            kam = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.joints.a_j)[joint_id].item())
            routed = abs(arm - kam) <= common.ROUTING_TOL
            lines.append(f"          kamino.a_j={kam:.4f}   routing: {common.short_verdict(routed, expected_routed=True)}")
        lines.append(f"  torque={state.torque:.2f} Nm   speed={drive.speed:+.2f} rad/s   initial alpha={drive.alpha0:+.2f}")
        lines.append("  higher armature -> more effective inertia -> slower spin-up (longer revolution)")
        state.lines = lines

    panel = common.PropertyPanel(
        "Joint armature (horizontal spinner)",
        state,
        [("armature", "armature", 0.0, 20.0), ("torque (N*m)", "torque", 0.0, 2.0)],
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
