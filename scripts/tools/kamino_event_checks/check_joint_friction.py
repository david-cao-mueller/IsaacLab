# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: does runtime joint-friction randomization reach Kamino?

Drives the real DR term :class:`isaaclab.envs.mdp.events.randomize_joint_parameters`
(friction, ``operation="abs"``), writing ``model.joint_friction``
(``JOINT_DOF_PROPERTIES``). Kamino aliases the array (``joints.b_j``) -> ROUTED. A
constant torque spins the horizontal spinner (gravity off); Coulomb friction opposes
it, so higher ``joint friction`` slows the spin-up (and stalls the bar once friction
exceeds the torque).

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_joint_friction.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime joint-friction changes route to Kamino.")
args_cli, simulation_app = common.launch(parser)

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_joint_parameters
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics.newton_manager import NewtonManager

# reset a stalled bar (friction > torque) so the slider effect stays observable
STALL_RESET_STEPS = 200


class State:
    def __init__(self) -> None:
        self.friction = 0.5
        self.torque = 2.0
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
    drive = common.HorizontalSpinnerDrive(robot, sim, joint_id, max_steps=STALL_RESET_STEPS)

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
            friction_distribution_params=(state.friction, state.friction),
            operation="abs",
        )

    def step() -> None:
        drive.torque = state.torque
        drive.step()

    def measure() -> None:
        fric = float(common.as_flat_tensor(model.joint_friction)[joint_id].item())
        lines = [f"joint friction: newton={fric:.4f}"]
        if kamino:
            kam = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.joints.b_j)[joint_id].item())
            routed = abs(fric - kam) <= common.ROUTING_TOL
            lines.append(f"                kamino.b_j={kam:.4f}   routing: {common.short_verdict(routed, expected_routed=True)}")
        lines.append(f"  torque={state.torque:.2f} Nm   speed={drive.speed:+.2f} rad/s   initial alpha={drive.alpha0:+.2f}")
        lines.append("  higher friction -> slower spin-up (stalls once friction > torque)")
        state.lines = lines

    panel = common.PropertyPanel(
        "Joint friction (horizontal spinner)",
        state,
        [("joint friction", "friction", 0.0, 4.0), ("torque (N*m)", "torque", 0.0, 5.0)],
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
