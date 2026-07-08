# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: do runtime joint drive-gain (kp/kd) changes reach Kamino?

Drives the real DR term :class:`isaaclab.envs.mdp.events.randomize_actuator_gains`
(``operation="abs"``) on the spinner's implicit actuator, writing
``model.joint_target_ke`` / ``joint_target_kd`` (``JOINT_DOF_PROPERTIES``). Kamino
aliases these into ``joints.k_p_j`` / ``k_d_j`` -> ROUTED. The bar is released off the
PD target (angle 0): raise ``kp`` to snap back harder, ``kd`` to damp the oscillation.

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_joint_gains.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime joint kp/kd changes route to Kamino.")
args_cli, simulation_app = common.launch(parser)

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_actuator_gains
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics.newton_manager import NewtonManager

TILT_ANGLE = 1.5  # rad off the PD target that the bar is released from
RETILT_STEPS = 200  # re-tilt at least this often so the response stays visible


class State:
    def __init__(self) -> None:
        self.kp = 50.0
        self.kd = 5.0
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
    common.set_joint_angle(robot, sim, joint_id, TILT_ANGLE, 0.0)
    state = State()
    n_env = robot.num_instances
    target = torch.zeros((n_env, 1), device=sim.device, dtype=torch.float32)
    since_retilt = [0]

    # Drive the real Isaac Lab DR term instead of hand-calling the asset API.
    asset_cfg = SceneEntityCfg("robot", joint_ids=[joint_id])
    gains_term, env_shim = common.make_event_term(
        randomize_actuator_gains, scene, params={"asset_cfg": asset_cfg, "operation": "abs"}
    )

    def apply() -> None:
        gains_term(
            env_shim,
            None,
            asset_cfg=asset_cfg,
            stiffness_distribution_params=(state.kp, state.kp),
            damping_distribution_params=(state.kd, state.kd),
            operation="abs",
        )

    def step() -> None:
        robot.set_joint_position_target_index(target=target, joint_ids=[joint_id])
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        since_retilt[0] += 1
        angle = float(robot.data.joint_pos.torch[0, joint_id].item())
        speed = float(robot.data.joint_vel.torch[0, joint_id].item())
        settled = abs(angle) < 0.03 and abs(speed) < 0.05
        if since_retilt[0] >= RETILT_STEPS or settled:
            common.set_joint_angle(robot, sim, joint_id, TILT_ANGLE, 0.0)
            robot.update(sim.get_physics_dt())
            since_retilt[0] = 0

    def measure() -> None:
        ke = float(common.as_flat_tensor(model.joint_target_ke)[joint_id].item())
        kd = float(common.as_flat_tensor(model.joint_target_kd)[joint_id].item())
        angle = float(robot.data.joint_pos.torch[0, joint_id].item())
        speed = float(robot.data.joint_vel.torch[0, joint_id].item())
        lines = [f"kp: newton={ke:.3f}", f"kd: newton={kd:.3f}"]
        if kamino:
            kam_ke = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.joints.k_p_j)[joint_id].item())
            kam_kd = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.joints.k_d_j)[joint_id].item())
            r_ke = abs(ke - kam_ke) <= common.ROUTING_TOL
            r_kd = abs(kd - kam_kd) <= common.ROUTING_TOL
            lines = [
                f"kp: newton={ke:.3f}  kamino.k_p_j={kam_ke:.3f}  {common.short_verdict(r_ke, True)}",
                f"kd: newton={kd:.3f}  kamino.k_d_j={kam_kd:.3f}  {common.short_verdict(r_kd, True)}",
            ]
        lines.append(f"  bar angle={angle:+.3f} rad   speed={speed:+.3f} rad/s   (PD target 0)")
        state.lines = lines

    panel = common.PropertyPanel(
        "Joint drive gains (horizontal spinner)",
        state,
        [("stiffness kp", "kp", 0.0, 300.0), ("damping kd", "kd", 0.0, 40.0)],
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
