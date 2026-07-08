# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: does runtime center-of-mass randomization reach Kamino?

Drives the real DR term :class:`isaaclab.envs.mdp.events.randomize_rigid_body_com`,
which writes ``model.body_com`` (``BODY_INERTIAL_PROPERTIES``). After PR #3386 Kamino
aliases it into ``bodies.i_r_com_i`` -> ROUTED. A CoM offset is only observable via a
torque about the CoM, so gravity is off and a +Z force is applied at the link origin:
centered CoM launches straight up, an offset CoM adds ``alpha_y = com_x * F_z / I_yy``
and the box tumbles. The body is periodically re-launched from rest so the kick is
repeatable.

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_body_com.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime center-of-mass changes route to Kamino.")
args_cli, simulation_app = common.launch(parser)

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_rigid_body_com
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics.newton_manager import NewtonManager

BOX_SIZE = 0.3
START_Z = 3.0
RELAUNCH_STEPS = 150  # re-launch from rest this often so the kick is repeatable
RELAUNCH_DRIFT = 3.5  # ...or sooner if the box drifts this far from the start height


class State:
    def __init__(self) -> None:
        self.com_x = 0.1
        self.force_z = 3.0
        self.dirty = False
        self.alpha0_meas = 0.0  # measured initial angular accel about y (clean, from rest)
        self.lines: list[str] = []


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=args_cli.dt, gravity=(0.0, 0.0, 0.0), physics=common.make_physics_cfg(args_cli.physics)
    )
    sim = SimulationContext(sim_cfg)
    scene = common.build_scene(sim, common.free_box_scene_cfg(size=BOX_SIZE, mass=1.0, start_z=START_Z))
    box = scene["box"]
    sim.set_camera_view([2.5, -2.5, 3.4], [0.0, 0.0, START_Z])
    common.settle(sim, box, args_cli.settle_steps)

    common.warn_if_not_kamino(sim)
    kamino = "kamino" in sim.physics_manager.__name__.lower()
    model = NewtonManager.get_model()
    state = State()
    prev_w = [0.0]
    since_launch = [0]
    forces = torch.zeros(1, 1, 3, device=sim.device)

    # Drive the real Isaac Lab DR term instead of hand-calling the asset API.
    asset_cfg = SceneEntityCfg("box")
    com_term, env_shim = common.make_event_term(randomize_rigid_body_com, scene, params={"asset_cfg": asset_cfg})

    def apply() -> None:
        com_term(env_shim, None, com_range={"x": (state.com_x, state.com_x)}, asset_cfg=asset_cfg)

    def relaunch() -> None:
        common.recenter_free_body(box, sim, START_Z)
        box.update(sim.get_physics_dt())
        prev_w[0] = 0.0
        since_launch[0] = 0

    def step() -> None:
        # Apply the +Z force at the body's link origin (a fixed material point, NOT
        # the CoM). is_global=True + positions -> torque about CoM = (P - com) x F.
        forces[0, 0, 2] = state.force_z
        positions = box.data.root_link_pos_w.torch.reshape(1, 1, 3)
        box.permanent_wrench_composer.set_forces_and_torques_index(forces=forces, positions=positions, is_global=True)
        box.write_data_to_sim()
        sim.step()
        box.update(sim.get_physics_dt())
        since_launch[0] += 1

        w_y = float(box.data.root_com_ang_vel_w.torch[0, 1].item())
        alpha_y = (w_y - prev_w[0]) / sim.get_physics_dt()
        prev_w[0] = w_y
        # capture the clean initial kick (first step after a launch from rest)
        if since_launch[0] == 1:
            state.alpha0_meas = alpha_y

        drift = abs(float(box.data.root_link_pos_w.torch[0, 2].item()) - START_Z)
        if since_launch[0] >= RELAUNCH_STEPS or drift > RELAUNCH_DRIFT:
            relaunch()

    def measure() -> None:
        comx = float(common.as_flat_tensor(model.body_com)[0].item())
        iyy = float(common.as_flat_tensor(model.body_inertia)[4].item())
        alpha0_theory = state.com_x * state.force_z / iyy if iyy > 0 else float("nan")
        lines = [f"com x: newton={comx:+.4f} m"]
        if kamino:
            kam = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.bodies.i_r_com_i)[0].item())
            routed = abs(comx - kam) <= common.ROUTING_TOL
            lines.append(f"       kamino.i_r_com_i={kam:+.4f}")
            lines.append(f"  routing: {common.short_verdict(routed, expected_routed=True)}")
        lines.append(f"  initial alpha_y: meas={state.alpha0_meas:+.2f}  theory com_x*Fz/Iyy={alpha0_theory:+.2f}")
        lines.append("  centered CoM -> straight launch;  offset CoM -> tumbling launch")
        state.lines = lines

    panel = common.PropertyPanel(
        "Center-of-mass randomization",
        state,
        [("com x (m)", "com_x", -0.15, 0.15), ("force z (N)", "force_z", 0.0, 8.0)],
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
