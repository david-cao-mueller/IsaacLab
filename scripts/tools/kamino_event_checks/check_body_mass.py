# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: does runtime mass randomization reach Kamino's dynamics?

Drives the real DR term :class:`isaaclab.envs.mdp.events.randomize_rigid_body_mass`
(``operation="abs"``), which runs the production pipeline (``set_masses_index`` +
``recompute_inertia``, raising ``BODY_INERTIAL_PROPERTIES``). ``model.body_mass`` is
aliased into Kamino's ``bodies.m_i`` (ROUTED), but nothing refreshes
``model.body_inv_mass`` which Kamino integrates with, so a = F/m does NOT track the
mass slider -> inverse mass is STALE (BUG). Gravity is off; a pure horizontal push
makes the box glide.

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_body_mass.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime mass changes route to Kamino dynamics.")
args_cli, simulation_app = common.launch(parser)

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_rigid_body_mass
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics.newton_manager import NewtonManager

START_Z = 3.0


class State:
    def __init__(self, m0: float) -> None:
        self.mass = m0
        self.force_x = 5.0
        self.dirty = False
        self.lines: list[str] = []


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=args_cli.dt, gravity=(0.0, 0.0, 0.0), physics=common.make_physics_cfg(args_cli.physics)
    )
    sim = SimulationContext(sim_cfg)
    scene = common.build_scene(sim, common.free_box_scene_cfg(mass=1.0, start_z=START_Z))
    box = scene["box"]
    sim.set_camera_view([4.0, -4.0, 3.0], [0.0, 0.0, START_Z])
    common.settle(sim, box, args_cli.settle_steps)

    common.warn_if_not_kamino(sim)
    kamino = "kamino" in sim.physics_manager.__name__.lower()
    model = NewtonManager.get_model()
    state = State(float(box.data.body_mass.torch.reshape(-1)[0].item()))
    prev_v = [0.0]
    forces = torch.zeros(1, 1, 3, device=sim.device)
    torques = torch.zeros(1, 1, 3, device=sim.device)

    # Drive the real Isaac Lab DR term instead of hand-calling the asset API.
    asset_cfg = SceneEntityCfg("box")
    mass_term, env_shim = common.make_event_term(
        randomize_rigid_body_mass, scene, params={"asset_cfg": asset_cfg, "operation": "abs"}
    )

    def apply() -> None:
        mass_term(
            env_shim,
            None,
            asset_cfg=asset_cfg,
            mass_distribution_params=(state.mass, state.mass),
            operation="abs",
            distribution="uniform",
            recompute_inertia=True,
            min_mass=1e-6,
        )

    def step() -> None:
        forces[0, 0, 0] = state.force_x
        box.permanent_wrench_composer.set_forces_and_torques_index(forces=forces, torques=torques, is_global=True)
        box.write_data_to_sim()
        sim.step()
        box.update(sim.get_physics_dt())
        if abs(float(box.data.root_link_pos_w.torch[0, 0].item())) > 4.0:
            common.recenter_free_body(box, sim, START_Z)
            box.update(sim.get_physics_dt())
            prev_v[0] = 0.0

    def measure() -> None:
        v = float(box.data.root_com_lin_vel_w.torch[0, 0].item())
        a_x = (v - prev_v[0]) / sim.get_physics_dt()
        prev_v[0] = v

        newton_m = float(common.as_flat_tensor(model.body_mass)[0].item())
        inv_m = float(common.as_flat_tensor(model.body_inv_mass)[0].item())
        lines = [f"body mass: newton={newton_m:.4f} kg"]
        if kamino:
            kam_m = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.bodies.m_i)[0].item())
            routed = abs(newton_m - kam_m) <= common.ROUTING_TOL
            lines.append(f"           kamino.m_i={kam_m:.4f}")
            lines.append(f"  mass routing: {common.short_verdict(routed, expected_routed=True)}")
            expected_inv = 1.0 / state.mass
            fresh = abs(inv_m - expected_inv) <= max(common.ROUTING_TOL, 1e-4 * expected_inv)
            lines.append(f"  inv_mass: model={inv_m:.4f}  expect 1/m={expected_inv:.4f}")
            lines.append(f"  inv_mass: {'OK (fresh)' if fresh else 'STALE! (Kamino integrates old 1/m)'}")
        lines.append(f"  a_x meas={a_x:+.3f}   expect F/m={state.force_x / state.mass:+.3f}")
        state.lines = lines

    panel = common.PropertyPanel(
        "Mass randomization",
        state,
        [("mass (kg)", "mass", 0.1, 10.0), ("force x (N)", "force_x", 0.0, 20.0)],
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
