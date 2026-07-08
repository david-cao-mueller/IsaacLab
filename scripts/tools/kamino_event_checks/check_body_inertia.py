# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: does runtime inertia randomization reach Kamino's dynamics?

Drives the real DR term :class:`isaaclab.envs.mdp.events.randomize_rigid_body_inertia`
(``operation="scale"``, ``diagonal_only=True``), scaling ``model.body_inertia``
(``BODY_INERTIAL_PROPERTIES``). The value is aliased into Kamino's ``bodies.i_I_i``
(ROUTED), but ``model.body_inv_inertia`` is never refreshed and Kamino integrates with
it, so alpha = tau / I does NOT track the inertia slider -> inverse inertia is STALE
(BUG, like body mass). A long bar is spun about z (gravity off) and reset from rest
each revolution for a clean initial-alpha probe.

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_body_inertia.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime inertia changes route to Kamino dynamics.")
args_cli, simulation_app = common.launch(parser)

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_rigid_body_inertia
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics.newton_manager import NewtonManager

BAR_SIZE = (0.9, 0.15, 0.15)  # long, heavy bar so spin about z is visible and not too fast
BAR_MASS = 5.0
START_Z = 3.0
FULL_ROTATION = common.FULL_ROTATION  # reset from rest after one complete revolution


class State:
    def __init__(self, izz0: float) -> None:
        self.inertia_scale = 1.0
        self.torque_z = 1.0
        self.izz0 = izz0
        self.dirty = False
        self.alpha0_meas = 0.0  # measured initial angular accel about z (from rest)
        self.lines: list[str] = []


def main() -> None:
    dt = max(args_cli.dt, common.SPIN_DT)  # honor a larger user --dt, otherwise use the bigger default
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=dt, gravity=(0.0, 0.0, 0.0), physics=common.make_physics_cfg(args_cli.physics)
    )
    sim = SimulationContext(sim_cfg)
    scene = common.build_scene(sim, common.free_box_scene_cfg(size=BAR_SIZE, mass=BAR_MASS, start_z=START_Z))
    box = scene["box"]
    sim.set_camera_view([3.0, -3.0, 3.6], [0.0, 0.0, START_Z])
    common.settle(sim, box, args_cli.settle_steps)

    common.warn_if_not_kamino(sim)
    kamino = "kamino" in sim.physics_manager.__name__.lower()
    model = NewtonManager.get_model()
    state = State(float(common.as_flat_tensor(model.body_inertia)[8].item()))
    prev_w = [0.0]
    since_respin = [0]
    rot_accum = [0.0]  # accumulated absolute rotation angle [rad] since last respin
    forces = torch.zeros(1, 1, 3, device=sim.device)
    torques = torch.zeros(1, 1, 3, device=sim.device)

    # Drive the real Isaac Lab DR term instead of hand-calling the asset API.
    asset_cfg = SceneEntityCfg("box")
    inertia_term, env_shim = common.make_event_term(
        randomize_rigid_body_inertia, scene, params={"asset_cfg": asset_cfg, "operation": "scale", "diagonal_only": True}
    )

    def apply() -> None:
        inertia_term(
            env_shim,
            None,
            asset_cfg=asset_cfg,
            inertia_distribution_params=(state.inertia_scale, state.inertia_scale),
            operation="scale",
            diagonal_only=True,
        )

    def respin() -> None:
        common.recenter_free_body(box, sim, START_Z)
        box.update(sim.get_physics_dt())
        prev_w[0] = 0.0
        since_respin[0] = 0
        rot_accum[0] = 0.0

    def step() -> None:
        dt = sim.get_physics_dt()
        torques[0, 0, 2] = state.torque_z
        box.permanent_wrench_composer.set_forces_and_torques_index(forces=forces, torques=torques, is_global=True)
        box.write_data_to_sim()
        sim.step()
        box.update(dt)
        since_respin[0] += 1

        w = float(box.data.root_com_ang_vel_w.torch[0, 2].item())
        alpha = (w - prev_w[0]) / dt
        prev_w[0] = w
        if since_respin[0] == 1:
            state.alpha0_meas = alpha
        # reset from rest once the bar has swept a full revolution
        rot_accum[0] += abs(w) * dt
        if rot_accum[0] >= FULL_ROTATION:
            respin()

    def measure() -> None:
        izz = float(common.as_flat_tensor(model.body_inertia)[8].item())
        inv_izz = float(common.as_flat_tensor(model.body_inv_inertia)[8].item())
        lines = [f"I_zz: newton={izz:.4f}  (base={state.izz0:.4f})"]
        if kamino:
            kam_izz = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.bodies.i_I_i)[8].item())
            routed = abs(izz - kam_izz) <= common.ROUTING_TOL
            lines.append(f"      kamino.i_I_i={kam_izz:.4f}   value routing: {common.short_verdict(routed, True)}")
            expected_inv = 1.0 / izz if izz > 0 else float("nan")
            fresh = abs(inv_izz - expected_inv) <= max(common.ROUTING_TOL, 1e-4 * expected_inv)
            lines.append(f"  inv_Izz: model={inv_izz:.4f}  expect 1/Izz={expected_inv:.4f}")
            lines.append(f"  inv_Izz: {'OK (fresh)' if fresh else 'STALE! (Kamino spins with old 1/I)'}")
        lines.append(f"  initial alpha_z: meas={state.alpha0_meas:+.2f}  expect(fresh) tau/Izz={state.torque_z / izz:+.2f}")
        state.lines = lines

    panel = common.PropertyPanel(
        "Inertia randomization",
        state,
        [("inertia scale", "inertia_scale", 0.2, 5.0), ("torque z (N*m)", "torque_z", 0.0, 3.0)],
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
