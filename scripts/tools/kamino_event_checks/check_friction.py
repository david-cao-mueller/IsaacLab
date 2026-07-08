# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive: does runtime material-friction randomization reach the solver (stick/slip)?

Drives the *real* Isaac Lab DR term
:class:`isaaclab.envs.mdp.events.randomize_rigid_body_material` (via a minimal env
shim). On Newton it samples per-shape ``shape_material_mu`` /
``shape_material_restitution`` and raises ``SolverNotifyFlags.SHAPE_PROPERTIES``; on
PhysX it writes the per-shape material properties directly.

A constant horizontal force pushes the box on a ground plane; vary the ``box mu``
slider and watch the stick/slip behavior against the Coulomb prediction
(``mu_required = force / (m * g)``). The ground friction (``--plane_mu``) and the
friction combine/mix mode (``--combine_mode``) are fixed at startup. PhysX authors
the combine mode on the spawn materials; the Newton backends ignore it (MJWarp
effectively averages, Kamino uses its built-in ``MaterialMuxMode`` max rule, which
is not exposed on the solver cfg).

Kamino bakes shape materials into a discrete ``MaterialManager`` table at build time
(``register_materials`` reads ``model.shape_material_mu`` once) and does not support
``SHAPE_PROPERTIES``, so on Kamino the stick/slip behavior should NOT track the mu
slider (NOT ROUTED). PhysX / MJWarp are live and serve as the reference.

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_friction.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime material-friction changes route to the solver.")
parser.add_argument(
    "--combine_mode",
    type=str,
    default="multiply",
    choices=("average", "min", "max", "multiply"),
    help=(
        "Friction combine / mix mode fixed at startup, authored on the PhysX spawn materials "
        "(friction_combine_mode). Ignored by the Newton backends: MJWarp effectively averages and "
        "Kamino uses its built-in MaterialMuxMode (max), which is not exposed on the solver cfg. "
        "Restart to change."
    ),
)
parser.add_argument("--plane_mu", type=float, default=1.0, help="Ground-plane friction coefficient.")
args_cli, simulation_app = common.launch(parser)

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext

BOX_SIZE = 0.2


def effective_friction(mu_a: float, mu_b: float, combine_mode: str) -> float:
    """Combine two friction coefficients according to ``combine_mode``."""
    if combine_mode == "min":
        return min(mu_a, mu_b)
    if combine_mode == "max":
        return max(mu_a, mu_b)
    if combine_mode == "multiply":
        return mu_a * mu_b
    if combine_mode == "average":
        return 0.5 * (mu_a + mu_b)
    raise ValueError(f"unknown combine mode: {combine_mode}")


def sim_combine_mode(backend: str, combine_mode: str) -> str:
    """Combine mode actually used by the active backend (for the theory readout).

    Only PhysX honors the requested mode. MJWarp effectively averages and Kamino
    uses its built-in ``MaterialMuxMode`` max rule, so ``--combine_mode`` does not
    apply to either Newton backend.
    """
    if backend == "physx":
        return combine_mode
    if backend == "newton_kamino":
        return "max"
    return "average"


def read_box_mu(box, sim) -> float:
    """Read the box's currently applied friction coefficient (backend-aware).

    Newton exposes a per-shape ``shape_material_mu`` view binding (the same one
    :class:`isaaclab.envs.mdp.events.randomize_rigid_body_material` writes); PhysX
    stores ``(static, dynamic, restitution)`` per shape in the material buffer.
    """
    if "newton" in sim.physics_manager.__name__.lower():
        from isaaclab_newton.physics.newton_manager import NewtonManager

        binding = box._root_view.get_attribute("shape_material_mu", NewtonManager.get_model())[:, 0]
        return float(wp.to_torch(binding)[0, 0].item())
    materials = wp.to_torch(box.root_view.get_material_properties())
    return float(materials.reshape(-1, 3)[0, 0].item())


class State:
    def __init__(self) -> None:
        self.box_mu = 0.5
        self.force_x = 5.0
        self.dirty = False
        self.lines: list[str] = []


def main() -> None:
    combine = args_cli.combine_mode
    # PhysX authors the combine mode on the spawn materials; Newton backends via the solver cfg.
    spawn_combine = combine if args_cli.physics == "physx" else None
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=args_cli.dt,
        physics=common.make_physics_cfg(
            args_cli.physics,
            mjwarp_kwargs={"impratio": 5.0},
            physx_kwargs={"enable_external_forces_every_iteration": True},
        ),
    )
    sim = SimulationContext(sim_cfg)
    gravity = abs(sim_cfg.gravity[2])
    scene = common.build_scene(
        sim,
        common.grounded_box_scene_cfg(
            size=BOX_SIZE, mass=1.0, mu=0.5, plane_mu=args_cli.plane_mu, friction_combine_mode=spawn_combine
        ),
    )
    box = scene["box"]
    start_z = 0.5 * BOX_SIZE + 1.25e-2
    sim.set_camera_view([2.5, -2.5, 1.2], [0.0, 0.0, start_z])
    common.settle(sim, box, max(args_cli.settle_steps, 60))

    common.warn_if_not_kamino(sim)
    kamino = "kamino" in sim.physics_manager.__name__.lower()
    mass = float(box.data.body_mass.torch.reshape(-1)[0].item())
    state = State()
    forces = torch.zeros(1, 1, 3, device=sim.device)
    torques = torch.zeros(1, 1, 3, device=sim.device)

    # Drive the real Isaac Lab DR term instead of hand-writing the material view.
    # ``randomize_rigid_body_material`` samples its friction/restitution ranges once at
    # construction (from ``cfg.params``) -- the ranges passed at call time are ignored --
    # so we rebuild the term whenever the slider moves to re-sample at the new mu.
    asset_cfg = SceneEntityCfg("box")

    def apply() -> None:
        params = {
            "asset_cfg": asset_cfg,
            "static_friction_range": (state.box_mu, state.box_mu),
            "dynamic_friction_range": (state.box_mu, state.box_mu),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": False,
        }
        term, env_shim = common.make_event_term(randomize_rigid_body_material, scene, params=params)
        term(env_shim, None, **params)

    def step() -> None:
        forces[0, 0, 0] = state.force_x
        box.permanent_wrench_composer.set_forces_and_torques_index(forces=forces, torques=torques, is_global=True)
        box.write_data_to_sim()
        sim.step()
        box.update(sim.get_physics_dt())
        if abs(float(box.data.root_link_pos_w.torch[0, 0].item())) > 3.0:
            common.recenter_free_body(box, sim, start_z)
            box.update(sim.get_physics_dt())

    def measure() -> None:
        applied_mu = read_box_mu(box, sim)
        used_combine = sim_combine_mode(args_cli.physics, combine)
        mu_eff = effective_friction(applied_mu, args_cli.plane_mu, used_combine)
        speed = float(box.data.root_com_lin_vel_w.torch[0, 0].item())
        normal = mass * gravity
        mu_required = state.force_x / normal if normal > 0 else float("inf")
        predicted = "SLIP" if mu_eff < mu_required else "STICK"
        observed = "SLIP" if abs(speed) > 0.02 else "STICK"
        lines = [
            f"box mu (applied) = {applied_mu:.3f}   plane mu = {args_cli.plane_mu:.3f}   combine = {used_combine}",
            f"effective mu = {mu_eff:.3f}   mu_required = {mu_required:.2f}",
            f"theory:   {predicted}",
            f"observed: {observed}   (v_x={speed:+.3f})",
        ]
        if kamino:
            lines += [
                "Kamino: SHAPE_PROPERTIES unsupported (warns) -> material baked at build",
                "  => dragging mu should NOT change observed behavior on Kamino (NOT ROUTED)",
            ]
        state.lines = lines

    panel = common.PropertyPanel(
        "Material friction (stick/slip)",
        state,
        [("box mu", "box_mu", 0.0, 1.5), ("force x (N)", "force_x", 0.0, 20.0)],
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
