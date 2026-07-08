# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: does runtime collider-offset randomization reach Kamino?

Drives the real DR term
:class:`isaaclab.envs.mdp.events.randomize_rigid_body_collider_offsets`, which on
Newton maps PhysX rest/contact offsets to ``shape_margin`` / ``shape_gap`` and raises
``SHAPE_PROPERTIES``. Kamino does not support that flag (warns and no-ops) and its
collision model is built once, so the box's resting height should not track the
sliders -> NOT ROUTED (BUG).

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_collider_offsets.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime collider-offset changes route to Kamino.")
args_cli, simulation_app = common.launch(parser)

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_rigid_body_collider_offsets
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext

BOX_SIZE = 0.2


def collider_offset_bindings(asset, sim):
    """Return the (margin/rest, gap/contact) offset bindings for the active backend.

    On Newton: ``(shape_margin, shape_gap)`` Warp arrays (what
    :class:`isaaclab.envs.mdp.events.randomize_rigid_body_collider_offsets` writes).
    On PhysX: ``(rest_offsets, contact_offsets)`` read from the ``root_view``.
    """
    if "newton" in sim.physics_manager.__name__.lower():
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager.get_model()
        root_view = asset._root_view
        return (
            root_view.get_attribute("shape_margin", model)[:, 0],
            root_view.get_attribute("shape_gap", model)[:, 0],
        )
    return asset.root_view.get_rest_offsets(), asset.root_view.get_contact_offsets()


class State:
    def __init__(self) -> None:
        self.rest_offset = 0.02
        self.contact_offset = 0.04
        self.dirty = False
        self.lines: list[str] = []


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=args_cli.dt, physics=common.make_physics_cfg(args_cli.physics)
    )
    sim = SimulationContext(sim_cfg)
    scene = common.build_scene(sim, common.grounded_box_scene_cfg(size=BOX_SIZE, mass=1.0, mu=0.8))
    box = scene["box"]
    start_z = 0.5 * BOX_SIZE + 1.25e-2
    sim.set_camera_view([2.0, -2.0, 0.8], [0.0, 0.0, start_z])
    common.settle(sim, box, max(args_cli.settle_steps, 60))

    common.warn_if_not_kamino(sim)
    kamino_check = "kamino" in sim.physics_manager.__name__.lower()
    margin_binding, gap_binding = collider_offset_bindings(box, sim)
    state = State()

    # Drive the real Isaac Lab DR term instead of hand-writing the view bindings.
    asset_cfg = SceneEntityCfg("box")
    offset_term, env_shim = common.make_event_term(
        randomize_rigid_body_collider_offsets, scene, params={"asset_cfg": asset_cfg}
    )

    def apply() -> None:
        offset_term(
            env_shim,
            None,
            asset_cfg=asset_cfg,
            rest_offset_distribution_params=(state.rest_offset, state.rest_offset),
            contact_offset_distribution_params=(state.contact_offset, state.contact_offset),
        )

    def step() -> None:
        box.write_data_to_sim()
        sim.step()
        box.update(sim.get_physics_dt())

    def measure() -> None:
        margin = float(wp.to_torch(margin_binding).reshape(-1)[0].item())
        gap = float(wp.to_torch(gap_binding).reshape(-1)[0].item())
        rest_z = float(box.data.root_link_pos_w.torch[0, 2].item())
        if "newton" in sim.physics_manager.__name__.lower():
            lines = [
                f"shape_margin (newton) = {margin:.4f}",
                f"shape_gap    (newton) = {gap:.4f}",
                f"box resting z = {rest_z:.4f} m",
            ]
            if kamino_check:
                lines += [
                    "Kamino: SHAPE_PROPERTIES unsupported (warns) -> collision baked at build",
                    "  => resting height should NOT track the sliders on Kamino (NOT ROUTED)",
                ]
            else:
                lines.append("Newton (MJWarp): offsets written to model; resting z should track sliders.")
        else:
            lines = [
                f"rest_offset    (physx) = {margin:.4f}",
                f"contact_offset (physx) = {gap:.4f}",
                f"box resting z = {rest_z:.4f} m",
                "PhysX: reference backend — resting height should track the sliders.",
            ]
        state.lines = lines

    panel = common.PropertyPanel(
        "Collider offsets (SHAPE_PROPERTIES)",
        state,
        [("rest offset -> margin", "rest_offset", 0.0, 0.1), ("contact offset -> gap", "contact_offset", 0.0, 0.15)],
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
