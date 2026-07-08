# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive check: does runtime gravity randomization reach Kamino?

Drives the real DR term
:class:`isaaclab.envs.mdp.events.randomize_physics_scene_gravity` (``operation="abs"``),
which writes ``model.gravity`` and raises ``MODEL_PROPERTIES``. Kamino handles that flag
via ``_update_gravity()`` -> ROUTED. Drag the ``gravity z`` slider and watch the
free-falling box's vertical acceleration follow it.

Run:
    ./isaaclab.sh -p scripts/tools/kamino_event_checks/check_gravity.py --physics newton_kamino
"""

from __future__ import annotations

import _kamino_check_common as common

parser = common.build_parser("Check that runtime gravity changes route to Kamino.")
args_cli, simulation_app = common.launch(parser)

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.events import randomize_physics_scene_gravity
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics.newton_manager import NewtonManager

START_Z = 6.0


class State:
    def __init__(self, g0: float) -> None:
        self.gravity_z = g0
        self.dirty = False
        self.newton_g = g0
        self.kamino_g = g0
        self.v_z = 0.0
        self.a_z = 0.0
        self.lines: list[str] = []


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device, dt=args_cli.dt, physics=common.make_physics_cfg(args_cli.physics)
    )
    sim = SimulationContext(sim_cfg)
    scene = common.build_scene(sim, common.free_box_scene_cfg(mass=1.0, start_z=START_Z))
    box = scene["box"]
    sim.set_camera_view([4.0, -4.0, 3.0], [0.0, 0.0, 3.0])
    common.settle(sim, box, args_cli.settle_steps)

    common.warn_if_not_kamino(sim)
    kamino = "kamino" in sim.physics_manager.__name__.lower()
    model = NewtonManager.get_model()
    state = State(float(sim_cfg.gravity[2]))
    prev_v = [0.0]

    # Drive the real Isaac Lab DR term instead of hand-writing model.gravity.
    g0 = list(sim_cfg.gravity)
    gravity_term, env_shim = common.make_event_term(
        randomize_physics_scene_gravity,
        scene,
        params={"operation": "abs", "gravity_distribution_params": (g0, g0)},
    )
    env_ids = torch.arange(scene.num_envs, device=sim.device)

    def apply() -> None:
        g = [g0[0], g0[1], state.gravity_z]
        gravity_term(env_shim, env_ids, gravity_distribution_params=(g, g), operation="abs")

    def step() -> None:
        box.write_data_to_sim()
        sim.step()
        box.update(sim.get_physics_dt())
        # keep the box in view: recenter once it falls too far
        if abs(float(box.data.root_link_pos_w.torch[0, 2].item()) - START_Z) > 4.0:
            common.recenter_free_body(box, sim, START_Z)
            box.update(sim.get_physics_dt())
            prev_v[0] = 0.0

    def measure() -> None:
        v = float(box.data.root_com_lin_vel_w.torch[0, 2].item())
        state.a_z = (v - prev_v[0]) / sim.get_physics_dt()
        prev_v[0] = v
        state.v_z = v
        state.newton_g = float(common.as_flat_tensor(model.gravity)[2].item())
        state.kamino_g = float(common.as_flat_tensor(NewtonManager._solver._model_kamino.gravity.vector)[2].item()) if kamino else float("nan")

        lines = [f"gravity z: newton={state.newton_g:+.4f}"]
        if kamino:
            diff = abs(state.newton_g - state.kamino_g)
            routed = diff <= common.ROUTING_TOL
            lines.append(f"           kamino={state.kamino_g:+.4f}  |d|={diff:.2g}")
            lines.append(f"  routing: {common.short_verdict(routed, expected_routed=True)}")
        lines.append(f"  measured a_z={state.a_z:+.3f}   v_z={state.v_z:+.3f}")
        state.lines = lines

    panel = common.PropertyPanel(
        "Gravity randomization", state, [("gravity z (m/s^2)", "gravity_z", -20.0, 5.0)], lambda: state.lines
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
