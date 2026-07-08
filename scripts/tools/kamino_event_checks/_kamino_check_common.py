# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared harness for the Kamino event-routing diagnostic scripts.

Each script is an interactive Newton-visualizer tool that answers, live, for one
randomizable physics property:

    When an Isaac Lab event / asset-write API changes a value at runtime, does the
    change actually reach the Kamino solver, or is Kamino operating on a stale
    snapshot taken at build time?

How it works:

* A Newton ``ViewerGL`` side panel exposes sliders for the property. Moving a
  slider triggers the *same* asset-write / ``add_model_change`` path an
  ``EventManager`` term uses (see ``isaaclab.envs.mdp.events``).
* Every frame we read back the *Newton model array* and the *Kamino internal
  array* (``solver._model_kamino.*``) and show both plus a routed / not-routed
  verdict, alongside a dynamics readout (acceleration, angular velocity, ...).

Kamino keeps a **direct reference** (alias) to some Newton arrays -- those update
live (ROUTED). Others are **cloned** in ``ModelKamino.from_newton`` -- writing the
Newton array leaves the clone untouched, so the change is silently ignored (NOT
ROUTED).

Import order matters: this module performs no Isaac Lab imports at module scope so
it is safe to import before the Kit app is launched. Helpers import lazily.
"""

from __future__ import annotations

import argparse

BACKENDS = ("physx", "newton_mjwarp", "newton_kamino")

# Max |Newton array - Kamino array| still counted as "aliased / routed to Kamino".
ROUTING_TOL = 1e-5


def build_parser(description: str) -> argparse.ArgumentParser:
    """Return an ``ArgumentParser`` with the flags shared by every check script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--physics",
        type=str,
        default="newton_kamino",
        choices=BACKENDS,
        help="Physics backend / solver to audit (routing check only meaningful for newton_kamino).",
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Physics timestep [s].")
    parser.add_argument("--settle_steps", type=int, default=30, help="Steps to run before enabling the UI.")
    return parser


def launch(parser: argparse.ArgumentParser):
    """Finalize CLI parsing and start the simulator. Returns ``(args, simulation_app)``.

    Always launches the Newton visualizer so the interactive slider panel is available.
    """
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["newton"])
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    return args, app_launcher.app


def make_physics_cfg(
    backend: str,
    kamino_kwargs: dict | None = None,
    mjwarp_kwargs: dict | None = None,
    physx_kwargs: dict | None = None,
):
    """Build the ``SimulationCfg.physics`` config for the requested backend.

    Args:
        backend: One of ``"physx"``, ``"newton_mjwarp"`` or ``"newton_kamino"``.
        kamino_kwargs: Extra keyword arguments forwarded to :class:`KaminoSolverCfg`.
        mjwarp_kwargs: Extra keyword arguments forwarded to :class:`MJWarpSolverCfg`.
        physx_kwargs: Extra keyword arguments forwarded to :class:`PhysxCfg`.
    """
    from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    if backend == "newton_mjwarp":
        return NewtonCfg(solver_cfg=MJWarpSolverCfg(**(mjwarp_kwargs or {})))
    if backend == "newton_kamino":
        return NewtonCfg(solver_cfg=KaminoSolverCfg(**(kamino_kwargs or {})))
    return PhysxCfg(**(physx_kwargs or {}))


def as_flat_tensor(wp_array):
    """Convert a Warp array to a flat float ``torch.Tensor`` (host-readable)."""
    import warp as wp

    return wp.to_torch(wp_array).reshape(-1).float()


def short_verdict(routed: bool, expected_routed: bool) -> str:
    """One-line verdict for the live UI panel."""
    if expected_routed and routed:
        return "ROUTED (ok)"
    if expected_routed and not routed:
        return "STALE! (regression)"
    if not expected_routed and not routed:
        return "NOT ROUTED (bug)"
    return "ROUTED (fixed?)"


# --------------------------------------------------------------------------- #
# Newton visualizer / interactive UI                                          #
# --------------------------------------------------------------------------- #


class PropertyPanel:
    """Generic Newton ``ViewerGL`` side panel: property sliders + a live readout.

    Args:
        title: Panel heading.
        state: Any object with the slider attributes plus a ``dirty`` bool. Moving
            a slider updates the attribute and sets ``state.dirty = True`` so the
            run loop re-applies the value (mirroring an event trigger).
        sliders: Tuples of ``(label, attr, min, max)``.
        readout_fn: Callable returning the list of text lines to render each frame.
    """

    def __init__(self, title, state, sliders, readout_fn):
        self._title = title
        self._state = state
        self._sliders = sliders
        self._readout_fn = readout_fn

    def draw(self, imgui) -> None:
        imgui.separator()
        imgui.text(self._title)
        for label, attr, lo, hi in self._sliders:
            changed, value = imgui.slider_float(label, float(getattr(self._state, attr)), lo, hi)
            if changed:
                setattr(self._state, attr, value)
                self._state.dirty = True
        imgui.separator()
        for line in self._readout_fn():
            imgui.text(line)


def register_panel(sim, panel) -> bool:
    """Register ``panel.draw`` on the live Newton ``ViewerGL`` side dock. Returns True if attached."""
    for viz in sim.visualizers:
        gl = getattr(viz, "_viewer", None)
        if gl is not None and hasattr(gl, "register_ui_callback") and hasattr(gl, "log_scalar"):
            gl.register_ui_callback(panel.draw, position="side")
            return True
    return False


# --------------------------------------------------------------------------- #
# Stepping / scene helpers                                                     #
# --------------------------------------------------------------------------- #


def settle(sim, asset, steps: int) -> None:
    """Advance the sim ``steps`` steps, flushing asset writes and refreshing buffers."""
    dt = sim.get_physics_dt()
    for _ in range(steps):
        asset.write_data_to_sim()
        sim.step()
        asset.update(dt)


def free_box_scene_cfg(size: float | tuple[float, float, float] = 0.2, mass: float = 1.0, start_z: float = 1.0):
    """Return an ``InteractiveSceneCfg`` subclass with a single free-floating box.

    No ground plane: the body is contact-free so linear / angular accelerations from
    an applied wrench are clean to measure. ``size`` may be a scalar (cube) or an
    ``(x, y, z)`` tuple (e.g. an elongated bar, whose rotation is easy to see).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    dims = (size, size, size) if isinstance(size, (int, float)) else tuple(size)

    @configclass
    class _FreeBoxSceneCfg(InteractiveSceneCfg):
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        )
        box = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box",
            spawn=sim_utils.CuboidCfg(
                size=dims,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.5, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, start_z)),
        )

    return _FreeBoxSceneCfg


def grounded_box_scene_cfg(
    size: float = 0.2,
    mass: float = 1.0,
    mu: float = 0.5,
    start_z: float | None = None,
    plane_mu: float | None = None,
    friction_combine_mode: str | None = None,
):
    """Return an ``InteractiveSceneCfg`` subclass with a box resting on a ground plane.

    Used by the shape-material / friction / collider-offset checks, where contact
    with the ground is what makes the property observable.

    Args:
        size: Box edge length [m].
        mass: Box mass [kg].
        mu: Box static/dynamic friction coefficient.
        start_z: Initial box height [m]. Defaults to resting on the plane.
        plane_mu: Ground-plane friction coefficient. Defaults to ``mu``.
        friction_combine_mode: Optional friction combine mode authored on the spawn
            materials (PhysX only; Newton authors the combine mode via the solver
            config). When ``None`` the engine default is used.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    z0 = 0.5 * size + 1.25e-2 if start_z is None else start_z
    ground_mu = mu if plane_mu is None else plane_mu

    def _material(coeff: float):
        kwargs: dict = {"static_friction": coeff, "dynamic_friction": coeff, "restitution": 0.0}
        if friction_combine_mode is not None:
            kwargs["friction_combine_mode"] = friction_combine_mode
        return sim_utils.RigidBodyMaterialCfg(**kwargs)

    ground_material = _material(ground_mu)
    box_material = _material(mu)

    @configclass
    class _GroundedBoxSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(physics_material=ground_material),
        )
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        box = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box",
            spawn=sim_utils.CuboidCfg(
                size=(size, size, size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=box_material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.5, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0)),
        )

    return _GroundedBoxSceneCfg


def _local_asset(name: str) -> str:
    """Resolve a bundled asset file next to this script's ``assets/`` folder."""
    from pathlib import Path

    local_asset = Path(__file__).resolve().parent / "assets" / name
    if local_asset.is_file():
        return str(local_asset)
    msg = f"Could not locate kamino event-check asset at {local_asset}"
    raise FileNotFoundError(msg)


def spinner_scene_cfg():
    """Return an ``InteractiveSceneCfg`` subclass with a fixed-base horizontal spinner.

    A single revolute joint (named ``grounding``) about the vertical Z axis with a bar
    link, so a constant torque spins the bar in the horizontal plane (gravity does not
    torque it). This mirrors the free-body inertia bar, but the effective rotational
    inertia is a *joint* property (link inertia + armature).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass

    robot_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_local_asset("horizontal_spinner.usda"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"grounding": 0.0}),
        actuators={
            "pendulum_actuator": ImplicitActuatorCfg(
                joint_names_expr=["grounding"],
                effort_limit_sim=400.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    @configclass
    class _SpinnerSceneCfg(InteractiveSceneCfg):
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        )
        robot = robot_cfg.replace(prim_path="/World/envs/env_.*/Robot")

    return _SpinnerSceneCfg


def build_scene(sim, scene_cfg_cls, num_envs: int = 1, env_spacing: float = 4.0):
    """Instantiate a scene from a cfg subclass and reset the sim. Returns the scene."""
    scene_cfg = scene_cfg_cls(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)
    scene = scene_cfg.class_type(scene_cfg)
    sim.reset()
    return scene


class EventEnvShim:
    """Minimal stand-in for ``ManagerBasedEnv`` so ``isaaclab.envs.mdp`` event terms
    can be driven directly outside a full manager-based environment.

    The randomization event terms only touch ``env.scene`` (``scene[name]`` /
    ``scene.num_envs``), ``env.device`` and ``env.sim`` (for backend detection and
    scene-wide gravity); this shim exposes exactly those.
    """

    def __init__(self, scene, sim=None):
        self.scene = scene
        self.num_envs = scene.num_envs
        self.device = scene.device
        if sim is None:
            from isaaclab.sim import SimulationContext

            sim = SimulationContext.instance()
        self.sim = sim


def make_event_term(term_cls, scene, params: dict, mode: str = "reset"):
    """Instantiate a class-based MDP event term against a scene via :class:`EventEnvShim`.

    Returns ``(term, env_shim)``; call ``term(env_shim, env_ids, **call_kwargs)`` to run it.
    """
    from isaaclab.managers import EventTermCfg

    env = EventEnvShim(scene)
    cfg = EventTermCfg(func=term_cls, mode=mode, params=params)
    return term_cls(cfg, env), env


def spinner_joint_id(robot) -> int:
    """Return the revolute joint index of the fixed-base horizontal spinner (fallback: joint 0)."""
    try:
        ids, _ = robot.find_joints("grounding")
        return int(ids[0])
    except Exception:  # noqa: BLE001 - robust to naming changes across assets
        return 0


def set_joint_angle(robot, sim, joint_id: int, angle: float, velocity: float = 0.0) -> None:
    """Teleport a single joint to ``angle`` with optional initial ``velocity``."""
    import torch

    n_env = robot.num_instances
    robot.write_joint_state_to_sim_index(
        position=torch.full((n_env, 1), angle, device=sim.device, dtype=torch.float32),
        velocity=torch.full((n_env, 1), velocity, device=sim.device, dtype=torch.float32),
        joint_ids=[joint_id],
    )


def set_spinner_camera(sim) -> None:
    """Elevated view looking down on the horizontal spinner's sweep plane."""
    sim.set_camera_view([1.6, -1.6, 1.9], [0.0, 0.0, 0.75])


SPIN_DT = 0.02  # larger timestep so a full revolution completes quickly (still stable)
FULL_ROTATION = 2.0 * 3.141592653589793  # one complete revolution [rad]


class HorizontalSpinnerDrive:
    """Apply a constant joint torque to the horizontal spinner and reset from rest.

    Mirrors the free-body inertia bar: a constant torque spins the bar in the
    horizontal plane (gravity off), and the joint is reset from rest once it has
    swept ``revolutions`` full turns -- or after ``max_steps`` steps, which lets a
    stalled joint (friction > torque) or a joint parked at a position limit recover
    so the slider effect can be re-observed.

    Read ``speed`` (rad/s), ``alpha`` (rad/s^2) and ``alpha0`` (clean initial
    acceleration measured on the first step after each reset) after :meth:`step`.
    """

    def __init__(self, robot, sim, joint_id: int, max_steps: int | None = None, revolutions: float = 1.0) -> None:
        import torch

        self._robot = robot
        self._sim = sim
        self._joint_id = joint_id
        self._max_steps = max_steps
        self._reset_rotation = revolutions * FULL_ROTATION
        self._torch = torch
        self.torque = 0.0
        self.speed = 0.0
        self.alpha = 0.0
        self.alpha0 = 0.0
        self._prev_w = 0.0
        self._since_respin = 0
        self._rot_accum = 0.0

    def reset(self) -> None:
        set_joint_angle(self._robot, self._sim, self._joint_id, 0.0, 0.0)
        self._robot.update(self._sim.get_physics_dt())
        self._prev_w = 0.0
        self._since_respin = 0
        self._rot_accum = 0.0

    def step(self) -> None:
        dt = self._sim.get_physics_dt()
        cmd = self._torch.full(
            (self._robot.num_instances, 1), self.torque, device=self._sim.device, dtype=self._torch.float32
        )
        self._robot.set_joint_effort_target_index(target=cmd, joint_ids=[self._joint_id])
        self._robot.write_data_to_sim()
        self._sim.step()
        self._robot.update(dt)
        self._since_respin += 1

        w = float(self._robot.data.joint_vel.torch[0, self._joint_id].item())
        self.alpha = (w - self._prev_w) / dt
        self._prev_w = w
        self.speed = w
        if self._since_respin == 1:
            self.alpha0 = self.alpha
        self._rot_accum += abs(w) * dt
        stalled = self._max_steps is not None and self._since_respin >= self._max_steps
        if self._rot_accum >= self._reset_rotation or stalled:
            self.reset()


def recenter_free_body(box, sim, start_z: float) -> None:
    """Teleport a free body back to the origin at rest so it stays in view."""
    import torch

    pose = torch.zeros(1, 7, device=sim.device)
    pose[0, 2] = start_z
    pose[0, 3] = 1.0  # identity quaternion (w, x, y, z) -> w first
    box.write_root_pose_to_sim_index(root_pose=pose)
    box.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=sim.device))


def warn_if_not_kamino(sim) -> None:
    """Print a note when the Newton-vs-Kamino routing check is not meaningful (non-Kamino backend)."""
    if "kamino" in sim.physics_manager.__name__.lower():
        return
    print(
        "  NOTE: the Newton-vs-Kamino array routing check only applies to newton_kamino.\n"
        "        Run with --physics newton_kamino for the routing verdict.\n"
    )


def print_banner(title: str, backend: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print(f"  backend = {backend}")
    print("  Move the sliders in the Newton viewer side panel to apply changes in real time.")
    print("=" * 78 + "\n")


def run_check(*, sim, simulation_app, state, panel, apply_fn, step_fn, measure_fn, backend) -> None:
    """Drive the interactive slider loop shared by every check script.

    Each frame: if a slider moved (``state.dirty``) re-apply the property via
    ``apply_fn`` (the event-mirroring write), advance the sim with ``step_fn``,
    then refresh the live readout via ``measure_fn``. Runs until the Newton viewer
    window is closed.
    """
    register_panel(sim, panel)
    print_banner(panel._title, backend)

    state.dirty = True  # apply the initial slider value once (mirrors an event trigger)
    while simulation_app.is_running():
        if getattr(state, "dirty", False):
            apply_fn()
            state.dirty = False
        step_fn()
        measure_fn()
