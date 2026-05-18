# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
from enum import Enum, auto

import carb
import carb.settings
import numpy as np
import omni.kit.app
from isaacsim.core.api.articulations import ArticulationSubset
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController

try:
    import omni.anim.navigation.core as nav_core
except Exception:
    nav_core = None

try:
    from isaacsim.util.debug_draw import _debug_draw
except Exception:
    _debug_draw = None


# -----------------------------------------------------------------------------
# Fill these values in for your scene.
# -----------------------------------------------------------------------------
SCENE_USD_PATH = r"C:\Users\user\NTHU\digital_twin\dataset\midterm\asset\hospital.usd"
ROBOT_USD_PATH = r"C:\Users\user\NTHU\digital_twin\dataset\midterm\asset\custom_robot.usd"
ROBOT_PRIM_PATH = "/World/nova_carter"
ROBOT_ARTICULATION_PRIM_PATH = ROBOT_PRIM_PATH + "/nova_carter"
CHASSIS_PRIM_PATH = ROBOT_ARTICULATION_PRIM_PATH + "/chassis_link"
FRANKA_BASE_PRIM_PATH = ROBOT_ARTICULATION_PRIM_PATH + "/franka/panda_link0"
END_EFFECTOR_PRIM_PATH = ROBOT_ARTICULATION_PRIM_PATH + "/franka/panda_hand"
ROBOT_INITIAL_POSITION = np.array([6.0, -1.0, 0.0])
ROBOT_INITIAL_YAW = 0.0
NAVMESH_MIN_RADIUS = 30
CUBE_PRIM_PATH = "/World/TaskCube"
PATH_LINE_COLOR = (1.0, 0.0, 0.0, 1.0)
PATH_LINE_WIDTH = 8
PATH_LINE_Z_OFFSET = 0.05
NAV_LINEAR_VELOCITY = 0.35
NAV_YAW_VELOCITY = 0.8
FACE_CUBE_YAW_TOLERANCE = 0.08
FACE_CUBE_YAW_VELOCITY = 0.5
ROBOT_SIDE_YAW_OFFSETS = {
    "+Y": np.pi / 2.0,
    "-Y": -np.pi / 2.0,
}

# Manual task variables requested in the homework.
cube_position = np.array([9.98115, -1.39854, 0.83727])
place_cube_pos = np.array([22.93955, -3.11071, 0.85179])

WHEEL_JOINT_NAMES = ["joint_wheel_left", "joint_wheel_right"]
ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]


class PickPlaceState(Enum):
    NAVIGATE_TO_PICK = auto()
    FACE_CUBE = auto()
    SETTLE = auto()
    GRASP_SEQUENCE = auto()
    LIFT = auto()
    ADJUST_ARM_X = auto()
    NAVIGATE_TO_PLACE = auto()
    FACE_PLACE = auto()
    PLACE_RELEASE = auto()
    DONE = auto()


class ArmSegment:
    def __init__(self, target_pos, gripper_width, settle_steps=0, ee_tolerance=0.025, target_joints=None):
        self.target_pos = None if target_pos is None else np.asarray(
            target_pos, dtype=float)
        self.gripper_width = gripper_width
        self.settle_steps = settle_steps
        self.ee_tolerance = ee_tolerance
        self.target_joints = None if target_joints is None else np.asarray(
            target_joints, dtype=float)
        self.trajectory = []
        self.index = 0
        self.waited = 0
        self.ik_solved = False


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 120.0
        self._world_settings["rendering_dt"] = 1.0 / 60.0
        self._robot = None
        self._cube = None
        self._robot_controller = None
        self._wheel_subset = None
        self._wheel_joint_names = list(WHEEL_JOINT_NAMES)
        self._arm_subset = None
        self._finger_subset = None
        self._pose_controller = None
        self._diff_controller = None
        self._ik_solver = None
        self._articulation_ik = None
        self._chassis_prim = None
        self._franka_base_prim = None
        self._ee_prim = None
        self._debug_draw = None
        self._state = PickPlaceState.NAVIGATE_TO_PICK
        self._pick_goal_pos = None
        self._place_goal_pos = None
        self._path = []
        self._waypoint_index = 0
        self._settle_count = 0
        self._nav_log_counter = 0
        self._arm_segments = []
        self._active_segment = None
        self._holding_object = False
        self._task_started = False
        self.waypoint_threshold = 0.18
        self.goal_threshold = 0.12
        self.settle_steps_after_stop = 55
        self.gripper_open = 0.05
        self.gripper_closed = 0.005
        self.pre_grasp_height = 0.1
        self.grasp_height_offset = 0.02
        self.lift_height = 0.1
        self.post_lift_straight_arm_joints = np.array(
            [0.0, -0.35, 0.0, -1.80, 0.0, 1.45, 0.75],
            dtype=float,
        )
        self.post_lift_adjust_settle_steps = 20
        self.arm_trajectory_steps = 90

    def setup_scene(self):
        world = self.get_world()
        add_reference_to_stage(SCENE_USD_PATH, "/World/LoadedScene")
        self._cube = world.scene.add(DynamicCuboid(
            prim_path=CUBE_PRIM_PATH,
            name="task_cube",
            position=cube_position,
            size=0.05,
            color=np.array([0.1, 0.55, 1.0]),
            mass=0.05,
        ))
        set_camera_view(
            eye=np.array([3.0, 3.0, 2.2]),
            target=np.array([0.0, 0.0, 0.5]),
            camera_prim_path="/OmniverseKit_Persp",
        )

    async def setup_post_load(self):
        world = self.get_world()
        await self._bake_navmesh()
        self._pick_goal_pos = self._resolve_pick_goal_pos()
        self._place_goal_pos = self._resolve_place_goal_pos()
        await self._load_robot_after_navmesh_bake()

        self._robot_controller = self._robot.get_articulation_controller()
        available_dofs = list(self._robot.dof_names)
        carb.log_info(f"Robot DOF names: {available_dofs}")
        self._wheel_joint_names = self._resolve_wheel_joint_names(
            available_dofs)
        if len(self._wheel_joint_names) == 2:
            self._wheel_subset = ArticulationSubset(
                self._robot, self._wheel_joint_names)
            carb.log_info(f"Using wheel joints: {self._wheel_joint_names}")
        else:
            self._wheel_subset = None
            carb.log_warn(
                "Could not resolve two wheel joints. Navigation is disabled; check Robot DOF names in the log.")
        self._arm_subset = ArticulationSubset(self._robot, ARM_JOINT_NAMES)
        self._finger_subset = ArticulationSubset(
            self._robot, FINGER_JOINT_NAMES)
        self._chassis_prim = SingleXFormPrim(
            CHASSIS_PRIM_PATH, name="nova_carter_chassis_pose")
        self._franka_base_prim = SingleXFormPrim(
            FRANKA_BASE_PRIM_PATH, name="franka_link0_pose")
        self._ee_prim = SingleXFormPrim(
            END_EFFECTOR_PRIM_PATH, name="panda_hand_pose")

        self._diff_controller = DifferentialController(
            name="nova_diff", wheel_radius=0.14, wheel_base=0.413)
        self._pose_controller = WheelBasePoseController(
            name="nova_pose_controller",
            open_loop_wheel_controller=self._diff_controller,
            is_holonomic=False,
        )
        kinematics_config = load_supported_lula_kinematics_solver_config(
            "Franka")
        self._ik_solver = LulaKinematicsSolver(**kinematics_config)
        self._articulation_ik = ArticulationKinematicsSolver(
            self._robot, self._ik_solver, "right_gripper")

        self._reset_state_machine()

        if world.physics_callback_exists("pick_place_state_machine"):
            world.remove_physics_callback("pick_place_state_machine")
        world.add_physics_callback(
            "pick_place_state_machine", self._on_physics_step)
        await world.play_async()

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("pick_place_state_machine"):
            world.remove_physics_callback("pick_place_state_machine")
        self._stop_wheels()

    async def setup_post_reset(self):
        self._reset_state_machine()
        world = self.get_world()
        if not world.physics_callback_exists("pick_place_state_machine"):
            world.add_physics_callback(
                "pick_place_state_machine", self._on_physics_step)
        await world.play_async()

    def world_cleanup(self):
        world = self.get_world()
        if world is not None and world.physics_callback_exists("pick_place_state_machine"):
            world.remove_physics_callback("pick_place_state_machine")
        self._clear_debug_path()
        self._release_debug_draw_interface()
        self._robot = None
        self._cube = None
        self._robot_controller = None
        self._wheel_subset = None
        self._arm_subset = None
        self._finger_subset = None
        self._pose_controller = None
        self._diff_controller = None
        self._ik_solver = None
        self._articulation_ik = None
        self._chassis_prim = None
        self._franka_base_prim = None
        self._ee_prim = None
        self._place_goal_pos = None
        self._arm_segments = []

    async def _load_robot_after_navmesh_bake(self):
        world = self.get_world()
        add_reference_to_stage(ROBOT_USD_PATH, ROBOT_PRIM_PATH)
        # Wrap the articulation authored in custom_robot.usd; this does not create a second robot.
        self._robot = world.scene.add(SingleArticulation(
            prim_path=ROBOT_ARTICULATION_PRIM_PATH, name="nova_carter"))
        await omni.kit.app.get_app().next_update_async()
        self._set_robot_initial_pose()
        await world.reset_async()
        self._set_robot_initial_pose()

    def _set_robot_initial_pose(self):
        if self._robot is None:
            return
        self._robot.set_default_state(
            position=ROBOT_INITIAL_POSITION,
            orientation=euler_angles_to_quat(
                np.array([0.0, 0.0, ROBOT_INITIAL_YAW])),
        )
        self._robot.set_world_pose(
            position=ROBOT_INITIAL_POSITION,
            orientation=euler_angles_to_quat(
                np.array([0.0, 0.0, ROBOT_INITIAL_YAW])),
        )

    def _reset_state_machine(self):
        self._state = PickPlaceState.NAVIGATE_TO_PICK
        if self._pick_goal_pos is None:
            self._pick_goal_pos = self._resolve_pick_goal_pos()
        if self._place_goal_pos is None:
            self._place_goal_pos = self._resolve_place_goal_pos()
        self._path = self._query_navmesh_path(self._pick_goal_pos)
        self._waypoint_index = 0
        self._settle_count = 0
        self._arm_segments = []
        self._active_segment = None
        self._holding_object = False
        self._task_started = True
        self._open_gripper()

    def _on_physics_step(self, step_size):
        if not self._task_started or self._state == PickPlaceState.DONE:
            return

        if self._holding_object:
            self._close_gripper()

        if self._state == PickPlaceState.NAVIGATE_TO_PICK:
            if self._navigate_path_step(self._pick_goal_pos):
                self._stop_wheels()
                self._state = PickPlaceState.FACE_CUBE

        elif self._state == PickPlaceState.FACE_CUBE:
            if self._face_target_step(cube_position, "pick"):
                self._stop_wheels()
                self._settle_count = 0
                self._state = PickPlaceState.SETTLE

        elif self._state == PickPlaceState.SETTLE:
            self._stop_wheels()
            self._settle_count += 1
            if self._settle_count >= self.settle_steps_after_stop:
                self._prepare_grasp_sequence()
                self._state = PickPlaceState.GRASP_SEQUENCE

        elif self._state == PickPlaceState.GRASP_SEQUENCE:
            if self._run_arm_sequence():
                self._holding_object = True
                lift_target = cube_position + \
                    np.array([0.0, 0.0, self.lift_height])
                self._arm_segments = [ArmSegment(
                    lift_target, self.gripper_closed, settle_steps=20)]
                self._active_segment = None
                self._state = PickPlaceState.LIFT

        elif self._state == PickPlaceState.LIFT:
            if self._run_arm_sequence():
                self._prepare_post_lift_x_adjust_sequence()
                self._state = PickPlaceState.ADJUST_ARM_X

        elif self._state == PickPlaceState.ADJUST_ARM_X:
            self._close_gripper()
            if self._run_arm_sequence():
                if self._place_goal_pos is None:
                    self._place_goal_pos = self._resolve_place_goal_pos()
                self._path = self._query_navmesh_path(self._place_goal_pos)
                self._waypoint_index = 0
                self._state = PickPlaceState.NAVIGATE_TO_PLACE

        elif self._state == PickPlaceState.NAVIGATE_TO_PLACE:
            self._close_gripper()
            if self._navigate_path_step(self._place_goal_pos):
                self._stop_wheels()
                self._state = PickPlaceState.FACE_PLACE

        elif self._state == PickPlaceState.FACE_PLACE:
            self._close_gripper()
            if self._face_target_step(place_cube_pos, "place"):
                self._stop_wheels()
                self._prepare_place_sequence()
                self._state = PickPlaceState.PLACE_RELEASE

        elif self._state == PickPlaceState.PLACE_RELEASE:
            if self._run_arm_sequence():
                self._holding_object = False
                self._open_gripper()
                self._stop_wheels()
                self._state = PickPlaceState.DONE
                carb.log_info("Pick and place task complete.")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _navigate_path_step(self, final_goal):
        if self._wheel_subset is None:
            return False

        robot_pos, robot_quat = self._robot.get_world_pose()
        if np.linalg.norm(robot_pos[:2] - np.asarray(final_goal)[:2]) <= self.goal_threshold:
            return True

        if not self._path:
            self._path = self._query_navmesh_path(final_goal)
            self._waypoint_index = 0

        target = np.asarray(
            self._path[min(self._waypoint_index, len(self._path) - 1)], dtype=float)
        if np.linalg.norm(robot_pos[:2] - target[:2]) <= self.waypoint_threshold:
            self._waypoint_index += 1
            if self._waypoint_index >= len(self._path):
                return np.linalg.norm(robot_pos[:2] - np.asarray(final_goal)[:2]) <= self.goal_threshold
            target = np.asarray(self._path[self._waypoint_index], dtype=float)

        wheel_action = self._pose_controller.forward(
            start_position=robot_pos,
            start_orientation=robot_quat,
            goal_position=target,
            lateral_velocity=NAV_LINEAR_VELOCITY,
            yaw_velocity=NAV_YAW_VELOCITY,
            position_tol=self.waypoint_threshold,
        )
        self._apply_wheel_velocity_action(wheel_action)
        return False

    def _face_target_step(self, target_pos, label):
        if self._wheel_subset is None:
            return True

        robot_pos, robot_quat = self._get_facing_pose()
        target_yaw = np.arctan2(
            target_pos[1] - robot_pos[1],
            target_pos[0] - robot_pos[0],
        )
        robot_yaw = quat_to_euler_angles(robot_quat)[-1]
        side_name, signed_error, side_yaw = self._choose_nearest_robot_side_error(
            robot_yaw, target_yaw)
        rotate_angle = abs(signed_error)

        if rotate_angle <= FACE_CUBE_YAW_TOLERANCE:
            return True

        self._apply_face_rotation_action(signed_error)
        return False

    def _apply_face_rotation_action(self, signed_error):
        if self._diff_controller is None:
            return

        yaw_velocity = np.clip(
            signed_error, -FACE_CUBE_YAW_VELOCITY, FACE_CUBE_YAW_VELOCITY)
        wheel_action = self._diff_controller.forward(
            np.array([0.0, yaw_velocity], dtype=float))
        self._apply_wheel_velocity_action(wheel_action)

    def _get_facing_pose(self):
        if self._chassis_prim is not None:
            try:
                return self._chassis_prim.get_world_pose()
            except Exception as exc:
                carb.log_warn(
                    f"Could not read chassis pose; falling back to robot pose. Details: {exc}")
        return self._robot.get_world_pose()

    def _choose_nearest_robot_side_error(self, robot_yaw, target_yaw):
        candidates = []
        for side_name, side_offset in ROBOT_SIDE_YAW_OFFSETS.items():
            side_yaw = self._wrap_angle(robot_yaw + side_offset)
            signed_error = self._wrap_angle(target_yaw - side_yaw)
            candidates.append((side_name, signed_error, side_yaw))
        return min(candidates, key=lambda item: abs(item[1]))

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _apply_wheel_velocity_action(self, wheel_action):
        velocities = getattr(wheel_action, "joint_velocities", wheel_action)
        velocities = np.asarray(velocities, dtype=float)
        if velocities.size != len(self._wheel_joint_names):
            velocities = velocities[: len(self._wheel_joint_names)]
        self._nav_log_counter += 1
        if self._nav_log_counter % 60 == 0:
            carb.log_info(
                f"[Nav] wheel velocity command {velocities.tolist()} on {self._wheel_joint_names}")
        self._wheel_subset.apply_action(joint_velocities=velocities)

    def _stop_wheels(self):
        if self._wheel_subset is not None:
            zero = np.zeros(len(self._wheel_joint_names))
            self._wheel_subset.apply_action(joint_velocities=zero)

    def _resolve_wheel_joint_names(self, available_dofs):
        if all(name in available_dofs for name in WHEEL_JOINT_NAMES):
            return list(WHEEL_JOINT_NAMES)

        lower_to_name = {name.lower(): name for name in available_dofs}
        left_aliases = (
            "left_wheel_joint", "wheel_left_joint", "left_wheel",
            "left_wheel_axle", "left_tire_joint", "left_tyre_joint",
            "rear_left_wheel_joint", "back_left_wheel_joint",
        )
        right_aliases = (
            "right_wheel_joint", "wheel_right_joint", "right_wheel",
            "right_wheel_axle", "right_tire_joint", "right_tyre_joint",
            "rear_right_wheel_joint", "back_right_wheel_joint",
        )
        left = next(
            (lower_to_name[name] for name in left_aliases if name in lower_to_name), None)
        right = next(
            (lower_to_name[name] for name in right_aliases if name in lower_to_name), None)
        if left is not None and right is not None:
            return [left, right]

        def is_wheel_like(name):
            lowered = name.lower()
            return (
                ("wheel" in lowered or "tire" in lowered or "tyre" in lowered)
                and "caster" not in lowered
                and "panda" not in lowered
                and "finger" not in lowered
            )

        wheel_like = [name for name in available_dofs if is_wheel_like(name)]
        left_like = [name for name in wheel_like if "left" in name.lower(
        ) or name.lower().endswith("_l")]
        right_like = [name for name in wheel_like if "right" in name.lower(
        ) or name.lower().endswith("_r")]
        if left_like and right_like:
            return [left_like[0], right_like[0]]
        if len(wheel_like) >= 2:
            carb.log_warn(
                f"Using first two wheel-like DOFs as differential wheels: {wheel_like[:2]}")
            return wheel_like[:2]
        return []

    def _resolve_pick_goal_pos(self):
        return self._resolve_walkable_goal_pos(cube_position, "pick")

    def _resolve_place_goal_pos(self):
        return self._resolve_walkable_goal_pos(place_cube_pos, "place")

    def _resolve_walkable_goal_pos(self, target_pos, label):
        fallback = np.array([target_pos[0], target_pos[1], 0.0], dtype=float)
        nav = self._get_navigation_interface()
        if nav is None:
            carb.log_warn(
                f"[NavMesh] Could not resolve {label} goal from navmesh; using XY fallback {fallback.tolist()}.")
            return fallback

        try:
            navmesh = nav.get_navmesh()
            if navmesh is None:
                carb.log_warn(
                    f"[NavMesh] No navmesh available for {label} goal; using XY fallback {fallback.tolist()}.")
                return fallback

            target_ground = carb.Float3(
                float(target_pos[0]), float(target_pos[1]), 0.0)
            closest, island_id = navmesh.query_closest_point(target_ground)
            if closest is None:
                carb.log_warn(
                    f"[NavMesh] No closest walkable point found for {label}; using fallback {fallback.tolist()}.")
                return fallback

            walkable_goal = np.array([closest.x, closest.y, 0.0], dtype=float)
            carb.log_info(
                f"[NavMesh] {label.capitalize()} goal resolved from XY {fallback.tolist()} to walkable {walkable_goal.tolist()} on island {island_id}.")
            return walkable_goal
        except Exception as exc:
            carb.log_warn(
                f"[NavMesh] {label.capitalize()} goal resolution failed; using fallback {fallback.tolist()}. Details: {exc}")
            return fallback

    def _query_navmesh_path(self, goal_pos):
        robot_pos, _ = self._robot.get_world_pose()
        start = np.asarray(robot_pos, dtype=float)
        goal = np.asarray(goal_pos, dtype=float)
        path = self._navmesh_path(start, goal)
        if len(path) < 2:
            path = [start, goal]
        self._draw_debug_path(path)
        return [np.asarray(p, dtype=float) for p in path]

    async def _bake_navmesh(self):
        nav = self._get_navigation_interface()
        if nav is None:
            carb.log_warn(
                "NavMesh extension unavailable; using direct fallback paths.")
            return

        try:
            carb.settings.get_settings().set(
                "/exts/omni.anim.navigation.core/navMesh/config/agentMinRadius",
                NAVMESH_MIN_RADIUS,
            )
            await omni.kit.app.get_app().next_update_async()
            carb.log_info("[NavMesh] Baking...")
            started = nav.start_navmesh_baking()
            if not started:
                carb.log_warn("[NavMesh] Baking did not start.")
                return
            while nav.is_navmesh_baking():
                await omni.kit.app.get_app().next_update_async()
            navmesh = nav.get_navmesh()
            if navmesh is None:
                carb.log_warn(
                    "[NavMesh] Bake finished but no navmesh was produced.")
                return
            draw_vertices = navmesh.get_draw_triangles(0)
            carb.log_info(
                f"[NavMesh] Bake complete. Draw triangle vertices: {len(draw_vertices)}")
        except Exception as exc:
            carb.log_warn(
                f"Runtime NavMesh baking failed; using direct fallback paths. Details: {exc}")

    def _navmesh_path(self, start, goal):
        nav = self._get_navigation_interface()
        if nav is None:
            return [start, goal]
        try:
            navmesh = nav.get_navmesh()
            if navmesh is None:
                return [start, goal]
            start_pos = carb.Float3(
                float(start[0]), float(start[1]), float(start[2]))
            goal_pos = carb.Float3(
                float(goal[0]), float(goal[1]), float(goal[2]))
            path = navmesh.query_shortest_path(
                start_pos=start_pos, end_pos=goal_pos)
            if path is not None and path.get_point_count() > 0:
                return [np.array([p.x, p.y, p.z], dtype=float) for p in path.get_points()]
        except Exception as exc:
            carb.log_warn(
                f"NavMesh path query failed; using straight path. Details: {exc}")
        return [start, goal]

    def _draw_debug_path(self, path):
        if len(path) < 2:
            self._clear_debug_path()
            return

        draw = self._get_debug_draw_interface()
        if draw is None:
            return

        self._clear_debug_path()
        points = [np.asarray(point, dtype=float).copy() for point in path]
        for point in points:
            point[2] += PATH_LINE_Z_OFFSET

        starts = [tuple(point.tolist()) for point in points[:-1]]
        ends = [tuple(point.tolist()) for point in points[1:]]
        colors = [PATH_LINE_COLOR] * len(starts)
        widths = [PATH_LINE_WIDTH] * len(starts)
        draw.draw_lines(starts, ends, colors, widths)
        carb.log_info(
            f"[NavMesh] Drew path with {len(starts)} red line segments.")

    def _clear_debug_path(self):
        if self._debug_draw is None:
            return
        try:
            self._debug_draw.clear_lines()
        except Exception as exc:
            carb.log_warn(f"Could not clear debug path lines: {exc}")

    def _release_debug_draw_interface(self):
        global _debug_draw
        if self._debug_draw is None or _debug_draw is None:
            self._debug_draw = None
            return
        try:
            _debug_draw.release_debug_draw_interface(self._debug_draw)
        except Exception:
            pass
        self._debug_draw = None

    def _get_debug_draw_interface(self):
        global _debug_draw
        if self._debug_draw is not None:
            return self._debug_draw

        if _debug_draw is None:
            try:
                extension_manager = omni.kit.app.get_app().get_extension_manager()
                extension_manager.set_extension_enabled_immediate(
                    "isaacsim.util.debug_draw", True)
                from isaacsim.util.debug_draw import _debug_draw as debug_draw_module
                _debug_draw = debug_draw_module
            except Exception as exc:
                carb.log_warn(
                    f"Could not enable debug draw path visualization: {exc}")
                return None

        try:
            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:
            carb.log_warn(f"Could not acquire debug draw interface: {exc}")
            self._debug_draw = None
        return self._debug_draw

    def _get_navigation_interface(self):
        global nav_core
        if nav_core is None:
            try:
                extension_manager = omni.kit.app.get_app().get_extension_manager()
                for extension_name in (
                    "omni.anim.navigation.bundle",
                    "omni.anim.navigation.core",
                    "omni.anim.navigation.ui",
                ):
                    try:
                        extension_manager.set_extension_enabled_immediate(
                            extension_name, True)
                    except Exception as exc:
                        carb.log_warn(
                            f"Could not enable {extension_name}: {exc}")
                nav_core = importlib.import_module("omni.anim.navigation.core")
            except Exception as exc:
                carb.log_warn(
                    f"Could not enable omni.anim.navigation.core: {exc}")
                return None
        try:
            return nav_core.acquire_interface()
        except Exception as exc:
            carb.log_warn(f"Could not acquire NavMesh interface: {exc}")
            return None

    # ------------------------------------------------------------------
    # Arm and gripper state execution
    # ------------------------------------------------------------------
    def _prepare_grasp_sequence(self):
        pre_grasp = cube_position + np.array([0.0, 0.0, self.pre_grasp_height])
        grasp = cube_position + np.array([0.0, 0.0, self.grasp_height_offset])
        self._arm_segments = [
            ArmSegment(pre_grasp, self.gripper_open, settle_steps=45),
            ArmSegment(grasp, self.gripper_open, settle_steps=10),
            ArmSegment(grasp, self.gripper_closed, settle_steps=45),
        ]
        self._active_segment = None

    def _prepare_place_sequence(self):
        place_above = place_cube_pos + \
            np.array([0.0, 0.0, self.pre_grasp_height])
        place = place_cube_pos + np.array([0.0, 0.0, 0.045])
        self._arm_segments = [
            ArmSegment(place_above, self.gripper_closed, settle_steps=20),
            ArmSegment(place, self.gripper_closed, settle_steps=20),
            ArmSegment(place, self.gripper_open, settle_steps=45),
            ArmSegment(place_above, self.gripper_open, settle_steps=0),
        ]
        self._active_segment = None

    def _prepare_post_lift_x_adjust_sequence(self):
        self._arm_segments = [
            ArmSegment(
                None,
                self.gripper_closed,
                settle_steps=self.post_lift_adjust_settle_steps,
                target_joints=self.post_lift_straight_arm_joints,
            )
        ]
        self._active_segment = None

    def _run_arm_sequence(self):
        if self._active_segment is None:
            if not self._arm_segments:
                return True
            self._active_segment = self._arm_segments.pop(0)
            self._build_segment_trajectory(self._active_segment)

        segment = self._active_segment
        self._apply_gripper_width(segment.gripper_width)

        if segment.index < len(segment.trajectory):
            target_joints = segment.trajectory[segment.index]
            self._arm_subset.apply_action(joint_positions=target_joints)
            segment.index += 1
            return False

        if not segment.ik_solved:
            return False

        segment.waited += 1
        if segment.waited >= segment.settle_steps:
            self._active_segment = None
        return False

    def _build_segment_trajectory(self, segment):
        current_joints = np.asarray(
            self._arm_subset.get_joint_positions(), dtype=float)
        if segment.target_joints is not None:
            target_joints = segment.target_joints
            segment.ik_solved = True
        else:
            target_joints = self._solve_ik(segment.target_pos)
            if target_joints is None:
                segment.ik_solved = False
                target_joints = current_joints
            else:
                segment.ik_solved = True
        alphas = np.linspace(0.0, 1.0, self.arm_trajectory_steps)
        smooth = 3.0 * alphas**2 - 2.0 * alphas**3
        segment.trajectory = [current_joints + alpha *
                              (target_joints - current_joints) for alpha in smooth]
        segment.index = 0
        segment.waited = 0

    def _solve_ik(self, target_pos):
        base_pos, base_quat = self._franka_base_prim.get_world_pose()
        self._ik_solver.set_robot_base_pose(base_pos, base_quat)
        downward_orientation = euler_angles_to_quat(
            np.array([np.pi, 0.0, 0.0]))
        action, success = self._articulation_ik.compute_inverse_kinematics(
            target_position=np.asarray(target_pos, dtype=float),
            target_orientation=downward_orientation,
        )
        if not success:
            carb.log_warn(f"IK failed for target {target_pos}")
            return None
        joint_positions = np.asarray(action.joint_positions, dtype=float)
        if joint_positions.size >= len(ARM_JOINT_NAMES):
            return joint_positions[: len(ARM_JOINT_NAMES)]
        return joint_positions

    def _ee_position_reached(self, target_pos, tolerance):
        ee_pos, _ = self._ee_prim.get_world_pose()
        return np.linalg.norm(np.asarray(ee_pos) - np.asarray(target_pos)) <= tolerance

    def _open_gripper(self):
        self._apply_gripper_width(self.gripper_open)

    def _close_gripper(self):
        self._apply_gripper_width(self.gripper_closed)

    def _apply_gripper_width(self, width):
        if self._finger_subset is None:
            return
        positions = np.array([width, width], dtype=float)
        self._finger_subset.apply_action(joint_positions=positions)


async def _load_hello_world_sample():
    previous_task = globals().get("_HELLO_WORLD_LOAD_TASK")
    if previous_task is not None and not previous_task.done():
        previous_task.cancel()
        try:
            await previous_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            carb.log_warn(f"Previous HelloWorld load task ended with: {exc}")

    previous_sample = globals().get("_HELLO_WORLD_SAMPLE")
    if previous_sample is not None:
        try:
            previous_world = previous_sample.get_world()
            if previous_world is not None:
                if previous_world.physics_callback_exists("pick_place_state_machine"):
                    previous_world.remove_physics_callback(
                        "pick_place_state_machine")
                previous_world.stop()
                previous_world.clear_all_callbacks()
                previous_world.clear_instance()
            previous_sample.world_cleanup()
        except Exception as exc:
            carb.log_warn(f"Previous HelloWorld cleanup failed: {exc}")
        finally:
            globals()["_HELLO_WORLD_SAMPLE"] = None

    sample = HelloWorld()
    globals()["_HELLO_WORLD_SAMPLE"] = sample
    await sample.load_world_async()


globals()["_HELLO_WORLD_LOAD_TASK"] = asyncio.ensure_future(
    _load_hello_world_sample())
