# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import random
from enum import Enum, auto

import carb
import carb.settings
import numpy as np
import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Usd, UsdGeom
from isaacsim.core.api.articulations import ArticulationSubset
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from isaacsim.core.utils.stage import add_reference_to_stage
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
# Scene asset paths
# -----------------------------------------------------------------------------
SCENE_USD_PATH = r"C:\Users\user\NTHU\digital_twin\dataset\midterm\asset\hospital.usd"
ROBOT_USD_PATH = r"C:\Users\user\NTHU\digital_twin\dataset\midterm\asset\custom_robot.usd"

# -----------------------------------------------------------------------------
# Robot prim paths
# -----------------------------------------------------------------------------
ROBOT_PRIM_PATH = "/World/nova_carter"
ROBOT_ARTICULATION_PRIM_PATH = ROBOT_PRIM_PATH + "/nova_carter"
CHASSIS_PRIM_PATH = ROBOT_ARTICULATION_PRIM_PATH + "/chassis_link"
FRANKA_BASE_PRIM_PATH = ROBOT_ARTICULATION_PRIM_PATH + "/franka/panda_link0"
END_EFFECTOR_PRIM_PATH = ROBOT_ARTICULATION_PRIM_PATH + "/franka/panda_hand"
ROBOT_CAMERA_PRIM_PATH = CHASSIS_PRIM_PATH + "/rear_follow_camera"

# -----------------------------------------------------------------------------
# Robot camera pose
# -----------------------------------------------------------------------------
ROBOT_CAMERA_LOCAL_EYE = np.array([-2, 0.0, 4])
ROBOT_CAMERA_LOCAL_TARGET = np.array([0.65, 0.0, 0.45])

# -----------------------------------------------------------------------------
# Robot start placement
# -----------------------------------------------------------------------------
ROBOT_INITIAL_POSITION = np.array([6.0, -1.0, 0.0])
ROBOT_INITIAL_YAW = 0.0
ROBOT_START_MIN_DISTANCE_FROM_CUBE = 5.0
ROBOT_START_MAX_DISTANCE_FROM_CUBE = 8.0
ROBOT_START_TARGET_DISTANCE_FROM_CUBE = 6.5
ROBOT_START_RING_SAMPLES = 72
ROBOT_START_CLEARANCE_RADIUS = 0.65
ROBOT_START_CLEARANCE_MAX_PROJECTION = 0.10
ROBOT_START_CLEARANCE_SAMPLES = 16

# -----------------------------------------------------------------------------
# Navigation and path drawing
# -----------------------------------------------------------------------------
NAVMESH_MIN_RADIUS = 40
PATH_LINE_COLOR = (1.0, 0.0, 0.0, 1.0)
PATH_LINE_WIDTH = 8
PATH_LINE_Z_OFFSET = 0.05
NAV_LINEAR_VELOCITY = 0.35
NAV_YAW_VELOCITY = 0.8
ENABLE_FACE_TARGET_STATES = True
FACE_CUBE_YAW_TOLERANCE = 0.02
FACE_CUBE_YAW_VELOCITY = 0.5

# -----------------------------------------------------------------------------
# Cube placement and task selection
# -----------------------------------------------------------------------------
CUBE_PRIM_PATH = "/World/TaskCube"
CUBE_SIZE = 0.05
RANDOMIZE_CUBE_POSITIONS = True
CUBE_POSITION_NAME_KEYWORDS = ["SideTable", "Desk"]
CUBE_PLACEMENT_SURFACE_MARGIN = 0.03
CUBE_PLACEMENT_MAX_WALKABLE_DISTANCE = 0.6
CUBE_PLACEMENT_MIN_OUTWARD_WALKABLE_OFFSET = 0.02
CUBE_MIN_PICK_PLACE_DISTANCE = 5.0
CUBE_MAX_PICK_PLACE_PATH_POINTS = 500
CUBE_CENTER_HEIGHT_RANGE = (0.8, 0.90)

# -----------------------------------------------------------------------------
# Scene loading timing
# -----------------------------------------------------------------------------
SCENE_ASSET_LOAD_MAX_FRAMES = 300
SCENE_ASSET_LOAD_SETTLE_FRAMES = 5

# -----------------------------------------------------------------------------
# Robot orientation helpers
# -----------------------------------------------------------------------------
ROBOT_SIDE_YAW_OFFSETS = {
    "+Y": np.pi / 2.0,
    "-Y": -np.pi / 2.0,
}

# -----------------------------------------------------------------------------
# Manual homework task positions
# -----------------------------------------------------------------------------
cube_position = np.array([9.98115, -1.39854, 0.83727])
place_cube_pos = np.array([22.93955, -3.11071, 0.85179])

# -----------------------------------------------------------------------------
# Robot joint names
# -----------------------------------------------------------------------------
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
    ADJUST_ARM_X = auto()
    NAVIGATE_TO_PLACE = auto()
    FACE_PLACE = auto()
    PLACE_RELEASE = auto()
    DONE = auto()


class ArmSegment:
    def __init__(
        self,
        target_pos,
        gripper_width,
        settle_steps=0,
        ee_tolerance=0.025,
        target_joints=None,
        label="arm_segment",
        reach_timeout_steps=240,
        trajectory_steps=None,
    ):
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
        self.reach_waited = 0
        self.ik_solved = False
        self.label = label
        self.reach_timeout_steps = reach_timeout_steps
        self.trajectory_steps = trajectory_steps


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        # World simulation timing
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 120.0
        self._world_settings["rendering_dt"] = 1.0 / 60.0

        # Runtime scene and robot handles
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

        # Pick/place task state
        self._state = PickPlaceState.NAVIGATE_TO_PICK
        self._cube_position = np.array(cube_position, dtype=float)
        self._place_cube_pos = np.array(place_cube_pos, dtype=float)
        self._robot_initial_position = np.array(
            ROBOT_INITIAL_POSITION, dtype=float)
        self._robot_initial_yaw = ROBOT_INITIAL_YAW
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

        # Navigation thresholds
        self.waypoint_threshold = 0.18
        self.goal_threshold = 0.02
        self.settle_steps_after_stop = 55

        # Gripper widths and end-effector heights
        self.gripper_open = 0.05
        self.gripper_closed = 0.005
        self.pre_grasp_height = 0.25
        self.grasp_height_offset = 0.02
        self.lift_height = 0.25

        # Fetch sequence timing
        self.fetch_open_settle_steps = 20
        self.fetch_pre_grasp_move_steps = 140
        self.fetch_pre_grasp_settle_steps = 20
        self.fetch_descent_move_steps = 140
        self.fetch_descent_settle_steps = 20
        self.fetch_close_settle_steps = 20
        self.fetch_lift_move_steps = 140
        self.fetch_lift_settle_steps = 20

        # Post-grasp arm adjustment
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
            position=self._cube_position,
            size=CUBE_SIZE,
            color=np.array([0.1, 0.55, 1.0]),
            mass=0.05,
        ))

    async def setup_post_load(self):
        world = self.get_world()
        carb.log_warn(
            "[Placement] setup_post_load reached; resolving randomized task and robot start.")
        await self._wait_for_scene_assets_loaded()
        await self._bake_navmesh()
        self._randomize_task_positions_from_keywords()
        self._resolve_robot_initial_pose_near_cube()
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
        self._setup_robot_follow_camera()

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
        orientation = euler_angles_to_quat(
            np.array([0.0, 0.0, self._robot_initial_yaw]))
        self._apply_robot_root_pose(orientation)
        self._robot.set_default_state(
            position=self._robot_initial_position,
            orientation=orientation,
        )
        self._robot.set_world_pose(
            position=self._robot_initial_position,
            orientation=orientation,
        )
        carb.log_warn(
            f"[Placement] Applied robot initial pose {self._robot_initial_position.tolist()} with yaw {self._robot_initial_yaw:.3f}.")

    def _apply_robot_root_pose(self, orientation):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not prim.IsValid():
            return

        xform_api = UsdGeom.XformCommonAPI(prim)
        xform_api.SetTranslate(Gf.Vec3d(
            float(self._robot_initial_position[0]),
            float(self._robot_initial_position[1]),
            float(self._robot_initial_position[2]),
        ))
        xform_api.SetRotate(
            Gf.Vec3f(0.0, 0.0, float(np.degrees(self._robot_initial_yaw))),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )

    async def _wait_for_scene_assets_loaded(self):
        context = omni.usd.get_context()
        app = omni.kit.app.get_app()
        carb.log_warn(
            "[Assets] Waiting for scene assets to finish loading before NavMesh bake.")

        for frame in range(SCENE_ASSET_LOAD_MAX_FRAMES):
            await app.next_update_async()
            try:
                loading = context.get_stage_loading_status()
                if isinstance(loading, (tuple, list)):
                    pending = sum(int(value) for value in loading)
                    if pending == 0:
                        break
                elif not loading:
                    break
            except Exception:
                # Some Isaac/Kit versions do not expose stage loading status.
                break
        else:
            carb.log_warn(
                f"[Assets] Scene asset loading did not report idle after {SCENE_ASSET_LOAD_MAX_FRAMES} frame(s); baking anyway.")

        for _ in range(SCENE_ASSET_LOAD_SETTLE_FRAMES):
            await app.next_update_async()
        carb.log_warn(
            f"[Assets] Scene asset wait complete; added {SCENE_ASSET_LOAD_SETTLE_FRAMES} settle frame(s).")

    def _setup_robot_follow_camera(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            carb.log_warn(
                "[Camera] Could not get USD stage; robot camera was not created.")
            return

        camera = UsdGeom.Camera.Define(stage, ROBOT_CAMERA_PRIM_PATH)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

        camera_xform = UsdGeom.Xformable(camera.GetPrim())
        camera_xform.ClearXformOpOrder()
        look_at = Gf.Matrix4d(1.0)
        look_at.SetLookAt(
            Gf.Vec3d(*ROBOT_CAMERA_LOCAL_EYE),
            Gf.Vec3d(*ROBOT_CAMERA_LOCAL_TARGET),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        camera_xform.AddTransformOp().Set(look_at.GetInverse())

        viewport = get_active_viewport()
        if viewport is None:
            carb.log_warn(
                "[Camera] Could not get active viewport; robot camera was created but not selected.")
            return
        viewport.camera_path = ROBOT_CAMERA_PRIM_PATH
        carb.log_info(
            f"[Camera] Active viewport attached to robot camera: {ROBOT_CAMERA_PRIM_PATH}")

    # ------------------------------------------------------------------
    # Random task placement
    # ------------------------------------------------------------------
    def _randomize_task_positions_from_keywords(self):
        if not RANDOMIZE_CUBE_POSITIONS:
            return

        candidates = self._find_keyword_placement_candidates(
            CUBE_POSITION_NAME_KEYWORDS)
        if len(candidates) < 2:
            carb.log_warn(
                f"[Placement] Need at least 2 valid keyword placement candidates; found {len(candidates)}. "
                "Using fixed cube/task positions.")
            return

        random.shuffle(candidates)
        pick_candidate = random.choice(candidates)
        place_candidates = []
        for candidate in candidates:
            if np.linalg.norm(candidate["position"][:2] - pick_candidate["position"][:2]) < CUBE_MIN_PICK_PLACE_DISTANCE:
                continue

            path_point_count = self._placement_path_point_count(
                pick_candidate["walkable_goal"], candidate["walkable_goal"])
            if path_point_count is None or path_point_count > CUBE_MAX_PICK_PLACE_PATH_POINTS:
                continue

            candidate = dict(candidate)
            candidate["path_point_count"] = path_point_count
            place_candidates.append(candidate)

        if not place_candidates:
            carb.log_warn(
                f"[Placement] No target candidate at least {CUBE_MIN_PICK_PLACE_DISTANCE:.1f} m away "
                f"and within {CUBE_MAX_PICK_PLACE_PATH_POINTS} nav path point(s) from the picked cube position. "
                "Using fixed target position.")
            return

        place_candidate = random.choice(place_candidates)
        self._cube_position = pick_candidate["position"].copy()
        self._place_cube_pos = place_candidate["position"].copy()
        self._pick_goal_pos = pick_candidate["walkable_goal"].copy()
        self._place_goal_pos = place_candidate["walkable_goal"].copy()

        try:
            self._cube.set_world_pose(position=self._cube_position)
            if hasattr(self._cube, "set_default_state"):
                self._cube.set_default_state(position=self._cube_position)
        except Exception as exc:
            carb.log_warn(
                f"[Placement] Could not move cube to randomized position {self._cube_position.tolist()}: {exc}")

        carb.log_info(
            f"[Placement] Cube randomized on {pick_candidate['prim_path']} at {self._cube_position.tolist()} "
            f"(walkable distance {pick_candidate['walkable_distance']:.3f} m).")
        carb.log_info(
            f"[Placement] Target randomized on {place_candidate['prim_path']} at {self._place_cube_pos.tolist()} "
            f"(walkable distance {place_candidate['walkable_distance']:.3f} m, "
            f"path points {place_candidate['path_point_count']}).")

    def _find_keyword_placement_candidates(self, keywords):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            carb.log_warn("[Placement] Could not get USD stage.")
            return []

        lowered_keywords = [keyword.lower() for keyword in keywords]
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        candidates = []

        for prim in stage.Traverse():
            if not prim.IsActive():
                continue
            prim_path = str(prim.GetPath())
            if prim_path == CUBE_PRIM_PATH:
                continue
            searchable_name = f"{prim.GetName()} {prim_path}".lower()
            if not any(keyword in searchable_name for keyword in lowered_keywords):
                continue

            try:
                aligned_box = bbox_cache.ComputeWorldBound(
                    prim).ComputeAlignedBox()
                bbox_min = np.array(aligned_box.GetMin(), dtype=float)
                bbox_max = np.array(aligned_box.GetMax(), dtype=float)
            except Exception as exc:
                carb.log_warn(
                    f"[Placement] Could not compute bounds for {prim_path}: {exc}")
                continue

            extents = bbox_max - bbox_min
            top_center_z = bbox_max[2] + CUBE_SIZE * 0.5
            if extents[0] <= CUBE_SIZE or extents[1] <= CUBE_SIZE:
                continue
            if not (CUBE_CENTER_HEIGHT_RANGE[0] <= top_center_z <= CUBE_CENTER_HEIGHT_RANGE[1]):
                continue

            for position, outward_normals in self._sample_surface_corner_positions(bbox_min, bbox_max):
                walkable_goal = self._closest_walkable_point(position)
                if walkable_goal is None:
                    continue
                walkable_distance = np.linalg.norm(
                    position[:2] - walkable_goal[:2])
                if walkable_distance > CUBE_PLACEMENT_MAX_WALKABLE_DISTANCE:
                    continue
                walkable_offset = walkable_goal[:2] - position[:2]
                outward_offset = min(
                    np.dot(walkable_offset, outward_normal)
                    for outward_normal in outward_normals
                )
                if outward_offset < CUBE_PLACEMENT_MIN_OUTWARD_WALKABLE_OFFSET:
                    continue
                candidates.append({
                    "position": position,
                    "walkable_goal": walkable_goal,
                    "walkable_distance": walkable_distance,
                    "prim_path": prim_path,
                })

        carb.log_warn(
            f"[Placement] Found {len(candidates)} placement candidates from keywords {keywords} "
            f"with cube center height in {CUBE_CENTER_HEIGHT_RANGE}, outward walkable offset >= "
            f"{CUBE_PLACEMENT_MIN_OUTWARD_WALKABLE_OFFSET:.2f} m.")
        return candidates

    def _sample_surface_corner_positions(self, bbox_min, bbox_max):
        x_min, y_min, _ = bbox_min
        x_max, y_max, z_max = bbox_max
        edge_inset = CUBE_SIZE * 0.5 + CUBE_PLACEMENT_SURFACE_MARGIN
        x_low = min(x_max, x_min + edge_inset)
        x_high = max(x_min, x_max - edge_inset)
        y_low = min(y_max, y_min + edge_inset)
        y_high = max(y_min, y_max - edge_inset)
        top_center_z = z_max + CUBE_SIZE * 0.5
        positions = [
            (
                np.array([x_low, y_low, top_center_z], dtype=float),
                (np.array([-1.0, 0.0], dtype=float),
                 np.array([0.0, -1.0], dtype=float)),
            ),
            (
                np.array([x_low, y_high, top_center_z], dtype=float),
                (np.array([-1.0, 0.0], dtype=float),
                 np.array([0.0, 1.0], dtype=float)),
            ),
            (
                np.array([x_high, y_low, top_center_z], dtype=float),
                (np.array([1.0, 0.0], dtype=float),
                 np.array([0.0, -1.0], dtype=float)),
            ),
            (
                np.array([x_high, y_high, top_center_z], dtype=float),
                (np.array([1.0, 0.0], dtype=float),
                 np.array([0.0, 1.0], dtype=float)),
            ),
        ]
        random.shuffle(positions)
        return positions

    def _closest_walkable_point(self, target_pos):
        nav = self._get_navigation_interface()
        if nav is None:
            return None

        try:
            navmesh = nav.get_navmesh()
            if navmesh is None:
                return None
            target_ground = carb.Float3(
                float(target_pos[0]), float(target_pos[1]), 0.0)
            closest, _ = navmesh.query_closest_point(target_ground)
            if closest is None:
                return None
            return np.array([closest.x, closest.y, 0.0], dtype=float)
        except Exception as exc:
            carb.log_warn(
                f"[Placement] Could not query closest walkable point for {target_pos.tolist()}: {exc}")
            return None

    def _placement_path_point_count(self, start_pos, goal_pos):
        nav = self._get_navigation_interface()
        if nav is None:
            return 2

        try:
            navmesh = nav.get_navmesh()
            if navmesh is None:
                return 2
            path = navmesh.query_shortest_path(
                start_pos=carb.Float3(
                    float(start_pos[0]), float(start_pos[1]), float(start_pos[2])),
                end_pos=carb.Float3(
                    float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2])),
            )
            if path is None or path.get_point_count() <= 0:
                return None
            return path.get_point_count()
        except Exception as exc:
            carb.log_warn(
                f"[Placement] Could not query pick-to-place path point count: {exc}")
            return None

    def _navmesh_path_distance(self, start_pos, goal_pos):
        nav = self._get_navigation_interface()
        if nav is None:
            return None

        try:
            navmesh = nav.get_navmesh()
            if navmesh is None:
                return None
            start = np.asarray(start_pos, dtype=float)
            goal = np.asarray(goal_pos, dtype=float)
            path = navmesh.query_shortest_path(
                start_pos=carb.Float3(
                    float(start[0]), float(start[1]), float(start[2])),
                end_pos=carb.Float3(
                    float(goal[0]), float(goal[1]), float(goal[2])),
            )
            if path is None or path.get_point_count() < 2:
                return None
            points = [np.array([p.x, p.y, p.z], dtype=float)
                      for p in path.get_points()]
            return sum(
                np.linalg.norm(points[index] - points[index - 1])
                for index in range(1, len(points))
            )
        except Exception as exc:
            carb.log_warn(
                f"[Placement] Could not query robot-start navmesh path distance: {exc}")
            return None

    def _resolve_robot_initial_pose_near_cube(self):
        best_position = None
        best_score = None
        best_path_distance_to_cube = None
        best_xy_distance_to_cube = None
        best_projection_error = None
        best_clearance_error = None
        best_uncleared = None
        cube_xy = self._cube_position[:2]
        attempted_samples = 0
        rejected_for_clearance = 0
        direct_fallback = self._direct_robot_start_near_cube()
        pick_goal = self._pick_goal_pos
        if pick_goal is None:
            pick_goal = self._resolve_pick_goal_pos()

        for index in range(ROBOT_START_RING_SAMPLES):
            angle = 2.0 * np.pi * index / ROBOT_START_RING_SAMPLES
            for distance in (
                ROBOT_START_MIN_DISTANCE_FROM_CUBE,
                ROBOT_START_TARGET_DISTANCE_FROM_CUBE,
                ROBOT_START_MAX_DISTANCE_FROM_CUBE,
            ):
                sample = np.array([
                    self._cube_position[0] + distance * np.cos(angle),
                    self._cube_position[1] + distance * np.sin(angle),
                    0.0,
                ], dtype=float)
                walkable = self._closest_walkable_point(sample)
                if walkable is None:
                    continue

                attempted_samples += 1
                path_distance_to_cube = self._navmesh_path_distance(
                    walkable, pick_goal)
                if path_distance_to_cube is None:
                    continue
                if not (ROBOT_START_MIN_DISTANCE_FROM_CUBE <= path_distance_to_cube <= ROBOT_START_MAX_DISTANCE_FROM_CUBE):
                    continue

                distance_error = abs(
                    path_distance_to_cube - ROBOT_START_TARGET_DISTANCE_FROM_CUBE)
                projection_error = np.linalg.norm(walkable[:2] - sample[:2])
                score = distance_error + 0.25 * projection_error
                candidate = (
                    score,
                    walkable,
                    path_distance_to_cube,
                    np.linalg.norm(walkable[:2] - cube_xy),
                    projection_error,
                )
                if best_uncleared is None or score < best_uncleared[0]:
                    best_uncleared = candidate

                clearance_ok, clearance_error = self._robot_start_has_clearance(
                    walkable)
                if not clearance_ok:
                    rejected_for_clearance += 1
                    continue

                score += 0.5 * clearance_error
                if best_score is None or score < best_score:
                    best_score = score
                    best_position = walkable
                    best_path_distance_to_cube = path_distance_to_cube
                    best_xy_distance_to_cube = np.linalg.norm(
                        walkable[:2] - cube_xy)
                    best_projection_error = projection_error
                    best_clearance_error = clearance_error

        if best_position is None:
            fallback_walkable = self._closest_walkable_point(direct_fallback)
            fallback_ok = False
            fallback_clearance_error = None
            if fallback_walkable is not None:
                fallback_path_distance = self._navmesh_path_distance(
                    fallback_walkable, pick_goal)
                fallback_ok, fallback_clearance_error = self._robot_start_has_clearance(
                    fallback_walkable)
            else:
                fallback_path_distance = None

            if fallback_ok:
                carb.log_warn(
                    f"[Placement] No ring-sampled robot start passed clearance after {attempted_samples} walkable sample(s); "
                    f"using projected direct annulus start {fallback_walkable.tolist()}.")
                best_position = fallback_walkable
                best_path_distance_to_cube = fallback_path_distance
                best_xy_distance_to_cube = np.linalg.norm(
                    best_position[:2] - cube_xy)
                best_projection_error = np.linalg.norm(
                    best_position[:2] - direct_fallback[:2])
                best_clearance_error = fallback_clearance_error
            elif best_uncleared is not None:
                _, best_position, best_path_distance_to_cube, best_xy_distance_to_cube, best_projection_error = best_uncleared
                best_clearance_error = None
                carb.log_warn(
                    f"[Placement] No robot start passed footprint clearance after {attempted_samples} walkable sample(s); "
                    "using best path-valid start anyway. It may be close to obstacles.")
            else:
                carb.log_warn(
                    f"[Placement] No robot start with navmesh path distance {ROBOT_START_MIN_DISTANCE_FROM_CUBE:.1f}-"
                    f"{ROBOT_START_MAX_DISTANCE_FROM_CUBE:.1f} m from cube after {attempted_samples} walkable sample(s); "
                    f"using direct annulus start {direct_fallback.tolist()}.")
                best_position = direct_fallback
                best_path_distance_to_cube = self._navmesh_path_distance(
                    best_position, pick_goal)
                best_xy_distance_to_cube = np.linalg.norm(
                    best_position[:2] - cube_xy)
                best_projection_error = 0.0
                best_clearance_error = None

        self._robot_initial_position = best_position
        target_yaw = np.arctan2(
            self._cube_position[1] - best_position[1],
            self._cube_position[0] - best_position[0],
        )
        self._robot_initial_yaw = self._wrap_angle(
            target_yaw - ROBOT_SIDE_YAW_OFFSETS["+Y"])
        carb.log_warn(
            f"[Placement] Robot start set to {self._robot_initial_position.tolist()} "
            f"(nav path distance {self._format_distance(best_path_distance_to_cube)}, "
            f"XY distance {best_xy_distance_to_cube:.2f} m, "
            f"projection error {best_projection_error:.2f} m, {attempted_samples} walkable sample(s)), "
            f"clearance error {self._format_distance(best_clearance_error)}, "
            f"clearance rejects {rejected_for_clearance}, "
            f"yaw {self._robot_initial_yaw:.3f}.")

    def _robot_start_has_clearance(self, position):
        center = np.asarray(position, dtype=float)
        max_projection_error = 0.0
        for index in range(ROBOT_START_CLEARANCE_SAMPLES):
            angle = 2.0 * np.pi * index / ROBOT_START_CLEARANCE_SAMPLES
            probe = center + np.array([
                ROBOT_START_CLEARANCE_RADIUS * np.cos(angle),
                ROBOT_START_CLEARANCE_RADIUS * np.sin(angle),
                0.0,
            ], dtype=float)
            walkable = self._closest_walkable_point(probe)
            if walkable is None:
                return False, float("inf")
            projection_error = np.linalg.norm(walkable[:2] - probe[:2])
            max_projection_error = max(max_projection_error, projection_error)
            if projection_error > ROBOT_START_CLEARANCE_MAX_PROJECTION:
                return False, max_projection_error
        return True, max_projection_error

    def _format_distance(self, distance):
        if distance is None:
            return "unavailable"
        return f"{distance:.2f} m"

    def _direct_robot_start_near_cube(self):
        distance = random.uniform(
            ROBOT_START_MIN_DISTANCE_FROM_CUBE,
            ROBOT_START_MAX_DISTANCE_FROM_CUBE,
        )
        angle = random.uniform(-np.pi, np.pi)
        return np.array([
            self._cube_position[0] + distance * np.cos(angle),
            self._cube_position[1] + distance * np.sin(angle),
            0.0,
        ], dtype=float)

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
                if ENABLE_FACE_TARGET_STATES:
                    self._state = PickPlaceState.FACE_CUBE
                else:
                    self._settle_count = 0
                    self._state = PickPlaceState.SETTLE

        elif self._state == PickPlaceState.FACE_CUBE:
            if self._face_target_step(self._get_current_cube_position(), "pick"):
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
                if ENABLE_FACE_TARGET_STATES:
                    self._state = PickPlaceState.FACE_PLACE
                else:
                    self._prepare_place_sequence()
                    self._state = PickPlaceState.PLACE_RELEASE

        elif self._state == PickPlaceState.FACE_PLACE:
            self._close_gripper()
            if self._face_target_step(self._place_cube_pos, "place"):
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
        return self._resolve_walkable_goal_pos(self._cube_position, "pick")

    def _resolve_place_goal_pos(self):
        return self._resolve_walkable_goal_pos(self._place_cube_pos, "place")

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
    def _get_current_cube_position(self, log_fail=False):
        if self._cube is None:
            return self._cube_position.copy()

        try:
            position, _ = self._cube.get_world_pose()
            position = np.asarray(position, dtype=float)
            self._cube_position = position.copy()
            return position
        except Exception as exc:
            if log_fail:
                carb.log_warn(
                    f"[Cube] Could not read current cube pose; using last known position {self._cube_position.tolist()}. Details: {exc}")
            return self._cube_position.copy()

    def _prepare_grasp_sequence(self):
        cube_pos = self._get_current_cube_position(log_fail=True)
        carb.log_info(
            f"[Cube] Preparing grasp using current cube position {cube_pos.tolist()}.")

        pre_grasp = cube_pos + np.array([0.0, 0.0, self.pre_grasp_height])
        grasp = cube_pos + np.array([0.0, 0.0, self.grasp_height_offset])

        current_joints = None
        if self._arm_subset is not None:
            try:
                current_joints = self._arm_subset.get_joint_positions()
            except Exception as exc:
                carb.log_warn(
                    f"[Arm] Could not read current arm joints before grasp: {exc}")

        self._arm_segments = []
        if current_joints is not None:
            self._arm_segments.append(
                ArmSegment(
                    None,
                    self.gripper_open,
                    settle_steps=self.fetch_open_settle_steps,
                    target_joints=current_joints,
                    label="open_gripper_before_fetch",
                    trajectory_steps=1,
                )
            )

        self._arm_segments.extend([
            ArmSegment(
                pre_grasp,
                self.gripper_open,
                settle_steps=self.fetch_pre_grasp_settle_steps,
                ee_tolerance=0.04,
                label="move_to_pre_grasp_above_cube",
                trajectory_steps=self.fetch_pre_grasp_move_steps,
            ),
            ArmSegment(
                grasp,
                self.gripper_open,
                settle_steps=self.fetch_descent_settle_steps,
                ee_tolerance=0.035,
                label="descend_to_grasp_cube",
                trajectory_steps=self.fetch_descent_move_steps,
            ),
            ArmSegment(
                grasp,
                self.gripper_closed,
                settle_steps=self.fetch_close_settle_steps,
                ee_tolerance=0.035,
                label="close_gripper_on_cube",
                trajectory_steps=1,
            ),
            ArmSegment(
                pre_grasp,
                self.gripper_closed,
                settle_steps=self.fetch_lift_settle_steps,
                ee_tolerance=0.04,
                label="lift_object_to_pre_grasp",
                trajectory_steps=self.fetch_lift_move_steps,
            ),
        ])
        self._active_segment = None

    def _prepare_place_sequence(self):
        place_above = self._place_cube_pos + \
            np.array([0.0, 0.0, self.pre_grasp_height])
        place = self._place_cube_pos + np.array([0.0, 0.0, 0.045])
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

        if segment.target_pos is not None and not self._ee_position_reached(segment.target_pos, segment.ee_tolerance):
            if segment.trajectory:
                self._arm_subset.apply_action(
                    joint_positions=segment.trajectory[-1])
            segment.reach_waited += 1
            if segment.reach_waited == 1 or segment.reach_waited % 60 == 0:
                ee_pos, _ = self._ee_prim.get_world_pose()
                error = np.linalg.norm(
                    np.asarray(ee_pos, dtype=float) - segment.target_pos)
                # carb.log_warn(
                #     f"[Arm] Waiting for {segment.label} to reach target; error {error:.3f} m.")
            if segment.reach_waited < segment.reach_timeout_steps:
                return False
            # carb.log_warn(
            #     f"[Arm] Continuing after timeout waiting for {segment.label}; target may not be reached.")

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
        trajectory_steps = (
            self.arm_trajectory_steps
            if segment.trajectory_steps is None
            else segment.trajectory_steps
        )
        alphas = np.linspace(0.0, 1.0, trajectory_steps)
        smooth = 3.0 * alphas**2 - 2.0 * alphas**3
        segment.trajectory = [current_joints + alpha *
                              (target_joints - current_joints) for alpha in smooth]
        segment.index = 0
        segment.waited = 0
        segment.reach_waited = 0

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
