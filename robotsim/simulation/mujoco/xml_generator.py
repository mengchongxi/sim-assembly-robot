"""MuJoCo XML 模型生成器模块。

从 2D tile 构型或 YAML 文件生成 MuJoCo XML 机器人模型。
包含体素图、运动学树构建和 MjSpec 生成的完整流水线。
"""
import collections
import math
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from robotsim.core.types import TileType

# ==============================================================================
# 全局配置
# ==============================================================================
GRID_SCALE = 0.12
BOX_HALF = 0.06
TYPE_CONFIG = {
    "joint": {"name_tag": "joint", "rgba": (0.6, 0.2, 0.8, 1.0)},
    "basic": {"name_tag": "basic", "rgba": (1.0, 0.0, 0.0, 1.0)},
    "wheel": {"name_tag": "wheel", "rgba": (0.2, 0.8, 0.2, 1.0)},
}
DEFAULT_TYPE = {"name_tag": "node", "rgba": (0.5, 0.5, 0.5, 1.0)}
MODEL_NAME = "generation"

# TileType → 内部类型字符串
_TILE_TYPE_MAP = {
    TileType.BASE: "basic",
    TileType.JOINT: "joint",
    TileType.WHEEL: "wheel",
}


# ==============================================================================
# 1) 数据层：VoxelGraph
# ==============================================================================
class VoxelGraph:
    """解析体素块数据，提供邻接查询与 purple→red 目标解析。

    支持两种输入模式：
    - YAML dict 输入（兼容 sim_mujoco.py 原始格式）
    - 2D tile dict 输入（来自 planner GUI）
    """

    def __init__(self, data: Dict, root_coord: Tuple = None, mode: str = "yaml"):
        self.grid: Dict[Tuple, Dict] = {}
        self.root_coord = root_coord
        if mode == "yaml":
            self._parse_yaml(data)
            if self.root_coord is None:
                self.root_coord = (0, 0, 1)
        elif mode == "tiles":
            self._parse_tiles(data)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _parse_yaml(self, yaml_data: Dict) -> None:
        """加载 blocks 到坐标索引表 self.grid。"""
        if "blocks" not in yaml_data:
            raise ValueError("YAML file missing 'blocks' key")
        for block in yaml_data["blocks"]:
            origin = block["origin"]
            pos = (origin["gx"], origin["gy"], origin["gz"])
            self.grid[pos] = block

    def _parse_tiles(self, tiles: Dict[TileType, set]) -> None:
        """从 planner 的 2D tile dict 构建 3D 体素图（z=0 平面）。"""
        for tile_type, positions in tiles.items():
            if tile_type not in _TILE_TYPE_MAP:
                continue
            internal_type = _TILE_TYPE_MAP[tile_type]
            color = "purple" if tile_type == TileType.JOINT else "red"
            for (x, y) in positions:
                coord = (x, y, 0)
                self.grid[coord] = {
                    "color": color,
                    "type": internal_type,
                    "origin": {"gx": x, "gy": y, "gz": 0},
                }
                # JOINT tiles 自动生成指向相邻 BASE 的 vector
                if tile_type == TileType.JOINT:
                    vector = self._find_adjacent_direction(
                        (x, y), tiles.get(TileType.BASE, set())
                    )
                    if vector:
                        self.grid[coord]["vector"] = vector

        # 自动选取 root：优先选第一个 BASE tile
        if self.root_coord is None:
            base_positions = tiles.get(TileType.BASE, set())
            if base_positions:
                bx, by = min(base_positions)
                self.root_coord = (bx, by, 0)
            elif self.grid:
                self.root_coord = next(iter(self.grid))

    @staticmethod
    def _find_adjacent_direction(
        pos: Tuple[int, int], base_set: set
    ) -> Optional[List[int]]:
        """找到 pos 相邻的某个 BASE tile 方向向量（4邻接）。"""
        x, y = pos
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (x + dx, y + dy) in base_set:
                return [dx, dy, 0]
        return None

    def get_neighbors(self, coord: Tuple) -> List[Tuple]:
        """返回 6 邻接且存在于 grid 的坐标列表。"""
        x, y, z = coord
        directions = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1),
        ]
        return [p for p in directions if p in self.grid]

    def get_purple_target(self, purple_coord: Tuple) -> Optional[Tuple]:
        """若 purple 块含 vector 且指向 red 块，返回该 red 坐标。"""
        block = self.grid[purple_coord]
        vector = block.get("vector")
        if not vector:
            return None
        target = (
            purple_coord[0] + vector[0],
            purple_coord[1] + vector[1],
            purple_coord[2] + vector[2],
        )
        if target in self.grid and self.grid[target].get("color") == "red":
            return target
        return None


# ==============================================================================
# 2) 逻辑层：KinematicTreeBuilder
# ==============================================================================
class KinematicTreeBuilder:
    """将无向体素邻接图转换为有向运动学树（root 为起点）。"""

    def __init__(self, graph: VoxelGraph):
        self.graph = graph

    def build(self) -> Dict:
        """构建递归 dict：{type,pos,children,(vector)}。"""
        structure = self._build_kinematic_tree_structure()
        return self._convert_to_recursive_dict(self.graph.root_coord, structure)

    def _build_kinematic_tree_structure(self) -> Dict:
        """BFS 构树：同层父节点按"未访问邻居数"优先扩展；purple 子节点优先。"""
        root = self.graph.root_coord
        if root not in self.graph.grid:
            raise ValueError(f"Root node {root} not found in data")

        tree_structure = {root: {"children": []}}
        visited = {root}
        queue = collections.deque([root])

        while queue:
            layer_size = len(queue)
            current_layer_parents = [queue.popleft() for _ in range(layer_size)]

            parents_with_priority = []
            for p_coord in current_layer_parents:
                neighbors = self.graph.get_neighbors(p_coord)
                unvisited_neighbors = [n for n in neighbors if n not in visited]
                parents_with_priority.append((len(unvisited_neighbors), p_coord))
            parents_with_priority.sort(key=lambda x: x[0], reverse=True)

            for _, parent_coord in parents_with_priority:
                neighbors = self.graph.get_neighbors(parent_coord)
                purple_candidates, red_candidates = [], []

                for n in neighbors:
                    if n in visited:
                        continue
                    if self.graph.grid[n].get("color") == "purple":
                        purple_candidates.append(n)
                    else:
                        red_candidates.append(n)

                for purple_child in purple_candidates:
                    if purple_child in visited:
                        continue
                    visited.add(purple_child)
                    tree_structure[parent_coord]["children"].append(purple_child)
                    tree_structure[purple_child] = {"children": []}

                    red_grandchild = self.graph.get_purple_target(purple_child)
                    if red_grandchild and red_grandchild not in visited:
                        visited.add(red_grandchild)
                        tree_structure[purple_child]["children"].append(red_grandchild)
                        tree_structure[red_grandchild] = {"children": []}
                        queue.append(red_grandchild)

                for red_child in red_candidates:
                    if red_child in visited:
                        continue
                    visited.add(red_child)
                    tree_structure[parent_coord]["children"].append(red_child)
                    tree_structure[red_child] = {"children": []}
                    queue.append(red_child)

        return tree_structure

    def _convert_to_recursive_dict(self, coord: Tuple, tree_structure: Dict) -> Dict:
        """把坐标树转换为递归节点字典，补齐 type/pos/children/(vector)。"""
        node_data = self.graph.grid[coord]
        color_to_type = {"red": "basic", "purple": "joint"}

        node = {
            "type": color_to_type.get(node_data.get("color"), node_data.get("color")),
            "pos": list(coord),
            "children": [],
        }
        if "vector" in node_data:
            node["vector"] = node_data["vector"]

        for child_coord in tree_structure[coord]["children"]:
            node["children"].append(
                self._convert_to_recursive_dict(child_coord, tree_structure)
            )
        return node


# ==============================================================================
# 3) 输出层：MujocoSpecBuilder
# ==============================================================================
class MujocoSpecBuilder:
    """将递归 dict 构建为 MuJoCo MjSpec，并补充 actuator 与 keyframe。"""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        grid_scale: float = GRID_SCALE,
        box_half: float = BOX_HALF,
        type_config: Dict = None,
        default_type: Dict = None,
    ):
        self.model_name = model_name
        self.grid_scale = float(grid_scale)
        self.box_half = float(box_half)
        self.type_config = type_config or TYPE_CONFIG
        self.default_type = default_type or DEFAULT_TYPE
        self._idx_counter = count()
        self.spec: Optional[mujoco.MjSpec] = None

    def build_spec(self, data: Dict) -> mujoco.MjSpec:
        """从树数据生成 spec：body/geom/joint + actuator + home keyframe。"""
        spec = mujoco.MjSpec()
        self.spec = spec

        spec.modelname = self.model_name
        spec.compiler.degree = False

        self._apply_global_options(spec)

        self._idx_counter = count()
        self._build_body(
            node=data,
            parent_body=spec.worldbody,
            parent_world_pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            override_pos=None,
            override_quat=None,
        )

        self._add_actuators_for_hinge_joints(ctrlrange=(-100.0, 100.0), ctrllimited=True)
        self._ensure_home_keyframe(name="home", time=0.0)
        return spec

    def _apply_global_options(self, spec: mujoco.MjSpec) -> None:
        """设置 timestep/gravity 以及默认 geom/joint 参数。"""
        spec.option.timestep = 0.02
        spec.option.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        spec.default.geom.density = 300
        spec.default.geom.friction = np.array([0.6, 0.1, 0.1], dtype=np.float64)
        spec.default.geom.solref = np.array([0.02, 1.0], dtype=np.float64)
        spec.default.geom.solimp = np.array([0.9, 0.95, 0.001, 0.5, 2.0], dtype=np.float64)
        spec.default.joint.damping = 0.1
        spec.default.joint.armature = 0.01
        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.05]
        )

    def _add_actuators_for_hinge_joints(
        self,
        ctrlrange: Tuple[float, float] = (-1.0, 1.0),
        ctrllimited: bool = True,
        name_prefix: str = "act_",
    ) -> None:
        """为所有具名 hinge joint 自动添加 motor actuator。"""
        for j in self.spec.joints:
            if int(j.type) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            jname = getattr(j, "name", "")
            if not jname:
                continue

            act_name = f"{name_prefix}{jname}"
            if self.spec.actuator(act_name) is not None:
                continue

            self.spec.add_actuator(
                name=act_name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=jname,
                ctrllimited=bool(ctrllimited),
                ctrlrange=np.array([ctrlrange[0], ctrlrange[1]], dtype=np.float64),
                gainprm=np.array([1.0] + [0.0] * 9, dtype=np.float64),
                biasprm=np.zeros((10,), dtype=np.float64),
            )

    def _ensure_home_keyframe(self, name: str = "home", time: float = 0.0) -> None:
        """添加 home keyframe（qpos 全 0；若有 freejoint 则保证单位四元数）。"""
        for k in getattr(self.spec, "keys", []):
            if getattr(k, "name", "") == name:
                return

        model_tmp = self.spec.compile()
        nq = int(model_tmp.nq)
        qpos = np.zeros((nq,), dtype=np.float64)
        if nq >= 7:
            qpos[3] = 1.0

        self.spec.add_key(name=name, time=float(time), qpos=qpos)

    def _build_body(
        self,
        node: Dict,
        parent_body: "mujoco.MjsBody",
        parent_world_pos: np.ndarray,
        override_pos: Optional[Iterable[float]] = None,
        override_quat: Optional[Iterable[float]] = None,
    ) -> "mujoco.MjsBody":
        """递归生成 body/geom；对 joint+vector 节点插入 disk 子体与 hinge。"""
        node_type = node.get("type", "basic")
        config = self.type_config.get(node_type, self.default_type)
        world_pos_grid = np.array(node.get("pos", [0, 0, 0]), dtype=np.float64)

        seq = next(self._idx_counter)
        name_tag = config["name_tag"]
        body_name = f"body_{name_tag}_{seq}"
        geom_name = f"geom_{name_tag}_{seq}"

        if override_pos is not None:
            rel_pos = np.array(list(override_pos), dtype=np.float64)
        else:
            rel_pos = (world_pos_grid - parent_world_pos) * self.grid_scale

        body_kwargs = {"name": body_name, "pos": rel_pos}
        if override_quat is not None:
            body_kwargs["quat"] = np.array(list(override_quat), dtype=np.float64)

        body = parent_body.add_body(**body_kwargs)

        if parent_body is self.spec.worldbody and seq == 0:
            body.add_freejoint(name="root_freejoint")

        children = node.get("children", [])
        is_leaf = len(children) == 0

        rgba = config.get("rgba", self.default_type["rgba"])
        if is_leaf:
            rgba = (0.0, 0.0, 0.0, 1.0)

        # WHEEL 类型使用 cylinder geom
        if node_type == "wheel":
            body.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=np.array([self.box_half, self.box_half * 0.5, 0.0], dtype=np.float64),
                rgba=np.array(list(rgba), dtype=np.float64),
                contype=0,
                conaffinity=1 if is_leaf else 0,
            )
            # wheel 添加绕 x 轴的 hinge
            body.add_joint(
                name=f"wheel_hinge_{seq}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                axis=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                limited=False,
            )
        else:
            body.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([self.box_half, self.box_half, self.box_half], dtype=np.float64),
                rgba=np.array(list(rgba), dtype=np.float64),
                contype=0,
                conaffinity=1 if is_leaf else 0,
            )

        children = node.get("children", [])
        special_child_node = None

        if node_type == "joint" and "vector" in node:
            vector = np.array(node["vector"], dtype=np.float64)
            target_pos = world_pos_grid + vector
            for child in children:
                child_pos = np.array(child.get("pos", [0, 0, 0]), dtype=np.float64)
                if np.array_equal(child_pos, target_pos):
                    special_child_node = child
                    break

        for child in children:
            if child is special_child_node:
                disk_pos_arr, disk_quat_arr = self._get_disk_transform(
                    node["vector"], box_half=self.box_half
                )
                disk_body = body.add_body(
                    name=f"joint_disk_{seq}",
                    pos=disk_pos_arr,
                    quat=disk_quat_arr,
                )

                disk_body.add_joint(
                    name=f"disk_hinge_{seq}",
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    limited=True,
                    range=np.array([-math.pi, math.pi], dtype=np.float64),
                )

                disk_body.add_geom(
                    name=f"disk_geom_{seq}",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                    size=np.array([0.06, 0.003, 0.0], dtype=np.float64),
                    rgba=np.array([0.2, 0.6, 0.9, 1.0], dtype=np.float64),
                    contype=0,
                    conaffinity=0,
                )

                child_target_pos = np.array([0.0, 0.0, 0.06], dtype=np.float64)
                w, x, y, z = disk_quat_arr
                inv_quat_arr = np.array([w, -x, -y, -z], dtype=np.float64)

                self._build_body(
                    node=child,
                    parent_body=disk_body,
                    parent_world_pos=world_pos_grid,
                    override_pos=child_target_pos,
                    override_quat=inv_quat_arr,
                )
            else:
                self._build_body(
                    node=child,
                    parent_body=body,
                    parent_world_pos=world_pos_grid,
                )

        return body

    @staticmethod
    def _get_disk_transform(
        direction_vec: List[float], box_half: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """由方向向量计算 disk 的 (pos, quat[wxyz])。"""
        vec_target = np.array(direction_vec, dtype=np.float64)
        norm = np.linalg.norm(vec_target)
        vec_target = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64) if norm == 0 else vec_target / norm
        )

        disk_pos = vec_target * float(box_half)

        default_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.allclose(vec_target, default_z):
            q_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        elif np.allclose(vec_target, -default_z):
            q_xyzw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            rot_axis = np.cross(default_z, vec_target)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            angle = float(
                np.arccos(np.clip(float(np.dot(default_z, vec_target)), -1.0, 1.0))
            )
            q_xyzw = R.from_rotvec(rot_axis * angle).as_quat()

        disk_quat_wxyz = np.array(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
        )
        return disk_pos, disk_quat_wxyz


# ==============================================================================
# 4) 高层 API：MujocoXmlGenerator
# ==============================================================================
class MujocoXmlGenerator:
    """从 2D tile 构型或 YAML 文件生成 MuJoCo XML 模型文件。"""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        grid_scale: float = GRID_SCALE,
        box_half: float = BOX_HALF,
    ):
        self.model_name = model_name
        self.builder = MujocoSpecBuilder(
            model_name=model_name,
            grid_scale=grid_scale,
            box_half=box_half,
        )

    def from_2d_config(
        self,
        tiles: Dict[TileType, set],
        output_path: Path,
    ) -> Path:
        """将 planner 的 2D 目标构型转换为 MuJoCo XML。

        Args:
            tiles: {TileType.BASE: {(x,y),...}, TileType.JOINT: {...}, ...}
            output_path: 输出目录路径

        Returns:
            生成的 XML 文件路径
        """
        graph = VoxelGraph(tiles, mode="tiles")
        tree = KinematicTreeBuilder(graph).build()
        spec = self.builder.build_spec(tree)

        xml_str = spec.to_xml()
        xml_file = output_path / "robot.xml"
        xml_file.parent.mkdir(parents=True, exist_ok=True)
        with xml_file.open("w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"[xml_generator] Exported robot XML to {xml_file}")
        return xml_file

    def from_yaml(self, yaml_path: Path, output_path: Path) -> Path:
        """从 YAML 文件生成 MuJoCo XML（兼容原 sim_mujoco.py 工作流）。

        Args:
            yaml_path: 输入 YAML 文件路径
            output_path: 输出目录路径

        Returns:
            生成的 XML 文件路径
        """
        with yaml_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        graph = VoxelGraph(yaml_data, mode="yaml")
        tree = KinematicTreeBuilder(graph).build()
        spec = self.builder.build_spec(tree)

        xml_str = spec.to_xml()
        xml_file = output_path / "robot.xml"
        xml_file.parent.mkdir(parents=True, exist_ok=True)
        with xml_file.open("w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"[xml_generator] Exported robot XML to {xml_file}")
        return xml_file
