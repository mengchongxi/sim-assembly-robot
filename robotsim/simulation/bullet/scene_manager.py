"""机器人仿真场景管理模块（PyBullet）。

提供 SceneManager 类，负责均方块的加载、删除、查询，
以及从 YAML 轨迹文件解析初始花片布局和动作序列。
"""
import pybullet as p
import numpy as np
import yaml

from robotsim.utils.grid_utils import GRID_CELL_SIZE, grid_to_world_pose


class SceneManager:
    """管理 PyBullet 环境中所有方块对象的生命周期管理器。

    包括将方块按类型着色、按位置或 ID 查询/删除，
    以及从 YAML 文件解析并初始化完整场景。
    """
    def __init__(self):
        self.loaded_cubes = {}
        self.cube_colors = {
            "base": [1.0, 0.0, 0.0, 1.0],
            "joint": [0.0, 0.0, 1.0, 1.0],
            "wheel": [0.0, 1.0, 0.0, 1.0],
        }

    def load_cube(self, position, orientation_quat=None, color=None, cube_type="base"):
        """加载一个方块 URDF，按类型着色并注册到已加载字典。

        Args:
            position: 世界坐标 [x, y, z]。
            orientation_quat: 季装数复数 [x, y, z, w]，默认为单位季装数。
            color: RGBA 颜色列表，默认从 cube_colors 映射。
            cube_type: 类型字符串 'base'、'joint' 或 'wheel'。

        Returns:
            PyBullet 对象 ID。
        """
        urdf_path = "assembly_bullet/robot_5dof/base_cube.urdf"
        if orientation_quat is None:
            orientation_quat = [0, 0, 0, 1]
        if color is None:
            color = self.cube_colors.get(cube_type.lower(), [0.5, 0.5, 0.5, 1.0])

        cube_id = p.loadURDF(urdf_path, position, orientation_quat, useFixedBase=True)
        p.changeVisualShape(cube_id, -1, rgbaColor=color)
        self.loaded_cubes[cube_id] = cube_type
        print(f"[场景管理器] 创建了 {cube_type} 类型方块，ID: {cube_id}，位置: {position}")
        return cube_id

    def remove_cube_by_id(self, cube_id):
        """从场景中移除指定 ID 的方块，返回其类型或 None。"""
        if cube_id not in self.loaded_cubes:
            print(f"[场景管理器] 警告：试图删除不存在的方块 ID: {cube_id}")
            return None

        cube_type = self.loaded_cubes[cube_id]
        p.removeBody(cube_id)
        del self.loaded_cubes[cube_id]
        print(f"[场景管理器] 删除了 {cube_type} 类型方块，ID: {cube_id}")
        return cube_type

    def remove_cube_at_position(self, position, tolerance=0.01):
        """删除世界坐标在允许误差范围内匹配到的方块。

        Returns:
            第一个删除方块的类型字符串，未找到时返回 None。
        """
        target_x, target_y, target_z = position
        target_z = 0.12

        cubes_to_remove = []
        for cube_id, cube_type in self.loaded_cubes.items():
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            cube_x, cube_y, cube_z = cube_pos

            if (abs(cube_x - target_x) < tolerance and
                abs(cube_y - target_y) < tolerance and
                abs(cube_z - target_z) < tolerance):
                cubes_to_remove.append((cube_id, cube_type))

        removed_types = []
        for cube_id, cube_type in cubes_to_remove:
            self.remove_cube_by_id(cube_id)
            removed_types.append(cube_type)

        if not cubes_to_remove:
            print(f"[场景管理器] 在位置 {position} 没找到方块")
            return None

        return removed_types[0] if removed_types else None

    def clear_scene(self):
        """删除场景中所有方块。"""
        print("[场景管理器] 正在清空所有方块...")
        cube_ids = list(self.loaded_cubes.keys())
        for cube_id in cube_ids:
            self.remove_cube_by_id(cube_id)
        print(f"[场景管理器] 场景已清空，共删除了 {len(cube_ids)} 个方块")

    def get_cube_type_by_id(self, cube_id):
        """返回指定 ID 方块的类型字符串，不存在则返回 None。"""
        return self.loaded_cubes.get(cube_id, None)

    def get_all_cubes_info(self):
        """返回包含所有方块 ID、类型、位置和姿态的字典列表。"""
        cubes_info = []
        for cube_id, cube_type in self.loaded_cubes.items():
            pos, ori = p.getBasePositionAndOrientation(cube_id)
            cubes_info.append({
                'id': cube_id,
                'type': cube_type,
                'position': pos,
                'orientation': ori
            })
        return cubes_info

    def get_cubes_count(self):
        """返回现在场景中已加载的方块总数。"""
        return len(self.loaded_cubes)

    def get_cubes_by_type(self, cube_type):
        """返回指定类型的所有方块 ID 列表。"""
        return [cube_id for cube_id, c_type in self.loaded_cubes.items() if c_type == cube_type]

    def load_scene_from_yaml(self, yaml_file_path):
        """从指定 YAML 文件完整解析并构建仿真场景。

        该方法外部接口，先清空当前场景，再依次解析机器人初始状态、
        初始瓦片布局和动作序列。

        Returns:
            包含 'robot_position'、'robot_orientation' 和 'action_sequence' 的字典，
            加载失败时返回 None。
        """
        try:
            with open(yaml_file_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)

            print("[场景管理器] 开始从YAML文件加载场景...")
            self.clear_scene()
            robot_info = self._parse_robot_initial_state(data)
            self._load_initial_tiles(data)
            action_sequence = self._parse_action_sequence(data)
            print("[场景管理器] YAML场景加载完成！")

            return {
                'robot_position': robot_info['position'],
                'robot_orientation': robot_info['orientation'],
                'action_sequence': action_sequence
            }

        except FileNotFoundError:
            print(f"[场景管理器] 错误：找不到文件 {yaml_file_path}")
            return None
        except yaml.YAMLError as e:
            print(f"[场景管理器] YAML解析错误: {e}")
            return None
        except Exception as e:
            print(f"[场景管理器] 加载场景时出错: {e}")
            return None

    def _parse_robot_initial_state(self, data):
        if 'initial_position' not in data:
            print("[场景管理器] 警告：YAML文件中没有机器人初始位置信息")
            return {'position': [0, 0, 0.025], 'orientation': [0, 0, 0, 1]}

        front_foot = data['initial_position']['front_foot']
        back_foot = data['initial_position']['back_foot']
        robot_pos = np.array(back_foot) * 0.12
        direction_vector = np.array(front_foot) - np.array(back_foot)

        orientation_map = {
            (1, 0): [0, 0, 0, 1],
            (0, 1): [0, 0, 0.707, 0.707],
            (0, -1): [0, 0, -0.707, 0.707],
            (-1, 0): [0, 0, 1, 0]
        }

        direction_tuple = tuple(direction_vector)
        robot_ori = orientation_map.get(direction_tuple, [0, 0, 0, 1])

        return {
            'position': [robot_pos[0], robot_pos[1], 0.15],
            'orientation': robot_ori
        }

    def _load_initial_tiles(self, data):
        if 'initial_tiles' not in data:
            print("[场景管理器] 警告：YAML文件中没有初始方块布局")
            return

        for tile in data['initial_tiles']:
            tile_type = tile['type']
            grid_position = tile['position']
            world_x = round(grid_position[0] * 0.12, 2)
            world_y = round(grid_position[1] * 0.12, 2)
            world_z = 0.0
            world_position = [world_x, world_y, world_z]
            self.load_cube(world_position, cube_type=tile_type)

    def _parse_action_sequence(self, data):
        if 'movements' not in data:
            print("[场景管理器] 警告：YAML文件中没有动作序列")
            return []

        action_sequence = []
        print("[场景管理器] 开始解析动作序列...")

        for movement in data['movements']:
            if movement.get('action_success') != True:
                continue

            action_name = movement['action']
            result_position = movement.get('result_position')
            target_pose = grid_to_world_pose(result_position) if result_position else None

            if 'PICKUP' in action_name:
                action_record = {
                    'type': 'PICKUP',
                    'action': action_name,
                    'position': movement['tile_position'],
                    'tile_type': movement.get('tile_type'),
                    'target_pose': target_pose
                }
                action_sequence.append(action_record)
            elif 'PLACE' in action_name:
                action_record = {
                    'type': 'PLACE',
                    'action': action_name,
                    'position': movement['tile_position'],
                    'tile_type': movement.get('tile_type'),
                    'target_pose': target_pose
                }
                action_sequence.append(action_record)
            else:
                action_record = {
                    'type': 'MOVE',
                    'action': action_name,
                    'target_pose': target_pose
                }
                action_sequence.append(action_record)

        print(f"[场景管理器] 动作序列解析完成，共 {len(action_sequence)} 个动作")
        return action_sequence
