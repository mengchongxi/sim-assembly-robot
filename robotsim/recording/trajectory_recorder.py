"""机器人运动轨迹记录器模块。

提供 RobotTrajectoryRecorder 类，记录机器人每步动作的坐标变化、
操作类型和瓦片信息，并将完整轨迹序列序列化到 YAML 文件，
供后续仿真器进行回放。
"""
import math
import yaml
from typing import Tuple, Optional

from robotsim.core.types import ActionType, TileType
from robotsim.core.data import MovementRecord


class RobotTrajectoryRecorder:
    """机器人动作轨迹记录器。

    维护运动记录列表，并将初始位置、初始瓦片布局和全部动作
    序列序列化到 YAML 文件。
    """
    def __init__(self):
        self.initial_position = None
        self.initial_tiles = None
        self.movements = []
        self.step_counter = 0

    def set_initial_position(self, robot_config):
        """记录机器人的初始位置信息，应在任何动作记录开始前调用。"""
        self.initial_position = {
            'front_foot': list(robot_config.front_foot),
            'back_foot': list(robot_config.back_foot)
        }
        print(f"轨迹记录器：设置初始位置 - 前脚{robot_config.front_foot}, 后脚{robot_config.back_foot}")

    def set_initial_tiles(self, base_set, joint_set, wheel_set):
        """记录初始瓦片布局，应在任何动作记录开始前调用。"""
        self.initial_tiles = []

        for pos in base_set:
            self.initial_tiles.append({
                'type': 'base',
                'position': list(pos)
            })

        for pos in joint_set:
            self.initial_tiles.append({
                'type': 'joint',
                'position': list(pos)
            })

        for pos in wheel_set:
            self.initial_tiles.append({
                'type': 'wheel',
                'position': list(pos)
            })

        print(f"轨迹记录器：设置初始瓦片配置 - 总计{len(self.initial_tiles)}个瓦片")
        print(f"  - Base: {len(base_set)}个")
        print(f"  - Joint: {len(joint_set)}个")
        print(f"  - Wheel: {len(wheel_set)}个")

    def calculate_distance_moved(self, old_config, new_config):
        """计算一个动作展前两脚位移的欧几里得距离之和（网格单位）。"""
        front_dist = math.sqrt((new_config.front_foot[0] - old_config.front_foot[0])**2 +
                              (new_config.front_foot[1] - old_config.front_foot[1])**2)
        back_dist = math.sqrt((new_config.back_foot[0] - old_config.back_foot[0])**2 +
                             (new_config.back_foot[1] - old_config.back_foot[1])**2)
        return front_dist + back_dist

    def record_movement(self, action: ActionType, old_config, new_config, action_success: bool,
                       tile_type: Optional[TileType] = None, tile_position: Optional[Tuple[int, int]] = None):
        """记录一步动作及其结果。

        Args:
            action: 执行的动作类型。
            old_config: 动作执行前的机器人配置。
            new_config: 动作执行后的机器人配置。
            action_success: 动作是否成功执行。
            tile_type: 拾取/放置操作此涉及的瓦片类型，非拾放为 None。
            tile_position: 拾取/放置操作的目标网格坐标，非拾放为 None。
        """
        self.step_counter += 1

        distance = self.calculate_distance_moved(old_config, new_config) if action_success else 0.0

        result_position = {
            'front_foot': list(new_config.front_foot),
            'back_foot': list(new_config.back_foot)
        }

        tile_type_str = None
        tile_pos_list = None

        pickup_actions = [ActionType.PICKUP_FRONT, ActionType.PICKUP_LEFT, ActionType.PICKUP_RIGHT]
        place_actions = [ActionType.PLACE_FRONT, ActionType.PLACE_LEFT, ActionType.PLACE_RIGHT]

        if action in pickup_actions or action in place_actions:
            if tile_type:
                tile_type_str = tile_type.name
            if tile_position:
                tile_pos_list = list(tile_position)

        movement = MovementRecord(
            step_number=self.step_counter,
            action=action.name,
            action_success=action_success,
            distance_moved=distance,
            result_position=result_position,
            tile_type=tile_type_str,
            tile_position=tile_pos_list
        )

        self.movements.append(movement)

    def save_to_yaml(self, filename: str = None):
        """将完整轨迹序列序列化为 YAML 文件。

        Args:
            filename: 输出文件路径；清空时默认为 'robot_trajectory.yaml'。

        Returns:
            成功时返回文件路径字符串，失败时返回 None。
        """
        if not filename:
            filename = f"robot_trajectory.yaml"

        trajectory_data = {
            'initial_position': self.initial_position,
            'initial_tiles': self.initial_tiles,
            'movements': []
        }

        for movement in self.movements:
            movement_dict = {
                'step_number': movement.step_number,
                'action': movement.action,
                'action_success': movement.action_success,
                'distance_moved': movement.distance_moved,
                'result_position': movement.result_position
            }

            if movement.tile_type is not None:
                movement_dict['tile_type'] = movement.tile_type
            if movement.tile_position is not None:
                movement_dict['tile_position'] = movement.tile_position

            trajectory_data['movements'].append(movement_dict)

        try:
            with open(filename, 'w') as f:
                yaml.dump(trajectory_data, f, default_flow_style=False, sort_keys=False)
            print(f"✅ 机器人轨迹成功保存到: {filename}")
            return filename
        except Exception as e:
            print(f"❌ 轨迹保存失败: {e}")
            return None

    def clear_trajectory(self):
        """清空所有已记录的轨迹数据，重置记录器到初始状态。"""
        self.initial_position = None
        self.initial_tiles = None
        self.movements = []
        self.step_counter = 0
        print("轨迹记录已清除")

    def get_trajectory_summary(self):
        """返回轨迹摘要信息字符串，包括总步数、成功次数、总距离等。"""
        if not self.movements:
            return "无轨迹记录"

        total_distance = sum(m.distance_moved for m in self.movements)
        successful_actions = sum(1 for m in self.movements if m.action_success)
        pickup_actions = sum(1 for m in self.movements if 'PICKUP' in m.action)
        place_actions = sum(1 for m in self.movements if 'PLACE' in m.action)

        return f"总步数:{len(self.movements)}, 成功:{successful_actions}, 总距离:{total_distance:.1f}, 拾取:{pickup_actions}, 放置:{place_actions}"
