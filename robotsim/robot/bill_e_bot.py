"""双脚机器人逻辑模型模块。

实现 BillEBot 类，不依赖物理引擎的纯逻辑模型，用于规划器
中进行动作合法性检查和状态转移。评佰融合轨迹、房间检查和集动位置计算。
"""
from typing import Tuple, Optional

from robotsim.core.types import ActionType
from robotsim.core.robot_state import RobotConfiguration


class BillEBot:
    """双脚机器人的纯逻辑（无物理）表示。

    接收一个 RobotConfiguration 并提供在瓦片网格上验证和执行动作的方法。
    主要由 RobotPathPlanner 和高层和低层规划器内部使用。

    Args:
        config: 指加给该机器人的初始 RobotConfiguration。
    """
    def __init__(self, config: RobotConfiguration):
        self.config = config

    def is_walkable_position(self, pos: Tuple[int, int], grid_tiles: set) -> bool:
        """如果指定网格位置在可行走瓦片集中，返回 True。"""
        return pos in grid_tiles

    def is_valid_position(self, pos: Tuple[int, int], grid_width: int, grid_height: int) -> bool:
        """如果指定位置在网格边界内，返回 True。"""
        x, y = pos
        return 0 <= x < grid_width and 0 <= y < grid_height

    def get_rotated_position(self, pivot: Tuple[int, int], point: Tuple[int, int],
                           clockwise: bool = True) -> Tuple[int, int]:
        """将 point 绕 pivot 旋转 90°（顺时针或逆时针）并返回新网格坐标。"""
        px, py = pivot
        x, y = point

        rel_x = x - px
        rel_y = y - py

        if clockwise:
            new_rel_x = rel_y
            new_rel_y = -rel_x
        else:
            new_rel_x = -rel_y
            new_rel_y = rel_x

        new_x = new_rel_x + px
        new_y = new_rel_y + py

        return (new_x, new_y)

    def get_rotated_position_180(self, pivot: Tuple[int, int], point: Tuple[int, int]) -> Tuple[int, int]:
        """将 point 绕 pivot 旋转 180° 并返回新网格坐标。"""
        px, py = pivot
        x, y = point

        rel_x = x - px
        rel_y = y - py

        new_rel_x = -rel_x
        new_rel_y = -rel_y

        new_x = new_rel_x + px
        new_y = new_rel_y + py

        return (new_x, new_y)

    def execute_action(self, action: ActionType, grid_tiles: set,
                      grid_width: int = 30, grid_height: int = 30) -> Optional[RobotConfiguration]:
        """在网格上尝试执行指定动作，返回更新后的配置或 None（动作受阻）。

        处理所有 ActionType，包括前进、后退、四种 90° 旋转和两种 180° 旋转。
        拾取/放置动作由专用方法处理，此处不注册。

        Args:
            action: 要执行的 ActionType。
            grid_tiles: 可行走网格坐标集合。
            grid_width: 网格宽度限制。
            grid_height: 网格高度限制。

        Returns:
            新的 RobotConfiguration，或 None（动作被网格边界/障碍物阻止）。
        """
        new_config = self.config.copy()

        if action == ActionType.WAIT:
            return new_config

        elif action == ActionType.STEP_FORWARD:
            direction = self.config.get_facing_direction()
            new_front = (self.config.front_foot[0] + direction[0],
                        self.config.front_foot[1] + direction[1])

            if self.is_walkable_position(new_front, grid_tiles):
                new_config.back_foot = self.config.front_foot
                new_config.front_foot = new_front
                new_config.orientation = new_config._calculate_orientation()
                return new_config

        elif action == ActionType.STEP_BACKWARD:
            direction = self.config.get_facing_direction()
            new_back = (self.config.back_foot[0] - direction[0],
                       self.config.back_foot[1] - direction[1])

            if self.is_walkable_position(new_back, grid_tiles):
                new_config.front_foot = self.config.back_foot
                new_config.back_foot = new_back
                new_config.orientation = new_config._calculate_orientation()
                return new_config

        elif action == ActionType.ROTATE_FRONT_RIGHT:
            new_back = self.get_rotated_position(self.config.front_foot,
                                               self.config.back_foot,
                                               clockwise=True)

            if self.is_walkable_position(new_back, grid_tiles):
                new_config.back_foot = new_back
                new_config.orientation = new_config._calculate_orientation()
                return new_config

        elif action == ActionType.ROTATE_FRONT_LEFT:
            new_back = self.get_rotated_position(self.config.front_foot,
                                               self.config.back_foot,
                                               clockwise=False)

            if self.is_walkable_position(new_back, grid_tiles):
                new_config.back_foot = new_back
                new_config.orientation = new_config._calculate_orientation()
                return new_config

        elif action == ActionType.ROTATE_BACK_RIGHT:
            new_front = self.get_rotated_position(self.config.back_foot,
                                                self.config.front_foot,
                                                clockwise=True)

            if self.is_walkable_position(new_front, grid_tiles):
                new_config.front_foot = new_front
                new_config.orientation = new_config._calculate_orientation()
                return new_config

        elif action == ActionType.ROTATE_BACK_LEFT:
            new_front = self.get_rotated_position(self.config.back_foot,
                                                self.config.front_foot,
                                                clockwise=False)

            if self.is_walkable_position(new_front, grid_tiles):
                new_config.front_foot = new_front
                new_config.orientation = new_config._calculate_orientation()
                return new_config

        elif action == ActionType.ROTATE_FRONT_180:
            new_back = self.get_rotated_position_180(self.config.front_foot,
                                                   self.config.back_foot)

            if self.is_walkable_position(new_back, grid_tiles):
                new_config.back_foot = new_back
                new_config.orientation = new_config._calculate_orientation()
                return new_config

        elif action == ActionType.ROTATE_BACK_180:
            new_front = self.get_rotated_position_180(self.config.back_foot,
                                                    self.config.front_foot)

            if self.is_walkable_position(new_front, grid_tiles):
                new_config.front_foot = new_front
                new_config.orientation = new_config._calculate_orientation()
                return new_config

        return None

    def can_pick_up_tile_at(self, position: str, grid_tiles: set) -> Optional[Tuple[int, int]]:
        """检查机器人能否从指定方向拾取瓦片。

        Args:
            position: 方向字符串，取值 'front'、'left' 或 'right'。
            grid_tiles: 当前可行走瓦片集合。

        Returns:
            目标瓦片的网格坐标，或 None（已携带瓦片或目标不在网格上）。
        """
        if self.config.carrying_tile:
            return None

        positions = self.config.get_front_left_right_positions()
        target_pos = positions.get(position)

        if target_pos and target_pos in grid_tiles:
            return target_pos
        return None

    def can_place_tile_at(self, position: str, grid_tiles: set,
                         grid_width: int = 30, grid_height: int = 30) -> Optional[Tuple[int, int]]:
        """检查机器人能否向指定方向放置持有的瓦片。

        目标位置必须在网格边界内、不被占据，且与现有瓦片相邻。

        Args:
            position: 方向字符串，取值 'front'、'left' 或 'right'。
            grid_tiles: 当前可行走瓦片集合。
            grid_width: 网格宽度限制。
            grid_height: 网格高度限制。

        Returns:
            目标放置位置的网格坐标，或 None（条件不满足）。
        """
        if not self.config.carrying_tile:
            return None

        positions = self.config.get_front_left_right_positions()
        target_pos = positions.get(position)

        if target_pos:
            x, y = target_pos
            if (self.is_valid_position(target_pos, grid_width, grid_height) and
                target_pos not in grid_tiles):
                adjacent_positions = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                for adj_pos in adjacent_positions:
                    if adj_pos in grid_tiles:
                        return target_pos
        return None
