"""机器人状态表示模块。

定义 RobotConfiguration 类，描述双脚机器人在网格上的完整几何状态，
包括前脚、后脚位置、机身朝向以及携带瓦片信息。
"""
import math
from typing import Tuple, Optional

from robotsim.core.types import TileType


class RobotConfiguration:
    """双脚机器人在网格上的完整状态描述。

    机器人由前脚和后脚两个节点定义，每个节点占据一个网格格子。
    機身朝向由前脚指向后脚的向量自动推算。

    Attributes:
        front_foot: 前脚网格坐标 (x, y)。
        back_foot: 后脚网格坐标 (x, y)。
        carrying_tile: 是否正在携带瓦片。
        tile_pos: 携带瓦片的原始网格坐标（仅在携带时有效）。
        carrying_tile_type: 携带瓦片的类型（默认为 BASE）。
        orientation: 机器人朝向角度（弧度），由前后脚坐标自动计算。
    """
    def __init__(self, front_foot: Tuple[int, int], back_foot: Tuple[int, int],
                 carrying_tile: bool = False, tile_pos: Optional[Tuple[int, int]] = None,
                 tile_type: TileType = TileType.BASE):
        self.front_foot = front_foot
        self.back_foot = back_foot
        self.carrying_tile = carrying_tile
        self.tile_pos = tile_pos
        self.carrying_tile_type = tile_type
        self.orientation = self._calculate_orientation()

    def _calculate_orientation(self) -> float:
        """根据前后脚坐标计算机器人朝向角（弧度，区间 [-π, π]）。"""
        dx = self.front_foot[0] - self.back_foot[0]
        dy = self.front_foot[1] - self.back_foot[1]
        return math.atan2(dy, dx)

    def copy(self):
        """返回当前配置的深拷贝，保留所有字段包括 orientation。"""
        config = RobotConfiguration(
            self.front_foot,
            self.back_foot,
            self.carrying_tile,
            self.tile_pos,
            self.carrying_tile_type
        )
        config.orientation = self.orientation
        return config

    def get_facing_direction(self) -> Tuple[int, int]:
        """返回机器人当前朝向对应的轴对齐单位向量 (dx, dy)。

        将连续朝向角圆整到最近的 90° 倍数，
        返回可能的四个单位向量之一：(1,0)、(-1,0)、(0,1)、(0,-1)。
        """
        angle = self.orientation
        rounded_angle = round(angle / (math.pi/2)) * (math.pi/2)
        dx = round(math.cos(rounded_angle))
        dy = round(math.sin(rounded_angle))
        return (dx, dy)

    def get_front_left_right_positions(self) -> dict:
        """返回前脚定义的前方、左侧、右侧网格坐标字典。

        基于当前朝向将相对方向映射为绝对网格坐标。

        Returns:
            包含 'front'、'left'、'right' 键的字典，对应的网格坐标与
            前脚相邻。
        """
        front_x, front_y = self.front_foot
        facing_dir = self.get_facing_direction()

        front_pos = (front_x + facing_dir[0], front_y + facing_dir[1])

        if facing_dir == (1, 0):
            left_pos = (front_x, front_y - 1)
            right_pos = (front_x, front_y + 1)
        elif facing_dir == (-1, 0):
            left_pos = (front_x, front_y + 1)
            right_pos = (front_x, front_y - 1)
        elif facing_dir == (0, 1):
            left_pos = (front_x + 1, front_y)
            right_pos = (front_x - 1, front_y)
        elif facing_dir == (0, -1):
            left_pos = (front_x - 1, front_y)
            right_pos = (front_x + 1, front_y)
        else:
            left_pos = (front_x - 1, front_y)
            right_pos = (front_x + 1, front_y)

        return {
            'front': front_pos,
            'left': left_pos,
            'right': right_pos
        }

    def __str__(self):
        """返回可读的配置描述，包含前后脚坐标、朝向角和携带瓦片信息。"""
        tile_info = f", carrying {self.carrying_tile_type.name} at {self.tile_pos}" if self.carrying_tile else ""
        angle_deg = math.degrees(self.orientation)
        return f"Front: {self.front_foot}, Back: {self.back_foot}, Angle: {angle_deg:.1f}°{tile_info}"
