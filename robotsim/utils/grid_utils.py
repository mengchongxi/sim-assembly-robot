"""网格与世界坐标转换工具模块。

提供网格坐标转世界坐标、世界坐标转网格坐标以及
机器人姿态转换等实用函数，常数 GRID_CELL_SIZE = 0.12 米。
"""
import numpy as np
from typing import Tuple, List

GRID_CELL_SIZE = 0.12


def grid_to_world(grid_pos: Tuple[int, int], z: float = 0.0) -> List[float]:
    """将网格坐标转换为世界坐标（米）。"""
    return [round(grid_pos[0] * GRID_CELL_SIZE, 4),
            round(grid_pos[1] * GRID_CELL_SIZE, 4),
            z]


def world_to_grid(world_pos, cell_size: float = GRID_CELL_SIZE) -> Tuple[int, int]:
    """将世界坐标（米）转换为网格坐标（整数格子索引）。"""
    return (round(world_pos[0] / cell_size), round(world_pos[1] / cell_size))


def grid_to_world_pose(result_position: dict) -> dict:
    """将网格坐标表示的机器人位置转换为带朝向的世界姿态字典。

    Args:
        result_position: 包含 'front_foot' 和 'back_foot' 网格坐标的字典。

    Returns:
        包含 'position'（世界坐标）和 'orientation'（四元数）的字典。
    """
    front_foot = result_position['front_foot']
    back_foot = result_position['back_foot']
    robot_pos = np.array(back_foot) * GRID_CELL_SIZE
    pos = [round(robot_pos[0], 2), round(robot_pos[1], 2), 0.15]

    direction_vector = (np.array(front_foot) - np.array(back_foot)).tolist()
    direction_tuple = tuple(direction_vector)
    orientation_map = {
        (1, 0): [0, 0, 0, 1],
        (0, 1): [0, 0, 0.707, 0.707],
        (0, -1): [0, 0, -0.707, 0.707],
        (-1, 0): [0, 0, 1, 0]
    }
    orn = orientation_map.get(direction_tuple, [0, 0, 0, 1])
    return {'position': pos, 'orientation': orn}
