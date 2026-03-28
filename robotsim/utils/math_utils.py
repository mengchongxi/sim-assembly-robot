"""共享数学工具函数模块。

提供曼哈顿距离计算、质心计算以及网格坐标旋转等常用几何函数。
"""
import math
from typing import Tuple, Set


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """计算两个二维网格坐标之间的曼哈顿距离。"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def calculate_centroid(tiles: Set[Tuple[int, int]]) -> Tuple[float, float]:
    """计算一组瓦片坐标的几何质心。"""
    if not tiles:
        return (0, 0)
    sum_x = sum(pos[0] for pos in tiles)
    sum_y = sum(pos[1] for pos in tiles)
    return (sum_x / len(tiles), sum_y / len(tiles))


def rotate_point_90(pivot: Tuple[int, int], point: Tuple[int, int],
                    clockwise: bool = True) -> Tuple[int, int]:
    """将网格坐标 point 绕 pivot 旋转 90°（顺时针或逆时针）。"""
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

    return (new_rel_x + px, new_rel_y + py)


def rotate_point_180(pivot: Tuple[int, int], point: Tuple[int, int]) -> Tuple[int, int]:
    """将网格坐标 point 绕 pivot 旋转 180°。"""
    px, py = pivot
    x, y = point
    return (-( x - px) + px, -(y - py) + py)
