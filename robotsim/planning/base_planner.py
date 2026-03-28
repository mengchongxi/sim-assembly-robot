"""规划器基类模块。

定义 BasePlanner 基类，封装所有具体规划器广泛共享的
功能，包括规划工作区管理、质心对齐、坐标平移、
连通性检查以及问题输入验证等通用逻辑。
"""
import numpy as np
import heapq
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from functools import lru_cache
from typing import Tuple, List, Set, Optional


class BasePlanner:
    """规划器基类 - 包含通用功能"""

    def __init__(self, grid_width, grid_height):
        self.full_grid_width = grid_width
        self.full_grid_height = grid_height
        self.work_area = None
        self.expanded_nodes = 0  # 为基准测试添加统计

        # 初始化网格尺寸
        self.reset_working_area()

    def reset_working_area(self):
        """重置工作区域到全局网格"""
        self.work_area = None
        self.grid_width = self.full_grid_width
        self.grid_height = self.full_grid_height

    def set_working_area(self, x_min, x_max, y_min, y_max):
        """设置工作区域"""
        self.work_area = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        self.grid_width = x_max - x_min
        self.grid_height = y_max - y_min
        print(f"Planner working area focused to: x_range=({x_min}, {x_max}), y_range=({y_min}, {y_max})")

    def calculate_centroid(self, base_set: Set, joint_set: Set, wheel_set: Set) -> Tuple[float, float]:
        """计算构型质心"""
        all_tiles = base_set | joint_set | wheel_set
        if not all_tiles:
            return (0, 0)

        sum_x = sum(pos[0] for pos in all_tiles)
        sum_y = sum(pos[1] for pos in all_tiles)
        return (sum_x / len(all_tiles), sum_y / len(all_tiles))

    def compute_translation_vector(self, current_base: Set, current_joint: Set, current_wheel: Set,
                                 goal_base: Set, goal_joint: Set, goal_wheel: Set) -> Tuple[int, int]:
        """计算质心对齐的平移向量"""
        c_initial = self.calculate_centroid(current_base, current_joint, current_wheel)
        c_goal = self.calculate_centroid(goal_base, goal_joint, goal_wheel)
        dx = round(c_goal[0] - c_initial[0])
        dy = round(c_goal[1] - c_initial[1])
        return (dx, dy)

    def apply_translation(self, tiles: Set, dx: int, dy: int) -> Set:
        """对瓦片集合应用平移"""
        return {(pos[0] + dx, pos[1] + dy) for pos in tiles}

    def compute_planning_bounds(self, *tile_sets) -> Tuple[int, int, int, int]:
        """计算规划区域边界"""
        all_tiles = set()
        for tile_set in tile_sets:
            all_tiles.update(tile_set)

        if not all_tiles:
            return (0, self.full_grid_width, 0, self.full_grid_height)

        min_x = min(pos[0] for pos in all_tiles)
        max_x = max(pos[0] for pos in all_tiles)
        min_y = min(pos[1] for pos in all_tiles)
        max_y = max(pos[1] for pos in all_tiles)

        return (min_x, max_x, min_y, max_y)

    def setup_planning_workspace(self, current_base: Set, current_joint: Set, current_wheel: Set,
                               goal_base: Set, goal_joint: Set, goal_wheel: Set, padding: int = 1):
        """设置规划工作空间，包括质心对齐和边界计算"""
        # 质心对齐
        dx, dy = self.compute_translation_vector(current_base, current_joint, current_wheel,
                                               goal_base, goal_joint, goal_wheel)

        aligned_base = self.apply_translation(current_base, dx, dy)
        aligned_joint = self.apply_translation(current_joint, dx, dy)
        aligned_wheel = self.apply_translation(current_wheel, dx, dy)

        # 计算规划边界
        min_x, max_x, min_y, max_y = self.compute_planning_bounds(
            aligned_base, aligned_joint, aligned_wheel,
            goal_base, goal_joint, goal_wheel
        )

        # 设置工作区域（带边界扩展）
        self.set_working_area(
            max(0, min_x - padding),
            min(self.full_grid_width, max_x + 1 + padding),
            max(0, min_y - padding),
            min(self.full_grid_height, max_y + 1 + padding)
        )

        return aligned_base, aligned_joint, aligned_wheel, (dx, dy)

    def validate_problem_input(self, current_base: Set, current_joint: Set, current_wheel: Set,
                             goal_base: Set, goal_joint: Set, goal_wheel: Set) -> Tuple[bool, Optional[str]]:
        """验证问题输入的有效性"""
        # 检查是否为空
        if not (current_base or current_joint or current_wheel):
            return False, "Current configuration is empty"

        if not (goal_base or goal_joint or goal_wheel):
            return False, "Goal configuration is empty"

        # 检查瓦片数量匹配
        if len(current_base) != len(goal_base):
            return False, f"Base tile count mismatch: {len(current_base)} vs {len(goal_base)}"

        if len(current_joint) != len(goal_joint):
            return False, f"Joint tile count mismatch: {len(current_joint)} vs {len(goal_joint)}"

        if len(current_wheel) != len(goal_wheel):
            return False, f"Wheel tile count mismatch: {len(current_wheel)} vs {len(goal_wheel)}"

        return True, None

    def translate_plan(self, plan: List, dx: int, dy: int) -> List:
        """将规划结果平移回原始坐标系"""
        return [
            ((pickup[0] - dx, pickup[1] - dy),
             (dropoff[0] - dx, dropoff[1] - dy),
             move_type)
            for pickup, dropoff, move_type in plan
        ]

    def _manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """计算曼哈顿距离"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def is_connected(self, tiles: set) -> bool:
        """检查瓦片集合是否连通"""
        if not tiles:
            return True
        if len(tiles) == 1:
            return True

        q, visited, start = [next(iter(tiles))], set(), next(iter(tiles))
        visited.add(start)
        head = 0

        while head < len(q):
            x, y = q[head]
            head += 1
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in tiles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        return len(visited) == len(tiles)

    def find_leaf_tiles(self, tiles: set) -> List[Tuple[int, int]]:
        """找到叶子瓦片（移除后仍保持连通性的瓦片）"""
        if len(tiles) <= 1:
            return list(tiles)

        leaf_tiles = []
        for tile in tiles:
            if self.is_connected(tiles - {tile}):
                leaf_tiles.append(tile)
        return leaf_tiles

    def _is_incrementally_connected(self, dropoff_pos: Tuple[int, int], tiles_after_pickup: set) -> bool:
        """检查放置位置是否能保持增量连通性"""
        x, y = dropoff_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (x + dx, y + dy) in tiles_after_pickup:
                return True
        return False
