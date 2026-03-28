"""启发式 A* 规划器模块。

实现基于双向 A* 搜索的 HeuristicPlanner，
支持贪婪（greedy）和匈牙利（hungarian）两种启发式函数。
利用 LRU 缓存加速反复评估，支持模块整体重排序规划。
"""
import numpy as np
import heapq
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from functools import lru_cache
from typing import Tuple, List

from robotsim.planning.base_planner import BasePlanner


class HeuristicPlanner(BasePlanner):
    """启发式规划器 - 继承自BasePlanner，支持不同启发式函数"""

    def __init__(self, grid_width, grid_height, heuristic_type="greedy"):
        """
        初始化启发式规划器

        Args:
            grid_width: 网格宽度
            grid_height: 网格高度
            heuristic_type: 启发式类型，可选 "greedy" 或 "hungarian"
        """
        super().__init__(grid_width, grid_height)

        # 设置启发式类型
        if heuristic_type not in ["greedy", "hungarian"]:
            raise ValueError(f"Unsupported heuristic type: {heuristic_type}. Must be 'greedy' or 'hungarian'")

        self.heuristic_type = heuristic_type
        print(f"HeuristicPlanner initialized with {heuristic_type} heuristic")

        # 采纳您的建议：保留LRU包装思路，并应用在您提供的、逻辑正确的函数上
        cache_r = 2000
        self.get_reachable_tiles_cached = lru_cache(maxsize=cache_r)(self._get_reachable_tiles_from_outside)

        self.__post_init_cache()

    def __post_init_cache(self):
        """初始化算法的缓存"""
        cache_h = 4000
        self.hungarian_cached = lru_cache(maxsize=cache_h)(self._hungarian_cost)
        # 采纳您的建议：为贪婪成本计算也添加LRU缓存
        self.greedy_cost_cached = lru_cache(maxsize=cache_h)(self._greedy_cost)
        print(f"A* Planner: 匈牙利和贪婪算法的LRU缓存已启动！")

    def reset_working_area(self):
        """重置工作区域，同时清理缓存"""
        super().reset_working_area()
        if hasattr(self, 'get_reachable_tiles_cached'):
            self.get_reachable_tiles_cached.cache_clear()
        if hasattr(self, 'hungarian_cached'):
            self.hungarian_cached.cache_clear()
        if hasattr(self, 'greedy_cost_cached'):
            self.greedy_cost_cached.cache_clear()

    def set_working_area(self, x_min, x_max, y_min, y_max):
        """设置工作区域，同时清理缓存"""
        super().set_working_area(x_min, x_max, y_min, y_max)
        # 采纳您的建议：工作区变化时清空缓存，避免风险
        if hasattr(self, 'get_reachable_tiles_cached'):
            self.get_reachable_tiles_cached.cache_clear()

    def _hungarian_cost(self, misplaced: frozenset, empty_goal: frozenset) -> int:
        """
        计算错位集合和空目标之间的最小曼哈顿距离总和（匈牙利算法核心）。
        """
        m, e = list(misplaced), list(empty_goal)
        if not m or not e:
            return 0

        cost_matrix = np.array([[self._manhattan_distance(x, y) for y in e] for x in m])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return int(cost_matrix[row_ind, col_ind].sum())

    def _solve_hungarian_for_type(self, current_set, goal_set):
        """
        计算特定类型的瓦片从当前状态到目标状态的匈牙利成本。
        """
        misplaced = frozenset(current_set - goal_set)
        empty_goal = frozenset(goal_set - current_set)
        return self.hungarian_cached(misplaced, empty_goal)

    def _compute_accessible_empty_space(self, tiles_to_walk_on: frozenset) -> frozenset:
        """
        在空域上从工作区边界做泛洪，求"外部可达"的空格集合。
        机器人不在空格里行走，但这个集合用于定义哪些表面模块"可从外部接近"。
        """
        x_min, x_max = (self.work_area['x_min'], self.work_area['x_max']) if self.work_area else (0, self.full_grid_width)
        y_min, y_max = (self.work_area['y_min'], self.work_area['y_max']) if self.work_area else (0, self.full_grid_height)

        empty = {(x, y) for x in range(x_min, x_max) for y in range(y_min, y_max)} - set(tiles_to_walk_on)
        if not empty:
            return frozenset()

        # 从工作区四周的空格作为种子开始BFS
        q = []
        visited = set()
        # 顶/底边
        for x in range(x_min, x_max):
            for y in [y_min, y_max - 1]:
                if (x, y) in empty:
                    q.append((x, y)); visited.add((x, y))
        # 左/右边（去掉角点重复也可）
        for y in range(y_min + 1, y_max - 1):
            for x in [x_min, x_max - 1]:
                if (x, y) in empty and (x, y) not in visited:
                    q.append((x, y)); visited.add((x, y))

        head = 0
        while head < len(q):
            cx, cy = q[head]; head += 1
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = cx + dx, cy + dy
                nb = (nx, ny)
                if (x_min <= nx < x_max and y_min <= ny < y_max) and nb in empty and nb not in visited:
                    visited.add(nb); q.append(nb)

        return frozenset(visited)

    def _get_reachable_tiles_from_outside(self, tiles_to_walk_on: frozenset) -> frozenset:
        """
        先找"外部可达空格"，再以其相邻的模块为种子，在模块图上BFS得到可达模块集合。
        """
        if not tiles_to_walk_on:
            return frozenset()

        # 1) 外部可达空格
        ext_empty = self._compute_accessible_empty_space(tiles_to_walk_on)

        # 2) 找到与 ext_empty 相邻的表面模块作为种子
        seeds = set()
        for tx, ty in tiles_to_walk_on:
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nb = (tx + dx, ty + dy)
                if nb in ext_empty:
                    seeds.add((tx, ty)); break

        if not seeds:
            return frozenset()

        # 3) 在模块图上BFS
        visited = set(seeds)
        q = list(seeds); head = 0
        tiles_set = set(tiles_to_walk_on)
        while head < len(q):
            cx, cy = q[head]; head += 1
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nb = (cx + dx, cy + dy)
                if nb in tiles_set and nb not in visited:
                    visited.add(nb); q.append(nb)

        return frozenset(visited)

    def _is_accessible(self, dropoff_pos: Tuple[int, int], tiles_after_pickup: frozenset) -> bool:
        """
        判断一个放置点(dropoff_pos)是否可达。
        """
        # 调用缓存的、逻辑正确的函数
        reachable_tiles = self.get_reachable_tiles_cached(tiles_after_pickup)

        if not reachable_tiles:
            return False

        x, y = dropoff_pos
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (x + dx, y + dy)
            if neighbor in reachable_tiles:
                return True

        return False

    def _greedy_cost(self, misplaced: frozenset, empty_goal: frozenset) -> int:
        """
        【可缓存的辅助函数】
        计算错位集合和空目标之间的贪婪曼哈顿距离总和。
        """
        # 如果没有错位或者没有空位，成本为0
        if not misplaced or not empty_goal:
            return 0

        dist_sum = 0
        # 为每个错位的瓦片，找到离它最近的那个空目标位置的距离
        for m_pos in misplaced:
            min_dist = min(self._manhattan_distance(m_pos, e_pos) for e_pos in empty_goal)
            dist_sum += min_dist

        return dist_sum

    def _calculate_heuristic_greedy(self, current_state: tuple, goal_state: tuple) -> int:
        """
        启发式：贪婪曼哈顿距离总和 + 惩罚项 (现在使用缓存)。
        """
        current_base, current_joint, current_wheel = current_state
        goal_base, goal_joint, goal_wheel = goal_state
        h_cost = 0

        # 1. 调用新的缓存函数来计算基础启发式成本
        h_cost += self.greedy_cost_cached(
            frozenset(current_base - goal_base),
            frozenset(goal_base - current_base)
        )
        h_cost += self.greedy_cost_cached(
            frozenset(current_joint - goal_joint),
            frozenset(goal_joint - current_joint)
        )
        h_cost += self.greedy_cost_cached(
            frozenset(current_wheel - goal_wheel),
            frozenset(goal_wheel - current_wheel)
        )

        # 2. 加上与之前完全相同的惩罚项
        all_current = current_base | current_joint | current_wheel
        for pos in goal_base:
            if pos in all_current and pos not in current_base:
                h_cost += 2
        for pos in goal_joint:
            if pos in all_current and pos not in current_joint:
                h_cost += 2
        for pos in goal_wheel:
            if pos in all_current and pos not in current_wheel:
                h_cost += 2

        return h_cost

    def _calculate_heuristic_hungarian(self, current_state: tuple, goal_state: tuple) -> int:
        """计算启发式代价（匈牙利算法）"""
        current_base, current_joint, current_wheel = current_state
        goal_base, goal_joint, goal_wheel = goal_state
        h_cost = 0
        h_cost += self._solve_hungarian_for_type(current_base, goal_base)
        h_cost += self._solve_hungarian_for_type(current_joint, goal_joint)
        h_cost += self._solve_hungarian_for_type(current_wheel, goal_wheel)

        all_current = current_base | current_joint | current_wheel
        for pos in goal_base:
            if pos in all_current and pos not in current_base: h_cost += 2
        for pos in goal_joint:
            if pos in all_current and pos not in current_joint: h_cost += 2
        for pos in goal_wheel:
            if pos in all_current and pos not in current_wheel: h_cost += 2
        return h_cost

    def _calculate_heuristic(self, current_state: tuple, goal_state: tuple) -> int:
        """根据设置的启发式类型计算启发式代价"""
        if self.heuristic_type == "greedy":
            return self._calculate_heuristic_greedy(current_state, goal_state)
        elif self.heuristic_type == "hungarian":
            return self._calculate_heuristic_hungarian(current_state, goal_state)
        else:
            raise ValueError(f"Unknown heuristic type: {self.heuristic_type}")

    def _reconstruct_path(self, came_from: dict, current_state: tuple) -> list:
        """重构路径"""
        path = []
        while current_state in came_from:
            prev_state, move = came_from[current_state]
            path.append(move)
            current_state = prev_state
        return path

    def _reconstruct_bidirectional_path(self, came_from_fwd: dict, came_from_bwd: dict, meeting_point: tuple) -> list:
        """重构双向搜索路径"""
        path_fwd = self._reconstruct_path(came_from_fwd, meeting_point)
        path_fwd.reverse()
        path_bwd = self._reconstruct_path(came_from_bwd, meeting_point)
        return path_fwd + path_bwd

    def generate_full_plan(self, current_base, goal_base, current_joint, goal_joint, current_wheel, goal_wheel):
        """生成完整规划方案"""
        # 验证输入
        is_valid, error_msg = self.validate_problem_input(
            current_base, current_joint, current_wheel,
            goal_base, goal_joint, goal_wheel
        )
        if not is_valid:
            print(f"Input validation failed: {error_msg}")
            return False, []

        # 设置规划工作空间（包括质心对齐）
        aligned_base, aligned_joint, aligned_wheel, (dx, dy) = self.setup_planning_workspace(
            current_base, current_joint, current_wheel,
            goal_base, goal_joint, goal_wheel
        )

        start_state = (frozenset(aligned_base), frozenset(aligned_joint), frozenset(aligned_wheel))
        goal_state = (frozenset(goal_base), frozenset(goal_joint), frozenset(goal_wheel))

        if start_state == goal_state:
            self.reset_working_area()
            return True, []

        # 执行双向A*搜索
        success, plan = self._execute_bidirectional_search(start_state, goal_state)

        # 重置工作区域
        self.reset_working_area()

        if success:
            # 将规划结果平移回原始坐标系
            translated_plan = self.translate_plan(plan, dx, dy)
            return True, translated_plan
        else:
            return False, []

    def _execute_bidirectional_search(self, start_state: tuple, goal_state: tuple) -> Tuple[bool, List]:
        """执行双向A*搜索"""
        open_set_fwd = [(self._calculate_heuristic(start_state, goal_state), start_state)]
        came_from_fwd = {}
        g_score_fwd = defaultdict(lambda: float('inf'), {start_state: 0})
        open_set_bwd = [(self._calculate_heuristic(goal_state, start_state), goal_state)]
        came_from_bwd = {}
        g_score_bwd = defaultdict(lambda: float('inf'), {goal_state: 0})
        meeting_point, best_path_cost = None, float('inf')
        max_iter, count = 20000, 0

        while open_set_fwd and open_set_bwd and count < max_iter:
            count += 1
            self.expanded_nodes += 1  # 为基准测试统计

            # --- 前向搜索 ---
            _, current_fwd = heapq.heappop(open_set_fwd)
            if current_fwd in g_score_bwd:
                current_cost = g_score_fwd[current_fwd] + g_score_bwd[current_fwd]
                if current_cost < best_path_cost:
                    best_path_cost = current_cost
                    meeting_point = current_fwd
            if open_set_fwd and open_set_bwd and (open_set_fwd[0][0] + open_set_bwd[0][0]) >= best_path_cost:
                break

            self._expand_node(current_fwd, goal_state, came_from_fwd, g_score_fwd, open_set_fwd, forward=True)

            # --- 后向搜索 ---
            _, current_bwd = heapq.heappop(open_set_bwd)
            if current_bwd in g_score_fwd:
                current_cost = g_score_fwd[current_bwd] + g_score_bwd[current_bwd]
                if current_cost < best_path_cost:
                    best_path_cost = current_cost
                    meeting_point = current_bwd
            if open_set_fwd and open_set_bwd and (open_set_fwd[0][0] + open_set_bwd[0][0]) >= best_path_cost:
                break

            self._expand_node(current_bwd, start_state, came_from_bwd, g_score_bwd, open_set_bwd, forward=False)

        if meeting_point:
            final_path = self._reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, meeting_point)
            return True, final_path

        return False, []

    def _expand_node(self, current_state: tuple, target_state: tuple, came_from: dict,
                    g_score: dict, open_set: list, forward: bool = True):
        """扩展节点（前向或后向搜索）"""
        current_base_set, current_joint_set, current_wheel_set = current_state
        all_current = current_base_set | current_joint_set | current_wheel_set
        leaf_tiles = self.find_leaf_tiles(all_current)

        neighbors_pos = set()
        x_min, x_max = (self.work_area['x_min'], self.work_area['x_max']) if self.work_area else (0, self.full_grid_width)
        y_min, y_max = (self.work_area['y_min'], self.work_area['y_max']) if self.work_area else (0, self.full_grid_height)
        for x, y in all_current:
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if (x_min <= nx < x_max and y_min <= ny < y_max and (nx, ny) not in all_current):
                    neighbors_pos.add((nx, ny))

        for leaf in leaf_tiles:
            tiles_after_pickup = frozenset(all_current - {leaf})
            for dropoff in neighbors_pos:
                if not self._is_accessible(dropoff, tiles_after_pickup) or not self._is_incrementally_connected(dropoff, tiles_after_pickup):
                    continue

                move_type = "base"
                if leaf in current_joint_set: move_type = "joint"
                elif leaf in current_wheel_set: move_type = "wheel"

                new_base = (current_base_set - {leaf}) | {dropoff} if move_type == "base" else current_base_set
                new_joint = (current_joint_set - {leaf}) | {dropoff} if move_type == "joint" else current_joint_set
                new_wheel = (current_wheel_set - {leaf}) | {dropoff} if move_type == "wheel" else current_wheel_set
                neighbor_state = (frozenset(new_base), frozenset(new_joint), frozenset(new_wheel))

                tentative_g_score = g_score[current_state] + 1
                if tentative_g_score < g_score[neighbor_state]:
                    # 根据搜索方向调整动作记录
                    if forward:
                        action = (leaf, dropoff, move_type)
                    else:
                        action = (dropoff, leaf, move_type)

                    came_from[neighbor_state] = (current_state, action)
                    g_score[neighbor_state] = tentative_g_score
                    f_score = tentative_g_score + self._calculate_heuristic(neighbor_state, target_state)
                    heapq.heappush(open_set, (f_score, neighbor_state))
