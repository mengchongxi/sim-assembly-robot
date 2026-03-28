"""机器人低层路径规划器模块。

实现 RobotPathPlanner 类，基于 A* 搜索为蛇双脚机器人在瓦片网格上
规划具体动作序列，支持拾取、放置和纯导航三种任务类型。
"""
import heapq
from typing import Tuple, Optional, List, Set

from robotsim.core.types import ActionType
from robotsim.core.data import SearchNode, TaskGoal
from robotsim.core.robot_state import RobotConfiguration


# 延迟导入以避免循环依赖
def _get_bill_e_bot():
    from robotsim.robot.bill_e_bot import BillEBot
    return BillEBot


class RobotPathPlanner:
    """基于 A* 的低层机器人路径规划器。

    在瓦片网格上为机器人搜索具体动作序列，使机器人能够到达拾取
    或放置瓦片所需的位置，或者导航至指定的目标配置。

    Args:
        robot: 符合 BillEBot 接口的机器人实例。
        grid_tiles: 当前可行走的网格坐标集合。
        grid_width: 网格宽度（列数）。
        grid_height: 网格高度（行数）。
    """
    def __init__(self, robot, grid_tiles: set, grid_width: int = 30, grid_height: int = 30):
        self.robot = robot
        self.grid_tiles = grid_tiles
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.action_costs = {
            ActionType.WAIT: 0,
            ActionType.STEP_FORWARD: 1,
            ActionType.STEP_BACKWARD: 1,
            ActionType.ROTATE_FRONT_RIGHT: 1,
            ActionType.ROTATE_FRONT_LEFT: 1,
            ActionType.ROTATE_BACK_RIGHT: 1,
            ActionType.ROTATE_BACK_LEFT: 1,
            ActionType.ROTATE_FRONT_180: 2,
            ActionType.ROTATE_BACK_180: 2,
            ActionType.PICKUP_FRONT: 1,
            ActionType.PICKUP_LEFT: 1,
            ActionType.PICKUP_RIGHT: 1,
            ActionType.PLACE_FRONT: 1,
            ActionType.PLACE_LEFT: 1,
            ActionType.PLACE_RIGHT: 1
        }

    def update_grid_tiles(self, new_tiles: set):
        """更新可行走网格地址集，在每次拾取/放置操作后调用。"""
        self.grid_tiles = new_tiles

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两个网格坐标之间的曼哈顿距离。"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def heuristic_cost(self, config: RobotConfiguration, target_pos: Tuple[int, int]) -> float:
        """返回前脚到目标位置的曼哈顿距离估计代价。"""
        return self.manhattan_distance(config.front_foot, target_pos)

    def heuristic_cost_to_goal(self, config: RobotConfiguration, goal: TaskGoal) -> float:
        """返回前后脚到目标 TaskGoal 的合并曼哈顿距离估计代价。"""
        front_dist = self.manhattan_distance(config.front_foot, goal.front_foot)
        back_dist = self.manhattan_distance(config.back_foot, goal.back_foot)
        return front_dist + back_dist

    def config_to_state_key(self, config: RobotConfiguration) -> tuple:
        """将机器人配置转换为可哈希的状态键，用于闭开集去重。"""
        return (config.front_foot, config.back_foot, config.carrying_tile)

    def find_reachable_pickup_positions(self, config: RobotConfiguration, pickup_pos: Tuple[int, int]) -> List[TaskGoal]:
        """枚举拾取指定位置瓦片时所有合法的机器人目标配置。

        Returns:
            TaskGoal 列表，每个元素描述一种合法的机器人偏开位置。
        """
        x, y = pickup_pos
        possible_goals = []

        directions = [
            ((x, y+1), (x, y+2), "front"),
            ((x, y-1), (x, y-2), "front"),
            ((x-1, y), (x-2, y), "front"),
            ((x+1, y), (x+2, y), "front"),
        ]

        left_directions = [
            ((x+1, y), (x+1, y+1), "left"),
            ((x-1, y), (x-1, y-1), "left"),
            ((x, y+1), (x-1, y+1), "left"),
            ((x, y-1), (x+1, y-1), "left"),
        ]

        right_directions = [
            ((x-1, y), (x-1, y+1), "right"),
            ((x+1, y), (x+1, y-1), "right"),
            ((x, y-1), (x-1, y-1), "right"),
            ((x, y+1), (x+1, y+1), "right"),
        ]

        all_directions = directions + left_directions + right_directions

        for front_pos, back_pos, direction in all_directions:
            if (front_pos in self.grid_tiles and back_pos in self.grid_tiles):
                goal = TaskGoal(
                    front_foot=front_pos,
                    back_foot=back_pos,
                    task_direction=direction,
                    task_type="pickup"
                )
                possible_goals.append(goal)

        return possible_goals

    def find_reachable_place_positions(self, config: RobotConfiguration, place_pos: Tuple[int, int]) -> List[TaskGoal]:
        """枚举将瓦片放置到指定位置时所有合法的机器人目标配置。

        Returns:
            TaskGoal 列表，每个元素描述一种合法的机器人偏开位置。
        """
        x, y = place_pos
        possible_goals = []

        directions = [
            ((x, y+1), (x, y+2), "front"),
            ((x, y-1), (x, y-2), "front"),
            ((x-1, y), (x-2, y), "front"),
            ((x+1, y), (x+2, y), "front"),
        ]

        left_directions = [
            ((x+1, y), (x+1, y+1), "left"),
            ((x-1, y), (x-1, y-1), "left"),
            ((x, y+1), (x-1, y+1), "left"),
            ((x, y-1), (x+1, y-1), "left"),
        ]

        right_directions = [
            ((x-1, y), (x-1, y+1), "right"),
            ((x+1, y), (x+1, y-1), "right"),
            ((x, y-1), (x-1, y-1), "right"),
            ((x, y+1), (x+1, y+1), "right"),
        ]

        all_directions = directions + left_directions + right_directions

        for front_pos, back_pos, direction in all_directions:
            if (front_pos in self.grid_tiles and back_pos in self.grid_tiles):
                goal = TaskGoal(
                    front_foot=front_pos,
                    back_foot=back_pos,
                    task_direction=direction,
                    task_type="place"
                )
                possible_goals.append(goal)

        return possible_goals

    def a_star_search_to_goal(self, start_config: RobotConfiguration,
                             goal: TaskGoal,
                             max_iterations: int = 2000) -> Optional[List[ActionType]]:
        """使用 A* 搜索找到最优动作序列以到达目标配置。

        Args:
            start_config: 起始机器人配置。
            goal: 目标配置（前后脚坐标）。
            max_iterations: 最大搜索迭代次数，防止搜索过长。

        Returns:
            动作列表（ActionType 列表），失败或超上迭代限制时返回 None。
        """
        BillEBot = _get_bill_e_bot()
        open_list = []
        closed_set: Set[tuple] = set()

        start_node = SearchNode(
            config=start_config.copy(),
            g_cost=0,
            h_cost=self.heuristic_cost_to_goal(start_config, goal)
        )

        heapq.heappush(open_list, start_node)
        best_costs: dict = {}

        iterations = 0

        while open_list and iterations < max_iterations:
            iterations += 1
            current_node = heapq.heappop(open_list)
            current_state = self.config_to_state_key(current_node.config)

            if (current_node.config.front_foot == goal.front_foot and
                current_node.config.back_foot == goal.back_foot):
                return self.reconstruct_path(current_node)

            if current_state in closed_set:
                continue

            closed_set.add(current_state)

            for action in [ActionType.WAIT, ActionType.STEP_FORWARD, ActionType.STEP_BACKWARD,
                          ActionType.ROTATE_FRONT_RIGHT, ActionType.ROTATE_FRONT_LEFT,
                          ActionType.ROTATE_BACK_RIGHT, ActionType.ROTATE_BACK_LEFT,
                          ActionType.ROTATE_FRONT_180, ActionType.ROTATE_BACK_180]:

                temp_robot = BillEBot(current_node.config.copy())
                new_config = temp_robot.execute_action(action, self.grid_tiles,
                                                     self.grid_width, self.grid_height)

                if new_config is None:
                    continue

                new_state = self.config_to_state_key(new_config)

                if new_state in closed_set:
                    continue

                action_cost = self.action_costs[action]
                new_g_cost = current_node.g_cost + action_cost

                if new_state in best_costs and new_g_cost >= best_costs[new_state]:
                    continue

                best_costs[new_state] = new_g_cost

                new_node = SearchNode(
                    config=new_config,
                    g_cost=new_g_cost,
                    h_cost=self.heuristic_cost_to_goal(new_config, goal),
                    parent=current_node,
                    action=action
                )

                heapq.heappush(open_list, new_node)

        return None

    def reconstruct_path(self, goal_node: SearchNode) -> List[ActionType]:
        """从目标节点倒推父节点链，返回正序动作列表。"""
        path = []
        current = goal_node

        while current.parent is not None:
            if current.action is not None:
                path.append(current.action)
            current = current.parent

        path.reverse()
        return path

    def get_task_action(self, direction: str, task_type: str) -> ActionType:
        """将方向字符串和任务类型映射为具体的 ActionType。"""
        if task_type == "pickup":
            direction_to_action = {
                "front": ActionType.PICKUP_FRONT,
                "left": ActionType.PICKUP_LEFT,
                "right": ActionType.PICKUP_RIGHT
            }
        elif task_type == "place":
            direction_to_action = {
                "front": ActionType.PLACE_FRONT,
                "left": ActionType.PLACE_LEFT,
                "right": ActionType.PLACE_RIGHT
            }
        else:
            return ActionType.PICKUP_FRONT

        return direction_to_action.get(direction, ActionType.PICKUP_FRONT)

    def plan_pickup_task(self, pickup_pos: Tuple[int, int]) -> Optional[Tuple[List[ActionType], str]]:
        """规划拾取指定位置瓦片的完整动作序列。

        枚举所有合法拾取位置，对每个位置运行 A* 并选择
        总代价最低的方案。

        Returns:
            (complete_path, task_direction) 元组，或 None（无可达方案）。
        """
        possible_goals = self.find_reachable_pickup_positions(self.robot.config, pickup_pos)
        if not possible_goals:
            return None

        best_total_cost = float('inf')
        best_complete_path = None
        best_task_direction = None

        for goal in possible_goals:
            path = self.a_star_search_to_goal(self.robot.config, goal)
            if path is None:
                continue

            task_action = self.get_task_action(goal.task_direction, goal.task_type)
            complete_path = path + [task_action]

            total_cost = (sum(self.action_costs[action] for action in path) +
                        self.action_costs[task_action])

            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_complete_path = complete_path
                best_task_direction = goal.task_direction

        if best_complete_path:
            return (best_complete_path, best_task_direction)
        else:
            return None

    def plan_place_task(self, place_pos: Tuple[int, int]) -> Optional[Tuple[List[ActionType], str]]:
        """规划将持有瓦片放置到指定位置的完整动作序列。

        若机器人未携带瓦片，直接返回 None。

        Returns:
            (complete_path, task_direction) 元组，或 None（无可达方案）。
        """
        if not self.robot.config.carrying_tile:
            return None

        possible_goals = self.find_reachable_place_positions(self.robot.config, place_pos)
        if not possible_goals:
            return None

        best_total_cost = float('inf')
        best_complete_path = None
        best_task_direction = None

        for goal in possible_goals:
            path = self.a_star_search_to_goal(self.robot.config, goal)
            if path is None:
                continue

            task_action = self.get_task_action(goal.task_direction, goal.task_type)
            complete_path = path + [task_action]

            total_cost = (sum(self.action_costs[action] for action in path) +
                        self.action_costs[task_action])

            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_complete_path = complete_path
                best_task_direction = goal.task_direction

        if best_complete_path:
            return (best_complete_path, best_task_direction)
        else:
            return None

    def plan_path_to_config(self, start_config: RobotConfiguration, end_config: RobotConfiguration) -> Optional[List[ActionType]]:
        print(f"   > 正在规划从 {start_config.front_foot} 到达停车位 {end_config.front_foot} 的路径...")
        parking_goal = TaskGoal(
            front_foot=end_config.front_foot,
            back_foot=end_config.back_foot,
            task_direction="parking",  # 这是一个纯移动任务，方向不重要
            task_type="move"         # 类型也不重要
        )

        # 2. 调用类里已经存在的、强大的A*搜索功能！
        path = self.a_star_search_to_goal(start_config, parking_goal)

        if path:
            print(f"   > ✅ 成功找到通往停车位的路径，共需 {len(path)} 个动作。")
        else:
            print(f"   > ⚠️ 未能找到通往停车位的路径。")

        return path
