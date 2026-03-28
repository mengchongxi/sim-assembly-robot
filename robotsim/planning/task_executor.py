"""机器人任务序列执行器模块。

实现 RobotTaskSequenceExecutor 类，将高层规划器输出的
移动序列（拾取/放置对）转化为逻步级动作流，
通过低层路径规划器逐部返回每个具体的 ActionType。
"""
from typing import Tuple, Optional, List

from robotsim.core.types import ActionType

class RobotTaskSequenceExecutor:
    """逐步式任务序列执行器。

    将高层规划的瓦片移动序列（列表 of (pickup, dropoff, type)）
    转化为交替的拾取和放置任务。每次调用 execute_next_action()
    只返回一个 ActionType，外部循环应将其应用到机器人后再继续。

    Args:
        robot: 符合 BillEBot 接口的机器人实例。
        path_planner: RobotPathPlanner 实例，用于为每个子任务规划路径。
    """
    def __init__(self, robot, path_planner):
        self.robot = robot
        self.path_planner = path_planner
        self.task_sequence = []
        self.current_task_index = 0
        self.current_task_actions = []
        self.current_action_index = 0
        self.is_executing = False

    def load_move_sequence(self, move_sequence: List[Tuple], current_base: set, current_joint: set, current_wheel: set):
        """将高层规划输出的移动序列转化为交替拾取/放置任务并重置执行状态。

        Args:
            move_sequence: 列表 of (pickup_pos, dropoff_pos, tile_type) 元组。
            current_base/joint/wheel: 当前瓦片在网格上的坐标集（暂未使用，为未来扩展保留）。
        """
        self.task_sequence = []

        for pickup_pos, dropoff_pos, move_type in move_sequence:
            self.task_sequence.append({
                'type': 'pickup',
                'position': pickup_pos,
                'tile_type': move_type
            })
            self.task_sequence.append({
                'type': 'place',
                'position': dropoff_pos,
                'tile_type': move_type
            })

        self.current_task_index = 0
        self.current_task_actions = []
        self.current_action_index = 0
        self.is_executing = True

        print(f"加载任务序列: {len(self.task_sequence)} 个任务")

    def update_path_planner(self, new_grid_tiles: set):
        """更新路径规划器的可行走网格地址集，在瓦片布局发生变化后调用。"""
        self.path_planner.grid_tiles = new_grid_tiles

    def get_current_task_info(self) -> str:
        """返回当前任务进度的可读字符串，用于调试日志。"""
        if not self.is_executing or self.current_task_index >= len(self.task_sequence):
            return "无任务"

        task = self.task_sequence[self.current_task_index]
        action_progress = f"{self.current_action_index}/{len(self.current_task_actions)}"
        return f"任务 {self.current_task_index + 1}/{len(self.task_sequence)}: {task['type']} at {task['position']} ({action_progress})"

    def plan_current_task(self) -> bool:
        """为当前子任务调用路径规划器，填充此任务的动作列表。

        Returns:
            True 表示规划成功，False 表示规划失败或无未完成任务。
        """
        if not self.is_executing or self.current_task_index >= len(self.task_sequence):
            return False

        task = self.task_sequence[self.current_task_index]

        if task['type'] == 'pickup':
            result = self.path_planner.plan_pickup_task(task['position'])
        elif task['type'] == 'place':
            result = self.path_planner.plan_place_task(task['position'])
        else:
            return False

        if result:
            self.current_task_actions, _ = result
            self.current_action_index = 0
            return True
        else:
            print(f"任务规划失败: {task['type']} at {task['position']}")
            return False

    def execute_next_action(self) -> Optional[ActionType]:
        """返回当前子任务的下一个 ActionType，并把指针向前移动一步。

        当当前子任务的所有动作均返回完毕后，自动进入下一个子任务并规划。

        Returns:
            下一个 ActionType；所有任务完成、规划失败或未在执行时返回 None。
        """
        if not self.is_executing:
            return None

        if self.current_action_index >= len(self.current_task_actions):
            self.current_task_index += 1

            if self.current_task_index >= len(self.task_sequence):
                self.is_executing = False
                print("所有任务执行完成！")
                return None

            if not self.plan_current_task():
                self.is_executing = False
                print("任务规划失败，停止执行")
                return None

        if self.current_action_index < len(self.current_task_actions):
            action = self.current_task_actions[self.current_action_index]
            self.current_action_index += 1
            return action

        return None

    def start_execution(self) -> bool:
        """重置执行状态并为第一个子任务规划路径。

        Returns:
            True 表示第一个子任务规划成功，False 表示任务序列为空或规划失败。
        """
        if not self.task_sequence:
            print("没有任务序列可执行")
            return False

        self.current_task_index = 0
        self.current_action_index = 0
        self.is_executing = True

        return self.plan_current_task()

    def stop_execution(self):
        """中止执行并清空当前动作列表。"""
        self.is_executing = False
        self.current_task_actions = []
        self.current_action_index = 0
        print("任务执行已停止")
