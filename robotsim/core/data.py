"""核心数据结构模块。

定义了整个仿真流水线中各模块之间传递数据所用的数据类和配置类，
包括运动记录（MovementRecord）、A* 搜索节点（SearchNode）、
任务目标（TaskGoal）以及规划器配置（PlannerConfig）。
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple, Type

from robotsim.core.robot_state import RobotConfiguration
from robotsim.core.types import ActionType, TileType


@dataclass
class MovementRecord:
    """单步运动记录，用于轨迹回放和 YAML 序列化。

    Attributes:
        step_number: 动作序号（从 1 开始递增）。
        action: 动作名称字符串（如 'STEP_FORWARD'）。
        action_success: 动作是否执行成功。
        distance_moved: 本步两脚位移之和（网格单位）。
        result_position: 动作结束后机器人的前脚/后脚网格坐标字典。
        tile_type: 拾取/放置操作涉及的瓦片类型名称，非拾放动作为 None。
        tile_position: 拾取/放置操作的目标网格坐标，非拾放动作为 None。
    """
    step_number: int
    action: str
    action_success: bool
    distance_moved: float
    result_position: dict
    tile_type: Optional[str] = None
    tile_position: Optional[List[int]] = None


@dataclass
class SearchNode:
    """A* 搜索树中的一个节点，记录机器人状态与路径代价。

    Attributes:
        config: 该节点对应的机器人配置（前脚/后脚位置及携带状态）。
        g_cost: 从起点到该节点的实际代价。
        h_cost: 从该节点到目标的启发式估计代价。
        parent: 父节点，用于路径回溯；起始节点为 None。
        action: 到达该节点所执行的动作；起始节点为 None。
    """
    config: 'RobotConfiguration'
    g_cost: float
    h_cost: float
    parent: Optional['SearchNode'] = None
    action: Optional[ActionType] = None

    @property
    def f_cost(self) -> float:
        """返回 f = g + h，用于优先队列排序。"""
        return self.g_cost + self.h_cost

    def __lt__(self, other):
        """按 f_cost 比较，使 heapq 可以正常排序 SearchNode。"""
        return self.f_cost < other.f_cost


@dataclass
class TaskGoal:
    """路径规划中机器人需要到达的目标配置及对应任务描述。

    Attributes:
        front_foot: 目标状态下前脚的网格坐标。
        back_foot: 目标状态下后脚的网格坐标。
        task_direction: 拾取/放置方向，取值为 'front'、'left' 或 'right'。
        task_type: 任务类型，取值为 'pickup'、'place' 或 'move'。
    """
    front_foot: Tuple[int, int]
    back_foot: Tuple[int, int]
    task_direction: str
    task_type: str


class PlannerConfig:
    """规划器配置类"""
    def __init__(self, planner_class, name: str, description: str, **kwargs):
        self.planner_class = planner_class
        self.name = name
        self.description = description
        self.params = kwargs
