"""核心抽象基类和协议接口模块。

定义各大模块的抽象基类（ABC）和运行时可检测协议（Protocol），
供规划器、场景管理器、机器人控制器和 GUI 组件等实现使用。
"""
from abc import ABC, abstractmethod
from typing import Protocol, Tuple, List, Set, Optional, runtime_checkable

import numpy as np


@runtime_checkable
class TrajectoryGenerator(Protocol):
    """两点之间的轨迹生成器协议。"""
    def generate(self, start: np.ndarray, end: np.ndarray,
                 num_points: int, arc_height: float = 0.0) -> np.ndarray: ...


class HighLevelPlanner(ABC):
    """高层任务规划器抽象基类。"""

    @abstractmethod
    def generate_full_plan(self, current_base: Set, goal_base: Set,
                           current_joint: Set, goal_joint: Set,
                           current_wheel: Set, goal_wheel: Set) -> Tuple[bool, List]:
        """从当前构型到目标构型生成完整规划方案。

        Returns:
            (success, plan) 元组，plan 为 (pickup_pos, dropoff_pos, tile_type) 列表。
        """
        ...


class SceneManagerBase(ABC):
    """场景对象管理抽象基类。"""

    @abstractmethod
    def load_object(self, position, object_type: str, **kwargs):
        """向场景中添加对象，返回对象标识符。"""
        ...

    @abstractmethod
    def remove_object(self, object_id) -> Optional[str]:
        """按 ID 移除对象，返回其类型或 None。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """移除场景中的所有对象。"""
        ...


class RobotControllerBase(ABC):
    """机器人关节控制抽象基类。"""

    @abstractmethod
    def get_joint_positions(self) -> list:
        """返回当前各关节角度（弧度）列表。"""
        ...

    @abstractmethod
    def set_joint_positions(self, positions: list) -> None:
        """设置目标关节角度。"""
        ...


@runtime_checkable
class GUIComponent(Protocol):
    """线程安全 GUI 组件协议。"""
    def start(self) -> None:
        """在守护线程中启动 GUI。"""
        ...
