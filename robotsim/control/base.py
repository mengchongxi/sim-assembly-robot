"""控制器抽象接口定义。"""

from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ControllerBase(Protocol):
    """所有控制算法的统一接口。

    控制器接受当前状态，返回控制动作。
    具体实现可以是 MPPI、MPC、PID、RL policy 等。
    """

    def reset(self, initial_state: Dict[str, Any]) -> None:
        """重置控制器内部状态。"""
        ...

    def compute_action(self, state: Dict[str, Any]) -> np.ndarray:
        """根据当前状态计算控制动作。

        Args:
            state: 包含机器人当前状态信息的字典
                   （位姿、速度、关节角等，具体内容由实现定义）

        Returns:
            控制动作数组（关节力矩或目标位置，取决于 control_mode）
        """
        ...

    @property
    def control_mode(self) -> str:
        """控制模式：'torque' 或 'position'。"""
        ...
