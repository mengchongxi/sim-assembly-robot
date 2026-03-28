"""控制算法模块。

提供统一的控制器接口（ControllerBase）和具体控制算法实现。
当前支持：MPPI (Model Predictive Path Integral)。
"""

from robotsim.control.base import ControllerBase

__all__ = ["ControllerBase"]
