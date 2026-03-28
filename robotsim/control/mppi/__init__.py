"""MPPI (Model Predictive Path Integral) 控制算法模块。

提供基于 DIAL-MPC 的四足机器人运动控制：
- Config / ConfigManager: 配置管理
- DreamerEnv: 四足机器人 Brax 环境
- DIALMPCController: MPPI 轨迹优化控制器
- RobotSimulator: 仿真编排器
- ResultsManager: 结果保存

变体增强（通过 mixin 组合）:
- HistoryMemoryMixin: 历史动作记忆采样
- VelocityTrackingMixin: 详细速度跟踪
"""

from robotsim.control.mppi.config import Config, ConfigManager

# JAX/Brax 依赖模块采用延迟导入，避免在缺少 JAX 的环境中阻塞
# 使用时通过显式导入子模块: from robotsim.control.mppi.controller import DIALMPCController


def __getattr__(name):
    """延迟导入 JAX 依赖模块，仅在实际使用时触发。"""
    _lazy_imports = {
        "DIALMPCController": "robotsim.control.mppi.controller",
        "BaseRobotEnv": "robotsim.control.mppi.environment",
        "DreamerEnv": "robotsim.control.mppi.environment",
        "MathUtils": "robotsim.control.mppi.math_utils",
        "ResultsManager": "robotsim.control.mppi.results",
        "RobotSimulator": "robotsim.control.mppi.simulator",
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Config",
    "ConfigManager",
    "BaseRobotEnv",
    "DreamerEnv",
    "DIALMPCController",
    "MathUtils",
    "ResultsManager",
    "RobotSimulator",
]
