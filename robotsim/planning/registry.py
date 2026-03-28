"""规划器注册中心模块。

提供 PlannerRegistry 类，负责注册、存储和按名称实例化
规划器，支持在运行时动态切换规划算法。
"""
from typing import Dict

from robotsim.core.data import PlannerConfig
from robotsim.planning.base_planner import BasePlanner
from robotsim.planning.heuristic_planner import HeuristicPlanner

class PlannerRegistry:
    """规划器注册中心"""
    def __init__(self):
        self._planners: Dict[str, PlannerConfig] = {}
        self._register_default_planners()

    def _register_default_planners(self):
        """注册默认规划器"""
        self.register(PlannerConfig(
            HeuristicPlanner,
            "heuristic",
            "启发式A*规划器 - 使用匈牙利算法优化"
        ))

    def register(self, config: PlannerConfig):
        """注册新的规划器"""
        self._planners[config.name] = config
        print(f"已注册规划器: {config.name} - {config.description}")

    def get_planner(self, name: str, grid_width: int, grid_height: int) -> BasePlanner:
        """获取指定规划器实例"""
        if name not in self._planners:
            raise ValueError(f"未知规划器: {name}")

        config = self._planners[name]
        return config.planner_class(grid_width, grid_height, **config.params)

    def list_planners(self) -> Dict[str, str]:
        """列出所有可用规划器"""
        return {name: config.description for name, config in self._planners.items()}
