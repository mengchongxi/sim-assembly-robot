"""核心类型、数据结构与接口模块。"""

from robotsim.core.types import TileType, ActionType
from robotsim.core.data import MovementRecord, SearchNode, TaskGoal, PlannerConfig
from robotsim.core.robot_state import RobotConfiguration
from robotsim.core.interfaces import (
    TrajectoryGenerator, HighLevelPlanner, SceneManagerBase,
    RobotControllerBase, GUIComponent,
)
from robotsim.core.trajectory import BezierTrajectoryGenerator, SinusoidalTrajectoryGenerator
