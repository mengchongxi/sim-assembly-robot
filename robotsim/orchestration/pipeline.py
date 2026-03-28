"""统一规划-生成-可视化流水线编排模块。

实现 PlannerPipeline 类，编排 GUI 设定 → 自动规划 → XML 生成 → 轨迹导出
的完整工作流。每一步可独立调用，也可通过 run_full() 执行完整流程。
"""
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from robotsim.core.types import TileType
from robotsim.orchestration.planner_logic import CompletePlannerLogic
from robotsim.orchestration.planner_simulator import CompletePlannerSimulator
from robotsim.simulation.mujoco.xml_generator import MujocoXmlGenerator
from robotsim.simulation.mujoco.model_manager import ModelManager


class PlannerPipeline:
    """统一编排 GUI设定 → 自动规划 → XML生成 → 轨迹导出。

    每一步可独立调用，也可通过 run_full() 执行完整流程。
    """

    def __init__(
        self,
        planner_name: str = "heuristic",
        grid_width: int = 30,
        grid_height: int = 30,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.planner_name = planner_name

        self.simulator = CompletePlannerSimulator(
            grid_width=grid_width,
            grid_height=grid_height,
            default_planner=planner_name,
        )
        self.xml_generator = MujocoXmlGenerator()
        self.model_manager = ModelManager()

    @property
    def logic(self) -> CompletePlannerLogic:
        return self.simulator.logic

    def run_full(self) -> Optional[Path]:
        """完整流水线：GUI → 规划 → 导出。

        Returns:
            输出目录路径，失败则返回 None。
        """
        # Step 1: GUI 设定
        goal_tiles, initial_tiles = self.run_gui_setup()
        if goal_tiles is None:
            print("Pipeline: GUI setup cancelled or failed.")
            return None

        # Step 2: 规划
        trajectory_data = self.run_planning(goal_tiles, initial_tiles)
        if trajectory_data is None:
            print("Pipeline: Planning failed.")
            return None

        # Step 3: 导出
        output_dir = self.export_results(goal_tiles, trajectory_data)
        print(f"\n🎉 Pipeline complete! Output: {output_dir}")
        return output_dir

    def run_gui_setup(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """启动 Pygame GUI 让用户设定目标和初始构型。

        Returns:
            (goal_tiles, initial_tiles) 元组，取消时返回 (None, None)。
            tiles 格式: {TileType.BASE: {(x,y),...}, TileType.JOINT: {...}, ...}
        """
        if not self.simulator.run_gui_setup():
            return None, None

        goal_tiles = {
            TileType.BASE: set(self.logic.goal_base),
            TileType.JOINT: set(self.logic.goal_joint),
            TileType.WHEEL: set(self.logic.goal_wheel),
        }
        initial_tiles = {
            TileType.BASE: set(self.logic.current_base),
            TileType.JOINT: set(self.logic.current_joint),
            TileType.WHEEL: set(self.logic.current_wheel),
        }
        return goal_tiles, initial_tiles

    def run_planning(
        self,
        goal_tiles: Dict[TileType, set],
        initial_tiles: Dict[TileType, set],
    ) -> Optional[Dict]:
        """执行规划算法。

        Args:
            goal_tiles: 目标构型
            initial_tiles: 初始构型

        Returns:
            轨迹数据字典（与 robot_trajectory.yaml 格式一致），失败返回 None。
        """
        success = self.simulator.run_console_planning()
        if not success:
            return None

        # 从 trajectory_recorder 提取轨迹数据
        recorder = self.logic.trajectory_recorder
        trajectory_data = {
            "initial_position": recorder.initial_position,
            "initial_tiles": recorder.initial_tiles,
            "movements": [],
        }
        for movement in recorder.movements:
            movement_dict = {
                "step_number": movement.step_number,
                "action": movement.action,
                "action_success": movement.action_success,
                "distance_moved": movement.distance_moved,
                "result_position": movement.result_position,
            }
            if movement.tile_type is not None:
                movement_dict["tile_type"] = movement.tile_type
            if movement.tile_position is not None:
                movement_dict["tile_position"] = movement.tile_position
            trajectory_data["movements"].append(movement_dict)

        return trajectory_data

    def export_results(
        self,
        goal_tiles: Dict[TileType, set],
        trajectory_data: Dict,
    ) -> Path:
        """一次性导出所有结果到 models/mujoco/generated/<timestamp>/。

        1. 生成 robot.xml
        2. 保存轨迹 robot_trajectory.yaml
        3. 保存构型快照 config.yaml

        Args:
            goal_tiles: 目标构型
            trajectory_data: 轨迹数据

        Returns:
            输出目录路径。
        """
        output_dir = self.model_manager.create_output_dir()

        # 生成 MuJoCo XML
        self.xml_generator.from_2d_config(goal_tiles, output_dir)

        # 保存轨迹
        self.model_manager.save_trajectory(output_dir, trajectory_data)

        # 保存构型快照
        self.model_manager.save_config_snapshot(output_dir, goal_tiles)

        print(f"[pipeline] All results exported to {output_dir}")
        return output_dir
