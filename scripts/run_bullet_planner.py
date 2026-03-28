"""完整机器人任务与运动规划器（Pygame 界面 + 控制台规划 + 自动导出）入口脚本。"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import dataclass
import tyro
from robotsim.orchestration.pipeline import PlannerPipeline


@dataclass
class PlannerArgs:
    """机器人任务与运动规划器"""
    planner: str = "heuristic"
    """Default planner to use"""
    width: int = 30
    """Grid width"""
    height: int = 30
    """Grid height"""


def main():
    """解析命令行参数并启动完整规划流水线。"""
    args = tyro.cli(PlannerArgs)

    pipeline = PlannerPipeline(
        planner_name=args.planner,
        grid_width=args.width,
        grid_height=args.height,
    )
    output_dir = pipeline.run_full()

    if output_dir:
        print(f"输出目录: {output_dir}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
