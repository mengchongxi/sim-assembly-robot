"""MPPI 四足机器人运动控制入口脚本。

用法:
    # 基础运行 (dog1)
    python scripts/run_mppi_control.py \\
        --config configs/control/mppi/base_trot.yaml \\
        --robot-id dog1

    # 启用历史记忆
    python scripts/run_mppi_control.py \\
        --config configs/control/mppi/base_trot.yaml \\
        --robot-id dog1 --enable-memory

    # 启用速度跟踪
    python scripts/run_mppi_control.py \\
        --config configs/control/mppi/base_trot.yaml \\
        --robot-id dog2 --enable-velocity-tracking

    # 全部启用
    python scripts/run_mppi_control.py \\
        --config configs/control/mppi/base_trot.yaml \\
        --robot-id dog1 --enable-memory --enable-velocity-tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tyro


@dataclass
class MPPIArgs:
    """MPPI 四足机器人运动控制。"""

    config: str
    """YAML 配置文件路径。"""

    robot_id: Literal["dog1", "dog2"]
    """机器人标识。"""

    enable_memory: bool = False
    """启用历史动作记忆采样先验。"""

    enable_velocity_tracking: bool = False
    """启用详细速度跟踪与 CSV 输出。"""


def main(args: MPPIArgs) -> None:
    # 延迟导入（JAX 初始化耗时）
    from robotsim.control.mppi import (
        ConfigManager,
        DIALMPCController,
        DreamerEnv,
        ResultsManager,
        RobotSimulator,
    )
    from robotsim.control.mppi.variants import (
        HistoryMemoryMixin,
        MemoryDreamerEnvMixin,
        VelocityTrackingMixin,
    )

    # 加载配置
    config = ConfigManager.load(
        args.config,
        robot_id=args.robot_id,
        enable_memory=args.enable_memory,
        enable_velocity_tracking=args.enable_velocity_tracking,
    )

    # 构建环境（根据变体选择类）
    if args.enable_memory:
        class MemoryEnv(MemoryDreamerEnvMixin, DreamerEnv):
            """带历史记忆的 DreamerEnv。"""
            pass
        env = MemoryEnv(config)
    else:
        env = DreamerEnv(config)

    # 构建控制器
    if args.enable_memory:
        class MemoryController(HistoryMemoryMixin, DIALMPCController):
            """带历史记忆采样的 MPPI 控制器。"""
            pass
        controller = MemoryController(config, env)
    else:
        controller = DIALMPCController(config, env)

    # 构建结果管理器
    results_manager = ResultsManager(config)

    # 构建仿真器
    if args.enable_velocity_tracking:
        class TrackedSimulator(VelocityTrackingMixin, RobotSimulator):
            """带速度跟踪的仿真器。"""
            pass
        simulator = TrackedSimulator(
            config=config,
            env=env,
            controller=controller,
            results_manager=results_manager,
        )
    else:
        simulator = RobotSimulator(
            config=config,
            env=env,
            controller=controller,
            results_manager=results_manager,
        )

    # 运行仿真
    print(f"配置: robot_id={args.robot_id}, "
          f"memory={args.enable_memory}, "
          f"velocity_tracking={args.enable_velocity_tracking}")
    results = simulator.run_simulation()

    # 保存速度跟踪数据（如果启用）
    if args.enable_velocity_tracking and hasattr(simulator, "save_velocity_tracking"):
        simulator.save_velocity_tracking(config.output_dir)

    print(f"\n仿真完成！共执行了 {len(results['rewards'])} 步")
    print(f"结果保存到 {config.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(MPPIArgs))
