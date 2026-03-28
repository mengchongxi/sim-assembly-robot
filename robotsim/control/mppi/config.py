"""MPPI 控制算法配置管理。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import yaml


@dataclass
class RobotConfig:
    """机器人特定参数配置。

    不同机器人（如 dog1、dog2）通过此配置区分模型路径、关节范围等差异。
    """
    model_path: str = "models/mujoco/icra2026_model/dog_1_dog.xml"
    pos_tar_z: float = 0.45
    foot_radius: float = 0.0175
    torso_body_name: str = "module1"
    feet_site_names: List[str] = field(
        default_factory=lambda: ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    )
    joint_range_override: Optional[List[List[float]]] = None
    gait_params_override: Optional[Dict[str, List[float]]] = None


@dataclass
class Config:
    """MPPI 算法统一配置类。"""

    # 实验基础设置
    seed: int = 0
    output_dir: str = "output/mppi"
    n_steps: int = 800

    # DIAL-MPC 算法参数
    Nsample: int = 2024
    Hsample: int = 16
    Hnode: int = 4
    Ndiffuse: int = 3
    Ndiffuse_init: int = 10
    temp_sample: float = 0.05
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5
    update_method: str = "mppi"

    # 环境物理参数
    dt: float = 0.02
    timestep: float = 0.02
    leg_control: str = "torque"
    action_scale: float = 1.0

    # PD 控制参数
    kp: float = 100.0
    kd: float = 2.0

    # 运动目标参数
    default_vx: float = 0.8
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 1.0
    gait: str = "trot"

    # 调试
    debug: bool = False
    randomize_tasks: bool = False

    # 机器人特定配置
    robot: RobotConfig = field(default_factory=RobotConfig)

    # 变体功能开关
    enable_memory: bool = False
    history_len: int = 64
    enable_velocity_tracking: bool = False


class ConfigManager:
    """配置管理器，支持 YAML 加载和运行时覆盖。"""

    @staticmethod
    def load(yaml_path: str, **overrides) -> Config:
        """从 YAML 文件加载配置并应用覆盖值。

        Args:
            yaml_path: YAML 配置文件路径。
            **overrides: 运行时覆盖的键值对，支持:
                - robot_id: 快捷设置 output_dir 中的机器人标识
                - enable_memory: 启用历史记忆
                - enable_velocity_tracking: 启用速度跟踪
                - 其他 Config 字段名
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 提取 robot 子配置
        robot_data = data.pop("robot", {})
        robot_config = ConfigManager._build_robot_config(robot_data)

        # 过滤 Config 级别的有效字段
        config_fields = {f for f in Config.__dataclass_fields__ if f != "robot"}
        filtered = {k: v for k, v in data.items() if k in config_fields}

        # 处理 robot_id 快捷覆盖
        robot_id = overrides.pop("robot_id", None)
        if robot_id and "output_dir" in filtered:
            filtered["output_dir"] = filtered["output_dir"].replace(
                "{robot_id}", robot_id
            )

        # 应用其他覆盖
        for key, value in overrides.items():
            if key in config_fields:
                filtered[key] = value

        return Config(robot=robot_config, **filtered)

    @staticmethod
    def _build_robot_config(data: dict) -> RobotConfig:
        """从字典构建 RobotConfig。"""
        if not data:
            return RobotConfig()
        valid_keys = RobotConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return RobotConfig(**filtered)
