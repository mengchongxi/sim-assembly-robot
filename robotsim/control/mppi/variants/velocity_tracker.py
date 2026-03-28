"""速度跟踪 Mixin，为 MPPI 仿真添加详细的速度记录与分析能力。

记录全局/机体坐标系的线速度和角速度、目标速度和误差，
通过 pandas DataFrame 输出到 CSV 并打印统计信息。

用法:
    class TrackedSimulator(VelocityTrackingMixin, RobotSimulator):
        pass
"""

import os
import time

from jax import numpy as jnp
import numpy as np
import pandas as pd

from robotsim.control.mppi.math_utils import MathUtils


class VelocityTrackingMixin:
    """为 RobotSimulator 添加详细速度跟踪的 Mixin。

    覆盖 _record_data 方法，记录 24 个速度相关字段：
    - 6 个全局坐标系速度 (global_vx/vy/vz, global_wx/wy/wz)
    - 6 个机体坐标系速度 (body_vx/vy/vz, body_wx/wy/wz)
    - 6 个目标速度 (target_vx/vy/vz, target_wx/wy/wz)
    - 6 个速度误差 (error_vx/vy/vz, error_wx/wy/wz)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._velocity_math = MathUtils()
        self._velocity_data = []

    def _record_data(self, state, pose_data, joint_data, body_vel_data, body_ang_vel_data):
        """扩展数据记录，添加详细速度跟踪。"""
        # 调用基类记录位姿和基础速度
        super()._record_data(state, pose_data, joint_data, body_vel_data, body_ang_vel_data)

        # 详细速度记录
        x, xd = state.pipeline_state.x, state.pipeline_state.xd
        torso_idx = self.env._torso_idx - 1

        global_linear_vel = xd.vel[torso_idx]
        global_angular_vel = xd.ang[torso_idx] * jnp.pi / 180.0

        body_linear_vel = self._velocity_math.global_to_body_velocity(
            global_linear_vel, x.rot[torso_idx]
        )
        body_angular_vel = self._velocity_math.global_to_body_velocity(
            global_angular_vel, x.rot[torso_idx]
        )

        target_linear_vel = state.info["vel_tar"]
        target_angular_vel = state.info["ang_vel_tar"]

        step = len(self._velocity_data)
        velocity_record = {
            "time": step * self.config.dt,
            "step": step,
            # 世界坐标系速度
            "global_vx": float(global_linear_vel[0]),
            "global_vy": float(global_linear_vel[1]),
            "global_vz": float(global_linear_vel[2]),
            "global_wx": float(global_angular_vel[0]),
            "global_wy": float(global_angular_vel[1]),
            "global_wz": float(global_angular_vel[2]),
            # 机体坐标系速度
            "body_vx": float(body_linear_vel[0]),
            "body_vy": float(body_linear_vel[1]),
            "body_vz": float(body_linear_vel[2]),
            "body_wx": float(body_angular_vel[0]),
            "body_wy": float(body_angular_vel[1]),
            "body_wz": float(body_angular_vel[2]),
            # 目标速度
            "target_vx": float(target_linear_vel[0]),
            "target_vy": float(target_linear_vel[1]),
            "target_vz": float(target_linear_vel[2]),
            "target_wx": float(target_angular_vel[0]),
            "target_wy": float(target_angular_vel[1]),
            "target_wz": float(target_angular_vel[2]),
            # 速度误差
            "error_vx": float(body_linear_vel[0] - target_linear_vel[0]),
            "error_vy": float(body_linear_vel[1] - target_linear_vel[1]),
            "error_vz": float(body_linear_vel[2] - target_linear_vel[2]),
            "error_wx": float(body_angular_vel[0] - target_angular_vel[0]),
            "error_wy": float(body_angular_vel[1] - target_angular_vel[1]),
            "error_wz": float(body_angular_vel[2] - target_angular_vel[2]),
        }
        self._velocity_data.append(velocity_record)

    def save_velocity_tracking(self, output_dir: str):
        """将详细速度数据保存到 CSV（使用 pandas）。"""
        if not self._velocity_data:
            return

        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        df = pd.DataFrame(self._velocity_data)
        csv_path = os.path.join(
            output_dir, f"{timestamp}_robot_velocities.csv"
        )
        df.to_csv(csv_path, index=False, float_format="%.6f")

        print(f"速度跟踪数据保存完成！文件路径: {csv_path}")
        print(f"共保存了 {len(self._velocity_data)} 个时间步的速度数据")

        print("\n速度数据统计:")
        for col in df.columns:
            if col not in ("time", "step"):
                print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
