"""MPPI 仿真结果收集与保存。"""

import csv
import os
import time

import numpy as np

from robotsim.control.mppi.config import Config


class ResultsManager:
    """仿真结果的收集、保存和可视化管理器。"""

    def __init__(self, config: Config):
        self.config = config

    def save_results(self, pose_data, joint_data, body_vel_data, body_ang_vel_data):
        """保存仿真结果（位姿、关节、速度数据）。

        Returns:
            timestamp 字符串，用于关联同一次仿真的所有输出文件。
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        self._save_command_data(pose_data, joint_data, timestamp)
        self._save_velocity_csv(body_vel_data, body_ang_vel_data, timestamp)

        return timestamp

    def _save_command_data(self, pose_data, joint_data, timestamp):
        """保存位姿和关节角数据到 command.txt。"""
        print("Saving pose and joint angle data to command.txt")
        command_file_path = os.path.join(self.config.output_dir, "command.txt")

        with open(command_file_path, "w") as f:
            f.write("# 数据格式：每行包含一个时间步的完整状态信息\n")
            f.write("# 列说明：\n")
            f.write("#   1-3:   base_pos (x, y, z) - 机器人基座位置 [m]\n")
            f.write("#   4-7:   base_quat (x, y, z, w) - 机器人基座姿态四元数\n")
            f.write("#   8-19:  joint_angles - 12个关节角度 [rad]\n")
            f.write(f"# 总步数: {len(pose_data)}\n")
            f.write(f"# 时间步长: {self.config.dt} 秒\n")
            f.write(f"# 生成时间: {timestamp}\n")
            f.write("# " + "=" * 60 + "\n")

            for i in range(len(pose_data)):
                combined_data = np.concatenate([pose_data[i], joint_data[i]])
                line = " ".join([f"{val:8.6f}" for val in combined_data])
                f.write(f"{line}\n")

        print(f"数据保存完成！文件路径: {command_file_path}")
        print(f"共保存了 {len(pose_data)} 个时间步的数据")

    def _save_velocity_csv(self, body_vel_data, body_ang_vel_data, timestamp):
        """保存机体线速度和角速度到 CSV 文件。"""
        csv_file_path = os.path.join(
            self.config.output_dir, f"{timestamp}_velocity.csv"
        )
        print("Saving body velocity and angular velocity data to CSV...")

        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "step",
                "body_lin_vel_x", "body_lin_vel_y", "body_lin_vel_z",
                "body_ang_vel_x", "body_ang_vel_y", "body_ang_vel_z",
            ])

            for i in range(len(body_vel_data)):
                row = (
                    [i]
                    + body_vel_data[i].tolist()
                    + body_ang_vel_data[i].tolist()
                )
                writer.writerow(row)

        print(f"速度数据保存完成！文件路径: {csv_file_path}")
        print(f"共保存了 {len(body_vel_data)} 个时间步的数据")

    def save_visualization_html(self, env, rollout, timestamp):
        """生成并保存 Brax 可视化 HTML 文件。"""
        from brax.io import html

        print("Processing rollout for visualization and saving to HTML...")
        webpage = html.render(
            env.sys.tree_replace({"opt.timestep": env.dt}),
            rollout,
            1080,
            True,
        )

        output_path = os.path.join(
            self.config.output_dir,
            f"{timestamp}_brax_visualization.html",
        )
        with open(output_path, "w") as f:
            f.write(webpage)
        print(f"Visualization saved to: {output_path}")
