"""机器人足部轨迹生成器模块。

实现两种三维轨迹生成器：
- BezierTrajectoryGenerator: 三次贝塞尔曲线轨迹，支持自定义突出方向。
- SinusoidalTrajectoryGenerator: 正弦弧轨迹，具有平滑的加速和减速特性。
"""
import numpy as np


class BezierTrajectoryGenerator:
    """在两个三维位置之间生成贝塞尔曲线轨迹。"""

    def generate(self, start: np.ndarray, end: np.ndarray,
                 num_points: int, arc_height: float = 0.0,
                 bulge_direction: np.ndarray = None) -> np.ndarray:
        """生成从 start 到 end 的三次贝塞尔曲线轨迹点数组。

        Args:
            start: 起始位置（三维数组）。
            end: 终止位置（三维数组）。
            num_points: 采样点数。
            arc_height: 弧高（控制点偏移量）。
            bulge_direction: 突出方向单位向量，默认为 Z 轴正方向。

        Returns:
            形状为 (num_points, 3) 的轨迹点数组。
        """
        if bulge_direction is None:
            bulge_direction = np.array([0, 0, 1])

        p0 = np.array(start)
        p3 = np.array(end)

        v = p3 - p0
        p1_on_line = p0 + v / 3.0
        p2_on_line = p0 + 2.0 * v / 3.0

        bulge_direction = np.array(bulge_direction)
        norm = np.linalg.norm(bulge_direction)
        if norm > 1e-6:
            offset_vector = (bulge_direction / norm) * arc_height
        else:
            offset_vector = np.array([0.0, 0.0, 0.0])

        p1 = p1_on_line + offset_vector
        p2 = p2_on_line + offset_vector

        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1)
            point = ((1-t)**3 * p0) + (3 * (1-t)**2 * t * p1) + (3 * (1-t) * t**2 * p2) + (t**3 * p3)
            trajectory.append(point)

        return np.array(trajectory)


class SinusoidalTrajectoryGenerator:
    """在两个三维位置之间生成平滑正弦弧轨迹。"""

    def generate(self, start: np.ndarray, end: np.ndarray,
                 num_points: int, arc_height: float = 0.0) -> np.ndarray:
        """生成从 start 到 end 的正弦插值轨迹点数组。

        Args:
            start: 起始位置（三维数组）。
            end: 终止位置（三维数组）。
            num_points: 采样点数。
            arc_height: Z 轴弧高，使用 sin(π·α) 叠加。

        Returns:
            形状为 (num_points, 3) 的轨迹点数组。
        """
        start = np.array(start)
        end = np.array(end)

        alpha = np.linspace(0.0, 1.0, num_points)
        blend = 0.5 - 0.5 * np.cos(np.pi * alpha)
        trajectory = start[None, :] + blend[:, None] * (end - start)[None, :]
        trajectory[:, 2] += arc_height * np.sin(np.pi * alpha)
        return trajectory
