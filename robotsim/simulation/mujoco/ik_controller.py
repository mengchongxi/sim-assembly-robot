"""MuJoCo IK 控制辅助函数模块。

提供三个内部辅助函数，供 runner.py 中的 IK 传输演示使用：
- _generate_sinusoidal_trajectory(): 生成正弦插值轨迹数组。
- _set_mocap_position(): 按名称设置 mocap 体的世界位置。
- _build_velocity_limits(): 为模型所有铰链关节构建速度限制。
"""
import mujoco
import numpy as np
import mink


def _generate_sinusoidal_trajectory(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    duration_s: float,
    dt: float,
    arc_height: float,
) -> np.ndarray:
    """在基坐标系中生成从起点到终点的平滑正弦弧轨迹点数组。

    Z 轴弧高叠加公式：z_bump = arc_height * sin(π * α)。
    """
    num_samples = max(2, int(duration_s / dt))
    alpha = np.linspace(0.0, 1.0, num_samples)
    blend = 0.5 - 0.5 * np.cos(np.pi * alpha)
    trajectory = start_xyz[None, :] + blend[:, None] * (end_xyz - start_xyz)[None, :]
    trajectory[:, 2] += arc_height * np.sin(np.pi * alpha)
    return trajectory


def _set_mocap_position(
    model: mujoco.MjModel, data: mujoco.MjData, mocap_body: str, target_pos: np.ndarray
) -> None:
    """按名称将 mocap 体的世界位置设置为目标坐标。"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body)
    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        raise ValueError(f"Body '{mocap_body}' is not a mocap body.")
    data.mocap_pos[mocap_id] = target_pos


def _build_velocity_limits(model: mujoco.MjModel) -> mink.VelocityLimit:
    """为所有铰链关节应用统一速度限制并返回 VelocityLimit 对象。"""
    max_velocities = {}
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname:
            max_velocities[jname] = np.pi

    if not max_velocities:
        raise ValueError("No hinge joints found for IK velocity limits.")

    return mink.VelocityLimit(model, max_velocities)
