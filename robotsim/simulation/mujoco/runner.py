"""基于 MuJoCo 的仿真运行器模块。

提供两个独立运行入口：
- run_ik_transport(): 使用 mink IK 解析器实现指定轨迹的末端执行器位置控制。
- run_joint_viewer(): 通过 Tkinter 滑块 GUI 直接控制关节坐标的可视化观察器。
"""
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

from robotsim.simulation.mujoco.viewer import draw_body_frames, _resolve_joints
from robotsim.simulation.mujoco.ik_controller import (
    _generate_sinusoidal_trajectory, _set_mocap_position, _build_velocity_limits
)
from robotsim.gui.trajectory_gui import TargetTrajectoryGUI
from robotsim.gui.joint_gui import JointGUI


def run_ik_transport():
    """启动基于 mink IK 的轨迹末端控制演示。

    加载 MuJoCo fixback 模型，在后台线程启动 TargetTrajectoryGUI。
    GUI 中提交的轨迹请求经正弦插值后逐步驱动 mocap 目标体，
    mink 计算关节速度满足组合限制，200 Hz 循环运行。
    """
    _HERE = Path(__file__).parent
    _XML = _HERE / "fixback_model" / "fixback.xml"

    _END_EFFECTOR_BODY = "end_link"
    _TARGET_MOCAP_BODY = "target"
    _CONTROL_FREQUENCY = 200.0

    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name=_END_EFFECTOR_BODY,
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        ),
        posture_task := mink.PostureTask(model, cost=1e-3),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        _build_velocity_limits(model),
    ]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # 从当前 qpos（通常全为零）开始
        configuration.update(data.qpos)
        posture_task.set_target(configuration.q)

        # 将 mocap 目标初始化到当前末端执行器姿态，避免跳跃
        mink.move_mocap_to_frame(
            model, data, _TARGET_MOCAP_BODY, _END_EFFECTOR_BODY, "body"
        )

        gui = TargetTrajectoryGUI()
        gui.start()

        trajectory = None
        trajectory_index = 0

        rate = RateLimiter(frequency=_CONTROL_FREQUENCY, warn=False)
        while viewer.is_running():
            request = gui.consume_request()
            if request is not None:
                if request[0] == "trajectory":
                    _, start_xyz, end_xyz, duration_s, arc_height = request
                    trajectory = _generate_sinusoidal_trajectory(
                        start_xyz=start_xyz,
                        end_xyz=end_xyz,
                        duration_s=duration_s,
                        dt=rate.dt,
                        arc_height=arc_height,
                    )
                else:
                    _, target_xyz, duration_s = request
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _TARGET_MOCAP_BODY)
                    mocap_id = model.body_mocapid[body_id]
                    current_xyz = np.array(data.mocap_pos[mocap_id], dtype=float)
                    trajectory = _generate_sinusoidal_trajectory(
                        start_xyz=current_xyz,
                        end_xyz=target_xyz,
                        duration_s=duration_s,
                        dt=rate.dt,
                        arc_height=0.0,
                    )
                trajectory_index = 0

            if trajectory is not None and trajectory_index < len(trajectory):
                _set_mocap_position(
                    model,
                    data,
                    _TARGET_MOCAP_BODY,
                    trajectory[trajectory_index],
                )
                trajectory_index += 1

            T_wt = mink.SE3.from_mocap_name(model, data, _TARGET_MOCAP_BODY)
            end_effector_task.set_target(T_wt)

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

            mujoco.mj_camlight(model, data)
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)
            draw_body_frames(model, data, viewer)

            viewer.sync()
            rate.sleep()


def run_joint_viewer():
    """启动直接滑块控制关节角度的 MuJoCo 可视化观察器。

    加载 MuJoCo fixfront 模型，自动解析所有铰链/滑动关节并构建
    JointGUI。GUI 滑块的用户输入直接写入 data.qpos，200 Hz 循环运行。
    """
    _HERE = Path(__file__).parent
    _XML = _HERE / "fixfront_model" / "fixfront.xml"
    # _XML = _HERE / "fixback_model" / "fixback.xml"
    # _XML = _HERE / "model" / "transport.xml"
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    configuration = mink.Configuration(model)
    model = configuration.model
    data = configuration.data

    resolved_names, joint_ids = _resolve_joints(model)
    if not joint_ids:
        raise RuntimeError("No controllable hinge/slide joints found in model.")
    print(f"[viewer_mj] Auto-resolved joints ({len(resolved_names)}): {resolved_names}")

    # 从模型中收集各关节的角度范围
    joint_limits = []
    for jid in joint_ids:
        lo = model.jnt_range[jid, 0]
        hi = model.jnt_range[jid, 1]
        joint_limits.append((lo, hi))

    # 在后台线程中启动 GUI
    gui = JointGUI(resolved_names, joint_limits)
    gui.start()

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # 将 GUI 滑块角度写入 qpos
            angles = gui.get_angles()
            for jid, angle in zip(joint_ids, angles):
                qadr = model.jnt_qposadr[jid]
                data.qpos[qadr] = angle

            mujoco.mj_camlight(model, data)
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)
            draw_body_frames(model, data, viewer)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    run_ik_transport()
