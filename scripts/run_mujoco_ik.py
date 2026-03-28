"""MuJoCo 逆运动学（IK）传输仿真入口脚本。"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

from robotsim.simulation.mujoco.viewer import draw_body_frames
from robotsim.simulation.mujoco.ik_controller import (
    _generate_sinusoidal_trajectory, _set_mocap_position, _build_velocity_limits
)
from robotsim.gui.trajectory_gui import TargetTrajectoryGUI


_HERE = Path(__file__).parent.parent / "models"
_XML = _HERE / "mujoco" / "fixback_model" / "fixback.xml"
_END_EFFECTOR_BODY = "end_link"
_TARGET_MOCAP_BODY = "target"
_CONTROL_FREQUENCY = 200.0


def main():
    """加载 MuJoCo 模型，配置 IK 任务与约束，通过 GUI 轨迹请求驱动末端执行器运动。"""
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
        configuration.update(data.qpos)
        posture_task.set_target(configuration.q)
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
                    model, data, _TARGET_MOCAP_BODY, trajectory[trajectory_index],
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

        gui.stop()


if __name__ == "__main__":
    main()
