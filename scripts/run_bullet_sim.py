"""PyBullet 机器人轨迹物理仿真入口脚本。"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import time
import numpy as np
import pybullet as p

from robotsim.robot.robot_5dof import Robot5DOF
from robotsim.simulation.bullet.movement_controller import MovementController
from robotsim.simulation.bullet.interaction_controller import InteractionController


def main():
    """加载场景配置，初始化机器人，按顺序执行移动与交互动作序列。"""
    robot = Robot5DOF()
    robot.setup_simulation()
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    scene_info = robot.scene_manager.load_scene_from_yaml("config/robot_trajectory.yaml")

    if scene_info is None:
        print("场景加载失败，程序退出")
        exit()

    robot.robot_pos = np.array(scene_info['robot_position'][:2])
    robot.robot_ori = np.array(scene_info['robot_orientation'])
    robot.action_sequence = scene_info['action_sequence']

    init_pos = scene_info['robot_position']
    init_ori = scene_info['robot_orientation']

    robot.load_robot(init_pos, init_ori)

    leg_length = 0.13
    cube_radius = 0.06
    init_joint = [0, math.asin(cube_radius/leg_length),
                  math.pi-math.asin(cube_radius/leg_length)*2,
                  math.asin(cube_radius/leg_length), 0, 0]
    robot.move_joints(init_joint)
    time.sleep(1)

    robot.movement_controller = MovementController(robot, leg_length, init_joint)
    robot.interaction_controller = InteractionController(robot)

    p.resetDebugVisualizerCamera(
        cameraDistance=1,
        cameraYaw=0,
        cameraPitch=-45,
        cameraTargetPosition=[1, 1, 1]
    )

    for i, action_item in enumerate(robot.action_sequence):
        robot.report_final_state()

        if action_item.get('type') == 'MOVE':
            robot.movement_controller.execute(
                action_name=action_item['action'],
                target_pose=action_item.get('target_pose')
            )

        elif action_item.get('type') in ['PICKUP', 'PLACE']:
            robot.interaction_controller.execute(action_item)

        else:
            print(f"[警告] 未知动作项: {action_item}")

    robot.report_final_state()
    robot.keep_simulation()


if __name__ == "__main__":
    main()
