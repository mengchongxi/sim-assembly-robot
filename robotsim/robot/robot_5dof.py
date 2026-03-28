"""入门级 5 自由度机械臂仿真模块。

实现 Robot5DOF 类，封装 PyBullet 中的机械臂加载、关节控制、
贝塞尔弧轨迹生成、附着物品管理和仿真后处理等功能。
"""
import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from robotsim.simulation.bullet.scene_manager import SceneManager
from robotsim.simulation.bullet.attachment import PositionSyncAttachment


class Robot5DOF:
    """5 自由度双足机械臂的 PyBullet 仿真封装。

    提供机器人模型加载、关节控制、轨迹执行以及
    方块拾取/放置等完整仿真接口。
    """
    def __init__(self, urdf_file_path="assembly_bullet/robot_5dof/carry_arm.urdf"):
        self.urdf_file_path = urdf_file_path
        self.robot_id = None
        self.constraint_id = None
        self.angles = [0, 0, 0, 0, 0]
        self.robot_base_index = -1
        self.robot_end_index = 5
        self.gripper_link_index = 6
        self.robot_pos = None
        self.robot_ori = None

        self.scene_manager = SceneManager()
        self.cube_attachment = PositionSyncAttachment()
        self.movement_controller = None
        self.interaction_controller = None

        self.action_sequence = []
        self.arm_indices = [1, 2, 3, 4, 5, 6]
        self.joint_wrist_yaw = 5
        self.joint_wrist_pitch = 6

    def setup_simulation(self):
        """初始化 PyBullet 仿真环境（连接 GUI、关闭重力、加载地面）。"""
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.loadURDF("plane.urdf")

    def load_robot(self, start_pos=[0, 0, 0], start_orientation_quat=[0, 0, 0, 1]):
        """从 URDF 文件加载机器人模型，返回 PyBullet 对象 ID。"""
        self.robot_id = p.loadURDF(self.urdf_file_path, start_pos,
                                 start_orientation_quat, useFixedBase=False)
        return self.robot_id

    def generate_bezier_trajectory(self, start_pos, end_pos, num_points, arc_height=0.0, bulge_direction=np.array([0, 0, 1])):
        """生成从 start_pos 到 end_pos 的三次贝塞尔曲线轨迹，返回坐标列表。"""
        trajectory = []

        p0 = np.array(start_pos)
        p3 = np.array(end_pos)

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

        for i in range(num_points):
            t = i / (num_points - 1)
            point = ((1-t)**3 * p0) + (3 * (1-t)**2 * t * p1) + (3 * (1-t) * t**2 * p2) + (t**3 * p3)
            trajectory.append(point.tolist())

        return trajectory

    def fix_constraint(self, start_pos, parentLinkIndex, start_orientation_quat,
                      constraint_max_force=1000.0):
        """在指定连杆上创建固定约束，将机器人基座固定在世界坐标中。"""
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=parentLinkIndex,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=start_pos,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=start_orientation_quat
        )
        p.changeConstraint(self.constraint_id, start_pos, maxForce=constraint_max_force)

    def remove_constraint(self):
        """移除当前固定约束（如存在）。"""
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id)
            self.constraint_id = None

    # 合并关节控制功能，统一为一个主要接口
    def control_joints_position(self, joint_indices, target_positions, max_forces=None, use_simulation=True):
        """统一的关节位置控制方法。

        use_simulation=True 时使用仿真力控制；False 时直接 resetJointState。
        """
        if max_forces is None:
            max_forces = [100] * len(joint_indices)

        if use_simulation:
            # 使用仿真力控制（POSITION_CONTROL 模式）
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_positions,
                forces=max_forces
            )

            for i in range(20):
                self.cube_attachment.update_positions()
                p.stepSimulation()
                time.sleep(1./10000.)
        else:
            # 使用直接设置（resetJointState，不启动马达）
            if self.robot_id is None:
                print("请先加载机器人！")
                return

            for i, joint_index in enumerate(joint_indices):
                p.resetJointState(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index,
                    targetValue=target_positions[i],
                    targetVelocity=0
                )

            self.cube_attachment.update_positions()
            p.stepSimulation()

    # 保留原有接口以确保兼容性
    def set_joints_position(self, joint_indices, target_positions):
        """兼容性方法：直接设置关节位置（不启用马达）。"""
        self.control_joints_position(joint_indices, target_positions, use_simulation=False)

    def get_joint_states(self, joint_indices=None):
        """返回指定关节（或所有关节）的状态字典列表，每项包含 'position' 键。"""
        if joint_indices is None:
            joint_indices = range(p.getNumJoints(self.robot_id))

        joint_states = []
        for joint_index in joint_indices:
            state = p.getJointState(self.robot_id, joint_index)
            position = state[0]
            joint_states.append({'position': position})
        return joint_states

    def move_joints(self, target):
        """直接设置全部臂关节到 target 角度列表（无仿真步骤）。"""
        self.remove_constraint()
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        self.fix_constraint(base_pos, self.robot_base_index, base_quat)
        p.resetJointState(self.robot_id, 0, 0.0)

        self.set_joints_position(self.arm_indices, target)
        self.remove_constraint()

    def move_front_leg(self,target,leg_length):
        """固定基座，用解析 IK 将前脚移动到目标世界坐标。"""
        self.remove_constraint()
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        self.fix_constraint(base_pos, self.robot_base_index, base_quat)

        target_x,target_y,target_z = target[0],target[1],target[2]

        ik_x = abs(target_z)
        ik_y = math.sqrt(target_x**2 + target_y**2)

        dist = math.sqrt(ik_x**2 + ik_y**2)

        cos_C = (2*leg_length**2 - dist**2) / (2*leg_length**2)
        cos_C = max(-1.0, min(1.0, cos_C))

        C = math.acos(cos_C)

        angle2 = math.pi - C

        if target_z > 0:
            angle3 = math.pi - math.atan2(ik_y,ik_x)-(math.pi/2-C/2)
            angle1 = math.atan2(ik_y,ik_x)-(math.pi/2-C/2)
        elif target_z == 0:
            angle1 = C/2
            angle3 = C/2
        else:
            angle3 = math.atan2(ik_y,ik_x)-(math.pi/2-C/2)
            angle1 = math.pi/2+C/2-math.atan2(ik_y,ik_x)
        angle0 = math.atan2(target_y, target_x)
        angle4 = 0.0
        angle5 = 0.0

        self.control_joints_position(self.arm_indices, [angle0,angle1, angle2, angle3,angle4,angle5])
        self.remove_constraint()

    def move_back_leg(self,target,leg_length):
        """固定末端连杆，用解析 IK 将后脚移动到目标世界坐标。"""
        self.remove_constraint()
        end_state = p.getLinkState(self.robot_id, self.robot_end_index)
        self.fix_constraint(end_state[0], self.robot_end_index, end_state[1])
        target_x,target_y,target_z = target[0],target[1],target[2]

        ik_x = abs(target_z)
        ik_y = math.sqrt(target_x**2 + target_y**2)

        dist = math.sqrt(ik_x**2 + ik_y**2)

        cos_C = (2*leg_length**2 - dist**2) / (2*leg_length**2)
        cos_C = max(-1.0, min(1.0, cos_C))

        C = math.acos(cos_C)
        angle2 = math.pi - C

        if target_z > 0:
            angle1 = math.pi - math.atan2(ik_y,ik_x)-(math.pi/2-C/2)
            angle3 = math.atan2(ik_y,ik_x)-(math.pi/2-C/2)
        elif target_z == 0:
            angle1 = C/2
            angle3 = C/2
        else:
            angle1 = math.atan2(ik_y,ik_x)-(math.pi/2-C/2)
            angle3 = math.pi/2+C/2-math.atan2(ik_y,ik_x)

        angle4 = math.atan2(target_y, target_x)
        angle0 = 0
        angle5 = 0.0

        self.control_joints_position(self.arm_indices, [angle0,angle1, angle2, angle3,angle4,angle5])
        self.remove_constraint()

    def keep_simulation(self):
        """保持仿真循环直到连接断开（可视化结束时退出）。"""
        while p.isConnected():
            try:
                self.cube_attachment.update_positions()
                time.sleep(1./240.)
            except p.error:
                break
        p.disconnect()

    # 简化方块交互，统一通过场景管理器处理
    def pickup_cube(self, cube_position):
        """拾取方块：通过场景管理器删除网格方块并附着到夹爪连杆。"""
        cube_type = self.scene_manager.remove_cube_at_position(cube_position)
        if cube_type is None:
            print(f"[CEO] 在位置 {cube_position} 没找到方块，拾取失败")
            return None, None

        pos = self.get_point_in_link(self.gripper_link_index, [0, 0, 0.13])
        cube_id = self.scene_manager.load_cube(pos, cube_type=cube_type)

        self.cube_attachment.attach_cube(
            cube_id=cube_id,
            robot_id=self.robot_id,
            robot_link_index=self.gripper_link_index,
            offset=[0, 0, 0.13]
        )
        time.sleep(0.2)

        return cube_id, cube_type

    def drop_cube(self, cube_id, cube_type, drop_position):
        """放下方块：分离夹爪附着，通过场景管理器在目标位置重新创建方块。"""
        if self.cube_attachment.is_cube_attached(cube_id):
            self.cube_attachment.detach_cube(cube_id)

        self.scene_manager.remove_cube_by_id(cube_id)
        new_cube_id = self.scene_manager.load_cube(drop_position, cube_type=cube_type)

        print(f"[CEO] 成功协调各部门完成放下：{cube_type} 方块到位置 {drop_position}")
        time.sleep(0.2)
        return new_cube_id

    def get_point_in_link(self, link_index, point_in_link_frame):
        """将连杆局部坐标转换为世界坐标，返回世界坐标。"""
        if self.robot_id is None:
            print("请先加载机器人！")
            return None
        link_state = p.getLinkState(self.robot_id, link_index)
        link_pos, link_orn = link_state[0], link_state[1]
        point_world_pos, _ = p.multiplyTransforms(
            link_pos, link_orn,
            point_in_link_frame, [0, 0, 0, 1]
        )
        return point_world_pos

    def get_point_in_world(self, point_in_end_frame):
        """将末端连杆局部坐标转换为世界坐标，返回 (世界坐标, None)。"""
        if self.robot_id is None:
            print("请先加载机器人！")
            return None, None

        end_state = p.getLinkState(self.robot_id, self.robot_end_index)
        end_pos_world = end_state[0]
        end_orn_world = end_state[1]

        point_world_pos, _ = p.multiplyTransforms(
            end_pos_world, end_orn_world,
            point_in_end_frame, [0, 0, 0, 1]
        )

        return point_world_pos

    def report_final_state(self):
        """打印机器人当前关节角度和末端执行器位置的状态报告。"""
        if self.robot_id is None:
            print("机器人未加载！")
            return

        print("\n=== 机器人最终状态报告 ===")

        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        print(f"基座位置: x={base_pos[0]:.3f}, y={base_pos[1]:.3f}, z={base_pos[2]:.3f}")
        print(f"基座姿态: {base_orn}")

        print("\n关节角度信息:")
        num_joints = p.getNumJoints(self.robot_id)
        joint_angles = []

        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            joint_name = joint_info[1].decode('utf-8')
            joint_state = p.getJointState(self.robot_id, joint_idx)
            current_angle = joint_state[0]
            joint_angles.append(current_angle)

            print(f"  关节 {joint_idx} ({joint_name}): "
                f"{math.degrees(current_angle):7.2f}° ({current_angle:7.4f} 弧度)")

        end_state = p.getLinkState(self.robot_id, self.robot_end_index)
        end_pos = end_state[0]
        print(f"\n末端执行器位置: x={end_pos[0]:.3f}, y={end_pos[1]:.3f}, z={end_pos[2]:.3f}")

    def set_base_pose(self, pos, orn):
        """直接重置机器人基座的位置和姿态。"""
        p.resetBasePositionAndOrientation(self.robot_id, pos, orn)

    def get_arm_positions(self):
        """返回当前臂关节角度列表（弧度）。"""
        return [p.getJointState(self.robot_id, idx)[0] for idx in self.arm_indices]

    def set_arm_positions(self, target_positions, steps=60):
        """插值平滑地将手臂运动到目标关节角度列表。"""
        current = self.get_arm_positions()
        assert len(target_positions) == len(current)
        for s in range(1, steps + 1):
            alpha = s / steps
            interp = [(1 - alpha) * c + alpha * t for c, t in zip(current, target_positions)]
            self.control_joints_position(self.arm_indices, interp)
        self.control_joints_position(self.arm_indices, target_positions)

    def set_wrist(self, yaw=None, pitch=None, duration=0.35, steps=80):
        """用仿真控制平滑地设置腕关节偏航/俯仰角。"""
        base = self.get_arm_positions()
        target = base[:]
        yaw_i = self.arm_indices.index(self.joint_wrist_yaw)
        pit_i = self.arm_indices.index(self.joint_wrist_pitch)
        if yaw is not None:
            target[yaw_i] = yaw
        if pitch is not None:
            target[pit_i] = pitch
        self.set_arm_positions(target, steps=int(steps * max(0.1, duration)))

    def compute_yaw_from_action_name(self, action_name):
        """
        不再通过坐标计算，而是直接根据动作名称（如 'PICKUP_LEFT'）来判断手腕偏航角。
        """
        print(f"   [智能对准] 根据动作名称 '{action_name}' 直接判断方向...")
        if 'LEFT' in action_name:
            print("   -> 判断为左侧，偏航角: +90度")
            return math.pi / 2   # 向左转90度
        elif 'RIGHT' in action_name:
            print("   -> 判断为右侧，偏航角: -90度")
            return -math.pi / 2 # 向右转90度
        else:  # 包含了 'PICKUP_FRONT', 'PLACE_FRONT' 或其他默认情况
            print("   -> 判断为前方/默认，偏航角: 0度")
            return 0.0
    def set_wrist_hard_interp(self, yaw=None, pitch=None, duration=0.25, fps=240):
        """
        用 resetJointState 做线性插值过渡（视觉平滑），仍然不启用马达。
        """
        j_yaw = self.joint_wrist_yaw
        j_pitch = self.joint_wrist_pitch

        curr = self.get_arm_positions()
        yaw_i = self.arm_indices.index(j_yaw)
        pit_i = self.arm_indices.index(j_pitch)
        curr_yaw = curr[yaw_i]
        curr_pit = curr[pit_i]
        des_yaw = curr_yaw if yaw is None else yaw
        des_pit = curr_pit if pitch is None else pitch

        # 关停电机
        p.setJointMotorControlArray(
            self.robot_id, [j_yaw, j_pitch],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0.0, 0.0],
            forces=[0.0, 0.0]
        )

        steps = max(1, int(duration * fps))
        for s in range(1, steps + 1):
            a = s / steps
            y = curr_yaw + a * (des_yaw - curr_yaw)
            pt = curr_pit + a * (des_pit - curr_pit)
            p.resetJointState(self.robot_id, j_yaw, y, targetVelocity=0.0)
            p.resetJointState(self.robot_id, j_pitch, pt, targetVelocity=0.0)
            self.cube_attachment.update_positions()
            p.stepSimulation()
            time.sleep(1.0 / fps)
    def perform_pickup_sequence(self, action_name, world_position, down_angle=-0.7, up_angle=0.0, yaw_left_right=True):
        """执行完整的拾取动作序列：对准腕关节方向、俯身、拾取、抬起。"""
        curr = self.get_arm_positions()
        yaw_i = self.arm_indices.index(self.joint_wrist_yaw)
        pit_i = self.arm_indices.index(self.joint_wrist_pitch)
        curr_yaw, curr_pitch = curr[yaw_i], curr[pit_i]

        yaw_offset = 0.0
        if yaw_left_right:
            # 根据动作名称计算手腕偏航偏移量
            yaw_offset = self.compute_yaw_from_action_name(action_name)

            if abs(yaw_offset) > 1e-3:
                self.set_wrist_hard_interp(yaw=curr_yaw + yaw_offset, pitch=curr_pitch, duration=0.25)

        self.set_wrist_hard_interp(yaw=None, pitch=down_angle, duration=0.25)
        cube_id, cube_type = self.pickup_cube(world_position)
        self.set_wrist(yaw=None, pitch=up_angle, duration=0.25)

        if abs(yaw_offset) > 1e-3:
            self.set_wrist(yaw=curr_yaw, pitch=None, duration=0.25)
        return cube_id, cube_type

    def perform_place_sequence(self, action_name, cube_id, cube_type, world_position, down_angle=-0.7, up_angle=0.0, yaw_left_right=True):
        """执行完整的放置动作序列：对准腕关节方向、俯身、放置、抬起。"""
        curr = self.get_arm_positions()
        yaw_i = self.arm_indices.index(self.joint_wrist_yaw)
        pit_i = self.arm_indices.index(self.joint_wrist_pitch)
        curr_yaw, curr_pitch = curr[yaw_i], curr[pit_i]

        yaw_offset = 0.0
        if yaw_left_right:
            # 根据动作名称计算手腕偏航偏移量
            yaw_offset = self.compute_yaw_from_action_name(action_name)

            if abs(yaw_offset) > 1e-3:
                self.set_wrist_hard_interp(yaw=curr_yaw + yaw_offset, pitch=curr_pitch, duration=0.25)

        self.set_wrist_hard_interp(yaw=None, pitch=down_angle, duration=0.25)
        new_cube_id = self.drop_cube(cube_id, cube_type, world_position)
        self.set_wrist(yaw=None, pitch=up_angle, duration=0.25)

        if abs(yaw_offset) > 1e-3:
            self.set_wrist(yaw=curr_yaw, pitch=None, duration=0.25)
        return new_cube_id
