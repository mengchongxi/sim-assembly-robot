"""基于 PyBullet 的机器人移动控制器模块。

实现 MovementController 类，经由轨迹生成器和关节位置控制
实现双足机器人的前进、后退、左转、右转和 180 度转向等动作。
"""


class MovementController:
    """双足机器人运动序列控制器。

    封装所有移动动作（前进、后退、90° 旋转、180° 旋转）的
    贝塞尔轨迹生成和关节驱动逻辑。

    Args:
        robot: Robot5DOF 实例。
        leg_length: 机器人腿长（米）。
        init_joint: 初始关节角度列表。
    """
    def __init__(self, robot, leg_length, init_joint):
        self.robot = robot
        self.leg_length = leg_length
        self.init_joint = init_joint

        self.action_map = {
            'STEP_FORWARD': self.front_action,
            'STEP_BACKWARD': self.back_action,
            'ROTATE_BACK_LEFT': self.fix_back_left_rot_90,
            'ROTATE_BACK_RIGHT': self.fix_back_right_rot_90,
            'ROTATE_FRONT_LEFT': self.fix_front_left_rot_90,
            'ROTATE_FRONT_RIGHT': self.fix_front_right_rot_90,
            'ROTATE_BACK_180': self.fix_back_rot_180,
            'ROTATE_FRONT_180': self.fix_front_rot_180,
        }

    def execute(self, action_name, target_pose=None):
        """执行指定名称的移动动作，完成后可选地对齐机器人基座姿态。"""
        action_to_perform = self.action_map.get(action_name)
        if action_to_perform:
            print(f"   [运动控制器] 正在执行: {action_name}")
            action_to_perform()
            if target_pose is not None:
                self.robot.set_base_pose(
                    target_pose['position'],
                    target_pose['orientation']
                )
        else:
            print(f"   [警告] 运动控制器无法识别动作: {action_name}")

    # 统一轨迹生成，使用标准参数
    def _generate_standard_trajectory(self, start_pos, end_pos):
        """统一的标准贝塞尔轨迹生成方法（弧高 0.3 m，方向 Z 轴正方向）。"""
        return self.robot.generate_bezier_trajectory(
            start_pos, end_pos, num_points=40,
            arc_height=0.3, bulge_direction=[0,0,1]
        )

    def front_action(self):
        """执行向前迈步动作：前脚抬起前移，后脚跟进。"""
        trajectory1 = self._generate_standard_trajectory([0.12,0,0], [0.24,0,0])
        for point in trajectory1:
            self.robot.move_front_leg(point, self.leg_length)

        trajectory2 = self._generate_standard_trajectory([0.24,0,0], [0.12,0,0])
        for point in trajectory2:
            self.robot.move_back_leg(point, self.leg_length)

        self.robot.move_joints(self.init_joint)

    def back_action(self):
        """执行向后退步动作：后脚抬起后移，前脚跟进。"""
        trajectory1 = self._generate_standard_trajectory([0.12,0,0], [0.24,0,0])
        for point in trajectory1:
            self.robot.move_back_leg(point, self.leg_length)

        trajectory2 = self._generate_standard_trajectory([0.24,0,0], [0.12,0,0])
        for point in trajectory2:
            self.robot.move_front_leg(point, self.leg_length)

        self.robot.move_joints(self.init_joint)

    def fix_back_left_rot_90(self):
        """固定后脚，前脚向左旋转 90°。"""
        trajectory1 = self._generate_standard_trajectory([0.12,0,0], [0,0.12,0])
        for point in trajectory1:
            self.robot.move_front_leg(point, self.leg_length)

        trajectory2 = self._generate_standard_trajectory([0.12,0,0], [0.12,0,0])
        for point in trajectory2:
            self.robot.move_back_leg(point, self.leg_length)

        self.robot.move_joints(self.init_joint)

    def fix_back_right_rot_90(self):
        """固定后脚，前脚向右旋转 90°。"""
        trajectory1 = self._generate_standard_trajectory([0.12,0,0], [0,-0.12,0])
        for point in trajectory1:
            self.robot.move_front_leg(point, self.leg_length)

        trajectory2 = self._generate_standard_trajectory([0.12,0,0], [0.12,0,0])
        for point in trajectory2:
            self.robot.move_back_leg(point, self.leg_length)

        self.robot.move_joints(self.init_joint)

    def fix_back_rot_180(self):
        """固定后脚，前脚旋转 180°（经左侧两步完成）。"""
        trajectory1 = self._generate_standard_trajectory([0.12,0,0], [0,0.12,0])
        for point in trajectory1:
            self.robot.move_front_leg(point, self.leg_length)

        trajectory2 = self._generate_standard_trajectory([0,0.12,0], [-0.12,0,0])
        for point in trajectory2:
            self.robot.move_front_leg(point, self.leg_length)

        trajectory3 = self._generate_standard_trajectory([0.12,0,0], [0.12,0,0])
        for point in trajectory3:
            self.robot.move_back_leg(point, self.leg_length)

        self.robot.move_joints(self.init_joint)

    def fix_front_left_rot_90(self):
        """固定前脚，后脚向左旋转 90°。"""
        trajectory1 = self._generate_standard_trajectory([0.12,0,0], [0,0.12,0])
        for point in trajectory1:
            self.robot.move_back_leg(point, self.leg_length)

        trajectory2 = self._generate_standard_trajectory([0.12,0,0], [0.12,0,0])
        for point in trajectory2:
            self.robot.move_front_leg(point, self.leg_length)

        self.robot.move_joints(self.init_joint)

    def fix_front_right_rot_90(self):
        """固定前脚，后脚向右旋转 90°。"""
        trajectory1 = self._generate_standard_trajectory([0.12,0,0], [0,-0.12,0])
        for point in trajectory1:
            self.robot.move_back_leg(point, self.leg_length)

        trajectory2 = self._generate_standard_trajectory([0.12,0,0], [0.12,0,0])
        for point in trajectory2:
            self.robot.move_front_leg(point, self.leg_length)

        self.robot.move_joints(self.init_joint)

    def fix_front_rot_180(self):
        """固定前脚，后脚旋转 180°（经左侧两步完成）。"""
        trajectory1 = self._generate_standard_trajectory([0.12,0,0], [0,0.12,0])
        for point in trajectory1:
            self.robot.move_back_leg(point, self.leg_length)

        trajectory2 = self._generate_standard_trajectory([0,0.12,0], [-0.12,0,0])
        for point in trajectory2:
            self.robot.move_back_leg(point, self.leg_length)

        trajectory3 = self._generate_standard_trajectory([0.12,0,0], [0.12,0,0])
        for point in trajectory3:
            self.robot.move_front_leg(point, self.leg_length)

        self.robot.move_joints(self.init_joint)
