"""机器人在 PyBullet 环境中的拾取/放置交互控制器模块。

实现 InteractionController 类，封装模块的拾取和放置动作执行逻辑。
"""
import time

from robotsim.utils.grid_utils import GRID_CELL_SIZE


class InteractionController:
    """执行机器人拾取和放置动作的交互控制器。

    封装 PICKUP 和 PLACE 指令的具体属物理序列，并维护当前
    手持方块的状态信息。

    Args:
        robot: Robot5DOF 实例，用于调用属物理动作序列。
    """
    def __init__(self, robot):
        self.robot = robot
        self.held_cube_info = None

    def _handle_pickup(self, action_info):
        """执行 PICKUP 指令：将指定网格位置的方块拾取到夹爪。"""
        if self.held_cube_info is not None:
            print("   [警告] 交互控制器检测到机器人手上已经有东西了，但又接到了PICKUP指令！")
            return

        # 从 action_info 中获取需要的所有信息
        action_name = action_info['action']
        grid_position = action_info['position']
        world_x = round(grid_position[0] * GRID_CELL_SIZE, 2)
        world_y = round(grid_position[1] * GRID_CELL_SIZE, 2)
        world_position = [world_x, world_y, 0.0]

        time.sleep(0.2)
        cube_id, cube_type = self.robot.perform_pickup_sequence(
            action_name=action_name,  # 【修改点】把动作名称传下去
            world_position=world_position,
            down_angle=1.57,
            up_angle=0.0,
            yaw_left_right=True
        )
        if cube_id is not None:
            self.held_cube_info = {'id': cube_id, 'type': cube_type}
            print(f"   成功拿起一个 {cube_type} 方块，已记录在交互控制器中。")
        else:
            print(f"   [错误] 捡起失败！")

    def _handle_place(self, action_info):
        """执行 PLACE 指令：将夹爪中的方块放置到指定网格位置。"""
        if self.held_cube_info is None:
            print("   [错误] 交互控制器检测到机器人手上没东西，无法执行PLACE指令！")
            return

        # 从 action_info 中获取需要的所有信息
        action_name = action_info['action']
        grid_position = action_info['position']
        world_x = round(grid_position[0] * GRID_CELL_SIZE, 2)
        world_y = round(grid_position[1] * GRID_CELL_SIZE, 2)
        world_position = [world_x, world_y, 0.0]

        time.sleep(0.2)
        new_cube_id = self.robot.perform_place_sequence(
            action_name=action_name,  # 【修改点】把动作名称传下去
            cube_id=self.held_cube_info['id'],
            cube_type=self.held_cube_info['type'],
            world_position=world_position,
            down_angle=1.57,
            up_angle=0.0,
            yaw_left_right=True
        )
        self.held_cube_info = None
        print(f"   成功放下了一个方块，交互控制器已将手部状态清空。")

    def execute(self, action_info):
        """根据 action_info 中的 type 字段分发到拾取或放置处理函数。

        如果 action_info 包含 'target_pose'，操作完成后调用
        robot.set_base_pose() 对齐机器人的逻辑位置。
        """
        action_type = action_info.get('type')
        target_pose = action_info.get('target_pose')

        print(f"   [交互控制器] 正在执行: {action_type}，动作: {action_info['action']}")

        if action_type == 'PICKUP':
            # 将完整 action_info 字典传给处理函数
            self._handle_pickup(action_info)
        elif action_type == 'PLACE':
            # 将完整 action_info 字典传给处理函数
            self._handle_place(action_info)
        else:
            print(f"   [警告] 交互控制器无法识别动作类型: {action_type}")

        if target_pose is not None:
            self.robot.set_base_pose(
                target_pose['position'],
                target_pose['orientation']
            )
