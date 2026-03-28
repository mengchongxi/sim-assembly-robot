"""方块与机器人连杆的位置同步附着模块。

提供 PositionSyncAttachment 类，将一个或多个方块附着到机器人的指定连杆，
使其在每个仿真步内保持与连杆的相对位置不变。
"""
import pybullet as p


class PositionSyncAttachment:
    """基于位置同步的方块附着器。

    每个仿真步调用 update_positions() 时，所有附着的方块会
    跟随其附着连杆的世界坐标实时移动。
    """
    def __init__(self):
        self.attached_cubes = {}

    def attach_cube(self, cube_id, robot_id, robot_link_index, offset=[0, 0, 0.1]):
        """将方块附着到机器人的指定连杆。

        Args:
            cube_id: PyBullet 方块对象 ID。
            robot_id: PyBullet 机器人对象 ID。
            robot_link_index: 附着目标连杆的索引。
            offset: 方块相对连杆原点的偏移量 [x, y, z]。
        """
        self.attached_cubes[cube_id] = {
            'robot_id': robot_id,
            'robot_link_index': robot_link_index,
            'offset': offset
        }
        print(f"[位置同步] 方块 {cube_id} 已附着到机器人连杆 {robot_link_index}")

    def detach_cube(self, cube_id):
        """将指定方块从附着列表中移除。"""
        if cube_id in self.attached_cubes:
            del self.attached_cubes[cube_id]
            print(f"[位置同步] 方块 {cube_id} 已分离")

    def detach_all_cubes(self):
        """移除所有已附着的方块。"""
        cube_ids = list(self.attached_cubes.keys())
        for cube_id in cube_ids:
            self.detach_cube(cube_id)

    def update_positions(self):
        """调用一次博个附着方块的位置和姿态同步更新，应在每个仿真步内调用。"""
        for cube_id, info in self.attached_cubes.items():
            try:
                link_state = p.getLinkState(info['robot_id'], info['robot_link_index'])
                link_pos = link_state[0]
                link_orn = link_state[1]

                cube_world_pos, cube_world_orn = p.multiplyTransforms(
                    link_pos, link_orn,
                    info['offset'], [0, 0, 0, 1]
                )

                p.resetBasePositionAndOrientation(cube_id, cube_world_pos, cube_world_orn)

            except Exception as e:
                print(f"更新方块 {cube_id} 位置时出错: {e}")

    def get_attached_cubes(self):
        """返回所有已附着方块的 ID 列表。"""
        return list(self.attached_cubes.keys())

    def is_cube_attached(self, cube_id):
        """如果指定方块当前已附着，返回 True。"""
        return cube_id in self.attached_cubes
