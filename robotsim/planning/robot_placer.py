"""机器人放置位置搜索模块。

提供 RobotPlacer 工具类，为机器人在初始构型上拣择最佳起始放置位置，
以及在任务完成后为机器人寻找靠近目标构型质心的停车位。
"""
from typing import Tuple, Optional, Set

from robotsim.core.robot_state import RobotConfiguration

class RobotPlacer:
    """机器人放置位置静态工具类。

    所有方法均为静态方法，无需实例化。
    采用三级候选策略（质心候选、就近候选、全局搜索）选取最优放置点。
    """
    @staticmethod
    def calculate_centroid(tiles: set) -> Tuple[float, float]:
        """计算瓦片集合的几何质心。

        Returns:
            (cx, cy) 浮点坐标对，集合为空时返回 (0, 0)。
        """
        if not tiles:
            return (0, 0)

        sum_x = sum(pos[0] for pos in tiles)
        sum_y = sum(pos[1] for pos in tiles)
        return (sum_x / len(tiles), sum_y / len(tiles))

    @staticmethod
    def find_final_parking_spot(goal_base: set, goal_joint: set, goal_wheel: set) -> Optional[RobotConfiguration]:
        """
        在任务完成后，为机器人寻找一个最佳的"停车位"。
        这个位置基于最终的目标构型。
        """
        all_tiles = goal_base | goal_joint | goal_wheel

        if len(all_tiles) < 2:
            print("   > 目标构型的方块数量不足，无法找到停车位。")
            return None

        # 核心思想：在"新家"（目标构型）的中心区域找个好车位
        centroid = RobotPlacer.calculate_centroid(all_tiles)
        print(f"   > 目标构型质心: ({centroid[0]:.2f}, {centroid[1]:.2f})，正在寻找最佳停车位...")

        parking_candidates = []

        # 方案一："黄金车位"策略 - 寻找位于或靠近质心的停车位
        centroid_int = (round(centroid[0]), round(centroid[1]))
        if centroid_int in all_tiles:
            base_pos = centroid_int
            # 寻找质心方块的邻居
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (base_pos[0] + dx, base_pos[1] + dy)
                if neighbor in all_tiles:
                    config = RobotConfiguration(front_foot=neighbor, back_foot=base_pos)
                    # 给一个很高的优先级（距离为1）
                    parking_candidates.append((config, 1))
                    print(f"   > 发现一个中心候选停车位: {config}")

        # 方案二："就近停车"策略 - 如果中心没位置，就在离中心最近的方块里找
        if not parking_candidates:
            print("   > 中心位置无合适车位，扩大搜索范围至离中心最近的方块...")
            distances = [(pos, ((pos[0] - centroid[0])**2 + (pos[1] - centroid[1])**2)**0.5)
                        for pos in all_tiles]
            distances.sort(key=lambda x: x[1])

            # 在离质心最近的8个方块中寻找相邻对
            nearest_tiles = [pos for pos, _ in distances[:min(8, len(distances))]]

            for i, tile1 in enumerate(nearest_tiles):
                for j, tile2 in enumerate(nearest_tiles):
                    if i != j and abs(tile1[0] - tile2[0]) + abs(tile1[1] - tile2[1]) == 1:
                        config = RobotConfiguration(front_foot=tile1, back_foot=tile2)
                        # 计算这个车位到总质心的距离，作为评判标准
                        config_center = ((tile1[0] + tile2[0])/2, (tile1[1] + tile2[1])/2)
                        dist_to_centroid = ((config_center[0] - centroid[0])**2 +
                                        (config_center[1] - centroid[1])**2)**0.5
                        parking_candidates.append((config, dist_to_centroid))

        # 方案三："无奈之举"策略 - 随便找个能停的就行
        if not parking_candidates:
            print("   > 核心区域也找不到车位，进行全局搜索...")
            tile_list = list(all_tiles)
            for i, tile1 in enumerate(tile_list):
                for j, tile2 in enumerate(tile_list):
                    if i != j and abs(tile1[0] - tile2[0]) + abs(tile1[1] - tile2[1]) == 1:
                        # 给一个很低的优先级（距离设为100）
                        parking_candidates.append((RobotConfiguration(front_foot=tile1, back_foot=tile2), 100))
                        break # 找到一个就行了
                if parking_candidates:
                    break

        if not parking_candidates:
            print("   > ❌ 无法在目标构型上找到任何合适的停车位。")
            return None

        # 从所有候选车位中，选择一个最优的（距离质心最近的）
        best_config = min(parking_candidates, key=lambda x: x[1])[0]

        print(f"   > ✅ 选定最终停车位: {best_config}")
        return best_config

    @staticmethod
    def find_best_robot_placement(initial_base: set, initial_joint: set, initial_wheel: set, first_move_tile: Optional[Tuple[int, int]] = None) -> Optional[RobotConfiguration]:
        """在初始构型上为机器人寻找最佳起始放置位置。

        Args:
            initial_base: 初始 Base 瓦片网格坐标集。
            initial_joint: 初始 Joint 瓦片网格坐标集。
            initial_wheel: 初始 Wheel 瓦片网格坐标集。
            first_move_tile: 第一个需要移动的瓦片位置，放置时将避开此位置。

        Returns:
            最佳 RobotConfiguration，找不到时返回 None。
        """
        all_tiles = initial_base | initial_joint | initial_wheel

        if len(all_tiles) < 2:
            print("方块数量不足，无法放置机器人")
            return None

        centroid = RobotPlacer.calculate_centroid(all_tiles)
        print(f"初始构型质心位置: ({centroid[0]:.2f}, {centroid[1]:.2f})")

        if first_move_tile:
            print(f"避开第一个移动的方块: {first_move_tile}")

        placement_candidates = []

        distances = [(pos, ((pos[0] - centroid[0])**2 + (pos[1] - centroid[1])**2)**0.5)
                    for pos in all_tiles]
        distances.sort(key=lambda x: x[1])

        centroid_int = (round(centroid[0]), round(centroid[1]))
        if centroid_int in all_tiles and centroid_int != first_move_tile:
            print(f"质心位置 {centroid_int} 有方块，在其附近寻找放置位置")
            base_pos = centroid_int

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (base_pos[0] + dx, base_pos[1] + dy)
                if neighbor in all_tiles and neighbor != first_move_tile:
                    config = RobotConfiguration(front_foot=neighbor, back_foot=base_pos)
                    placement_candidates.append((config, 1))
                    print(f"候选位置: 后脚={base_pos}, 前脚={neighbor}")

        if not placement_candidates:
            print("质心位置无方块或被占用，使用最近的方块组合")

            nearest_tiles = [pos for pos, _ in distances[:min(8, len(distances))]]

            for i, tile1 in enumerate(nearest_tiles):
                if tile1 == first_move_tile:
                    continue
                for j, tile2 in enumerate(nearest_tiles):
                    if i != j and tile2 != first_move_tile:
                        if abs(tile1[0] - tile2[0]) + abs(tile1[1] - tile2[1]) == 1:
                            config = RobotConfiguration(front_foot=tile1, back_foot=tile2)
                            config_center = ((tile1[0] + tile2[0])/2, (tile1[1] + tile2[1])/2)
                            dist_to_centroid = ((config_center[0] - centroid[0])**2 +
                                              (config_center[1] - centroid[1])**2)**0.5
                            placement_candidates.append((config, dist_to_centroid))

        if not placement_candidates:
            print("使用任意相邻方块对")
            tile_list = list(all_tiles)
            for i, tile1 in enumerate(tile_list):
                if tile1 == first_move_tile:
                    continue
                for j, tile2 in enumerate(tile_list):
                    if i != j and tile2 != first_move_tile and abs(tile1[0] - tile2[0]) + abs(tile1[1] - tile2[1]) == 1:
                        config = RobotConfiguration(front_foot=tile1, back_foot=tile2)
                        placement_candidates.append((config, 100))
                        break
                if placement_candidates:
                    break

        if not placement_candidates:
            print("无法找到合适的机器人放置位置")
            return None

        best_config = min(placement_candidates, key=lambda x: x[1])[0]

        print(f"选择机器人放置位置: {best_config}")
        return best_config
