"""规划器核心逻辑模块。

实现 CompletePlannerLogic 类，为整个规划-执行工作流提供状态机，
包括高层规划、机器人放置、路径规划、刹車和轨迹存档。
"""
import time
import threading
import random
from typing import Tuple, Dict, Any

from robotsim.core.types import TileType, ActionType
from robotsim.core.robot_state import RobotConfiguration
from robotsim.planning.registry import PlannerRegistry
from robotsim.planning.base_planner import BasePlanner
from robotsim.planning.path_planner import RobotPathPlanner
from robotsim.planning.task_executor import RobotTaskSequenceExecutor
from robotsim.planning.robot_placer import RobotPlacer
from robotsim.robot.bill_e_bot import BillEBot
from robotsim.recording.trajectory_recorder import RobotTrajectoryRecorder


class CompletePlannerLogic:
    """完整规划器逻辑类 - 支持多种规划器"""
    def __init__(self, grid_width=30, grid_height=30, planner_name="heuristic"):
        self.grid_width = grid_width
        self.grid_height = grid_height

        # 规划器管理
        self.planner_registry = PlannerRegistry()
        self.current_planner_name = planner_name
        self.algorithm = self._create_planner(planner_name)
        self.robot_placer = RobotPlacer()

        # 游戏状态
        self.app_state = 'STATE_SET_GOAL'
        self.edit_mode = 'base'

        # 瓦片集合
        self.current_base = set()
        self.current_joint = set()
        self.current_wheel = set()
        self.goal_base = set()
        self.goal_joint = set()
        self.goal_wheel = set()

        # 规划结果
        self.move_sequence = []
        self.error_message = None

        # 机器人相关
        self.robot = None
        self.robot_placed = False
        self.robot_path_planner = None
        self.task_executor = None

        # 轨迹记录器
        self.trajectory_recorder = RobotTrajectoryRecorder()

        print(f"\n--- Complete Planning System Initialized with {self.algorithm.__class__.__name__} ---")
        print(f"Available planners: {list(self.planner_registry.list_planners().keys())}")
        print("Please set goal configuration first.")

    def _create_planner(self, planner_name: str) -> BasePlanner:
        """创建指定的规划器实例"""
        try:
            return self.planner_registry.get_planner(planner_name, self.grid_width, self.grid_height)
        except ValueError as e:
            print(f"警告: {e}，回退到默认启发式规划器")
            return self.planner_registry.get_planner("heuristic", self.grid_width, self.grid_height)

    def switch_planner(self, planner_name: str) -> bool:
        """切换规划器"""
        try:
            new_planner = self._create_planner(planner_name)
            self.algorithm = new_planner
            self.current_planner_name = planner_name
            print(f"✅ 已切换到规划器: {planner_name}")
            return True
        except Exception as e:
            print(f"❌ 切换规划器失败: {e}")
            return False

    def get_planner_info(self) -> Dict[str, Any]:
        """获取当前规划器信息"""
        return {
            'current': self.current_planner_name,
            'class': self.algorithm.__class__.__name__,
            'available': self.planner_registry.list_planners()
        }

    def reset_to_start(self):
        """重置到开始状态"""
        print(f"\n--- System Reset with {self.algorithm.__class__.__name__}. Please set goal configuration first. ---")
        self.app_state = 'STATE_SET_GOAL'
        self.current_base.clear()
        self.current_joint.clear()
        self.current_wheel.clear()
        self.goal_base.clear()
        self.goal_joint.clear()
        self.goal_wheel.clear()

        self.edit_mode = 'base'
        self.move_sequence = []
        self.error_message = None

        self.robot = None
        self.robot_placed = False
        self.robot_path_planner = None
        self.task_executor = None

        self.trajectory_recorder.clear_trajectory()

        # 重置规划器状态
        self.algorithm.reset_working_area()

    def handle_tile_click(self, tile: Tuple[int, int], is_goal=False):
        """处理瓦片点击"""
        self.error_message = None
        sets_group = (self.goal_base, self.goal_joint, self.goal_wheel) if is_goal else (self.current_base, self.current_joint, self.current_wheel)
        target_set = {'base': sets_group[0], 'joint': sets_group[1], 'wheel': sets_group[2]}[self.edit_mode]

        if tile in target_set:
            target_set.remove(tile)
        else:
            # 先从其他集合中移除
            for s in sets_group:
                if tile in s:
                    s.remove(tile)
            target_set.add(tile)

    def generate_random_initial_config(self) -> bool:
        """Generate a random initial configuration matching the goal tile counts."""
        n_base = len(self.goal_base)
        n_joint = len(self.goal_joint)
        n_wheel = len(self.goal_wheel)

        if n_base == 0 and n_joint == 0 and n_wheel == 0:
            return False

        pool = ([TileType.BASE] * n_base +
                [TileType.JOINT] * n_joint +
                [TileType.WHEEL] * n_wheel)
        random.shuffle(pool)

        for _ in range(100):
            self.current_base.clear()
            self.current_joint.clear()
            self.current_wheel.clear()

            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)

            pos = (x, y)
            if pool[0] == TileType.BASE:
                self.current_base.add(pos)
            elif pool[0] == TileType.JOINT:
                self.current_joint.add(pos)
            else:
                self.current_wheel.add(pos)

            placed_tiles = {pos}
            success = True

            for tile_type in pool[1:]:
                neighbors = set()
                for (px, py) in placed_tiles:
                    for nx, ny in [(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)]:
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and (nx, ny) not in placed_tiles:
                            neighbors.add((nx, ny))
                if not neighbors:
                    success = False
                    break

                chosen = random.choice(list(neighbors))
                if tile_type == TileType.BASE:
                    self.current_base.add(chosen)
                elif tile_type == TileType.JOINT:
                    self.current_joint.add(chosen)
                else:
                    self.current_wheel.add(chosen)
                placed_tiles.add(chosen)

            if success:
                return True

            self.current_base.clear()
            self.current_joint.clear()
            self.current_wheel.clear()

        return False

    def trigger_complete_planning_with_progress(self):
        """触发完整规划 - 包括任务规划和机器人运动规划（带进度显示）"""
        self.error_message = None

        # 使用当前规划器的验证功能
        is_valid, error_msg = self.algorithm.validate_problem_input(
            self.current_base, self.current_joint, self.current_wheel,
            self.goal_base, self.goal_joint, self.goal_wheel
        )

        if not is_valid:
            print(f"Error: {error_msg}")
            self.error_message = error_msg
            return False

        self.app_state = 'STATE_PLANNING'
        print(f"\n{'='*70}")
        print(f"    STARTING COMPLETE PLANNING PROCESS")
        print(f"    Using: {self.algorithm.__class__.__name__}")
        print(f"{'='*70}")

        overall_start = time.time()

        # 第一步：高层任务规划
        print(f"\n[Step 1/5] High-level task planning with {self.current_planner_name}...")
        step1_start = time.time()
        if not self.execute_high_level_planning_with_progress():
            return False
        step1_time = time.time() - step1_start
        print(f"✅ Step 1 completed in {step1_time:.2f} seconds")

        # 第二步：智能放置机器人
        print(f"\n[Step 2/5] Intelligent robot placement...")
        step2_start = time.time()
        if not self.place_robot_intelligently():
            return False
        step2_time = time.time() - step2_start
        print(f"✅ Step 2 completed in {step2_time:.2f} seconds")

        # 第三步：详细机器人路径规划
        print(f"\n[Step 3/5] Detailed robot path planning...")
        step3_start = time.time()
        if not self.execute_detailed_robot_planning_with_progress():
            return False
        step3_time = time.time() - step3_start
        print(f"✅ Step 3 completed in {step3_time:.2f} seconds")

        # 第四步：机器人回家
        step4_start = time.time()
        print(f"\n[Step 4/5] Robot parking...")
        final_parking_spot_config = self.robot_placer.find_final_parking_spot(
            self.goal_base, self.goal_joint, self.goal_wheel
        )

        if final_parking_spot_config:
            print(f"   > Found final parking spot at: {final_parking_spot_config}")
            all_goal_tiles = self.goal_base | self.goal_joint | self.goal_wheel
            self.robot_path_planner.update_grid_tiles(all_goal_tiles)

            path_to_parking = self.robot_path_planner.plan_path_to_config(
                start_config=self.robot.config,
                end_config=final_parking_spot_config
            )

            if path_to_parking:
                print(f"   > Found path to parking spot with {len(path_to_parking)} actions.")
                for action in path_to_parking:
                    self.execute_robot_action(action)
                print(f"   > ✅ Robot has parked successfully.")
            else:
                print("   > ⚠️ Could not find a path to the final parking spot.")
        else:
            print("   > ⚠️ Could not find a suitable parking spot for the robot.")

        step4_time = time.time() - step4_start
        print(f"✅ Step 4 completed in {step4_time:.2f} seconds")

        # 第五步：保存完整轨迹
        print(f"\n[Step 5/5] Saving complete trajectory...")
        step5_start = time.time()
        self.save_complete_trajectory()
        step5_time = time.time() - step5_start
        print(f"✅ Step 5 completed in {step5_time:.2f} seconds")

        total_time = time.time() - overall_start
        print(f"\n{'='*70}")
        print(f"    PLANNING COMPLETED SUCCESSFULLY!")
        print(f"    Planner: {self.algorithm.__class__.__name__}")
        print(f"    Total time: {total_time:.2f} seconds")
        print(f"    High-level moves: {len(self.move_sequence)}")
        print(f"    Robot actions: {len(self.trajectory_recorder.movements)}")
        print(f"{'='*70}")

        self.app_state = 'STATE_DONE'
        return True

    def execute_high_level_planning_with_progress(self):
        """执行高层任务规划（带进度显示）"""
        all_current = self.current_base | self.current_joint | self.current_wheel
        all_goal = self.goal_base | self.goal_joint | self.goal_wheel

        if not all_current or not all_goal:
            print("❌ Error: Empty configuration!")
            self.reset_to_start()
            return False

        print(f"   > Running {self.algorithm.__class__.__name__} algorithm...")
        start_time = time.time()

        # 创建一个显示进度的线程函数
        def show_progress():
            dots = 0
            while True:
                elapsed = time.time() - start_time
                print(f"\r   > Planning with {self.current_planner_name}{'.' * (dots % 4):<4} ({elapsed:.1f}s)", end='', flush=True)
                dots += 1
                time.sleep(0.5)
                if elapsed > 300:  # 5分钟超时
                    break

        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()

        # 使用当前规划器的完整规划功能
        success, plan = self.algorithm.generate_full_plan(
            self.current_base, self.goal_base,
            self.current_joint, self.goal_joint,
            self.current_wheel, self.goal_wheel
        )
        planning_time = time.time() - start_time

        # 清除进度显示
        print(f"\r   > Planning completed in {planning_time:.2f} seconds" + " " * 30)

        if success:
            print(f"   > ✅ {self.algorithm.__class__.__name__} found solution with {len(plan)} high-level moves")
            self.move_sequence = plan
            return True
        else:
            print(f"   > ❌ {self.algorithm.__class__.__name__} found no solution")
            self.app_state = 'STATE_FAILED'
            return False

    # ... [其余方法保持不变] ...
    def place_robot_intelligently(self):
        """智能放置机器人"""
        if self.robot_placed:
            print("   > Robot already placed")
            return True

        print("   > Finding optimal robot placement...")

        first_move_tile = None
        if self.move_sequence:
            first_move_tile = self.move_sequence[0][0]
            print(f"   > First tile to move: {first_move_tile}")

        robot_config = self.robot_placer.find_best_robot_placement(
            self.current_base, self.current_joint, self.current_wheel, first_move_tile
        )

        if robot_config:
            self.robot = BillEBot(robot_config)
            self.robot_placed = True

            all_tiles = self.current_base | self.current_joint | self.current_wheel
            self.robot_path_planner = RobotPathPlanner(
                self.robot, all_tiles, self.grid_width, self.grid_height
            )

            self.task_executor = RobotTaskSequenceExecutor(self.robot, self.robot_path_planner)

            # 设置轨迹记录器
            self.trajectory_recorder.set_initial_position(robot_config)
            self.trajectory_recorder.set_initial_tiles(self.current_base, self.current_joint, self.current_wheel)

            print(f"   > ✅ BILL-E robot placed at: {robot_config}")
            return True
        else:
            print("   > ❌ Cannot find suitable robot placement")
            return False

    def execute_detailed_robot_planning_with_progress(self):
        """执行详细机器人路径规划（带进度显示）"""
        if not self.task_executor or not self.move_sequence:
            print("   > ❌ Task executor not initialized or no move sequence")
            return False

        print(f"   > Planning robot actions for {len(self.move_sequence)} tasks...")

        # 加载任务序列
        self.task_executor.load_move_sequence(self.move_sequence, self.current_base, self.current_joint, self.current_wheel)

        start_time = time.time()
        total_tasks = len(self.task_executor.task_sequence)

        # 模拟执行所有任务，记录轨迹
        task_count = 0
        while self.task_executor.is_executing:
            task_count += 1
            current_task = self.task_executor.task_sequence[self.task_executor.current_task_index] if self.task_executor.current_task_index < len(self.task_executor.task_sequence) else None

            # 显示进度
            elapsed = time.time() - start_time
            progress = (task_count / total_tasks) * 100 if total_tasks > 0 else 0

            # 规划当前任务
            if not self.task_executor.plan_current_task():
                print(f"\n   > ❌ Failed to plan task {task_count}")
                return False

            # 执行当前任务的所有动作
            while (self.task_executor.current_action_index < len(self.task_executor.current_task_actions) and
                   self.task_executor.is_executing):

                action = self.task_executor.current_task_actions[self.task_executor.current_action_index]
                self.task_executor.current_action_index += 1

                # 执行动作并记录轨迹
                self.execute_robot_action(action)

            # 移动到下一个任务
            self.task_executor.current_task_index += 1
            self.task_executor.current_action_index = 0

            # 检查是否完成所有任务
            if self.task_executor.current_task_index >= len(self.task_executor.task_sequence):
                self.task_executor.is_executing = False
                break

        total_time = time.time() - start_time
        print(f"\n   > ✅ All tasks completed! Generated {len(self.trajectory_recorder.movements)} robot actions")
        return True

    def execute_robot_action(self, action: ActionType):
        """执行机器人动作并记录轨迹"""
        if not self.robot:
            return

        all_tiles = self.current_base | self.current_joint | self.current_wheel
        action_success = False
        old_config = self.robot.config.copy()

        # 用于记录瓦片操作信息
        operated_tile_type = None
        operated_tile_position = None

        if action in [ActionType.PICKUP_FRONT, ActionType.PICKUP_LEFT, ActionType.PICKUP_RIGHT]:
            direction_map = {
                ActionType.PICKUP_FRONT: "front",
                ActionType.PICKUP_LEFT: "left",
                ActionType.PICKUP_RIGHT: "right"
            }
            direction = direction_map[action]
            picked_pos = self.robot.can_pick_up_tile_at(direction, all_tiles)
            if picked_pos:
                # 确定被拾取的瓦片类型
                if picked_pos in self.current_base:
                    self.current_base.remove(picked_pos)
                    self.robot.config.carrying_tile_type = TileType.BASE
                    operated_tile_type = TileType.BASE
                elif picked_pos in self.current_joint:
                    self.current_joint.remove(picked_pos)
                    self.robot.config.carrying_tile_type = TileType.JOINT
                    operated_tile_type = TileType.JOINT
                elif picked_pos in self.current_wheel:
                    self.current_wheel.remove(picked_pos)
                    self.robot.config.carrying_tile_type = TileType.WHEEL
                    operated_tile_type = TileType.WHEEL

                operated_tile_position = picked_pos
                self.robot.config.carrying_tile = True
                self.robot.config.tile_pos = picked_pos
                action_success = True

                # 更新路径规划器的瓦片集合
                all_tiles = self.current_base | self.current_joint | self.current_wheel
                self.robot_path_planner.grid_tiles = all_tiles
                self.task_executor.update_path_planner(all_tiles)

        elif action in [ActionType.PLACE_FRONT, ActionType.PLACE_LEFT, ActionType.PLACE_RIGHT]:
            direction_map = {
                ActionType.PLACE_FRONT: "front",
                ActionType.PLACE_LEFT: "left",
                ActionType.PLACE_RIGHT: "right"
            }
            direction = direction_map[action]
            placed_pos = self.robot.can_place_tile_at(direction, all_tiles, self.grid_width, self.grid_height)
            if placed_pos:
                operated_tile_type = self.robot.config.carrying_tile_type
                operated_tile_position = placed_pos

                if self.robot.config.carrying_tile_type == TileType.BASE:
                    self.current_base.add(placed_pos)
                elif self.robot.config.carrying_tile_type == TileType.JOINT:
                    self.current_joint.add(placed_pos)
                elif self.robot.config.carrying_tile_type == TileType.WHEEL:
                    self.current_wheel.add(placed_pos)

                self.robot.config.carrying_tile = False
                self.robot.config.tile_pos = None
                action_success = True

                # 更新路径规划器的瓦片集合
                all_tiles = self.current_base | self.current_joint | self.current_wheel
                self.robot_path_planner.grid_tiles = all_tiles
                self.task_executor.update_path_planner(all_tiles)

        else:
            # 运动动作
            new_config = self.robot.execute_action(action, all_tiles, self.grid_width, self.grid_height)
            if new_config:
                self.robot.config = new_config
                action_success = True

        # 记录轨迹
        self.trajectory_recorder.record_movement(
            action=action,
            old_config=old_config,
            new_config=self.robot.config,
            action_success=action_success,
            tile_type=operated_tile_type,
            tile_position=operated_tile_position
        )

    def save_complete_trajectory(self):
        """保存完整轨迹"""
        if not self.trajectory_recorder.movements:
            print("   > No trajectory to save")
            self.error_message = "No trajectory to save"
            return

        filename = self.trajectory_recorder.save_to_yaml()
        if filename:
            summary = self.trajectory_recorder.get_trajectory_summary()
            success_msg = f"Saved: {filename}"
            print(f"   > ✅ {success_msg}")
            print(f"   > {summary}")
            self.error_message = success_msg
        else:
            error_msg = "Failed to save trajectory"
            print(f"   > ❌ {error_msg}")
            self.error_message = error_msg
