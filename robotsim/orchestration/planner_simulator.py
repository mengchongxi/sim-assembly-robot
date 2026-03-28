"""规划器仿真器编排器模块。

实现 CompletePlannerSimulator 类，将 Pygame GUI、规划逻辑和
渲染器整合为一个完整互动仿真工作流的顶层运行器。
"""
import sys
import pygame

from robotsim.orchestration.planner_logic import CompletePlannerLogic
from robotsim.gui.planner_gui import CompletePlannerRenderer


class CompletePlannerSimulator:
    """完整规划仿真器主类 - 支持规划器切换"""
    def __init__(self, grid_width=30, grid_height=30, default_planner="heuristic"):
        self.clock = pygame.time.Clock()

        # 初始化逻辑和渲染器
        self.logic = CompletePlannerLogic(grid_width, grid_height, default_planner)
        self.renderer = CompletePlannerRenderer(grid_width, grid_height)

        print(f"Complete Robot Task & Motion Planner started (Grid: {grid_width}x{grid_height})")
        print(f"Default planner: {default_planner}")

    def handle_click(self, pos, is_goal=False):
        """处理鼠标点击"""
        grid_x, grid_y = self.renderer.pixel_to_grid(pos[0], pos[1])
        if grid_x is None:
            return

        tile = (grid_x, grid_y)
        self.logic.handle_tile_click(tile, is_goal)

    def show_planner_selection(self):
        """显示规划器选择菜单"""
        planner_info = self.logic.get_planner_info()
        available_planners = list(planner_info['available'].keys())
        current_idx = available_planners.index(planner_info['current'])

        print(f"\n=== 规划器选择 ===")
        for i, (name, desc) in enumerate(planner_info['available'].items()):
            marker = "→" if name == planner_info['current'] else " "
            print(f"{marker} {i+1}. {name}: {desc}")

        print("\n请选择规划器 (输入数字，回车确认, 或直接回车保持当前):")
        try:
            choice = input().strip()
            if choice == "":
                print(f"保持当前规划器: {planner_info['current']}")
                return

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_planners):
                new_planner = available_planners[choice_idx]
                if self.logic.switch_planner(new_planner):
                    print(f"✅ 已切换到: {new_planner}")
                else:
                    print(f"❌ 切换失败，保持: {planner_info['current']}")
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入有效数字")
        except KeyboardInterrupt:
            print(f"\n保持当前规划器: {planner_info['current']}")

    def run_gui_setup(self):
        """运行GUI设置阶段"""
        print("GUI Setup Phase - Configure your problem")

        running = True

        while running and self.logic.app_state in ['STATE_SET_GOAL', 'STATE_SHOW_INITIAL']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.VIDEORESIZE:
                    self.renderer.handle_resize(event.w, event.h)

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.logic.reset_to_start()

                    # 规划器切换
                    elif event.key == pygame.K_p:
                        pygame.display.minimize()
                        self.show_planner_selection()
                        pygame.display.restore()

                    # 编辑模式切换（仅在设置目标阶段）
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3] and self.logic.app_state == 'STATE_SET_GOAL':
                        self.logic.edit_mode = {pygame.K_1: 'base', pygame.K_2: 'joint', pygame.K_3: 'wheel'}[event.key]

                # 状态特定的事件处理
                if self.logic.app_state == 'STATE_SET_GOAL':
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.handle_click(event.pos, is_goal=True)
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        total_goal = len(self.logic.goal_base) + len(self.logic.goal_joint) + len(self.logic.goal_wheel)
                        if total_goal == 0:
                            self.logic.error_message = "Goal configuration is empty!"
                        else:
                            print("\n--- Goal set. Generating random initial configuration... ---")
                            if self.logic.generate_random_initial_config():
                                self.logic.app_state = 'STATE_SHOW_INITIAL'
                                print("--- Random initial configuration generated. ---")
                            else:
                                self.logic.error_message = "Failed to generate random initial config!"

                elif self.logic.app_state == 'STATE_SHOW_INITIAL':
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            print("--- Regenerating random initial configuration... ---")
                            if self.logic.generate_random_initial_config():
                                print("--- New random initial configuration generated. ---")
                            else:
                                self.logic.error_message = "Failed to regenerate!"
                        elif event.key == pygame.K_RETURN:
                            print("\n--- Closing GUI and starting planning in terminal ---")
                            running = False

            # 渲染
            self.renderer.draw(self.logic)
            pygame.display.flip()
            self.clock.tick(60)

        # 关闭pygame GUI
        pygame.quit()

        return self.logic.app_state == 'STATE_SHOW_INITIAL'

    def run_console_planning(self):
        """运行命令行规划阶段"""
        print("\n" + "="*60)
        print("    COMMAND LINE PLANNING PHASE")
        print("="*60)

        # 执行规划
        success = self.logic.trigger_complete_planning_with_progress()

        if success:
            print(f"\n🎉 PLANNING COMPLETED SUCCESSFULLY! 🎉")
            print(f"Results have been saved to trajectory file.")
        else:
            print(f"\n❌ PLANNING FAILED!")
            print(f"Error: {self.logic.error_message}")

        return success

    def run(self):
        """主运行循环"""
        print("Starting Complete Robot Task & Motion Planner...")

        # 阶段1：GUI设置
        if not self.run_gui_setup():
            print("Setup cancelled or failed.")
            return

        # 阶段2：命令行规划
        success = self.run_console_planning()

        # 程序结束
        sys.exit(0 if success else 1)
