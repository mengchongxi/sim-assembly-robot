"""基于 Pygame 的规划器可视化渲染器模块。

提供 CompletePlannerRenderer 类，将规划器的网格状态、瓦片布局、
短暂信息以及操作提示渲染到 Pygame 窗口。
"""
import pygame


class CompletePlannerRenderer:
    """完整规划器渲染器 - 支持显示规划器信息"""
    def __init__(self, grid_width=30, grid_height=30):
        pygame.init()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.base_cell_size = 40
        self.cell_size = 40
        self.margin = 80

        # 增大UI面板宽度以显示规划器信息
        self.ui_width = 500
        self.window_width = grid_width * self.cell_size + 2 * self.margin + self.ui_width
        self.window_height = grid_height * self.cell_size + 2 * self.margin

        self.ui_ratio = self.ui_width / self.window_width
        self.min_width = 800
        self.min_height = 600

        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Complete Robot Task & Motion Planner")

        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 22)
        self.big_font = pygame.font.Font(None, 32)

        # 颜色
        self.colors = {
            'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (200, 200, 200),
            'dark_green': (0, 150, 0),
            'base_current': (200, 50, 50), 'base_goal': (255, 150, 150),
            'joint_current': (50, 50, 200), 'joint_goal': (150, 150, 255),
            'wheel_current': (50, 150, 50), 'wheel_goal': (150, 255, 150),
            'goal_shadow': (220, 220, 220),
            'error_red': (255, 0, 0),
            'success_green': (0, 180, 0),
            'planning_blue': (0, 100, 200),
            'planner_purple': (128, 0, 128),
        }

    def _scale(self, px):
        """Scale a pixel value proportionally to current cell_size."""
        return max(1, int(px * self.cell_size / self.base_cell_size))

    def handle_resize(self, new_width, new_height):
        """Handle window resize - recalculate all layout parameters."""
        new_width = max(new_width, self.min_width)
        new_height = max(new_height, self.min_height)

        self.window_width = new_width
        self.window_height = new_height

        # Recalculate UI width proportionally
        self.ui_width = int(new_width * self.ui_ratio)

        # Available space for grid
        grid_area_width = new_width - self.ui_width
        grid_area_height = new_height

        # margin ≈ 2 * cell_size (ratio from original: 80/40 = 2)
        margin_ratio = 2.0
        cell_w = grid_area_width / (self.grid_width + 2 * margin_ratio)
        cell_h = grid_area_height / (self.grid_height + 2 * margin_ratio)
        self.cell_size = max(1, int(min(cell_w, cell_h)))
        self.margin = max(10, int(self.cell_size * margin_ratio))

        # Recreate display surface
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)

        # Rebuild fonts proportionally
        scale = self.cell_size / self.base_cell_size
        self.font = pygame.font.Font(None, max(12, int(26 * scale)))
        self.small_font = pygame.font.Font(None, max(10, int(22 * scale)))
        self.big_font = pygame.font.Font(None, max(14, int(32 * scale)))

    def grid_to_pixel(self, x, y):
        """网格坐标转像素坐标"""
        return (self.margin + x * self.cell_size, self.margin + y * self.cell_size)

    def pixel_to_grid(self, x, y):
        """像素坐标转网格坐标"""
        if not (self.margin < x < self.margin + self.grid_width * self.cell_size and
                self.margin < y < self.margin + self.grid_height * self.cell_size):
            return None, None
        return int((x - self.margin) // self.cell_size), int((y - self.margin) // self.cell_size)

    def draw_grid(self):
        """绘制网格"""
        self.screen.fill(self.colors['white'])

        # 绘制垂直线
        for i in range(self.grid_width + 1):
            pygame.draw.line(self.screen, self.colors['gray'],
                           (self.margin + i * self.cell_size, self.margin),
                           (self.margin + i * self.cell_size, self.margin + self.grid_height * self.cell_size), 2)

        # 绘制水平线
        for i in range(self.grid_height + 1):
            pygame.draw.line(self.screen, self.colors['gray'],
                           (self.margin, self.margin + i * self.cell_size),
                           (self.margin + self.grid_width * self.cell_size, self.margin + i * self.cell_size), 2)

    def draw_cells(self, logic):
        """绘制瓦片"""
        goal_sets = logic.goal_base | logic.goal_joint | logic.goal_wheel

        # 在初始展示阶段绘制目标阴影
        if logic.app_state in ['STATE_SHOW_INITIAL']:
            for x, y in goal_sets:
                pygame.draw.rect(self.screen, self.colors['goal_shadow'],
                               (self.grid_to_pixel(x,y)[0]+2, self.grid_to_pixel(x,y)[1]+2,
                                self.cell_size-4, self.cell_size-4))

        # 决定绘制哪个集合
        if logic.app_state == 'STATE_SET_GOAL':
            sets_to_draw = goal_sets
            is_goal_context = True
        else:
            sets_to_draw = logic.current_base | logic.current_joint | logic.current_wheel
            is_goal_context = False

        # 绘制瓦片
        for x, y in sets_to_draw:
            color_map = self.colors['base_goal'] if is_goal_context else self.colors['base_current']

            if (x,y) in (logic.goal_joint if is_goal_context else logic.current_joint):
                color_map = self.colors['joint_goal'] if is_goal_context else self.colors['joint_current']
            elif (x,y) in (logic.goal_wheel if is_goal_context else logic.current_wheel):
                color_map = self.colors['wheel_goal'] if is_goal_context else self.colors['wheel_current']

            pygame.draw.rect(self.screen, color_map,
                           (self.grid_to_pixel(x,y)[0]+2, self.grid_to_pixel(x,y)[1]+2,
                            self.cell_size-4, self.cell_size-4))

    def draw_text_wrapped(self, text, x, y, max_width, font, color):
        """绘制自动换行文本"""
        words = text.split(' ')
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + word + " "
            if font.size(test_line)[0] < max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())

        current_y = y
        for line in lines:
            if current_y + font.get_height() > self.window_height - 10:
                break
            text_surface = font.render(line, True, color)
            self.screen.blit(text_surface, (x, current_y))
            current_y += font.get_height() + 3

        return current_y

    def draw_ui(self, logic):
        """绘制UI - 包括规划器信息"""
        ui_x = self.margin + self.grid_width * self.cell_size + self._scale(30)
        max_text_width = self.ui_width - self._scale(50)
        y_offset = self._scale(30)

        # 状态标题
        if logic.app_state == 'STATE_SET_GOAL':
            title = "Set Goal"
            color = self.colors['black']
        elif logic.app_state == 'STATE_SHOW_INITIAL':
            title = "Random Initial"
            color = self.colors['success_green']
        else:
            title = "Planning..."
            color = self.colors['planning_blue']

        self.screen.blit(self.big_font.render(title, True, color), (ui_x, y_offset))
        y_offset += self._scale(45)

        # 规划器信息
        planner_info = logic.get_planner_info()
        planner_text = f"Planner: {planner_info['current']}"
        self.screen.blit(self.small_font.render(planner_text, True, self.colors['planner_purple']), (ui_x, y_offset))
        y_offset += self._scale(30)

        # 根据状态显示不同内容
        if logic.app_state in ['STATE_SET_GOAL', 'STATE_SHOW_INITIAL']:
            if logic.app_state == 'STATE_SET_GOAL':
                # 编辑模式
                self.screen.blit(self.font.render(f"Mode: {logic.edit_mode.upper()}", True, self.colors['black']), (ui_x, y_offset))
                y_offset += self._scale(35)

                # 简化的图例
                self.screen.blit(self.small_font.render("B=Base J=Joint W=Wheel", True, self.colors['gray']), (ui_x, y_offset))
                y_offset += self._scale(25)

                # 瓦片计数
                total_goal = len(logic.goal_base) + len(logic.goal_joint) + len(logic.goal_wheel)
                self.screen.blit(self.font.render(f"Goal tiles: {total_goal}", True, self.colors['black']), (ui_x, y_offset))
                y_offset += self._scale(35)

                # 操作提示
                hint = "Click to place tile\nPress 1/2/3 to switch type\nPress P to switch planner\nPress ENTER when done"
            else:
                # STATE_SHOW_INITIAL: 展示随机初始构型
                counts_text = f"B:{len(logic.current_base)}/{len(logic.goal_base)} J:{len(logic.current_joint)}/{len(logic.goal_joint)} W:{len(logic.current_wheel)}/{len(logic.goal_wheel)}"
                self.screen.blit(self.small_font.render(counts_text, True, self.colors['black']), (ui_x, y_offset))
                y_offset += self._scale(25)

                # 错误信息
                if logic.error_message:
                    y_offset = self.draw_text_wrapped(logic.error_message, ui_x, y_offset, max_text_width,
                                                    self.small_font, self.colors['error_red'])
                    y_offset += self._scale(15)

                # 操作提示
                hint = "Press R to regenerate\nPress ENTER to start planning\nPress ESC to reset goal\n(Planning runs in terminal)"

            y_offset = self.draw_text_wrapped(hint, ui_x, y_offset, max_text_width,
                                            self.small_font, self.colors['dark_green'])

        # 重置提示
        reset_y = self.window_height - self._scale(60)
        self.screen.blit(self.small_font.render("ESC = Reset", True, self.colors['gray']), (ui_x, reset_y))
        self.screen.blit(self.small_font.render("P = Switch Planner", True, self.colors['gray']), (ui_x, reset_y + self._scale(20)))

    def draw(self, logic):
        """绘制所有内容"""
        self.draw_grid()
        self.draw_cells(logic)
        self.draw_ui(logic)
