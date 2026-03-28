"""核心枚举类型定义模块。

本模块定义了仿真系统中所有共享的枚举类型，包括网格瓦片类型（TileType）
和机器人可执行的动作类型（ActionType）。
"""
from enum import Enum


class TileType(Enum):
    """网格瓦片的功能类型枚举。

    用于标识网格中每个格子所承载的功能模块，同时也标记机器人携带的模块类型。
    """
    EMPTY = 0      # 空格，机器人可在此行走
    BASE = 1       # 基础结构模块
    JOINT = 2      # 关节模块
    WHEEL = 3      # 轮子模块
    OBSTACLE = -1  # 障碍物，不可通行


class ActionType(Enum):
    """机器人可执行的原子动作枚举。

    涵盖等待、前后移动、以前/后脚为支点的旋转（90° 和 180°），
    以及从前/左/右三个方向拾取或放置瓦片的操作。
    """
    WAIT = 0              # 原地等待
    STEP_FORWARD = 1      # 向前迈步
    STEP_BACKWARD = 2     # 向后退步
    ROTATE_FRONT_RIGHT = 3  # 固定前脚，后脚向右旋转 90°
    ROTATE_FRONT_LEFT = 4   # 固定前脚，后脚向左旋转 90°
    ROTATE_BACK_RIGHT = 5   # 固定后脚，前脚向右旋转 90°
    ROTATE_BACK_LEFT = 6    # 固定后脚，前脚向左旋转 90°
    ROTATE_FRONT_180 = 7    # 固定前脚，后脚旋转 180°
    ROTATE_BACK_180 = 8     # 固定后脚，前脚旋转 180°
    PICKUP_FRONT = 9    # 从前方拾取瓦片
    PICKUP_LEFT = 10    # 从左侧拾取瓦片
    PICKUP_RIGHT = 11   # 从右侧拾取瓦片
    PLACE_FRONT = 12    # 向前方放置瓦片
    PLACE_LEFT = 13     # 向左侧放置瓦片
    PLACE_RIGHT = 14    # 向右侧放置瓦片
