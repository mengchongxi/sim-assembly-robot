"""Brax 物理仿真运行器。

封装 Brax 环境的创建和步进逻辑，
提供与 MuJoCo runner 对称的接口。
未来 MPPI 迁移到 MuJoCo 时，只需切换 runner。
"""

from typing import Any, Dict, Optional, Tuple

import jax
from brax.envs.base import PipelineEnv


class BraxRunner:
    """Brax 物理仿真运行器。

    封装了 Brax PipelineEnv 的创建和步进，
    提供统一的仿真后端接口。
    """

    def __init__(self, env: PipelineEnv, seed: int = 0):
        """初始化 Brax 运行器。

        Args:
            env: 已构建的 Brax PipelineEnv 实例。
            seed: 随机数种子。
        """
        self.env = env
        self.seed = seed
        self._rng = jax.random.PRNGKey(seed)
        self._state = None

        # 预编译
        self._jit_reset = jax.jit(env.reset)
        self._jit_step = jax.jit(env.step)

    def reset(self) -> Any:
        """重置环境到初始状态。

        Returns:
            初始 Brax State 对象。
        """
        self._rng, rng_reset = jax.random.split(self._rng)
        self._state = self._jit_reset(rng_reset)
        return self._state

    def step(self, action) -> Tuple[Any, float, bool]:
        """执行一步仿真。

        Args:
            action: 控制动作数组。

        Returns:
            (state, reward, done) 三元组。
        """
        if self._state is None:
            raise RuntimeError("必须先调用 reset()")
        self._state = self._jit_step(self._state, action)
        return self._state, float(self._state.reward), bool(self._state.done)

    @property
    def state(self) -> Optional[Any]:
        """当前环境状态。"""
        return self._state

    @property
    def sys(self):
        """底层 Brax 物理系统。"""
        return self.env.sys
