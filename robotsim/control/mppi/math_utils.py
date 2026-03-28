"""MPPI 所需的 JAX 向量化数学工具。"""

import jax
from jax import numpy as jnp
from brax import math


class MathUtils:
    """坐标变换与步态计算工具类。

    所有方法均使用 @jax.jit 编译以获得最佳性能。
    """

    @staticmethod
    @jax.jit
    def body_to_global_velocity(v, q):
        """身体坐标系速度转换到世界坐标系。"""
        return math.rotate(v, q)

    @staticmethod
    @jax.jit
    def global_to_body_velocity(v, q):
        """世界坐标系速度转换到身体坐标系。"""
        return math.inv_rotate(v, q)

    @staticmethod
    @jax.jit
    def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
        """计算步态中各足端的抬腿高度。

        Args:
            duty_ratio: 占空比（地面接触时间比例）。
            cadence: 步频。
            amplitude: 抬腿幅度。
            phases: 各腿相位偏移数组。
            time: 当前仿真时间。

        Returns:
            各腿抬起高度数组。
        """
        def step_height(t, footphase, duty_ratio):
            angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
            angle = jnp.where(
                duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle
            )
            clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
            value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
            return jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)

        h_steps = amplitude * jax.vmap(
            step_height, in_axes=(None, 0, None)
        )(
            time * 2 * jnp.pi * cadence + jnp.pi,
            2 * jnp.pi * phases,
            duty_ratio,
        )
        return h_steps
