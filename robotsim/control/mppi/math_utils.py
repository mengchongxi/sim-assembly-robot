"""MPPI 所需的 JAX 向量化数学工具。"""

import functools

import jax
from jax import numpy as jnp
from brax import math


@functools.partial(jax.jit, static_argnums=(3,))
def _interpolate_1d(x_knots, y_knots, x_query, k=2):
    """JAX-native 1-D 多项式插值（替代 jax_cosmo InterpolatedUnivariateSpline）。

    对每个查询点，取最近 k+1 个节点做拉格朗日插值。
    支持批量 y_knots (shape [..., n_knots])，对最后一轴插值。

    Args:
        x_knots: 节点 x 坐标，shape (n_knots,)，需单调递增。
        y_knots: 节点 y 值，shape (..., n_knots)。
        x_query: 查询点 x，shape (n_query,)。
        k: 插值阶数（默认 2 = 二次）。

    Returns:
        插值结果，shape (..., n_query)。
    """
    n = x_knots.shape[0]
    # 为每个查询点找到最近节点的起始索引
    idx = jnp.searchsorted(x_knots, x_query, side="right") - 1
    # 以该节点为中心取 k+1 个点的窗口
    half = k // 2
    start = jnp.clip(idx - half, 0, n - (k + 1))

    def _lagrange_at(i):
        """对第 i 个查询点做 k+1 点拉格朗日插值。"""
        s = start[i]
        xs = jax.lax.dynamic_slice(x_knots, (s,), (k + 1,))
        # y_knots 可能是 2-D (..., n_knots)，需要对最后一轴切片
        ys = jax.lax.dynamic_slice_in_dim(y_knots, s, k + 1, axis=-1)
        xq = x_query[i]
        # 拉格朗日基函数
        bases = jnp.ones(k + 1)
        for j in range(k + 1):
            for m in range(k + 1):
                bases = bases.at[j].mul(
                    jnp.where(j == m, 1.0, (xq - xs[m]) / (xs[j] - xs[m]))
                )
        return jnp.tensordot(ys, bases, axes=([-1], [0]))

    return jax.vmap(_lagrange_at)(jnp.arange(x_query.shape[0]))


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
