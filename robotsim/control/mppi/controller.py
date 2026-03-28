"""DIAL-MPC 控制器核心实现。

实现 MPPI 轨迹优化：采样候选轨迹 → 环境滚动仿真 → softmax 加权更新。
"""

import functools

import jax
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

from robotsim.control.mppi.config import Config


class DIALMPCController:
    """DIAL-MPC (Diffusion-Inspired Annealing for Legged MPC) 控制器。

    核心 MPPI 轨迹优化循环：
    1. 采样候选轨迹（node2u 样条插值）
    2. 在环境中滚动仿真
    3. 通过 softmax 加权更新最优轨迹
    4. 返回第一步动作
    """

    def __init__(self, config: Config, env):
        self.config = config
        self.env = env
        self.nu = env.action_size

        # 更新函数
        self.update_fn = {"mppi": self.softmax_update}[config.update_method]

        # 噪声调度
        sigma0, sigma1 = 1e-2, 1.0
        A = sigma0
        B = jnp.log(sigma1 / sigma0) / config.Ndiffuse
        self.sigmas = A * jnp.exp(B * jnp.arange(config.Ndiffuse))
        self.sigma_control = (
            config.horizon_diffuse_factor ** jnp.arange(config.Hnode + 1)[::-1]
        )

        # 时间设置
        self.ctrl_dt = config.dt
        self.step_us = jnp.linspace(
            0, self.ctrl_dt * config.Hsample, config.Hsample + 1
        )
        self.step_nodes = jnp.linspace(
            0, self.ctrl_dt * config.Hsample, config.Hnode + 1
        )

        # 预编译向量化函数
        self.rollout_us_vmap = jax.jit(
            jax.vmap(self.rollout_us, in_axes=(None, 0))
        )
        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=1, out_axes=1)
        )
        self.u2node_vmap = jax.jit(
            jax.vmap(self.u2node, in_axes=1, out_axes=1)
        )
        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=0)
        )
        self.u2node_vvmap = jax.jit(
            jax.vmap(self.u2node_vmap, in_axes=0)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def rollout_us(self, state, us):
        """滚动仿真一条轨迹，返回奖励序列和状态序列。"""
        def step(state, u):
            state = self.env.step(state, u)
            return state, (state.reward, state.pipeline_state)
        _, (rews, pipeline_states) = jax.lax.scan(step, state, us)
        return rews, pipeline_states

    @functools.partial(jax.jit, static_argnums=(0,))
    def softmax_update(self, weights, Y0s, sigma, mu_0t):
        """Softmax 权重更新轨迹均值。"""
        return jnp.einsum("n,nij->ij", weights, Y0s), sigma

    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        """控制节点到完整轨迹的样条插值。"""
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
        return spline(self.step_us)

    @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        """完整轨迹到控制节点的重采样。"""
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
        return spline(self.step_nodes)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale):
        """执行一次反向扩散优化步。

        Args:
            state: 当前环境状态。
            rng: JAX 随机数密钥。
            Ybar_i: 当前均值轨迹 (Hnode+1, nu)。
            noise_scale: 节点噪声尺度 (Hnode+1,)。

        Returns:
            (rng, Ybar_updated, info) 三元组。
        """
        # 采样候选轨迹
        rng, Y0s_rng = jax.random.split(rng)
        eps_Y = jax.random.normal(
            Y0s_rng,
            (self.config.Nsample, self.config.Hnode + 1, self.nu),
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        Y0s = Y0s.at[:, 0].set(Ybar_i[0, :])
        Y0s = jnp.concatenate([Y0s, Ybar_i[None]], axis=0)
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # 评估轨迹
        us = self.node2u_vvmap(Y0s)
        rewss, pipeline_statess = self.rollout_us_vmap(state, us)
        rews = rewss.mean(axis=-1)

        # 计算 softmax 权重
        rew_Ybar_i = rewss[-1].mean()
        logp0 = (
            (rews - rew_Ybar_i) / rews.std(axis=-1) / self.config.temp_sample
        )
        weights = jax.nn.softmax(logp0)

        # 更新轨迹
        Ybar, new_noise_scale = self.update_fn(
            weights, Y0s, noise_scale, Ybar_i
        )
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)

        info = {"rews": rews, "new_noise_scale": new_noise_scale}
        return rng, Ybar, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        """时间步推进：滚动轨迹并补零。"""
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        return self.u2node_vmap(u)
