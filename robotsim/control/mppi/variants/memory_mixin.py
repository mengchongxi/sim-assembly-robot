"""历史动作记忆 Mixin，为 MPPI 控制器添加基于历史动作的采样先验。

通过维护固定长度的动作历史缓冲区，结合 FIR 低通滤波，
为 MPPI 采样提供 history-repeat 先验分布，减少探索空间，提升控制稳定性。

用法:
    class MemoryController(HistoryMemoryMixin, DIALMPCController):
        pass
"""

import functools

import jax
from jax import numpy as jnp


class MemoryDreamerEnvMixin:
    """为 DreamerEnv 添加历史动作缓冲区的 Mixin。

    在环境状态中维护 history_u 字段，每一步自动滚动更新。

    用法:
        class MemoryDreamerEnv(MemoryDreamerEnvMixin, DreamerEnv):
            pass
    """

    def _get_initial_state_info(self) -> dict:
        """扩展初始状态，添加历史动作缓冲区。"""
        info = super()._get_initial_state_info()
        info["history_u"] = jnp.zeros(
            (64, self.action_size), dtype=jnp.float32
        )
        return info

    def step(self, state, action):
        """扩展步进逻辑，维护历史动作缓冲区。"""
        # 更新历史缓冲
        hist = state.info["history_u"]
        hist = jnp.roll(hist, -1, axis=0)
        hist = hist.at[-1].set(action)
        state.info["history_u"] = hist

        return super().step(state, action)


class HistoryMemoryMixin:
    """为 DIALMPCController 添加混合采样策略的 Mixin。

    将采样分为两部分：
    1. warm-start 分支：标准高斯噪声采样（保持探索）
    2. history-repeat 分支：基于历史动作的周期重复采样（利用经验）

    FIR 低通滤波器用于平滑 history-repeat 分支的噪声。

    用法:
        class MemoryController(HistoryMemoryMixin, DIALMPCController):
            pass
    """

    def __init__(self, config, env):
        super().__init__(config, env)

        # 时间长度做成静态常量，避免 JAX ConcretizationTypeError
        self.T = int(config.Hsample + 1)
        self._t_idx = jnp.arange(self.T, dtype=jnp.int32)

        # FIR 低通滤波系数
        self._fir_coeffs = jnp.array(
            [
                0.10422766377112629,
                0.3239870556899027,
                0.3658903830367387,
                0.3239870556899027,
                0.10422766377112629,
            ],
            dtype=jnp.float32,
        )

        # history 缓冲区长度 (>= 2*T)
        self.history_len = int(max(64, 2 * self.T))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _fir_smooth(self, noise_u, coeffs):
        """对每个 action 维做独立 FIR 低通滤波。

        Args:
            noise_u: (T+L-1, nu) 含额外前缀的噪声序列。
            coeffs: (L,) FIR 滤波器系数。

        Returns:
            (T, nu) 平滑后的序列。
        """
        nu = noise_u.shape[1]
        kernel = (
            jnp.eye(nu, dtype=noise_u.dtype)[None, :, :]
            * coeffs[:, None, None]
        )
        x = noise_u[None, :, :]  # (1, T+L-1, nu) NWC
        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        return y[0]  # (T, nu)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _build_repeat_u(self, history_u, repeat_len):
        """从历史缓冲区末尾截取并周期重复到长度 T。

        Args:
            history_u: (K, nu) 历史动作序列。
            repeat_len: 重复段长度。

        Returns:
            (T, nu) 周期重复的动作序列。
        """
        K = history_u.shape[0]
        repeat_len = repeat_len.astype(jnp.int32)
        start = K - repeat_len
        idx = start + (self._t_idx % repeat_len)
        return history_u[idx]

    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, noise_scale):
        """执行一次反向扩散（混合 warm-start + history-repeat 采样）。

        将采样预算一分为二：
        - 前半 warm-start：标准高斯采样
        - 后半 history-repeat：基于历史动作的周期采样 + FIR 平滑
        """
        Nh = self.config.Nsample // 2
        Nw = self.config.Nsample - Nh
        nu = self.nu
        T = self.T
        L = self._fir_coeffs.shape[0]

        rng, rng_warm, rng_hist = jax.random.split(rng, 3)

        # 1) warm-start 分支
        eps_Y = jax.random.normal(
            rng_warm, (Nw, self.config.Hnode + 1, nu)
        )
        Y_warm = eps_Y * noise_scale[None, :, None] + Ybar_i
        Y_warm = Y_warm.at[:, 0].set(Ybar_i[0, :])
        Y_warm = jnp.clip(Y_warm, -1.0, 1.0)

        # 2) history-repeat 分支
        history_u = state.info["history_u"]
        have_hist = state.info["step"] >= (T + 1)

        def make_hist_samples(_):
            u_bar = self.node2u_vmap(Ybar_i)

            min_len = T // 2 + 1
            span = T - min_len + 1
            rep_lens = (
                (jnp.arange(Nh, dtype=jnp.int32) % span) + min_len
            )

            u_rep = jax.vmap(
                lambda rlen: self._build_repeat_u(history_u, rlen)
            )(rep_lens)

            alpha = (self._t_idx.astype(jnp.float32) + 1.0) / float(T)
            alpha = alpha[:, None]
            u_mix = (
                alpha[None, :, :] * u_rep
                + (1.0 - alpha[None, :, :]) * u_bar[None, :, :]
            )

            sigma_u = self.node2u(noise_scale)[:, None]
            white = jax.random.normal(rng_hist, (Nh, T + L - 1, nu))
            smooth = jax.vmap(
                lambda w: self._fir_smooth(w, self._fir_coeffs)
            )(white)

            history_noise_ratio = 0.5
            u_mix = u_mix + smooth * (
                history_noise_ratio * sigma_u[None, :, :]
            )

            u_mix = jnp.clip(u_mix, -1.0, 1.0)
            Y_hist = jax.vmap(self.u2node_vmap)(u_mix)
            Y_hist = Y_hist.at[:, 0].set(Ybar_i[0, :])
            Y_hist = jnp.clip(Y_hist, -1.0, 1.0)
            return Y_hist

        Y_hist = jax.lax.cond(
            have_hist, make_hist_samples, lambda _: Y_warm[:Nh], operand=None
        )

        # 合并候选轨迹
        Y0s = jnp.concatenate(
            [Y_hist, Y_warm, Ybar_i[None]], axis=0
        )

        # 3) rollout + softmax 权重更新
        us = self.node2u_vvmap(Y0s)
        rewss, pipeline_statess = self.rollout_us_vmap(state, us)
        rews = rewss.mean(axis=-1)

        rew_Ybar_i = rewss[-1].mean()
        denom = rews.std(axis=-1) + 1e-6
        logp0 = (rews - rew_Ybar_i) / denom / self.config.temp_sample
        weights = jax.nn.softmax(logp0)

        Ybar, new_noise_scale = self.update_fn(
            weights, Y0s, noise_scale, Ybar_i
        )
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)

        info = {"rews": rews, "new_noise_scale": new_noise_scale}
        return rng, Ybar, info
