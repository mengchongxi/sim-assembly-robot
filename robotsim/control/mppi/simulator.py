"""MPPI 仿真编排器。

组装环境、控制器、结果管理器，驱动主仿真循环。
"""

import os
import time

import jax
import numpy as np
from jax import numpy as jnp
from tqdm import tqdm

from robotsim.control.mppi.config import Config, ConfigManager
from robotsim.control.mppi.controller import DIALMPCController
from robotsim.control.mppi.environment import DreamerEnv
from robotsim.control.mppi.math_utils import MathUtils
from robotsim.control.mppi.results import ResultsManager


class RobotSimulator:
    """MPPI 仿真主编排器。

    负责：
    1. 从配置创建或接收环境、控制器、结果管理器
    2. 驱动主仿真循环（step → MPPI优化 → 记录）
    3. 保存结果和可视化

    支持两种构造方式：
    - 从 config_path 自动创建所有组件
    - 通过关键字参数注入已有组件（用于 mixin 组合）
    """

    def __init__(
        self,
        config_path: str = None,
        *,
        config: Config = None,
        env: DreamerEnv = None,
        controller: DIALMPCController = None,
        results_manager: ResultsManager = None,
    ):
        self._setup_xla()

        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = ConfigManager.load(config_path)
        else:
            raise ValueError("必须提供 config_path 或 config 参数")

        self.env = env or DreamerEnv(self.config)
        self.controller = controller or DIALMPCController(
            self.config, self.env
        )
        self.results_manager = results_manager or ResultsManager(self.config)

        self.reset_env = jax.jit(self.env.reset)
        self.step_env = jax.jit(self.env.step)
        self.math_utils = MathUtils()

    def _setup_xla(self):
        """设置 XLA GPU 优化选项。"""
        xla_flags = os.environ.get("XLA_FLAGS", "")
        xla_flags += " --xla_gpu_triton_gemm_any=True"
        os.environ["XLA_FLAGS"] = xla_flags

    def run_simulation(self):
        """运行完整 MPPI 仿真循环。

        Returns:
            包含 rewards、rollout、controls、infos、pose_data、joint_data 的结果字典。
        """
        print("🚀 Creating environment")

        # 初始化
        rng = jax.random.PRNGKey(seed=self.config.seed)
        rng, rng_reset = jax.random.split(rng)
        state = self.reset_env(rng_reset)
        Y0 = jnp.zeros([self.config.Hnode + 1, self.controller.nu])

        # 数据收集
        rews, rollout, us, infos = [], [], [], []
        pose_data, joint_data = [], []
        body_vel_data, body_ang_vel_data = [], []

        # 反向扫描函数
        def reverse_scan(rng_Y0_state, factor):
            rng, Y0, state = rng_Y0_state
            rng, Y0, info = self.controller.reverse_once(
                state, rng, Y0, factor
            )
            return (rng, Y0, state), info

        # 主循环
        with tqdm(range(self.config.n_steps), desc="Rollout") as pbar:
            for t in pbar:
                # 执行控制
                state = self.step_env(state, Y0[0])
                rollout.append(state.pipeline_state)
                rews.append(state.reward)
                us.append(Y0[0])

                # 记录数据
                self._record_data(
                    state, pose_data, joint_data,
                    body_vel_data, body_ang_vel_data,
                )

                # 时间推进和优化
                Y0 = self.controller.shift(Y0)
                n_diffuse = (
                    self.config.Ndiffuse_init
                    if t == 0
                    else self.config.Ndiffuse
                )

                if t == 0:
                    print("Performing JIT on DIAL-MPC")

                t0 = time.time()
                traj_diffuse_factors = (
                    self.controller.sigma_control
                    * self.config.traj_diffuse_factor
                    ** (jnp.arange(n_diffuse))[:, None]
                )
                (rng, Y0, _), info = jax.lax.scan(
                    reverse_scan, (rng, Y0, state), traj_diffuse_factors
                )
                infos.append(info)
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({
                    "rew": f"{state.reward:.2e}",
                    "freq": f"{freq:.2f}",
                })

        # 保存结果
        timestamp = self.results_manager.save_results(
            pose_data, joint_data, body_vel_data, body_ang_vel_data
        )
        self.results_manager.save_visualization_html(
            self.env, rollout, timestamp
        )

        return {
            "rewards": rews,
            "rollout": rollout,
            "controls": us,
            "infos": infos,
            "pose_data": pose_data,
            "joint_data": joint_data,
        }

    def _record_data(
        self, state, pose_data, joint_data, body_vel_data, body_ang_vel_data
    ):
        """记录单步仿真数据（位姿、关节角、机体速度）。"""
        torso_idx = self.env._torso_idx - 1

        # 位姿数据
        base_pos = state.pipeline_state.x.pos[torso_idx]
        base_quat = state.pipeline_state.x.rot[torso_idx]
        base_quat_xyzw = jnp.array([
            base_quat[1], base_quat[2], base_quat[3], base_quat[0]
        ])
        pose = jnp.concatenate([base_pos, base_quat_xyzw])
        pose_data.append(np.array(pose))
        joint_data.append(np.array(state.pipeline_state.qpos[7:]))

        # 机体速度数据
        x, xd = state.pipeline_state.x, state.pipeline_state.xd
        torso_rot = x.rot[torso_idx]

        global_lin_vel = xd.vel[torso_idx]
        body_lin_vel = self.math_utils.global_to_body_velocity(
            global_lin_vel, torso_rot
        )
        body_vel_data.append(np.array(body_lin_vel))

        global_ang_vel_rad = xd.ang[torso_idx]
        body_ang_vel = self.math_utils.global_to_body_velocity(
            global_ang_vel_rad, torso_rot
        )
        body_ang_vel_data.append(np.array(body_ang_vel))
