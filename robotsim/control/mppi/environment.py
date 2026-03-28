"""MPPI 环境抽象与四足机器人实现。

BaseRobotEnv 定义了 MPPI 所需的环境交互模板，
DreamerEnv 实现了具体的四足机器人物理模型。
"""

import functools
from abc import ABC, abstractmethod
from typing import Any, List, Sequence

import jax
import mujoco
import numpy as np
from brax import math
from brax.base import System
from brax.envs.base import PipelineEnv, State
import brax.base as base
from jax import numpy as jnp

from robotsim.control.mppi.config import Config
from robotsim.control.mppi.math_utils import MathUtils


class BaseRobotEnv(PipelineEnv, ABC):
    """MPPI 环境基类（模板方法模式）。

    定义了环境交互的通用流程（reset → step → render），
    子类负责实现具体的物理模型、奖励函数和终止条件。
    """

    def __init__(self, config: Config):
        assert jnp.allclose(config.dt % config.timestep, 0.0), (
            "timestep must be divisible by dt"
        )

        self._config = config
        self._math_utils = MathUtils()
        n_frames = int(config.dt / config.timestep)

        sys = self._make_system()
        super().__init__(sys, "mjx", n_frames, config.debug)

        self._setup_constants()
        self._nv = self.sys.nv
        self._nq = self.sys.nq

    @abstractmethod
    def _make_system(self) -> System:
        """创建并返回 Brax 物理系统（子类必须实现）。"""
        pass

    @abstractmethod
    def _setup_constants(self):
        """设置环境相关的常量，如步态参数、关节范围等（子类必须实现）。"""
        pass

    @abstractmethod
    def _get_initial_state_info(self) -> dict:
        """获取初始化的状态信息字典（子类必须实现）。"""
        pass

    @abstractmethod
    def _update_targets(
        self, state: State, cmd_rng: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """更新目标速度和角速度（子类必须实现）。"""
        pass

    @abstractmethod
    def _compute_reward(
        self, pipeline_state: base.State, state_info: dict
    ) -> jax.Array:
        """计算当前状态的奖励（子类必须实现）。"""
        pass

    @abstractmethod
    def _check_done(self, pipeline_state: base.State) -> jax.Array:
        """检查是否达到终止条件（子类必须实现）。"""
        pass

    @abstractmethod
    def _update_state_info(self, pipeline_state: base.State, state_info: dict):
        """更新每一步的状态信息（子类必须实现）。"""
        pass

    @abstractmethod
    def _get_obs(
        self, pipeline_state: base.State, state_info: dict
    ) -> jax.Array:
        """获取当前状态的观察值（子类必须实现）。"""
        pass

    @functools.partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        """动作空间映射到关节目标角度。"""
        act_normalized = (act * self._config.action_scale + 1.0) / 2.0
        joint_targets = self.joint_range[:, 0] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )
        return jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: jax.Array, pipeline_state) -> jax.Array:
        """动作转换为关节扭矩（PD 控制）。"""
        joint_target = self.act2joint(act)
        q = pipeline_state.qpos[7:][: len(joint_target)]
        qd = pipeline_state.qvel[6:][: len(joint_target)]
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd
        return jnp.clip(
            tau,
            self.joint_torque_range[:, 0],
            self.joint_torque_range[:, 1],
        )

    def reset(self, rng: jax.Array) -> State:
        """重置环境到初始状态。"""
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = self._get_initial_state_info()
        state_info["rng"] = rng

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        return State(pipeline_state, obs, reward, done, {}, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        """环境步进（通用流程）。"""
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        if self._config.leg_control == "position":
            ctrl = self.act2joint(action)
        else:
            ctrl = self.act2tau(action, state.pipeline_state)

        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        vel_tar, ang_vel_tar = self._update_targets(state, cmd_rng)

        ramp_factor = jnp.minimum(
            state.info["step"] * self.dt / self._config.ramp_up_time, 1.0
        )
        state.info["vel_tar"] = vel_tar * ramp_factor
        state.info["ang_vel_tar"] = ang_vel_tar * ramp_factor

        reward = self._compute_reward(pipeline_state, state.info)
        done = self._check_done(pipeline_state)
        self._update_state_info(pipeline_state, state.info)
        obs = self._get_obs(pipeline_state, state.info)
        state.info["rng"] = rng
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        """渲染轨迹序列。"""
        return super().render(
            trajectory, camera=camera or "track", width=width, height=height
        )


class DreamerEnv(BaseRobotEnv):
    """四足机器人（Dreamer）环境实现。

    通过 Config.robot 段参数化模型路径、关节范围等，
    支持 dog1/dog2 等不同机器人构型。
    """

    # 默认关节范围（dog1）
    _DEFAULT_JOINT_RANGE = [
        [-0.5, 0.5], [0.4, 1.4], [-2.35, -1.35],  # 前左腿
        [-0.5, 0.5], [0.4, 1.4], [-2.35, -1.35],  # 前右腿
        [-0.5, 0.5], [0.4, 1.4], [-2.3, -1.3],    # 后左腿
        [-0.5, 0.5], [0.4, 1.4], [-2.3, -1.3],    # 后右腿
    ]

    # 默认步态参数
    _DEFAULT_GAIT_PHASE = {
        "stand": [0.0, 0.0, 0.0, 0.0],
        "walk": [0.0, 0.5, 0.75, 0.25],
        "trot": [0.0, 0.5, 0.5, 0.0],
        "canter": [0.0, 0.33, 0.33, 0.66],
        "gallop": [0.0, 0.05, 0.4, 0.35],
    }

    _DEFAULT_GAIT_PARAMS = {
        "stand": [1.0, 1.0, 0.0],
        "walk": [0.75, 1.0, 0.08],
        "trot": [0.45, 2, 0.08],
        "canter": [0.4, 4, 0.06],
        "gallop": [0.3, 3.5, 0.10],
    }

    def __init__(self, config: Config):
        super().__init__(config)

    def _make_system(self) -> System:
        """加载 MuJoCo XML 并创建 Brax 系统。"""
        from brax.io import mjcf

        model_path = self._config.robot.model_path
        sys = mjcf.load(model_path)
        return sys.tree_replace({"opt.timestep": self._config.timestep})

    def _setup_constants(self):
        """设置环境常量，支持配置覆盖。"""
        robot_cfg = self._config.robot

        self._foot_radius = robot_cfg.foot_radius

        # 步态配置（支持覆盖）
        self._gait_phase = {
            k: jnp.array(v) for k, v in self._DEFAULT_GAIT_PHASE.items()
        }

        gait_params = dict(self._DEFAULT_GAIT_PARAMS)
        if robot_cfg.gait_params_override:
            gait_params.update(robot_cfg.gait_params_override)
        self._gait_params = {
            k: jnp.array(v) for k, v in gait_params.items()
        }

        # 机器人结构索引
        self._torso_idx = mujoco.mj_name2id(
            self.sys.mj_model,
            mujoco.mjtObj.mjOBJ_BODY.value,
            robot_cfg.torso_body_name,
        )
        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self._default_pose = self.sys.mj_model.keyframe("home").qpos[7:]

        # 关节范围（支持覆盖）
        self.physical_joint_range = self.sys.jnt_range[1:]
        if robot_cfg.joint_range_override:
            self.joint_range = jnp.array(robot_cfg.joint_range_override)
        else:
            self.joint_range = jnp.array(self._DEFAULT_JOINT_RANGE)
        self.joint_torque_range = self.sys.actuator_ctrlrange

        # 足部站点 ID
        feet_site_id = [
            mujoco.mj_name2id(
                self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, name
            )
            for name in robot_cfg.feet_site_names
        ]
        self._feet_site_id = jnp.array(feet_site_id)

    def _get_initial_state_info(self) -> dict:
        """获取 Dreamer 初始状态信息字典。"""
        return {
            "pos_tar": jnp.array([0.282, 0.0, self._config.robot.pos_tar_z]),
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(4),
            "z_feet_tar": jnp.zeros(4),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4),
        }

    def _update_targets(
        self, state: State, cmd_rng: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """更新目标速度（支持随机目标模式）。"""
        def sample_command():
            _, key1, key2, key3 = jax.random.split(cmd_rng, 4)
            lin_vel_x = jax.random.uniform(key1, (1,), minval=-1.5, maxval=1.5)
            lin_vel_y = jax.random.uniform(key2, (1,), minval=-0.5, maxval=0.5)
            ang_vel_yaw = jax.random.uniform(
                key3, (1,), minval=-1.5, maxval=1.5
            )
            return (
                jnp.array([lin_vel_x[0], lin_vel_y[0], 0.0]),
                jnp.array([0.0, 0.0, ang_vel_yaw[0]]),
            )

        def default_command():
            return (
                jnp.array(
                    [self._config.default_vx, self._config.default_vy, 0.0]
                ),
                jnp.array([0.0, 0.0, self._config.default_vyaw]),
            )

        return jax.lax.cond(
            (state.info["randomize_target"])
            & (state.info["step"] % 500 == 0),
            sample_command,
            default_command,
        )

    def _compute_reward(self, pipeline_state, state_info):
        """计算综合奖励（步态 + 直立 + 偏航 + 速度 + 高度）。"""
        x, xd = pipeline_state.x, pipeline_state.xd

        # 步态奖励
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._config.gait]
        phases = self._gait_phase[self._config.gait]
        z_feet_tar = self._math_utils.get_foot_step(
            duty_ratio, cadence, amplitude, phases,
            state_info["step"] * self.dt,
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)

        # 直立奖励
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))

        # 偏航角奖励
        yaw_tar = (
            state_info["yaw_tar"]
            + state_info["ang_vel_tar"][2] * self.dt * state_info["step"]
        )
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        d_yaw = yaw - yaw_tar
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))

        # 速度奖励
        vb = self._math_utils.global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = self._math_utils.global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0,
            x.rot[self._torso_idx - 1],
        )
        reward_vel = -jnp.sum((vb[:2] - state_info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum(
            (ab[2] - state_info["ang_vel_tar"][2]) ** 2
        )

        # 高度奖励
        reward_height = -jnp.sum(
            (x.pos[self._torso_idx - 1, 2] - state_info["pos_tar"][2]) ** 2
        )

        return (
            reward_gaits * 0.0
            + reward_upright * 1.0
            + reward_yaw * 1.0
            + reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_height * 2.0
        )

    def _check_done(self, pipeline_state):
        """检查终止条件（倾倒、关节超限、高度过低）。"""
        x = pipeline_state.x
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.qpos[7:]

        done = (
            jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        )
        done |= jnp.any(joint_angles < self.joint_range[:, 0])
        done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.4
        return done.astype(jnp.float32)

    def _update_state_info(self, pipeline_state, state_info):
        """更新足部接触状态和步计数。"""
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        foot_contact_z = z_feet - self._foot_radius
        contact = foot_contact_z < 1e-3
        contact_filt = contact | state_info["last_contact"]

        state_info["step"] += 1
        state_info["z_feet"] = z_feet
        state_info["feet_air_time"] += self.dt
        state_info["feet_air_time"] *= ~contact_filt
        state_info["last_contact"] = contact

    def _get_obs(
        self, pipeline_state: base.State, state_info: dict[str, Any]
    ) -> jax.Array:
        """获取观察向量（速度目标 + 控制量 + 关节状态）。"""
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = self._math_utils.global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = self._math_utils.global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0,
            x.rot[self._torso_idx - 1],
        )

        return jnp.concatenate([
            state_info["vel_tar"],
            state_info["ang_vel_tar"],
            pipeline_state.ctrl,
            pipeline_state.qpos,
            vb,
            ab,
            pipeline_state.qvel[6:],
        ])
