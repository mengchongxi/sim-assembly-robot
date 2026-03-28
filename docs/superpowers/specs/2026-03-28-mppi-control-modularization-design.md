# MPPI 控制模块模块化拆分设计

## 问题陈述

`tmp/` 目录中包含约 4000 行未模块化的 MPPI 控制算法和显示脚本代码。这些代码实现了基于 JAX/Brax 的四足机器人运动控制（DIAL-MPC/MPPI），但存在以下问题：

1. 4 个 MPPI 变体之间约 80% 代码重复
2. 所有逻辑（配置、数学工具、环境、控制器、仿真、结果管理）混在单文件中
3. 未与现有 robotsim 框架集成，缺乏接口抽象
4. 配置文件重复（两份几乎相同的 YAML）

## 设计目标

- 将控制算法作为 `robotsim/control/` 顶级模块集成到框架中
- 通过基类 + mixin 模式消除 MPPI 变体间的代码重复
- 定义清晰的控制器接口（`ControllerBase`），为未来扩展（MPC、RL 等）做铺垫
- 将 Brax 仿真能力作为 `simulation/brax/` 后端接入
- MPPI 未来要迁移到 MuJoCo 后端，当前保持 Brax 实现但做好接口抽象
- 合并 YAML 配置到集中目录，支持 override 机制

## 不在范围内

- 通用运动生成（当前只接受 `models/mujoco/icra2026_model/` 中的指定 XML）
- Display 脚本迁移（现有 `simulation/mujoco/viewer` 已覆盖需求）
- 修改现有 robotsim 框架的已有模块

---

## 架构设计

### 目录结构

```
robotsim/
├── robotsim/
│   ├── control/                          # 新增：控制算法顶级模块
│   │   ├── __init__.py                   # 导出 ControllerBase 和公共接口
│   │   ├── base.py                       # ControllerBase 协议定义
│   │   └── mppi/                         # MPPI 算法族
│   │       ├── __init__.py               # 导出 MPPI 公共 API
│   │       ├── config.py                 # Config 数据类 + ConfigManager
│   │       ├── math_utils.py             # JAX 向量化数学工具
│   │       ├── environment.py            # BaseRobotEnv + DreamerEnv
│   │       ├── controller.py             # DIALMPCController 基类
│   │       ├── simulator.py              # RobotSimulator 编排器
│   │       ├── results.py                # ResultsManager 输出管理
│   │       └── variants/                 # MPPI 变体
│   │           ├── __init__.py
│   │           ├── memory_mixin.py       # HistoryMemoryMixin（历史动作缓冲）
│   │           └── velocity_tracker.py   # VelocityTrackingMixin（速度跟踪增强）
│   │
│   ├── simulation/
│   │   ├── brax/                         # 新增：Brax 仿真后端
│   │   │   ├── __init__.py
│   │   │   └── runner.py                 # BraxRunner（Brax 仿真运行器）
│   │   ├── mujoco/                       # 现有，不修改
│   │   └── bullet/                       # 现有，不修改
│   │
│   └── core/                             # 现有，可能新增控制器接口
│       └── interfaces.py                 # 追加 ControllerBase 协议（或放 control/base.py）
│
├── configs/
│   └── control/
│       └── mppi/
│           ├── base_trot.yaml            # 合并后的基础 trot 步态配置
│           └── README.md                 # 配置参数说明
│
├── scripts/
│   ├── run_mppi_control.py               # 新增：MPPI 控制入口脚本
│   └── ...existing...
│
├── models/mujoco/icra2026_model/         # 现有 XML 模型，不修改
└── tmp/                                  # 重构完成后可清理
```

---

### 模块详细设计

#### 1. `control/base.py` — 控制器抽象接口

定义所有控制算法必须遵循的协议，使框架可以透明切换不同控制器。

```python
from typing import Protocol, runtime_checkable, Any, Dict
import numpy as np

@runtime_checkable
class ControllerBase(Protocol):
    """所有控制算法的统一接口。
    
    控制器接受当前状态，返回控制动作。
    具体实现可以是 MPPI、MPC、PID、RL policy 等。
    """

    def reset(self, initial_state: Dict[str, Any]) -> None:
        """重置控制器状态。"""
        ...

    def compute_action(self, state: Dict[str, Any]) -> np.ndarray:
        """根据当前状态计算控制动作。
        
        Args:
            state: 包含机器人当前状态信息的字典
                   （位姿、速度、关节角等，具体内容由实现定义）
        
        Returns:
            控制动作数组（关节力矩或目标位置，取决于 control_mode）
        """
        ...

    @property
    def control_mode(self) -> str:
        """控制模式：'torque' 或 'position'。"""
        ...
```

遵循现有框架的 Protocol 模式（与 `core/interfaces.py` 中的 `TrajectoryGenerator` 一致）。

#### 2. `control/mppi/config.py` — 配置管理

从 4 个 MPPI 文件中提取统一的配置数据类和加载器。

```python
@dataclass
class MPPIConfig:
    """MPPI 算法统一配置。"""
    # 算法参数
    seed: int = 0
    n_steps: int = 800
    Nsample: int = 2024
    Hsample: int = 16
    Hnode: int = 4
    Ndiffuse: int = 3
    Ndiffuse_init: int = 10
    temp_sample: float = 0.05
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5
    update_method: str = "mppi"

    # 物理参数
    dt: float = 0.02
    timestep: float = 0.02
    leg_control: str = "torque"
    action_scale: float = 1.0

    # PD 增益
    kp: float = 100.0
    kd: float = 2.0

    # 运动目标
    default_vx: float = 0.8
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 1.0
    gait: str = "trot"

    # 输出
    output_dir: str = "output/mppi"

    # 变体功能开关
    enable_memory: bool = False
    history_len: int = 64
    enable_velocity_tracking: bool = False

class ConfigManager:
    """YAML 配置加载器，支持 override 机制。"""

    @staticmethod
    def load(yaml_path: str, **overrides) -> MPPIConfig:
        """加载 YAML 配置并应用覆盖值。"""
        ...

    @staticmethod
    def to_jax_arrays(config: MPPIConfig) -> Dict[str, jnp.ndarray]:
        """将配置转换为 JAX 数组以便 JIT 编译。"""
        ...
```

**改进点：**
- 合并 dog1/dog2 两份配置为一份 `base_trot.yaml`
- 通过 `overrides` 参数区分不同机器人
- 功能开关 `enable_memory` / `enable_velocity_tracking` 替代 4 个独立文件

#### 3. `control/mppi/math_utils.py` — 数学工具

从 4 份重复代码中提取唯一的数学工具模块。

```python
class MathUtils:
    """MPPI 所需的 JAX 向量化数学操作。"""

    @staticmethod
    @jax.jit
    def body_to_global_velocity(
        velocity: jnp.ndarray, quaternion: jnp.ndarray
    ) -> jnp.ndarray:
        """将体坐标系速度转换为全局坐标系。"""
        ...

    @staticmethod
    @jax.jit
    def get_foot_step(
        duty_ratio: float, phase: jnp.ndarray, ...
    ) -> jnp.ndarray:
        """计算步态足端高度。"""
        ...

    @staticmethod
    @jax.jit
    def quat_rotate(quaternion: jnp.ndarray, vector: jnp.ndarray) -> jnp.ndarray:
        """四元数旋转向量。"""
        ...
```

#### 4. `control/mppi/environment.py` — 环境抽象

```python
class BaseRobotEnv(ABC):
    """MPPI 环境基类（模板方法模式）。

    定义了 MPPI 所需的环境交互接口：
    - 动作到关节的映射
    - PD 控制力矩计算
    - 奖励函数
    - 状态信息提取
    """

    @abstractmethod
    def act2joint(self, action: jnp.ndarray) -> jnp.ndarray:
        """动作空间到关节位置的映射。"""
        ...

    @abstractmethod
    def act2tau(self, action: jnp.ndarray, pipeline_state) -> jnp.ndarray:
        """通过 PD 控制计算关节力矩。"""
        ...

    @abstractmethod
    def compute_reward(self, state, action) -> float:
        """计算奖励值。"""
        ...


class DreamerEnv(BaseRobotEnv):
    """四足机器人环境实现（基于 Brax PipelineEnv）。

    当前绑定到 icra2026_model 的 XML 定义。
    未来可扩展为从任意 XML 自动配置。
    """
    ...
```

#### 5. `control/mppi/controller.py` — MPPI 控制器核心

```python
class DIALMPCController:
    """DIAL-MPC (Diffusion-Inspired Annealing for Legged MPC) 控制器。

    实现核心 MPPI 轨迹优化循环：
    1. 采样候选轨迹（node2u 样条插值）
    2. 在环境中滚动仿真
    3. 通过 softmax 加权更新最优轨迹
    4. 返回第一步动作
    """

    def __init__(self, config: MPPIConfig, env: BaseRobotEnv):
        ...

    def run_mppi_step(self, state, rng_key) -> Tuple[jnp.ndarray, Any]:
        """执行单步 MPPI 优化。"""
        ...

    def softmax_update(self, costs, trajectories) -> jnp.ndarray:
        """基于代价的 softmax 加权轨迹更新。"""
        ...

    def node2u(self, nodes) -> jnp.ndarray:
        """控制节点到完整轨迹的样条插值。"""
        ...
```

#### 6. `control/mppi/variants/` — 变体 Mixin

**memory_mixin.py:**
```python
class HistoryMemoryMixin:
    """为 MPPI 控制器添加历史动作记忆能力。

    维护固定长度的动作历史缓冲区，作为 MPPI 采样的先验分布，
    减少探索空间，提升控制稳定性。

    包含 FIR 低通滤波器用于动作平滑。
    """
    history_len: int = 64

    def init_history_buffer(self, action_size: int) -> jnp.ndarray:
        ...

    def update_history(self, history: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        ...

    def get_history_prior(self, history: jnp.ndarray) -> jnp.ndarray:
        ...
```

**velocity_tracker.py:**
```python
class VelocityTrackingMixin:
    """为 MPPI 添加详细的速度跟踪与记录能力。

    记录体坐标系/世界坐标系的线速度和角速度，
    计算与目标速度的误差，输出到 CSV。
    """

    def init_velocity_records(self) -> Dict:
        ...

    def record_velocity(self, state, target) -> Dict:
        ...

    def save_velocity_csv(self, records: List[Dict], path: str) -> None:
        ...
```

**组合使用：**
```python
# 基础版本
simulator = RobotSimulator(config, controller, env)

# 带记忆的版本
class MemoryMPPIController(HistoryMemoryMixin, DIALMPCController):
    pass

# 带速度跟踪的版本
class TrackedMPPISimulator(VelocityTrackingMixin, RobotSimulator):
    pass

# 完整版本（dog2_mem 的等价物）
class FullMPPIController(HistoryMemoryMixin, DIALMPCController):
    pass
class FullMPPISimulator(VelocityTrackingMixin, RobotSimulator):
    pass
```

#### 7. `control/mppi/simulator.py` — 编排器

```python
class RobotSimulator:
    """MPPI 仿真编排器。

    负责组装环境、控制器、结果管理器，
    并驱动主仿真循环。
    """

    def __init__(
        self,
        config: MPPIConfig,
        controller: DIALMPCController,
        env: BaseRobotEnv,
        results_manager: Optional[ResultsManager] = None,
    ):
        ...

    def run_simulation(self) -> Dict[str, Any]:
        """执行完整仿真循环。

        Returns:
            包含轨迹、关节数据、性能指标的结果字典
        """
        ...
```

#### 8. `control/mppi/results.py` — 结果管理

```python
class ResultsManager:
    """仿真结果的收集、保存和可视化。"""

    def __init__(self, config: MPPIConfig):
        ...

    def record_step(self, step: int, state: Dict, action: np.ndarray) -> None:
        """记录单步数据。"""
        ...

    def save_all(self, output_dir: str) -> None:
        """保存所有结果（CSV、轨迹、汇总）。"""
        ...
```

#### 9. `simulation/brax/runner.py` — Brax 仿真后端

```python
class BraxRunner:
    """Brax 物理仿真运行器。

    封装 Brax 环境的创建和步进逻辑，
    提供与 MuJoCo runner 对称的接口。
    未来 MPPI 迁移到 MuJoCo 时，
    只需切换 runner 而不影响控制逻辑。
    """

    def __init__(self, model_path: str, config: Dict):
        ...

    def create_env(self, env_class: type) -> Any:
        """创建 Brax 环境实例。"""
        ...

    def step(self, state, action) -> Tuple[Any, float, bool]:
        """执行一步仿真。"""
        ...
```

---

### 配置设计

#### `configs/control/mppi/base_trot.yaml`

合并两份重复的 YAML 为一份模板：

```yaml
# MPPI 控制器基础配置 - trot 步态
seed: 0
output_dir: "output/mppi/{robot_id}"  # 支持模板变量
n_steps: 800

# DIAL-MPC 算法参数
algorithm:
  Nsample: 2024
  Hsample: 16
  Hnode: 4
  Ndiffuse: 3
  Ndiffuse_init: 10
  temp_sample: 0.05
  horizon_diffuse_factor: 0.9
  traj_diffuse_factor: 0.5
  update_method: "mppi"

# 物理仿真
physics:
  dt: 0.02
  timestep: 0.02
  leg_control: "torque"
  action_scale: 1.0

# PD 控制
pd_gains:
  kp: 100.0
  kd: 2.0

# 运动目标
motion:
  default_vx: 0.8
  default_vy: 0.0
  default_vyaw: 0.0
  ramp_up_time: 1.0
  gait: "trot"

# 变体功能（按需启用）
variants:
  enable_memory: false
  history_len: 64
  enable_velocity_tracking: false
```

#### 使用方式

```bash
# 基础运行
python scripts/run_mppi_control.py --config configs/control/mppi/base_trot.yaml --robot-id dog1

# 启用记忆
python scripts/run_mppi_control.py --config configs/control/mppi/base_trot.yaml --robot-id dog1 --enable-memory

# 启用速度跟踪
python scripts/run_mppi_control.py --config configs/control/mppi/base_trot.yaml --robot-id dog2 --enable-velocity-tracking
```

---

### 入口脚本

#### `scripts/run_mppi_control.py`

```python
"""MPPI 控制算法入口脚本。

用法:
    python scripts/run_mppi_control.py --config configs/control/mppi/base_trot.yaml --robot-id dog1
    python scripts/run_mppi_control.py --config configs/control/mppi/base_trot.yaml --robot-id dog2 --enable-memory --enable-velocity-tracking
"""

import argparse
from robotsim.control.mppi import ConfigManager, DIALMPCController, DreamerEnv, RobotSimulator, ResultsManager
from robotsim.control.mppi.variants import HistoryMemoryMixin, VelocityTrackingMixin


def main():
    parser = argparse.ArgumentParser(description="MPPI 四足机器人运动控制")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--robot-id", required=True, help="机器人标识 (dog1, dog2)")
    parser.add_argument("--enable-memory", action="store_true", help="启用历史动作记忆")
    parser.add_argument("--enable-velocity-tracking", action="store_true", help="启用速度跟踪")
    args = parser.parse_args()

    # 加载配置
    config = ConfigManager.load(
        args.config,
        robot_id=args.robot_id,
        enable_memory=args.enable_memory,
        enable_velocity_tracking=args.enable_velocity_tracking,
    )

    # 构建环境
    env = DreamerEnv(config)

    # 构建控制器（根据变体选择类）
    controller_cls = DIALMPCController
    if config.enable_memory:
        class MemoryController(HistoryMemoryMixin, DIALMPCController):
            pass
        controller_cls = MemoryController

    controller = controller_cls(config, env)

    # 构建仿真器
    simulator_cls = RobotSimulator
    if config.enable_velocity_tracking:
        class TrackedSimulator(VelocityTrackingMixin, RobotSimulator):
            pass
        simulator_cls = TrackedSimulator

    results_manager = ResultsManager(config)
    simulator = simulator_cls(config, controller, env, results_manager)

    # 运行
    results = simulator.run_simulation()
    results_manager.save_all(config.output_dir)
    print(f"仿真完成，结果保存到 {config.output_dir}")


if __name__ == "__main__":
    main()
```

---

### 数据流

```
YAML Config
    │
    ▼
ConfigManager.load() → MPPIConfig
    │
    ├──► DreamerEnv(config)              # 环境：状态转移 + 奖励
    ├──► DIALMPCController(config, env)  # 控制器：轨迹优化
    ├──► ResultsManager(config)          # 结果：收集 + 保存
    │
    ▼
RobotSimulator(config, controller, env, results_mgr)
    │
    ▼ run_simulation()
    │
    for step in range(n_steps):
    │   ├── state = env.get_state()
    │   ├── action = controller.compute_action(state)   # MPPI 优化
    │   ├── next_state = env.step(action)               # Brax 物理仿真
    │   └── results_mgr.record_step(step, state, action)
    │
    ▼
ResultsManager.save_all()
    ├── trajectory.csv
    ├── joints.csv
    └── velocity_errors.csv (如果启用 velocity tracking)
```

---

### 接口边界

| 边界 | 接口 | 说明 |
|------|------|------|
| 控制器 ↔ 框架 | `ControllerBase` Protocol | 未来可替换为 MPC、RL 等 |
| 控制器 ↔ 环境 | `BaseRobotEnv` ABC | 环境抽象，可替换物理后端 |
| 控制器 ↔ 配置 | `MPPIConfig` dataclass | 统一配置入口 |
| 仿真 ↔ 框架 | `BraxRunner` | 与 MuJoCo runner 对称 |
| 变体 ↔ 基类 | Mixin 模式 | 可自由组合功能增强 |

---

### 从 tmp/ 到目标位置的映射

| 源文件 (tmp/) | 目标位置 | 说明 |
|---|---|---|
| `annealed_mppi_dog_*.py` 中的 `Config` + `ConfigManager` | `control/mppi/config.py` | 合并为统一配置 |
| `annealed_mppi_dog_*.py` 中的 `MathUtils` | `control/mppi/math_utils.py` | 提取唯一副本 |
| `annealed_mppi_dog_*.py` 中的 `BaseRobotEnv` + `DreamerEnv` | `control/mppi/environment.py` | 提取环境抽象 |
| `annealed_mppi_dog_*.py` 中的 `DIALMPCController` | `control/mppi/controller.py` | 基类实现 |
| `annealed_mppi_dog_*.py` 中的 `RobotSimulator` | `control/mppi/simulator.py` | 编排器 |
| `annealed_mppi_dog_*.py` 中的 `ResultsManager` | `control/mppi/results.py` | 结果管理 |
| `*_mem.py` 中的历史缓冲逻辑 | `control/mppi/variants/memory_mixin.py` | Mixin |
| `dog_2*.py` 中的速度跟踪逻辑 | `control/mppi/variants/velocity_tracker.py` | Mixin |
| `annealed_mppi_trot_dog*.yaml` | `configs/control/mppi/base_trot.yaml` | 合并为模板 |
| Brax 环境创建/仿真逻辑 | `simulation/brax/runner.py` | 仿真后端 |
| N/A (新增) | `control/base.py` | 控制器协议 |
| N/A (新增) | `scripts/run_mppi_control.py` | 入口脚本 |

### 不迁移的文件

| 文件 | 原因 |
|---|---|
| `display_*.py` (11个) | 现有 `simulation/mujoco/viewer` 已覆盖需求 |

---

### 测试策略

1. **单元测试**：`MathUtils` 的数学函数（坐标变换、四元数运算）
2. **集成测试**：`ConfigManager` 能正确加载 YAML 并应用 override
3. **功能验证**：重构后的 MPPI 运行 dog1/dog2 任务，输出结果与 tmp/ 原始版本一致
4. **Mixin 组合测试**：memory + velocity tracking 同时启用时行为正确

### 命名约定

- 遵循现有框架的蛇形命名（snake_case）
- 中文 docstring（与现有框架一致）
- 模块级私有函数前缀 `_`
- `__init__.py` 中显式导出公共 API
