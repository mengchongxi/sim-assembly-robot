"""MuJoCo 模型文件管理器模块。

统一管理 models/mujoco/ 下所有 XML 模型文件（手工 + 自动生成），
提供输出目录创建、模型扫描、构型快照和轨迹保存功能。
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from robotsim.core.types import TileType


class ModelManager:
    """统一管理 models/mujoco/ 下所有 XML 模型。"""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # 相对于项目根目录
            project_root = Path(__file__).resolve().parents[3]
            base_dir = project_root / "models" / "mujoco"
        self.base_dir = Path(base_dir)
        self.generated_dir = self.base_dir / "generated"

    def create_output_dir(self, name: Optional[str] = None) -> Path:
        """在 generated/ 下创建新的输出目录。

        Args:
            name: 自定义目录名。默认使用时间戳命名（YYYY-MM-DD_HHMMSS）。

        Returns:
            创建的目录路径。
        """
        if name is None:
            name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = self.generated_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def list_all_models(self) -> List[Dict]:
        """扫描 models/mujoco/ 下所有 XML 文件。

        Returns:
            [{"name": str, "path": Path, "source": "manual"|"generated"}, ...]
        """
        models = []

        if not self.base_dir.exists():
            return models

        for item in sorted(self.base_dir.iterdir()):
            if item.name == "generated":
                continue
            if item.is_dir():
                for xml_file in sorted(item.glob("*.xml")):
                    models.append({
                        "name": item.name,
                        "path": xml_file,
                        "source": "manual",
                    })
            elif item.suffix == ".xml":
                models.append({
                    "name": item.stem,
                    "path": item,
                    "source": "manual",
                })

        if self.generated_dir.exists():
            for gen_dir in sorted(self.generated_dir.iterdir()):
                if not gen_dir.is_dir() or gen_dir.name.startswith("."):
                    continue
                for xml_file in sorted(gen_dir.glob("*.xml")):
                    models.append({
                        "name": gen_dir.name,
                        "path": xml_file,
                        "source": "generated",
                    })

        return models

    def get_latest_generated(self) -> Optional[Path]:
        """返回最新生成的模型目录路径，无则返回 None。"""
        if not self.generated_dir.exists():
            return None

        dirs = [
            d for d in self.generated_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if not dirs:
            return None

        return max(dirs, key=lambda d: d.stat().st_mtime)

    def save_config_snapshot(
        self, output_dir: Path, tiles: Dict[TileType, set]
    ) -> Path:
        """将目标构型保存为 config.yaml。

        Args:
            output_dir: 输出目录路径。
            tiles: {TileType.BASE: {(x,y),...}, ...}

        Returns:
            保存的配置文件路径。
        """
        config_data = {"tiles": {}}
        for tile_type, positions in tiles.items():
            type_name = tile_type.name.lower()
            config_data["tiles"][type_name] = [
                list(pos) for pos in sorted(positions)
            ]

        config_file = output_dir / "config.yaml"
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        print(f"[model_manager] Config snapshot saved to {config_file}")
        return config_file

    def save_trajectory(self, output_dir: Path, trajectory_data: Dict) -> Path:
        """将轨迹数据保存到指定输出目录。

        Args:
            output_dir: 输出目录路径。
            trajectory_data: 轨迹数据字典（与 robot_trajectory.yaml 格式一致）。

        Returns:
            保存的轨迹文件路径。
        """
        traj_file = output_dir / "robot_trajectory.yaml"
        with traj_file.open("w", encoding="utf-8") as f:
            yaml.dump(trajectory_data, f, default_flow_style=False, sort_keys=False)
        print(f"[model_manager] Trajectory saved to {traj_file}")
        return traj_file
