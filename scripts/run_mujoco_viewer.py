"""MuJoCo 交互式关节查看器入口脚本。

启动时弹出 Tkinter 文件选择对话框，选择 models/mujoco/ 下的任意 XML 模型，
然后在 MuJoCo 查看器中加载并提供关节滑块控制。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

from robotsim.simulation.mujoco.viewer import draw_body_frames, _resolve_joints
from robotsim.simulation.mujoco.model_manager import ModelManager
from robotsim.gui.joint_gui import JointGUI


def select_xml_file(initial_dir: Path) -> Path | None:
    """Tkinter 文件选择对话框，返回选中的 XML 路径。"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择 MuJoCo 模型文件",
        initialdir=initial_dir,
        filetypes=[("MuJoCo XML", "*.xml"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(file_path) if file_path else None


def load_and_view(xml_path: Path):
    """加载 MuJoCo 模型并启动交互式查看器。"""
    print(f"[viewer_mj] Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    configuration = mink.Configuration(model)
    model = configuration.model
    data = configuration.data

    resolved_names, joint_ids = _resolve_joints(model)
    if not joint_ids:
        print("[viewer_mj] No controllable hinge/slide joints found in model.")
        print("[viewer_mj] Launching viewer in view-only mode.")
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            rate = RateLimiter(frequency=50.0, warn=False)
            while viewer.is_running():
                mujoco.mj_camlight(model, data)
                mujoco.mj_fwdPosition(model, data)
                draw_body_frames(model, data, viewer)
                viewer.sync()
                rate.sleep()
        return

    print(f"[viewer_mj] Auto-resolved joints ({len(resolved_names)}): {resolved_names}")

    joint_limits = []
    for jid in joint_ids:
        lo = model.jnt_range[jid, 0]
        hi = model.jnt_range[jid, 1]
        joint_limits.append((lo, hi))

    gui = JointGUI(resolved_names, joint_limits)
    gui.start()

    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            rate = RateLimiter(frequency=200.0, warn=False)
            while viewer.is_running() and gui.is_running:
                angles = gui.get_angles()
                for jid, angle in zip(joint_ids, angles):
                    qadr = model.jnt_qposadr[jid]
                    data.qpos[qadr] = angle

                mujoco.mj_camlight(model, data)
                mujoco.mj_fwdPosition(model, data)
                mujoco.mj_sensorPos(model, data)
                draw_body_frames(model, data, viewer)
                viewer.sync()
                rate.sleep()
    finally:
        gui.stop()


def main():
    """启动 MuJoCo 交互式关节查看器。"""
    manager = ModelManager()

    # 列出可用模型
    models = manager.list_all_models()
    if models:
        print("\n可用模型:")
        for m in models:
            print(f"  [{m['source']}] {m['name']}: {m['path']}")
        print()

    # 文件选择对话框
    xml_path = select_xml_file(manager.base_dir)
    if xml_path is None or not xml_path.exists():
        print("未选择文件，退出。")
        sys.exit(0)

    load_and_view(xml_path)
    # 强制退出，避免 Tcl 跨线程 GC 导致的崩溃
    os._exit(0)


if __name__ == "__main__":
    main()
