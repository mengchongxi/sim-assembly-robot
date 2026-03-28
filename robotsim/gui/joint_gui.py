"""
joint_gui.py — 基于 Tkinter 的关节角度实时控制 GUI。

独立测试用法：
    python joint_gui.py

从其他模块调用：
    from joint_gui import JointGUI
    gui = JointGUI(joint_names, joint_limits)
    gui.start()          # 在后台线程中启动 GUI
    angles = gui.get_angles()   # 返回浮点数列表（弧度）
"""

import math
import threading
import tkinter as tk
from tkinter import ttk


class JointGUI:
    """每个关节对应一个滑块的非阻塞式 GUI。

    Parameters
    ----------
    joint_names  : list[str]   – 每个关节的显示名称
    joint_limits : list[tuple] – 每个关节的 (最小弧度, 最大弧度) 限制
    """

    def __init__(self, joint_names, joint_limits):
        self._names = joint_names
        self._limits = joint_limits
        self._n = len(joint_names)

        # 共享状态：以弧度存储各关节角度，由锁保护
        self._angles = [0.0] * self._n
        self._lock = threading.Lock()

        self._root = None
        self._vars = []   # tk.DoubleVar 列表（度数）
        self._labels = [] # 当前值显示标签列表

        self._stop_requested = False
        self._closed_event = threading.Event()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def start(self):
        """在守护线程中启动 GUI（立即返回，不阻塞调用方）。"""
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    @property
    def is_running(self):
        """GUI 是否仍在运行。"""
        return not self._closed_event.is_set()

    def stop(self):
        """请求关闭 GUI 并等待其完成。"""
        if self._closed_event.is_set():
            return
        self._stop_requested = True
        self._closed_event.wait(timeout=3.0)

    def get_angles(self):
        """返回当前各关节角度的浮点数列表（单位：弧度）。"""
        with self._lock:
            return list(self._angles)

    def set_angle(self, index, radians):
        """以编程方式设置指定关节的角度（单位：弧度）。"""
        with self._lock:
            self._angles[index] = radians
        if self._vars:
            deg = math.degrees(radians)
            self._vars[index].set(deg)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _run(self):
        self._root = tk.Tk()
        self._root.title("Joint Controller")
        self._root.resizable(False, False)

        style = ttk.Style(self._root)
        style.theme_use("clam")

        header = tk.Label(
            self._root, text="Joint Angle Control",
            font=("Helvetica", 13, "bold"), pady=6
        )
        header.pack(fill="x")

        frame = tk.Frame(self._root, padx=10, pady=6)
        frame.pack(fill="both", expand=True)

        for i, (name, (lo, hi)) in enumerate(zip(self._names, self._limits)):
            lo_deg = math.degrees(lo)
            hi_deg = math.degrees(hi)

            row = tk.Frame(frame)
            row.pack(fill="x", pady=3)

            # 关节名称标签
            tk.Label(row, text=f"{name}", width=10, anchor="w",
                     font=("Helvetica", 10)).grid(row=0, column=0, sticky="w")

            # 滑块（角度单位：度）
            var = tk.DoubleVar(value=0.0)
            self._vars.append(var)

            slider = ttk.Scale(
                row, from_=lo_deg, to=hi_deg,
                orient="horizontal", length=280,
                variable=var,
                command=lambda v, idx=i: self._on_slide(idx, v)
            )
            slider.grid(row=0, column=1, padx=6)

            # 当前值显示标签
            val_label = tk.Label(row, text="  0.00°", width=8,
                                 font=("Courier", 10), anchor="e")
            val_label.grid(row=0, column=2, sticky="e")
            self._labels.append(val_label)

            # 范围提示
            tk.Label(row, text=f"[{lo_deg:.0f}°, {hi_deg:.0f}°]",
                     font=("Helvetica", 8), fg="gray").grid(row=0, column=3, padx=4)

        # 重置按钮
        btn_frame = tk.Frame(self._root, pady=6)
        btn_frame.pack()
        tk.Button(btn_frame, text="Reset All", width=12,
                  command=self._reset_all).pack()

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.after(100, self._check_stop)
        self._root.mainloop()

    def _on_slide(self, index, value_str):
        deg = float(value_str)
        rad = math.radians(deg)
        with self._lock:
            self._angles[index] = rad
        if self._labels:
            self._labels[index].config(text=f"{deg:+7.2f}°")

    def _reset_all(self):
        for i, var in enumerate(self._vars):
            var.set(0.0)
            self._on_slide(i, "0.0")

    def _check_stop(self):
        if self._stop_requested:
            self._on_close()
        else:
            self._root.after(100, self._check_stop)

    def _on_close(self):
        try:
            self._root.quit()
            self._root.destroy()
        except Exception:
            pass
        finally:
            self._root = None
            self._vars = []
            self._labels = []
            self._closed_event.set()


# ---------------------------------------------------------------------------
# 独立演示程序
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    names = [f"joint{i+1}" for i in range(6)]
    limits = [(-math.pi, math.pi)] * 6

    gui = JointGUI(names, limits)
    gui.start()

    print("GUI 已运行，每秒打印一次关节角度（按 Ctrl-C 退出）...")
    try:
        while True:
            angles = gui.get_angles()
            print("  " + "  ".join(f"{math.degrees(a):+7.2f}°" for a in angles))
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
