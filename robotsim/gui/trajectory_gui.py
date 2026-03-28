"""基于 Tkinter 的轨迹请求 GUI 模块。

提供 TargetTrajectoryGUI 类，在独立线程中运行 Tkinter 窗口，允许用户
输入轨迹参数（起始/终止 XYZ、持续时间、弧高）或“Go-To”目标坐标，
并通过线程安全的请求队列将请求传递给仿真循环。
"""
import threading
import tkinter as tk
import numpy as np


class TargetTrajectoryGUI:
    """在独立线程中运行的轨迹请求提交 GUI。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending_request = None
        self._root = None
        self._stop_requested = False
        self._closed_event = threading.Event()

    def start(self):
        """在后台守护线程中启动 Tkinter GUI。"""
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def stop(self):
        """请求关闭 GUI 并等待其完成。"""
        self._stop_requested = True
        self._closed_event.wait(timeout=2.0)

    def consume_request(self):
        """返回并清除最新的请求元组。

        请求格式：
        - ("trajectory", start_xyz, end_xyz, duration_s, arc_height)
        - ("goto", target_xyz, duration_s)
        """
        with self._lock:
            request = self._pending_request
            self._pending_request = None
        return request

    def _run(self):
        """在守护线程中运行的内部方法，构建并进入 Tkinter 主循环。

        窗口包含：
        - 轨迹模式：起始 XYZ、终止 XYZ、持续时间和弧高输入。
        - Go-To 模式：目标 XYZ 和持续时间输入。
        """
        self._root = tk.Tk()
        self._root.title("Target Trajectory")
        self._root.resizable(False, False)

        container = tk.Frame(self._root, padx=10, pady=10)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="Start XYZ (base frame)", font=("Helvetica", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 3)
        )
        start_vars = [tk.StringVar(value="0.0"), tk.StringVar(value="-0.12"), tk.StringVar(value="0.0")]
        for i, axis in enumerate(("x", "y", "z")):
            tk.Label(container, text=axis).grid(row=1, column=i, sticky="w")
            tk.Entry(container, width=10, textvariable=start_vars[i]).grid(
                row=2, column=i, padx=(0 if i == 0 else 5, 0)
            )

        tk.Label(container, text="End XYZ (base frame)", font=("Helvetica", 10, "bold")).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 3)
        )
        end_vars = [tk.StringVar(value="0.0"), tk.StringVar(value="-0.24"), tk.StringVar(value="0.0")]
        for i, axis in enumerate(("x", "y", "z")):
            tk.Label(container, text=axis).grid(row=4, column=i, sticky="w")
            tk.Entry(container, width=10, textvariable=end_vars[i]).grid(
                row=5, column=i, padx=(0 if i == 0 else 5, 0)
            )

        tk.Label(container, text="Duration (s)", font=("Helvetica", 10, "bold")).grid(
            row=6, column=0, sticky="w", pady=(10, 3)
        )
        duration_var = tk.StringVar(value="4.0")
        tk.Entry(container, width=10, textvariable=duration_var).grid(row=7, column=0, sticky="w")

        tk.Label(container, text="Arc Height (m)", font=("Helvetica", 10, "bold")).grid(
            row=6, column=1, sticky="w", pady=(10, 3)
        )
        arc_height_var = tk.StringVar(value="0.1")
        tk.Entry(container, width=10, textvariable=arc_height_var).grid(row=7, column=1, sticky="w")

        status_var = tk.StringVar(value="Fill values and click Play")
        tk.Label(container, textvariable=status_var, fg="gray").grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

        def submit():
            try:
                start = np.array([float(v.get()) for v in start_vars], dtype=float)
                end = np.array([float(v.get()) for v in end_vars], dtype=float)
                duration_s = float(duration_var.get())
                arc_height = float(arc_height_var.get())
            except ValueError:
                status_var.set("Invalid number format")
                return

            if duration_s <= 0.0:
                status_var.set("Duration must be > 0")
                return

            with self._lock:
                self._pending_request = ("trajectory", start, end, duration_s, arc_height)
            status_var.set("Trajectory submitted")

        tk.Button(container, text="Generate & Play", width=16, command=submit).grid(
            row=7, column=2, sticky="e"
        )

        tk.Label(container, text="Go To XYZ (base frame)", font=("Helvetica", 10, "bold")).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(14, 3)
        )
        goto_vars = [
            tk.StringVar(value="0.0"),
            tk.StringVar(value="-0.18"),
            tk.StringVar(value="0.05"),
        ]
        for i, axis in enumerate(("x", "y", "z")):
            tk.Label(container, text=axis).grid(row=10, column=i, sticky="w")
            tk.Entry(container, width=10, textvariable=goto_vars[i]).grid(
                row=11, column=i, padx=(0 if i == 0 else 5, 0)
            )

        tk.Label(container, text="Go Duration (s)", font=("Helvetica", 10, "bold")).grid(
            row=12, column=0, sticky="w", pady=(8, 3)
        )
        goto_duration_var = tk.StringVar(value="2.0")
        tk.Entry(container, width=10, textvariable=goto_duration_var).grid(row=13, column=0, sticky="w")

        def submit_goto():
            try:
                target_xyz = np.array([float(v.get()) for v in goto_vars], dtype=float)
                duration_s = float(goto_duration_var.get())
            except ValueError:
                status_var.set("Invalid go-to number format")
                return

            if duration_s <= 0.0:
                status_var.set("Go duration must be > 0")
                return

            with self._lock:
                self._pending_request = ("goto", target_xyz, duration_s)
            status_var.set("Go-to target submitted")

        tk.Button(container, text="Go To XYZ", width=16, command=submit_goto).grid(
            row=13, column=2, sticky="e"
        )

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.after(100, self._check_stop)
        self._root.mainloop()

    def _check_stop(self):
        if self._stop_requested:
            self._on_close()
        else:
            self._root.after(100, self._check_stop)

    def _on_close(self):
        self._root.destroy()
        self._closed_event.set()
