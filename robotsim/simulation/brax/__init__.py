"""Brax 物理仿真后端。"""


def __getattr__(name):
    """延迟导入，避免在缺少 JAX/Brax 的环境中阻塞。"""
    if name == "BraxRunner":
        from robotsim.simulation.brax.runner import BraxRunner
        return BraxRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BraxRunner"]
