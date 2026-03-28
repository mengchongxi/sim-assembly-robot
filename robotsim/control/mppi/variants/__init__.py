"""MPPI 变体增强模块。"""

"""MPPI 变体增强模块。"""


def __getattr__(name):
    """延迟导入 JAX 依赖的 mixin 类。"""
    _lazy_imports = {
        "HistoryMemoryMixin": "robotsim.control.mppi.variants.memory_mixin",
        "MemoryDreamerEnvMixin": "robotsim.control.mppi.variants.memory_mixin",
        "VelocityTrackingMixin": "robotsim.control.mppi.variants.velocity_tracker",
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HistoryMemoryMixin",
    "MemoryDreamerEnvMixin",
    "VelocityTrackingMixin",
]
