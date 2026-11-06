"""External Maa strategy, isolated from the official agent's import namespace."""
from pathlib import Path

_runtime = None


def install(app_dir):
    """Keep one runtime in this Python agent process; task state lives on it."""
    global _runtime
    if _runtime is None:
        from .maa_hook import ValueModelRuntime

        runtime = ValueModelRuntime(Path(__file__).resolve().parent, app_dir)
        runtime.register()
        _runtime = runtime
    return _runtime
