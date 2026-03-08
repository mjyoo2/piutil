# Backward-compatible re-export. Actual implementation in piutil.profiling.timer.
from piutil.profiling.timer import *  # noqa: F401,F403
from piutil.profiling.timer import _check_cuda, _cuda_sync, _format_time, _timers, _active_stack
