"""
Lifecycle management module for the autonomous agent system.
"""

from .hooks import LifecycleHook, PostStartHook, PostStopHook, PreStartHook, PreStopHook
from .manager import LifecycleEvent, LifecycleManager, LifecycleState
from .monitor import HealthCheck, LifecycleMonitor, PerformanceMonitor

__all__ = [
    "HealthCheck",
    "LifecycleEvent",
    # Hook classes
    "LifecycleHook",
    # Main classes
    "LifecycleManager",
    # Monitor classes
    "LifecycleMonitor",
    "LifecycleState",
    "PerformanceMonitor",
    "PostStartHook",
    "PostStopHook",
    "PreStartHook",
    "PreStopHook",
]
