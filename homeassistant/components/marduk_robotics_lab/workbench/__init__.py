"""Composable workbench modules for robotics middleware."""

from __future__ import annotations

from .hypergraph import HypergraphManager
from .devices import DeviceManager  
from .sensors import SensorManager
from .actuators import ActuatorManager
from .agents import AgentManager
from .tensors import TensorManager
from .registry import WorkbenchRegistry

__all__ = [
    "HypergraphManager",
    "DeviceManager", 
    "SensorManager",
    "ActuatorManager",
    "AgentManager",
    "TensorManager",
    "WorkbenchRegistry",
]