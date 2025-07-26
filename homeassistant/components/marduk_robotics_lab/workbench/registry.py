"""Workbench registry for managing all components."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .hypergraph import HypergraphManager
from .tensors import TensorManager
from .devices import DeviceManager
from .sensors import SensorManager
from .actuators import ActuatorManager
from .agents import AgentManager

_LOGGER = logging.getLogger(__name__)


class WorkbenchRegistry:
    """Central registry for all workbench components and modules."""
    
    def __init__(self) -> None:
        """Initialize workbench registry."""
        # Core managers
        self.hypergraph_manager = HypergraphManager()
        self.tensor_manager = TensorManager()
        
        # Component managers
        self.device_manager = DeviceManager(self.hypergraph_manager)
        self.sensor_manager = SensorManager(self.hypergraph_manager, self.tensor_manager)
        self.actuator_manager = ActuatorManager(self.hypergraph_manager, self.tensor_manager)
        self.agent_manager = AgentManager(self.hypergraph_manager)
        
        _LOGGER.info("Initialized workbench registry with all component managers")
    
    def get_complete_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all managers."""
        return {
            "hypergraph": self.hypergraph_manager.get_hypergraph_stats(),
            "tensors": self.tensor_manager.get_tensor_stats(),
            "devices": self.device_manager.get_device_stats(),
            "sensors": self.sensor_manager.get_sensor_stats(),
            "actuators": self.actuator_manager.get_actuator_stats(),
            "agents": self.agent_manager.get_agent_stats(),
        }
    
    def export_complete_workbench(self) -> Dict[str, Any]:
        """Export the complete workbench state."""
        return {
            "hypergraph": self.hypergraph_manager.export_graph(),
            "tensors": self.tensor_manager.export_tensor_definitions(),
            "devices": self.device_manager.export_device_definitions(),
            "stats": self.get_complete_stats(),
        }
    
    def clear_all(self) -> None:
        """Clear all data from all managers."""
        self.hypergraph_manager.clear()
        self.tensor_manager.clear()
        
        # Recreate managers to reset state
        self.device_manager = DeviceManager(self.hypergraph_manager)
        self.sensor_manager = SensorManager(self.hypergraph_manager, self.tensor_manager)
        self.actuator_manager = ActuatorManager(self.hypergraph_manager, self.tensor_manager)
        self.agent_manager = AgentManager(self.hypergraph_manager)
        
        _LOGGER.info("Cleared all workbench registry data")
    
    def validate_consistency(self) -> List[str]:
        """Validate consistency across all managers."""
        issues = []
        
        # Check that all device nodes exist in hypergraph
        for device_id in self.device_manager.get_all_device_configs():
            if not self.hypergraph_manager.get_node(device_id):
                issues.append(f"Device {device_id} missing from hypergraph")
        
        # Check that all agent nodes exist in hypergraph
        for agent_id in self.agent_manager.get_all_agent_kernels():
            if not self.hypergraph_manager.get_node(agent_id):
                issues.append(f"Agent {agent_id} missing from hypergraph")
        
        # Check tensor spec consistency
        hypergraph_specs = set()
        for node in self.hypergraph_manager.get_hypergraph_stats():
            for spec in getattr(node, 'tensor_specs', []):
                hypergraph_specs.add(spec.name)
        
        tensor_specs = set(self.tensor_manager.export_tensor_definitions()["specifications"].keys())
        
        missing_in_tensor_manager = hypergraph_specs - tensor_specs
        for spec_name in missing_in_tensor_manager:
            issues.append(f"Tensor spec {spec_name} in hypergraph but not in tensor manager")
        
        return issues