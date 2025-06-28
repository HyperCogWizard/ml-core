"""Data models for Marduk's Robotics Lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from homeassistant.config_entries import ConfigEntry


@dataclass
class TensorField:
    """Represents a tensor field in the robotics environment."""
    
    name: str
    dimensions: Tuple[int, ...]
    dtype: str = "float32"
    data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_gguf_dict(self) -> Dict[str, Any]:
        """Convert tensor field to GGUF-compatible dictionary."""
        return {
            "name": self.name,
            "shape": self.dimensions,
            "dtype": self.dtype,
            "data": self.data.tolist() if self.data is not None else None,
            "metadata": self.metadata
        }


@dataclass
class HypergraphNode:
    """Represents a node in the robotics hypergraph."""
    
    node_id: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    tensor_fields: List[TensorField] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    
    def add_tensor_field(self, tensor_field: TensorField) -> None:
        """Add a tensor field to this node."""
        self.tensor_fields.append(tensor_field)
    
    def get_tensor_field(self, name: str) -> Optional[TensorField]:
        """Get a tensor field by name."""
        for field in self.tensor_fields:
            if field.name == name:
                return field
        return None


@dataclass
class AgentKernel:
    """Represents an agentic kernel with cognitive capabilities."""
    
    kernel_id: str
    agent_type: str
    state: str
    cognitive_grammar: str = "scheme"
    memory_banks: List[TensorField] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    learning_rate: float = 0.01
    exploration_factor: float = 0.1
    
    def to_gguf_dict(self) -> Dict[str, Any]:
        """Export kernel state to GGUF format."""
        return {
            "kernel_id": self.kernel_id,
            "agent_type": self.agent_type,
            "state": self.state,
            "cognitive_grammar": self.cognitive_grammar,
            "memory_banks": [bank.to_gguf_dict() for bank in self.memory_banks],
            "functions": self.functions,
            "parameters": {
                "learning_rate": self.learning_rate,
                "exploration_factor": self.exploration_factor
            }
        }


@dataclass
class DeviceConfiguration:
    """Configuration for a robotics device."""
    
    device_id: str
    device_type: str
    degrees_of_freedom: int
    sensor_channels: int
    actuator_channels: int
    tensor_dimensions: Tuple[int, ...] = field(default_factory=tuple)
    modalities: List[str] = field(default_factory=list)
    
    def get_total_complexity(self) -> int:
        """Calculate total device complexity."""
        base_complexity = self.degrees_of_freedom + self.sensor_channels + self.actuator_channels
        tensor_complexity = sum(self.tensor_dimensions) if self.tensor_dimensions else 0
        return base_complexity + tensor_complexity


@dataclass
class MardukRoboticsData:
    """Container for Marduk Robotics Lab runtime data."""
    
    middleware: Any  # RoboticsMiddleware - avoiding circular import
    coordinator: Any  # MardukCoordinator - avoiding circular import  
    config_entry: ConfigEntry
    
    # Runtime state
    active_agents: Dict[str, AgentKernel] = field(default_factory=dict)
    device_configs: Dict[str, DeviceConfiguration] = field(default_factory=dict)
    hypergraph_nodes: Dict[str, HypergraphNode] = field(default_factory=dict)
    tensor_fields: Dict[str, TensorField] = field(default_factory=dict)
    
    def add_agent(self, agent: AgentKernel) -> None:
        """Add an agent kernel to the lab."""
        self.active_agents[agent.kernel_id] = agent
    
    def add_device(self, device: DeviceConfiguration) -> None:
        """Add a device configuration to the lab."""
        self.device_configs[device.device_id] = device
    
    def get_agent(self, agent_id: str) -> Optional[AgentKernel]:
        """Get an agent by ID."""
        return self.active_agents.get(agent_id)
    
    def get_device(self, device_id: str) -> Optional[DeviceConfiguration]:
        """Get a device configuration by ID."""
        return self.device_configs.get(device_id)
    
    def export_to_gguf(self) -> Dict[str, Any]:
        """Export entire lab state to GGUF format."""
        return {
            "version": "1.0.0",
            "lab_id": self.config_entry.entry_id,
            "agents": {aid: agent.to_gguf_dict() for aid, agent in self.active_agents.items()},
            "devices": {did: {
                "device_id": device.device_id,
                "device_type": device.device_type,
                "degrees_of_freedom": device.degrees_of_freedom,
                "sensor_channels": device.sensor_channels,
                "actuator_channels": device.actuator_channels,
                "tensor_dimensions": device.tensor_dimensions,
                "modalities": device.modalities,
                "complexity": device.get_total_complexity()
            } for did, device in self.device_configs.items()},
            "hypergraph": {nid: {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "properties": node.properties,
                "tensor_fields": [field.to_gguf_dict() for field in node.tensor_fields],
                "connections": node.connections
            } for nid, node in self.hypergraph_nodes.items()},
            "tensor_fields": {tid: field.to_gguf_dict() for tid, field in self.tensor_fields.items()}
        }