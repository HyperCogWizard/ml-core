"""Data models for Marduk's Robotics Lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from homeassistant.config_entries import ConfigEntry


@dataclass
class TensorField:
    """Represents a tensor field in the robotics environment with GGUF support."""
    
    name: str
    dimensions: Tuple[int, ...]
    dtype: str = "float32"
    data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced fields for comprehensive GGUF support
    tensor_type: str = "general"
    tensor_id: Optional[str] = None
    parent_entity: Optional[str] = None
    update_frequency: float = 1.0
    compression_algorithm: Optional[str] = None
    
    def to_gguf_dict(self) -> Dict[str, Any]:
        """Convert tensor field to comprehensive GGUF-compatible dictionary."""
        return {
            "name": self.name,
            "shape": self.dimensions,
            "dtype": self.dtype,
            "data": self.data.tolist() if self.data is not None else None,
            "metadata": self.metadata,
            # Enhanced comprehensive fields
            "tensor_type": self.tensor_type,
            "tensor_id": self.tensor_id,
            "parent_entity": self.parent_entity,
            "update_frequency": self.update_frequency,
            "compression_algorithm": self.compression_algorithm
        }
    
    def set_tensor_identity(self, tensor_id: str, parent_entity: str, tensor_type: str = "general") -> None:
        """Set tensor identity for comprehensive tracking."""
        self.tensor_id = tensor_id
        self.parent_entity = parent_entity
        self.tensor_type = tensor_type


@dataclass
class HypergraphNode:
    """Represents a node in the robotics hypergraph with comprehensive GGUF support."""
    
    node_id: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    tensor_fields: List[TensorField] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    
    # Enhanced fields for comprehensive GGUF support
    node_state: Dict[str, Any] = field(default_factory=dict)
    processing_load: float = 0.0
    communication_latency: Dict[str, float] = field(default_factory=dict)
    membrane_bindings: List[str] = field(default_factory=list)
    
    def add_tensor_field(self, tensor_field: TensorField) -> None:
        """Add a tensor field to this node."""
        # Set tensor identity for comprehensive tracking
        tensor_field.set_tensor_identity(
            tensor_id=f"{self.node_id}_{tensor_field.name}",
            parent_entity=self.node_id,
            tensor_type=f"hypergraph_{self.node_type}"
        )
        self.tensor_fields.append(tensor_field)
    
    def get_tensor_field(self, name: str) -> Optional[TensorField]:
        """Get a tensor field by name."""
        for field in self.tensor_fields:
            if field.name == name:
                return field
        return None
    
    def to_gguf_dict(self) -> Dict[str, Any]:
        """Export hypergraph node to comprehensive GGUF format."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "properties": self.properties,
            "tensor_fields": [field.to_gguf_dict() for field in self.tensor_fields],
            "connections": self.connections,
            # Enhanced comprehensive state
            "node_state": self.node_state,
            "processing_load": self.processing_load,
            "communication_latency": self.communication_latency,
            "membrane_bindings": self.membrane_bindings
        }


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
    
    # Enhanced fields for comprehensive GGUF support
    state_history: List[str] = field(default_factory=list)
    cognitive_parameters: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    communication_channels: List[str] = field(default_factory=list)
    membrane_id: Optional[str] = None
    parent_membrane: Optional[str] = None
    child_membranes: List[str] = field(default_factory=list)
    
    def to_gguf_dict(self) -> Dict[str, Any]:
        """Export kernel state to GGUF format with comprehensive coverage."""
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
            },
            # Enhanced comprehensive state
            "state_history": self.state_history,
            "cognitive_parameters": self.cognitive_parameters,
            "execution_context": self.execution_context,
            "communication_channels": self.communication_channels,
            # P-System membrane structure
            "membrane_structure": {
                "membrane_id": self.membrane_id,
                "parent_membrane": self.parent_membrane,
                "child_membranes": self.child_membranes
            }
        }
    
    def add_state_transition(self, new_state: str) -> None:
        """Record a state transition in the agent's history."""
        if self.state != new_state:
            self.state_history.append(f"{self.state}->{new_state}")
            self.state = new_state
    
    def set_membrane_hierarchy(self, membrane_id: str, parent: Optional[str] = None) -> None:
        """Set the P-System membrane hierarchy for this agent."""
        self.membrane_id = membrane_id
        self.parent_membrane = parent
    
    def add_child_membrane(self, child_membrane_id: str) -> None:
        """Add a child membrane to this agent's hierarchy."""
        if child_membrane_id not in self.child_membranes:
            self.child_membranes.append(child_membrane_id)


@dataclass
class DeviceConfiguration:
    """Configuration for a robotics device with comprehensive GGUF support."""
    
    device_id: str
    device_type: str
    degrees_of_freedom: int
    sensor_channels: int
    actuator_channels: int
    tensor_dimensions: Tuple[int, ...] = field(default_factory=tuple)
    modalities: List[str] = field(default_factory=list)
    
    # Enhanced fields for comprehensive GGUF support
    device_parameters: Dict[str, Any] = field(default_factory=dict)
    sensor_specifications: List[Dict[str, Any]] = field(default_factory=list)
    actuator_specifications: List[Dict[str, Any]] = field(default_factory=list)
    calibration_data: Dict[str, Any] = field(default_factory=dict)
    communication_protocols: List[str] = field(default_factory=list)
    firmware_version: str = "unknown"
    device_status: str = "operational"
    
    # P-System membrane integration
    membrane_id: Optional[str] = None
    membrane_type: str = "device_membrane"
    input_channels: List[str] = field(default_factory=list)
    output_channels: List[str] = field(default_factory=list)
    
    def get_total_complexity(self) -> int:
        """Calculate total device complexity."""
        base_complexity = self.degrees_of_freedom + self.sensor_channels + self.actuator_channels
        tensor_complexity = sum(self.tensor_dimensions) if self.tensor_dimensions else 0
        modality_complexity = len(self.modalities) * 2
        return base_complexity + tensor_complexity + modality_complexity
    
    def to_gguf_dict(self) -> Dict[str, Any]:
        """Export device configuration to comprehensive GGUF format."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "degrees_of_freedom": self.degrees_of_freedom,
            "sensor_channels": self.sensor_channels,
            "actuator_channels": self.actuator_channels,
            "tensor_dimensions": list(self.tensor_dimensions),
            "modalities": self.modalities,
            "complexity": self.get_total_complexity(),
            # Enhanced comprehensive configuration
            "device_parameters": self.device_parameters,
            "sensor_specifications": self.sensor_specifications,
            "actuator_specifications": self.actuator_specifications,
            "calibration_data": self.calibration_data,
            "communication_protocols": self.communication_protocols,
            "firmware_version": self.firmware_version,
            "device_status": self.device_status,
            # P-System membrane structure
            "membrane_structure": {
                "membrane_id": self.membrane_id,
                "membrane_type": self.membrane_type,
                "input_channels": self.input_channels,
                "output_channels": self.output_channels
            }
        }
    
    def add_sensor_specification(self, sensor_spec: Dict[str, Any]) -> None:
        """Add a sensor specification to the device."""
        self.sensor_specifications.append(sensor_spec)
    
    def add_actuator_specification(self, actuator_spec: Dict[str, Any]) -> None:
        """Add an actuator specification to the device."""
        self.actuator_specifications.append(actuator_spec)
    
    def set_membrane_configuration(self, membrane_id: str, input_channels: List[str], output_channels: List[str]) -> None:
        """Configure P-System membrane settings for this device."""
        self.membrane_id = membrane_id
        self.input_channels = input_channels
        self.output_channels = output_channels


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
        """Export entire lab state to comprehensive GGUF format with P-System membranes."""
        return {
            "version": "2.0.0",  # Updated version for comprehensive support
            "lab_id": self.config_entry.entry_id,
            "format_type": "marduk_comprehensive_gguf",
            "p_system_compatible": True,
            
            # Comprehensive agent export
            "agents": {aid: agent.to_gguf_dict() for aid, agent in self.active_agents.items()},
            
            # Comprehensive device export
            "devices": {did: device.to_gguf_dict() for did, device in self.device_configs.items()},
            
            # Comprehensive hypergraph export
            "hypergraph": {nid: node.to_gguf_dict() for nid, node in self.hypergraph_nodes.items()},
            
            # Environment-level tensor fields
            "tensor_fields": {tid: field.to_gguf_dict() for tid, field in self.tensor_fields.items()},
            
            # P-System membrane hierarchy
            "membrane_system": self._generate_membrane_hierarchy(),
            
            # System-wide metadata
            "metadata": {
                "total_agents": len(self.active_agents),
                "total_devices": len(self.device_configs),
                "total_hypergraph_nodes": len(self.hypergraph_nodes),
                "total_tensor_fields": len(self.tensor_fields),
                "export_timestamp": "runtime_generated"
            }
        }
    
    def _generate_membrane_hierarchy(self) -> Dict[str, Any]:
        """Generate P-System membrane hierarchy structure."""
        membranes = {}
        
        # Create agent membranes
        for agent_id, agent in self.active_agents.items():
            if agent.membrane_id:
                membranes[agent.membrane_id] = {
                    "membrane_type": "agent_membrane",
                    "entity_id": agent_id,
                    "entity_type": "agent",
                    "parent_membrane": agent.parent_membrane,
                    "child_membranes": agent.child_membranes,
                    "communication_channels": agent.communication_channels
                }
        
        # Create device membranes
        for device_id, device in self.device_configs.items():
            if device.membrane_id:
                membranes[device.membrane_id] = {
                    "membrane_type": device.membrane_type,
                    "entity_id": device_id,
                    "entity_type": "device",
                    "input_channels": device.input_channels,
                    "output_channels": device.output_channels
                }
        
        # Create lab-level membrane
        membranes["lab_root"] = {
            "membrane_type": "lab_membrane",
            "entity_id": self.config_entry.entry_id,
            "entity_type": "laboratory",
            "child_membranes": list(membranes.keys())
        }
        
        return membranes