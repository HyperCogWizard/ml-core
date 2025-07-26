"""Workbench hypergraph node representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from ..tensors import TensorFieldSpec


@dataclass
class WorkbenchNode:
    """Enhanced hypergraph node for workbench components."""
    
    node_id: str
    node_type: str
    component_class: str
    properties: Dict[str, Any] = field(default_factory=dict)
    tensor_specs: List[TensorFieldSpec] = field(default_factory=list)
    connections: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_connection(self, target_node_id: str) -> None:
        """Add a connection to another node."""
        self.connections.add(target_node_id)
    
    def remove_connection(self, target_node_id: str) -> None:
        """Remove a connection to another node."""
        self.connections.discard(target_node_id)
    
    def add_tensor_spec(self, spec: TensorFieldSpec) -> None:
        """Add a tensor field specification."""
        # Remove existing spec with same name if present
        self.tensor_specs = [s for s in self.tensor_specs if s.name != spec.name]
        self.tensor_specs.append(spec)
    
    def get_tensor_spec(self, name: str) -> Optional[TensorFieldSpec]:
        """Get tensor field specification by name."""
        for spec in self.tensor_specs:
            if spec.name == name:
                return spec
        return None
    
    def get_total_tensor_dimensions(self) -> int:
        """Calculate total tensor dimensions for this node."""
        total = 0
        for spec in self.tensor_specs:
            total += spec.get_total_elements()
        return total
    
    def is_connected_to(self, node_id: str) -> bool:
        """Check if this node is connected to another node."""
        return node_id in self.connections
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "component_class": self.component_class,
            "properties": self.properties,
            "tensor_specs": [spec.to_dict() for spec in self.tensor_specs],
            "connections": list(self.connections),
            "metadata": self.metadata,
            "total_dimensions": self.get_total_tensor_dimensions(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkbenchNode:
        """Create node from dictionary representation."""
        from ..tensors import TensorFieldSpec
        
        node = cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            component_class=data["component_class"],
            properties=data.get("properties", {}),
            connections=set(data.get("connections", [])),
            metadata=data.get("metadata", {}),
        )
        
        # Reconstruct tensor specs
        for spec_data in data.get("tensor_specs", []):
            spec = TensorFieldSpec.from_dict(spec_data)
            node.tensor_specs.append(spec)
            
        return node