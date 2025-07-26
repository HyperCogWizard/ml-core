"""Workbench hypergraph edge representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class EdgeType(Enum):
    """Types of hypergraph edges."""
    
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    DEPENDENCY = "dependency"
    COMMUNICATION = "communication"
    TENSOR_COUPLING = "tensor_coupling"


@dataclass
class WorkbenchEdge:
    """Enhanced hypergraph edge for workbench components."""
    
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tensor flow specifications
    tensor_mappings: Dict[str, str] = field(default_factory=dict)  # source_field -> target_field
    bandwidth_requirements: Optional[float] = None  # MB/s
    latency_requirements: Optional[float] = None  # ms
    
    def add_tensor_mapping(self, source_field: str, target_field: str) -> None:
        """Add a tensor field mapping between nodes."""
        self.tensor_mappings[source_field] = target_field
    
    def remove_tensor_mapping(self, source_field: str) -> None:
        """Remove a tensor field mapping."""
        self.tensor_mappings.pop(source_field, None)
    
    def get_mapped_field(self, source_field: str) -> Optional[str]:
        """Get the target field mapped to a source field."""
        return self.tensor_mappings.get(source_field)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type.value,
            "properties": self.properties,
            "metadata": self.metadata,
            "tensor_mappings": self.tensor_mappings,
            "bandwidth_requirements": self.bandwidth_requirements,
            "latency_requirements": self.latency_requirements,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkbenchEdge:
        """Create edge from dictionary representation."""
        return cls(
            edge_id=data["edge_id"],
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            edge_type=EdgeType(data["edge_type"]),
            properties=data.get("properties", {}),
            metadata=data.get("metadata", {}),
            tensor_mappings=data.get("tensor_mappings", {}),
            bandwidth_requirements=data.get("bandwidth_requirements"),
            latency_requirements=data.get("latency_requirements"),
        )