"""Workbench component base class."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..hypergraph import WorkbenchNode
from ..tensors import TensorFieldSpec, TensorField

_LOGGER = logging.getLogger(__name__)


class WorkbenchComponent(ABC):
    """Base class for all workbench components."""
    
    def __init__(self, component_id: str, component_type: str) -> None:
        """Initialize workbench component."""
        self.component_id = component_id
        self.component_type = component_type
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self._tensor_fields: Dict[str, TensorField] = {}
        self._is_active = False
        
    @property
    def is_active(self) -> bool:
        """Check if component is active."""
        return self._is_active
    
    @abstractmethod
    def get_tensor_specs(self) -> List[TensorFieldSpec]:
        """Get tensor field specifications for this component."""
        pass
    
    @abstractmethod
    async def async_initialize(self) -> bool:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def async_shutdown(self) -> None:
        """Shutdown the component."""
        pass
    
    @abstractmethod
    async def async_update(self) -> None:
        """Update component state."""
        pass
    
    def add_tensor_field(self, field: TensorField) -> None:
        """Add a tensor field to this component."""
        self._tensor_fields[field.spec.name] = field
        self.last_updated = datetime.now()
    
    def get_tensor_field(self, name: str) -> Optional[TensorField]:
        """Get tensor field by name."""
        return self._tensor_fields.get(name)
    
    def get_all_tensor_fields(self) -> Dict[str, TensorField]:
        """Get all tensor fields."""
        return self._tensor_fields.copy()
    
    def remove_tensor_field(self, name: str) -> bool:
        """Remove tensor field by name."""
        if name in self._tensor_fields:
            del self._tensor_fields[name]
            self.last_updated = datetime.now()
            return True
        return False
    
    def create_hypergraph_node(self) -> WorkbenchNode:
        """Create hypergraph node representation of this component."""
        node = WorkbenchNode(
            node_id=self.component_id,
            node_type=self.component_type,
            component_class=self.__class__.__name__,
            metadata={
                "created_at": self.created_at.isoformat(),
                "last_updated": self.last_updated.isoformat(),
                "is_active": self.is_active,
            }
        )
        
        # Add tensor specifications
        for spec in self.get_tensor_specs():
            node.add_tensor_spec(spec)
        
        return node
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "component_class": self.__class__.__name__,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "is_active": self.is_active,
            "tensor_field_count": len(self._tensor_fields),
            "tensor_specs": [spec.to_dict() for spec in self.get_tensor_specs()],
        }