"""Tensor manager for workbench components."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict
from datetime import datetime

from .spec import TensorFieldSpec, TensorModalityType, StandardTensorSpecs
from .field import TensorField

_LOGGER = logging.getLogger(__name__)


class TensorManager:
    """Manages tensor fields and specifications for workbench components."""
    
    def __init__(self) -> None:
        """Initialize tensor manager."""
        self._tensor_fields: Dict[str, TensorField] = {}
        self._tensor_specs: Dict[str, TensorFieldSpec] = {}
        self._modality_index: Dict[TensorModalityType, List[str]] = defaultdict(list)
        
        # Load standard tensor specifications
        self._load_standard_specs()
    
    def _load_standard_specs(self) -> None:
        """Load standard tensor specifications."""
        for spec in StandardTensorSpecs.get_all_specs():
            self.register_tensor_spec(spec)
    
    def register_tensor_spec(self, spec: TensorFieldSpec) -> bool:
        """Register a tensor field specification."""
        if spec.name in self._tensor_specs:
            _LOGGER.warning("Tensor spec %s already exists, updating", spec.name)
        
        self._tensor_specs[spec.name] = spec
        
        # Update modality index
        if spec.name not in self._modality_index[spec.modality]:
            self._modality_index[spec.modality].append(spec.name)
        
        _LOGGER.debug("Registered tensor spec %s for modality %s", 
                     spec.name, spec.modality.value)
        return True
    
    def get_tensor_spec(self, name: str) -> Optional[TensorFieldSpec]:
        """Get tensor specification by name."""
        return self._tensor_specs.get(name)
    
    def get_specs_by_modality(self, modality: TensorModalityType) -> List[TensorFieldSpec]:
        """Get all tensor specs for a specific modality."""
        spec_names = self._modality_index.get(modality, [])
        return [self._tensor_specs[name] for name in spec_names if name in self._tensor_specs]
    
    def create_tensor_field(self, spec_name: str, data: Optional[Any] = None) -> Optional[TensorField]:
        """Create a tensor field from a registered specification."""
        spec = self.get_tensor_spec(spec_name)
        if spec is None:
            _LOGGER.error("Tensor spec %s not found", spec_name)
            return None
        
        try:
            tensor_field = TensorField(spec=spec)
            if data is not None:
                import numpy as np
                tensor_field.set_data(np.array(data))
            
            self._tensor_fields[spec_name] = tensor_field
            _LOGGER.debug("Created tensor field %s", spec_name)
            return tensor_field
            
        except Exception as err:
            _LOGGER.error("Failed to create tensor field %s: %s", spec_name, err)
            return None
    
    def update_tensor_field(self, name: str, data: Any, timestamp: Optional[datetime] = None) -> bool:
        """Update an existing tensor field with new data."""
        if name not in self._tensor_fields:
            _LOGGER.error("Tensor field %s not found", name)
            return False
        
        try:
            import numpy as np
            tensor_field = self._tensor_fields[name]
            tensor_field.set_data(np.array(data), timestamp)
            _LOGGER.debug("Updated tensor field %s", name)
            return True
            
        except Exception as err:
            _LOGGER.error("Failed to update tensor field %s: %s", name, err)
            return False
    
    def get_tensor_field(self, name: str) -> Optional[TensorField]:
        """Get tensor field by name."""
        return self._tensor_fields.get(name)
    
    def remove_tensor_field(self, name: str) -> bool:
        """Remove a tensor field."""
        if name not in self._tensor_fields:
            _LOGGER.warning("Tensor field %s not found", name)
            return False
        
        del self._tensor_fields[name]
        _LOGGER.debug("Removed tensor field %s", name)
        return True
    
    def get_all_tensor_fields(self) -> Dict[str, TensorField]:
        """Get all tensor fields."""
        return self._tensor_fields.copy()
    
    def get_stale_tensor_fields(self, max_age_seconds: float = 1.0) -> List[str]:
        """Get names of stale tensor fields."""
        stale_fields = []
        for name, field in self._tensor_fields.items():
            if field.is_stale(max_age_seconds):
                stale_fields.append(name)
        return stale_fields
    
    def cleanup_stale_fields(self, max_age_seconds: float = 10.0) -> int:
        """Remove stale tensor fields and return count removed."""
        stale_fields = self.get_stale_tensor_fields(max_age_seconds)
        for name in stale_fields:
            self.remove_tensor_field(name)
        
        if stale_fields:
            _LOGGER.debug("Cleaned up %d stale tensor fields", len(stale_fields))
        
        return len(stale_fields)
    
    def calculate_total_memory_usage(self) -> int:
        """Calculate total memory usage of all tensor fields in bytes."""
        total_bytes = 0
        for field in self._tensor_fields.values():
            total_bytes += field.spec.get_memory_size_bytes()
        return total_bytes
    
    def get_tensor_stats(self) -> Dict[str, Any]:
        """Get statistics about tensor fields."""
        total_fields = len(self._tensor_fields)
        valid_fields = sum(1 for field in self._tensor_fields.values() if field.is_valid())
        stale_fields = len(self.get_stale_tensor_fields(1.0))
        
        modality_counts = {}
        for modality, spec_names in self._modality_index.items():
            active_count = sum(1 for name in spec_names if name in self._tensor_fields)
            modality_counts[modality.value] = {
                "registered_specs": len(spec_names),
                "active_fields": active_count,
            }
        
        return {
            "total_fields": total_fields,
            "valid_fields": valid_fields,
            "stale_fields": stale_fields,
            "total_memory_bytes": self.calculate_total_memory_usage(),
            "modality_counts": modality_counts,
        }
    
    def export_tensor_definitions(self) -> Dict[str, Any]:
        """Export all tensor specifications and current field states."""
        return {
            "specifications": {
                name: spec.to_dict() for name, spec in self._tensor_specs.items()
            },
            "fields": {
                name: field.to_gguf_dict() for name, field in self._tensor_fields.items()
            },
            "stats": self.get_tensor_stats(),
        }
    
    def clear(self) -> None:
        """Clear all tensor fields but keep specifications."""
        self._tensor_fields.clear()
        _LOGGER.debug("Cleared all tensor fields")