"""Enhanced tensor field implementation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime

from .spec import TensorFieldSpec


@dataclass  
class TensorField:
    """Enhanced tensor field with data and metadata."""
    
    spec: TensorFieldSpec
    data: Optional[np.ndarray] = None
    timestamp: Optional[datetime] = None
    quality: float = 1.0  # Quality metric [0.0 - 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize tensor field after creation."""
        if self.data is not None:
            self._validate_data()
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def _validate_data(self) -> None:
        """Validate that data matches the specification."""
        if self.data is None:
            return
            
        if self.data.shape != self.spec.dimensions:
            raise ValueError(
                f"Data shape {self.data.shape} does not match spec dimensions {self.spec.dimensions}"
            )
        
        # Check data type compatibility
        expected_dtype = self._get_numpy_dtype()
        if self.data.dtype != expected_dtype:
            # Try to convert if possible
            try:
                self.data = self.data.astype(expected_dtype)
            except (ValueError, TypeError) as err:
                raise ValueError(f"Cannot convert data to expected dtype {expected_dtype}") from err
    
    def _get_numpy_dtype(self) -> np.dtype:
        """Get numpy dtype from tensor spec."""
        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
            "bool": bool,
        }
        return dtype_map.get(self.spec.dtype.value, np.float32)
    
    def set_data(self, data: np.ndarray, timestamp: Optional[datetime] = None) -> None:
        """Set tensor data with validation."""
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self._validate_data()
    
    def get_data_copy(self) -> Optional[np.ndarray]:
        """Get a copy of the tensor data."""
        return self.data.copy() if self.data is not None else None
    
    def is_valid(self) -> bool:
        """Check if tensor field has valid data."""
        return (
            self.data is not None and
            self.data.shape == self.spec.dimensions and
            self.quality > 0.0
        )
    
    def get_age_seconds(self) -> float:
        """Get age of data in seconds."""
        if self.timestamp is None:
            return float('inf')
        return (datetime.now() - self.timestamp).total_seconds()
    
    def is_stale(self, max_age_seconds: float = 1.0) -> bool:
        """Check if data is stale."""
        return self.get_age_seconds() > max_age_seconds
    
    def apply_range_limits(self) -> None:
        """Apply min/max value limits to the data."""
        if self.data is None:
            return
        
        if self.spec.min_value is not None:
            self.data = np.maximum(self.data, self.spec.min_value)
        
        if self.spec.max_value is not None:
            self.data = np.minimum(self.data, self.spec.max_value)
    
    def normalize(self) -> Optional[np.ndarray]:
        """Normalize data to [0, 1] range if min/max are specified."""
        if (self.data is None or 
            self.spec.min_value is None or 
            self.spec.max_value is None):
            return None
        
        value_range = self.spec.max_value - self.spec.min_value
        if value_range == 0:
            return np.zeros_like(self.data)
        
        return (self.data - self.spec.min_value) / value_range
    
    def to_gguf_dict(self) -> Dict[str, Any]:
        """Convert tensor field to GGUF-compatible dictionary."""
        return {
            "name": self.spec.name,
            "shape": self.spec.dimensions,
            "dtype": self.spec.dtype.value,
            "modality": self.spec.modality.value,
            "data": self.data.tolist() if self.data is not None else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "quality": self.quality,
            "unit": self.spec.unit,
            "update_rate_hz": self.spec.update_rate_hz,
            "metadata": {**self.spec.metadata, **self.metadata},
        }
    
    @classmethod
    def from_gguf_dict(cls, data: Dict[str, Any]) -> TensorField:
        """Create tensor field from GGUF dictionary."""
        from .spec import TensorFieldSpec, TensorDataType, TensorModalityType
        
        spec = TensorFieldSpec(
            name=data["name"],
            dimensions=tuple(data["shape"]),
            dtype=TensorDataType(data["dtype"]),
            modality=TensorModalityType(data["modality"]),
            unit=data.get("unit"),
            update_rate_hz=data.get("update_rate_hz"),
            metadata=data.get("metadata", {}),
        )
        
        tensor_data = None
        if data.get("data") is not None:
            tensor_data = np.array(data["data"])
        
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            spec=spec,
            data=tensor_data,
            timestamp=timestamp,
            quality=data.get("quality", 1.0),
        )