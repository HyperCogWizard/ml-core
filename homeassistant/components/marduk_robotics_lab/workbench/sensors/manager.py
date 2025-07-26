"""Sensor manager for workbench components."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..tensors import TensorFieldSpec, TensorModalityType, TensorManager
from ..hypergraph import HypergraphManager

_LOGGER = logging.getLogger(__name__)


class SensorManager:
    """Manages sensor components in the workbench."""
    
    def __init__(self, hypergraph_manager: HypergraphManager, tensor_manager: TensorManager) -> None:
        """Initialize sensor manager."""
        self._hypergraph_manager = hypergraph_manager
        self._tensor_manager = tensor_manager
        self._sensor_specs: Dict[str, List[TensorFieldSpec]] = {}
    
    def register_sensor_array(self, device_id: str, sensor_specs: List[TensorFieldSpec]) -> bool:
        """Register sensor array for a device."""
        if not sensor_specs:
            _LOGGER.warning("No sensor specs provided for device %s", device_id)
            return False
        
        # Validate all specs are sensor modalities
        sensor_modalities = {
            TensorModalityType.POSITION,
            TensorModalityType.VELOCITY,
            TensorModalityType.ACCELERATION,
            TensorModalityType.ORIENTATION,
            TensorModalityType.ANGULAR_VELOCITY,
            TensorModalityType.FORCE,
            TensorModalityType.TORQUE,
            TensorModalityType.TEMPERATURE,
            TensorModalityType.PRESSURE,
            TensorModalityType.PROXIMITY,
            TensorModalityType.VISION,
            TensorModalityType.AUDIO,
            TensorModalityType.IMU,
            TensorModalityType.GPS,
            TensorModalityType.LIDAR,
            TensorModalityType.ULTRASONIC,
        }
        
        for spec in sensor_specs:
            if spec.modality not in sensor_modalities:
                _LOGGER.error("Invalid sensor modality %s for spec %s", spec.modality, spec.name)
                return False
            
            # Register with tensor manager
            self._tensor_manager.register_tensor_spec(spec)
        
        self._sensor_specs[device_id] = sensor_specs
        
        # Create sensor array node in hypergraph
        self._create_sensor_array_node(device_id, sensor_specs)
        
        _LOGGER.info("Registered sensor array for device %s with %d sensors", 
                     device_id, len(sensor_specs))
        return True
    
    def _create_sensor_array_node(self, device_id: str, sensor_specs: List[TensorFieldSpec]) -> None:
        """Create hypergraph node for sensor array."""
        from ..hypergraph import WorkbenchNode
        
        total_channels = sum(spec.get_total_elements() for spec in sensor_specs)
        
        node = WorkbenchNode(
            node_id=f"{device_id}_sensors",
            node_type="sensor_array",
            component_class="SensorArray",
            properties={
                "device_id": device_id,
                "sensor_count": len(sensor_specs),
                "total_channels": total_channels,
            }
        )
        
        # Add tensor specifications
        for spec in sensor_specs:
            node.add_tensor_spec(spec)
        
        self._hypergraph_manager.add_node(node)
    
    def get_sensor_specs(self, device_id: str) -> List[TensorFieldSpec]:
        """Get sensor specifications for a device."""
        return self._sensor_specs.get(device_id, [])
    
    def get_all_sensor_specs(self) -> Dict[str, List[TensorFieldSpec]]:
        """Get all sensor specifications."""
        return self._sensor_specs.copy()
    
    def get_sensor_stats(self) -> Dict[str, Any]:
        """Get statistics about sensors."""
        total_devices = len(self._sensor_specs)
        total_sensors = sum(len(specs) for specs in self._sensor_specs.values())
        total_channels = sum(
            sum(spec.get_total_elements() for spec in specs)
            for specs in self._sensor_specs.values()
        )
        
        modality_counts = {}
        for specs in self._sensor_specs.values():
            for spec in specs:
                modality = spec.modality.value
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            "total_devices_with_sensors": total_devices,
            "total_sensors": total_sensors,
            "total_sensor_channels": total_channels,
            "modality_counts": modality_counts,
        }