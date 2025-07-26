"""Actuator manager for workbench components."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..tensors import TensorFieldSpec, TensorModalityType, TensorManager
from ..hypergraph import HypergraphManager

_LOGGER = logging.getLogger(__name__)


class ActuatorManager:
    """Manages actuator components in the workbench."""
    
    def __init__(self, hypergraph_manager: HypergraphManager, tensor_manager: TensorManager) -> None:
        """Initialize actuator manager."""
        self._hypergraph_manager = hypergraph_manager
        self._tensor_manager = tensor_manager
        self._actuator_specs: Dict[str, List[TensorFieldSpec]] = {}
    
    def register_actuator_array(self, device_id: str, actuator_specs: List[TensorFieldSpec]) -> bool:
        """Register actuator array for a device."""
        if not actuator_specs:
            _LOGGER.warning("No actuator specs provided for device %s", device_id)
            return False
        
        # Validate all specs are actuator modalities
        actuator_modalities = {
            TensorModalityType.MOTOR_POSITION,
            TensorModalityType.MOTOR_VELOCITY,
            TensorModalityType.MOTOR_TORQUE,
            TensorModalityType.SERVO_ANGLE,
            TensorModalityType.PNEUMATIC_PRESSURE,
            TensorModalityType.HYDRAULIC_PRESSURE,
            TensorModalityType.LED_BRIGHTNESS,
            TensorModalityType.SPEAKER_VOLUME,
            TensorModalityType.HEATER_POWER,
            TensorModalityType.COMMAND_VECTOR,
        }
        
        for spec in actuator_specs:
            if spec.modality not in actuator_modalities:
                _LOGGER.error("Invalid actuator modality %s for spec %s", spec.modality, spec.name)
                return False
            
            # Register with tensor manager
            self._tensor_manager.register_tensor_spec(spec)
        
        self._actuator_specs[device_id] = actuator_specs
        
        # Create actuator array node in hypergraph
        self._create_actuator_array_node(device_id, actuator_specs)
        
        _LOGGER.info("Registered actuator array for device %s with %d actuators", 
                     device_id, len(actuator_specs))
        return True
    
    def _create_actuator_array_node(self, device_id: str, actuator_specs: List[TensorFieldSpec]) -> None:
        """Create hypergraph node for actuator array."""
        from ..hypergraph import WorkbenchNode
        
        total_channels = sum(spec.get_total_elements() for spec in actuator_specs)
        
        node = WorkbenchNode(
            node_id=f"{device_id}_actuators",
            node_type="actuator_array",
            component_class="ActuatorArray",
            properties={
                "device_id": device_id,
                "actuator_count": len(actuator_specs),
                "total_channels": total_channels,
            }
        )
        
        # Add tensor specifications
        for spec in actuator_specs:
            node.add_tensor_spec(spec)
        
        self._hypergraph_manager.add_node(node)
    
    def get_actuator_specs(self, device_id: str) -> List[TensorFieldSpec]:
        """Get actuator specifications for a device."""
        return self._actuator_specs.get(device_id, [])
    
    def get_all_actuator_specs(self) -> Dict[str, List[TensorFieldSpec]]:
        """Get all actuator specifications."""
        return self._actuator_specs.copy()
    
    def get_actuator_stats(self) -> Dict[str, Any]:
        """Get statistics about actuators."""
        total_devices = len(self._actuator_specs)
        total_actuators = sum(len(specs) for specs in self._actuator_specs.values())
        total_channels = sum(
            sum(spec.get_total_elements() for spec in specs)
            for specs in self._actuator_specs.values()
        )
        
        modality_counts = {}
        for specs in self._actuator_specs.values():
            for spec in specs:
                modality = spec.modality.value
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            "total_devices_with_actuators": total_devices,
            "total_actuators": total_actuators,
            "total_actuator_channels": total_channels,
            "modality_counts": modality_counts,
        }