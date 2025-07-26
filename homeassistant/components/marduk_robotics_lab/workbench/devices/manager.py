"""Device manager for workbench components."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import DeviceConfig
from .component import WorkbenchComponent
from ..hypergraph import HypergraphManager

_LOGGER = logging.getLogger(__name__)


class DeviceManager:
    """Manages robotics devices in the workbench."""
    
    def __init__(self, hypergraph_manager: HypergraphManager) -> None:
        """Initialize device manager."""
        self._hypergraph_manager = hypergraph_manager
        self._device_configs: Dict[str, DeviceConfig] = {}
        self._device_components: Dict[str, WorkbenchComponent] = {}
    
    def register_device_config(self, config: DeviceConfig) -> bool:
        """Register a device configuration."""
        validation_issues = config.validate_configuration()
        if validation_issues:
            _LOGGER.error("Device config validation failed: %s", validation_issues)
            return False
        
        self._device_configs[config.device_id] = config
        
        # Create hypergraph node for the device
        node = self._create_device_node(config)
        self._hypergraph_manager.add_node(node)
        
        _LOGGER.info("Registered device %s (%s)", config.device_id, config.device_type.value)
        return True
    
    def _create_device_node(self, config: DeviceConfig) -> Any:
        """Create hypergraph node from device configuration."""
        from ..hypergraph import WorkbenchNode
        
        node = WorkbenchNode(
            node_id=config.device_id,
            node_type="device",
            component_class="DeviceConfig",
            properties={
                "name": config.name,
                "device_type": config.device_type.value,
                "degrees_of_freedom": config.degrees_of_freedom,
                "sensor_channels": config.get_sensor_channel_count(),
                "actuator_channels": config.get_actuator_channel_count(),
                "connection_type": config.connection_type.value,
                "update_rate_hz": config.update_rate_hz,
            },
            metadata=config.metadata,
        )
        
        # Add all tensor specifications
        for spec in config.get_all_tensor_specs():
            node.add_tensor_spec(spec)
        
        return node
    
    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device."""
        if device_id not in self._device_configs:
            _LOGGER.warning("Device %s not found", device_id)
            return False
        
        # Remove from hypergraph
        self._hypergraph_manager.remove_node(device_id)
        
        # Remove device component if exists
        if device_id in self._device_components:
            del self._device_components[device_id]
        
        # Remove configuration
        del self._device_configs[device_id]
        
        _LOGGER.info("Unregistered device %s", device_id)
        return True
    
    def get_device_config(self, device_id: str) -> Optional[DeviceConfig]:
        """Get device configuration by ID."""
        return self._device_configs.get(device_id)
    
    def get_all_device_configs(self) -> Dict[str, DeviceConfig]:
        """Get all device configurations."""
        return self._device_configs.copy()
    
    def register_device_component(self, component: WorkbenchComponent) -> bool:
        """Register a device component."""
        if component.component_id not in self._device_configs:
            _LOGGER.error("Device config for %s not found", component.component_id)
            return False
        
        self._device_components[component.component_id] = component
        
        # Update hypergraph node
        node = component.create_hypergraph_node()
        self._hypergraph_manager.add_node(node)
        
        _LOGGER.info("Registered device component %s", component.component_id)
        return True
    
    def get_device_component(self, device_id: str) -> Optional[WorkbenchComponent]:
        """Get device component by ID."""
        return self._device_components.get(device_id)
    
    def get_devices_by_type(self, device_type: str) -> List[DeviceConfig]:
        """Get all devices of a specific type."""
        from .config import DeviceType
        
        try:
            target_type = DeviceType(device_type)
            return [
                config for config in self._device_configs.values()
                if config.device_type == target_type
            ]
        except ValueError:
            _LOGGER.error("Invalid device type: %s", device_type)
            return []
    
    def get_total_tensor_dimensions(self) -> int:
        """Get total tensor dimensions across all devices."""
        return sum(config.get_total_tensor_dimensions() for config in self._device_configs.values())
    
    def get_total_bandwidth_requirements(self) -> float:
        """Get total bandwidth requirements in Mbps."""
        return sum(config.get_bandwidth_requirements_mbps() for config in self._device_configs.values())
    
    def get_device_stats(self) -> Dict[str, Any]:
        """Get statistics about registered devices."""
        total_devices = len(self._device_configs)
        active_components = len(self._device_components)
        
        device_type_counts = {}
        total_dof = 0
        total_sensors = 0
        total_actuators = 0
        
        for config in self._device_configs.values():
            device_type = config.device_type.value
            device_type_counts[device_type] = device_type_counts.get(device_type, 0) + 1
            total_dof += config.degrees_of_freedom
            total_sensors += config.get_sensor_channel_count()
            total_actuators += config.get_actuator_channel_count()
        
        return {
            "total_devices": total_devices,
            "active_components": active_components,
            "device_type_counts": device_type_counts,
            "total_degrees_of_freedom": total_dof,
            "total_sensor_channels": total_sensors,
            "total_actuator_channels": total_actuators,
            "total_tensor_dimensions": self.get_total_tensor_dimensions(),
            "total_bandwidth_mbps": self.get_total_bandwidth_requirements(),
        }
    
    def export_device_definitions(self) -> Dict[str, Any]:
        """Export all device configurations."""
        return {
            "devices": {
                device_id: config.to_dict() 
                for device_id, config in self._device_configs.items()
            },
            "stats": self.get_device_stats(),
        }