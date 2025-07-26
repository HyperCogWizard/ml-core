"""Data update coordinator for Marduk's Robotics Lab."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any, Dict, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    DOMAIN,
    UPDATE_INTERVAL_NORMAL,
    AGENT_STATE_IDLE,
    AGENT_STATE_ERROR,
)
from .middleware import RoboticsMiddleware
from .gguf_handler import GGUFHandler
from .models import MardukRoboticsData

_LOGGER = logging.getLogger(__name__)


class MardukCoordinator(DataUpdateCoordinator):
    """Coordinator for managing Marduk's Robotics Lab data updates."""
    
    def __init__(
        self,
        hass: HomeAssistant,
        middleware: RoboticsMiddleware,
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=UPDATE_INTERVAL_NORMAL),
            config_entry=config_entry,
        )
        
        self.middleware = middleware
        self.config_entry = config_entry
        self.gguf_handler = GGUFHandler(hass, config_entry)
        
        # Register for middleware events
        self.middleware.register_event_listener(self._handle_middleware_event)
        
        # Internal state
        self._last_update_data: Dict[str, Any] = {}
        self._update_lock = asyncio.Lock()
    
    async def async_setup(self) -> None:
        """Set up the coordinator."""
        try:
            # Initialize GGUF handler
            await self.gguf_handler.async_setup()
            
            # Connect to middleware
            success = await self.middleware.async_connect()
            if not success:
                raise ConfigEntryNotReady("Failed to connect to robotics workbench")
            
            _LOGGER.info("Marduk Coordinator successfully initialized")
            
        except Exception as err:
            _LOGGER.error("Failed to setup coordinator: %s", err)
            raise
    
    async def async_validate_connection(self) -> None:
        """Validate connection to the robotics workbench."""
        if not self.middleware.connected:
            raise ConfigEntryNotReady("Middleware not connected")
        
        try:
            # Test basic communication
            test_command = {"type": "ping", "timestamp": asyncio.get_event_loop().time()}
            response = await self.middleware.send_command(test_command)
            
            if response.get("type") != "pong":
                raise ConfigEntryNotReady("Invalid response from workbench")
            
            _LOGGER.debug("Connection validation successful")
            
        except Exception as err:
            _LOGGER.error("Connection validation failed: %s", err)
            raise ConfigEntryNotReady(f"Connection validation failed: {err}") from err
    
    async def _async_update_data(self) -> Dict[str, Any]:
        """Fetch the latest data from the robotics workbench."""
        async with self._update_lock:
            try:
                if not self.middleware.connected:
                    await self.middleware.async_connect()
                    if not self.middleware.connected:
                        raise UpdateFailed("Cannot connect to robotics workbench")
                
                # Gather data from various sources
                data = await self._fetch_comprehensive_data()
                
                # Export to GGUF if needed
                await self._export_gguf_state(data)
                
                self._last_update_data = data
                return data
                
            except Exception as err:
                _LOGGER.error("Error updating coordinator data: %s", err)
                raise UpdateFailed(f"Failed to update data: {err}") from err
    
    async def _fetch_comprehensive_data(self) -> Dict[str, Any]:
        """Fetch comprehensive data from all lab components with enhanced GGUF support."""
        data = {
            "timestamp": asyncio.get_event_loop().time(),
            "hypergraph_nodes": {},
            "registered_devices": {},
            "active_agents": {},
            "tensor_fields": {},
            "system_status": "operational",
        }
        
        try:
            # Get enhanced hypergraph state with comprehensive tensor fields
            hypergraph_nodes = self.middleware.get_hypergraph_nodes()
            data["hypergraph_nodes"] = {
                node_id: {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "properties": node.properties,
                    "tensor_fields": [field.to_gguf_dict() for field in node.tensor_fields],
                    "connections": node.connections,
                    # Enhanced comprehensive fields
                    "node_state": getattr(node, 'node_state', {}),
                    "processing_load": getattr(node, 'processing_load', 0.0),
                    "communication_latency": getattr(node, 'communication_latency', {}),
                    "membrane_bindings": getattr(node, 'membrane_bindings', []),
                    "tensor_field_count": len(node.tensor_fields),
                    "connection_count": len(node.connections),
                }
                for node_id, node in hypergraph_nodes.items()
            }
            
            # Get enhanced device registry with comprehensive configuration
            devices = self.middleware.get_registered_devices()
            data["registered_devices"] = {
                device_id: device.to_gguf_dict() if hasattr(device, 'to_gguf_dict') else {
                    "device_id": device.device_id,
                    "device_type": device.device_type,
                    "degrees_of_freedom": device.degrees_of_freedom,
                    "sensor_channels": device.sensor_channels,
                    "actuator_channels": device.actuator_channels,
                    "tensor_dimensions": getattr(device, 'tensor_dimensions', ()),
                    "modalities": getattr(device, 'modalities', []),
                    "complexity": device.get_total_complexity(),
                    # Enhanced fields with fallbacks
                    "device_parameters": getattr(device, 'device_parameters', {}),
                    "sensor_specifications": getattr(device, 'sensor_specifications', []),
                    "actuator_specifications": getattr(device, 'actuator_specifications', []),
                    "calibration_data": getattr(device, 'calibration_data', {}),
                    "communication_protocols": getattr(device, 'communication_protocols', []),
                    "firmware_version": getattr(device, 'firmware_version', "unknown"),
                    "device_status": getattr(device, 'device_status', "operational"),
                    "membrane_structure": {
                        "membrane_id": getattr(device, 'membrane_id', None),
                        "membrane_type": getattr(device, 'membrane_type', "device_membrane"),
                        "input_channels": getattr(device, 'input_channels', []),
                        "output_channels": getattr(device, 'output_channels', [])
                    }
                }
                for device_id, device in devices.items()
            }
            
            # Get enhanced active agents with comprehensive state
            agents = self.middleware.get_active_agents()
            data["active_agents"] = {
                agent_id: agent.to_gguf_dict() if hasattr(agent, 'to_gguf_dict') else {
                    "kernel_id": agent.kernel_id,
                    "agent_type": agent.agent_type,
                    "state": agent.state,
                    "cognitive_grammar": getattr(agent, 'cognitive_grammar', 'scheme'),
                    "memory_bank_count": len(getattr(agent, 'memory_banks', [])),
                    "function_count": len(getattr(agent, 'functions', [])),
                    "learning_rate": getattr(agent, 'learning_rate', 0.01),
                    "exploration_factor": getattr(agent, 'exploration_factor', 0.1),
                    # Enhanced comprehensive fields with fallbacks
                    "functions": getattr(agent, 'functions', []),
                    "state_history": getattr(agent, 'state_history', []),
                    "cognitive_parameters": getattr(agent, 'cognitive_parameters', {}),
                    "execution_context": getattr(agent, 'execution_context', {}),
                    "communication_channels": getattr(agent, 'communication_channels', []),
                    "memory_banks": [
                        bank.to_gguf_dict() if hasattr(bank, 'to_gguf_dict') else {
                            "name": getattr(bank, 'name', f'memory_{i}'),
                            "dimensions": getattr(bank, 'dimensions', []),
                            "dtype": getattr(bank, 'dtype', 'float32'),
                            "data": getattr(bank, 'data', [])
                        }
                        for i, bank in enumerate(getattr(agent, 'memory_banks', []))
                    ],
                    "membrane_structure": {
                        "membrane_id": getattr(agent, 'membrane_id', None),
                        "parent_membrane": getattr(agent, 'parent_membrane', None),
                        "child_membranes": getattr(agent, 'child_membranes', [])
                    }
                }
                for agent_id, agent in agents.items()
            }
            
            # Enhanced system metrics calculation
            data["metrics"] = {
                "total_nodes": len(hypergraph_nodes),
                "total_devices": len(devices),
                "total_agents": len(agents),
                "active_agents": len([a for a in agents.values() if a.state != AGENT_STATE_IDLE]),
                "error_agents": len([a for a in agents.values() if a.state == AGENT_STATE_ERROR]),
                "total_complexity": sum(d.get_total_complexity() for d in devices.values()),
                # Enhanced metrics
                "total_tensor_fields": sum(
                    len(node.tensor_fields) for node in hypergraph_nodes.values()
                ),
                "total_memory_banks": sum(
                    len(getattr(agent, 'memory_banks', [])) for agent in agents.values()
                ),
                "total_membranes": len([
                    agent for agent in agents.values() 
                    if getattr(agent, 'membrane_id', None)
                ]) + len([
                    device for device in devices.values() 
                    if getattr(device, 'membrane_id', None)
                ]),
                "average_processing_load": sum(
                    getattr(node, 'processing_load', 0.0) for node in hypergraph_nodes.values()
                ) / max(len(hypergraph_nodes), 1)
            }
            
        except Exception as err:
            _LOGGER.warning("Error fetching some comprehensive data components: %s", err)
            data["system_status"] = "degraded"
        
        return data
    
    async def _export_gguf_state(self, data: Dict[str, Any]) -> None:
        """Export current state to GGUF format."""
        try:
            # Only export if there are significant changes
            if self._should_export_gguf(data):
                await self.gguf_handler.export_lab_state(data)
                _LOGGER.debug("Exported lab state to GGUF")
        except Exception as err:
            _LOGGER.warning("Failed to export GGUF state: %s", err)
    
    def _should_export_gguf(self, data: Dict[str, Any]) -> bool:
        """Determine if comprehensive GGUF export is needed."""
        if not self._last_update_data:
            return True
        
        # Check for significant changes in comprehensive metrics
        current_metrics = data.get("metrics", {})
        last_metrics = self._last_update_data.get("metrics", {})
        
        # Export if agent count, device count, or membrane count changed
        significant_changes = [
            current_metrics.get("total_agents") != last_metrics.get("total_agents"),
            current_metrics.get("total_devices") != last_metrics.get("total_devices"),
            current_metrics.get("total_membranes") != last_metrics.get("total_membranes"),
            current_metrics.get("total_tensor_fields") != last_metrics.get("total_tensor_fields"),
        ]
        
        if any(significant_changes):
            return True
        
        # Export if any agent state changed significantly
        current_agents = data.get("active_agents", {})
        last_agents = self._last_update_data.get("active_agents", {})
        
        for agent_id, agent_data in current_agents.items():
            last_agent = last_agents.get(agent_id, {})
            # Check state changes, memory bank changes, or cognitive parameter changes
            if (agent_data.get("state") != last_agent.get("state") or
                agent_data.get("memory_bank_count") != last_agent.get("memory_bank_count") or
                len(agent_data.get("state_history", [])) != len(last_agent.get("state_history", []))):
                return True
        
        # Export if any device configuration changed
        current_devices = data.get("registered_devices", {})
        last_devices = self._last_update_data.get("registered_devices", {})
        
        for device_id, device_data in current_devices.items():
            last_device = last_devices.get(device_id, {})
            # Check for device status or specification changes
            if (device_data.get("device_status") != last_device.get("device_status") or
                len(device_data.get("sensor_specifications", [])) != len(last_device.get("sensor_specifications", [])) or
                len(device_data.get("actuator_specifications", [])) != len(last_device.get("actuator_specifications", []))):
                return True
        
        # Export if hypergraph structure changed significantly
        current_nodes = data.get("hypergraph_nodes", {})
        last_nodes = self._last_update_data.get("hypergraph_nodes", {})
        
        if len(current_nodes) != len(last_nodes):
            return True
        
        for node_id, node_data in current_nodes.items():
            last_node = last_nodes.get(node_id, {})
            if (node_data.get("tensor_field_count") != last_node.get("tensor_field_count") or
                node_data.get("processing_load") != last_node.get("processing_load")):
                return True
        
        return False
    
    async def _handle_middleware_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle events from the middleware."""
        _LOGGER.debug("Received middleware event: %s", event_type)
        
        # Trigger immediate update for important events
        if event_type in ["agent_state_change", "device_update", "hypergraph_modification"]:
            # Schedule an immediate update
            asyncio.create_task(self.async_request_refresh())
    
    async def async_shutdown(self) -> None:
        """Shutdown the coordinator and cleanup resources."""
        try:
            # Unregister from middleware events
            self.middleware.unregister_event_listener(self._handle_middleware_event)
            
            # Disconnect middleware
            await self.middleware.async_disconnect()
            
            # Shutdown GGUF handler
            await self.gguf_handler.async_shutdown()
            
            _LOGGER.info("Marduk Coordinator shutdown complete")
            
        except Exception as err:
            _LOGGER.error("Error during coordinator shutdown: %s", err)
    
    async def async_spawn_agent(self, agent_type: str, agent_id: Optional[str] = None) -> str:
        """Spawn a new agent kernel."""
        from .models import AgentKernel
        
        if not agent_id:
            agent_id = f"agent_{len(self.middleware.get_active_agents())}"
        
        agent = AgentKernel(
            kernel_id=agent_id,
            agent_type=agent_type,
            state=AGENT_STATE_IDLE,
        )
        
        success = await self.middleware.spawn_agent(agent)
        if not success:
            raise HomeAssistantError(f"Failed to spawn agent {agent_id}")
        
        # Trigger immediate update
        await self.async_request_refresh()
        
        return agent_id
    
    async def async_register_device(
        self,
        device_id: str,
        device_type: str,
        degrees_of_freedom: int = 0,
        sensor_channels: int = 0,
        actuator_channels: int = 0,
    ) -> bool:
        """Register a new robotics device."""
        from .models import DeviceConfiguration
        
        device_config = DeviceConfiguration(
            device_id=device_id,
            device_type=device_type,
            degrees_of_freedom=degrees_of_freedom,
            sensor_channels=sensor_channels,
            actuator_channels=actuator_channels,
        )
        
        success = await self.middleware.register_device(device_config)
        if success:
            # Trigger immediate update
            await self.async_request_refresh()
        
        return success
    
    def get_lab_summary(self) -> Dict[str, Any]:
        """Get a summary of the current lab state with comprehensive metrics."""
        if not self.data:
            return {"status": "no_data"}
        
        metrics = self.data.get("metrics", {})
        return {
            "status": self.data.get("system_status", "unknown"),
            "total_nodes": metrics.get("total_nodes", 0),
            "total_devices": metrics.get("total_devices", 0),
            "total_agents": metrics.get("total_agents", 0),
            "active_agents": metrics.get("active_agents", 0),
            "error_agents": metrics.get("error_agents", 0),
            "total_complexity": metrics.get("total_complexity", 0),
            # Enhanced comprehensive metrics
            "total_tensor_fields": metrics.get("total_tensor_fields", 0),
            "total_memory_banks": metrics.get("total_memory_banks", 0),
            "total_membranes": metrics.get("total_membranes", 0),
            "average_processing_load": metrics.get("average_processing_load", 0.0),
            "last_update": self.data.get("timestamp"),
            "gguf_compatible": True,
            "p_system_ready": metrics.get("total_membranes", 0) > 0
        }
    
    async def async_create_comprehensive_snapshot(self) -> str:
        """Create a comprehensive GGUF snapshot of the entire lab state."""
        try:
            if not self.data:
                raise HomeAssistantError("No lab data available for snapshot")
            
            # Create comprehensive snapshot using enhanced GGUF handler
            snapshot_path = await self.gguf_handler.create_comprehensive_snapshot(self.data)
            
            _LOGGER.info("Created comprehensive lab snapshot: %s", snapshot_path)
            return snapshot_path
            
        except Exception as err:
            _LOGGER.error("Failed to create comprehensive snapshot: %s", err)
            raise HomeAssistantError(f"Snapshot creation failed: {err}") from err
    
    async def async_export_p_system_membranes(self) -> str:
        """Export P-System membrane hierarchy to dedicated GGUF file."""
        try:
            if not self.data:
                raise HomeAssistantError("No lab data available for membrane export")
            
            # Generate membrane hierarchy from current lab state
            membrane_hierarchy = self._generate_membrane_hierarchy_from_data()
            
            # Export using enhanced GGUF handler
            p_system_path = await self.gguf_handler.export_p_system_membranes(membrane_hierarchy)
            
            _LOGGER.info("Exported P-System membranes: %s", p_system_path)
            return p_system_path
            
        except Exception as err:
            _LOGGER.error("Failed to export P-System membranes: %s", err)
            raise HomeAssistantError(f"P-System export failed: {err}") from err
    
    def _generate_membrane_hierarchy_from_data(self) -> Dict[str, Any]:
        """Generate P-System membrane hierarchy from current lab data."""
        membranes = {}
        
        # Extract agent membranes
        for agent_id, agent_data in self.data.get("active_agents", {}).items():
            membrane_struct = agent_data.get("membrane_structure", {})
            if membrane_struct.get("membrane_id"):
                membranes[membrane_struct["membrane_id"]] = {
                    "membrane_type": "agent_membrane",
                    "entity_id": agent_id,
                    "entity_type": "agent",
                    "parent_membrane": membrane_struct.get("parent_membrane"),
                    "child_membranes": membrane_struct.get("child_membranes", []),
                    "communication_channels": agent_data.get("communication_channels", [])
                }
        
        # Extract device membranes
        for device_id, device_data in self.data.get("registered_devices", {}).items():
            membrane_struct = device_data.get("membrane_structure", {})
            if membrane_struct.get("membrane_id"):
                membranes[membrane_struct["membrane_id"]] = {
                    "membrane_type": membrane_struct.get("membrane_type", "device_membrane"),
                    "entity_id": device_id,
                    "entity_type": "device",
                    "input_channels": membrane_struct.get("input_channels", []),
                    "output_channels": membrane_struct.get("output_channels", [])
                }
        
        # Create lab-level root membrane
        membranes["lab_root"] = {
            "membrane_type": "lab_membrane",
            "entity_id": self.config_entry.entry_id,
            "entity_type": "laboratory",
            "child_membranes": list(membranes.keys()),
            "total_agents": len(self.data.get("active_agents", {})),
            "total_devices": len(self.data.get("registered_devices", {})),
            "total_tensors": self.data.get("metrics", {}).get("total_tensor_fields", 0)
        }
        
        return membranes