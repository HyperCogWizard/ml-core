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
        """Fetch comprehensive data from all lab components."""
        data = {
            "timestamp": asyncio.get_event_loop().time(),
            "hypergraph_nodes": {},
            "registered_devices": {},
            "active_agents": {},
            "tensor_fields": {},
            "system_status": "operational",
        }
        
        try:
            # Get hypergraph state
            hypergraph_nodes = self.middleware.get_hypergraph_nodes()
            data["hypergraph_nodes"] = {
                node_id: {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "properties": node.properties,
                    "tensor_field_count": len(node.tensor_fields),
                    "connection_count": len(node.connections),
                }
                for node_id, node in hypergraph_nodes.items()
            }
            
            # Get device registry
            devices = self.middleware.get_registered_devices()
            data["registered_devices"] = {
                device_id: {
                    "device_id": device.device_id,
                    "device_type": device.device_type,
                    "degrees_of_freedom": device.degrees_of_freedom,
                    "sensor_channels": device.sensor_channels,
                    "actuator_channels": device.actuator_channels,
                    "complexity": device.get_total_complexity(),
                    "modalities": device.modalities,
                }
                for device_id, device in devices.items()
            }
            
            # Get active agents
            agents = self.middleware.get_active_agents()
            data["active_agents"] = {
                agent_id: {
                    "kernel_id": agent.kernel_id,
                    "agent_type": agent.agent_type,
                    "state": agent.state,
                    "cognitive_grammar": agent.cognitive_grammar,
                    "memory_bank_count": len(agent.memory_banks),
                    "function_count": len(agent.functions),
                    "learning_rate": agent.learning_rate,
                    "exploration_factor": agent.exploration_factor,
                }
                for agent_id, agent in agents.items()
            }
            
            # Calculate system metrics
            data["metrics"] = {
                "total_nodes": len(hypergraph_nodes),
                "total_devices": len(devices),
                "total_agents": len(agents),
                "active_agents": len([a for a in agents.values() if a.state != AGENT_STATE_IDLE]),
                "error_agents": len([a for a in agents.values() if a.state == AGENT_STATE_ERROR]),
                "total_complexity": sum(d.get_total_complexity() for d in devices.values()),
            }
            
        except Exception as err:
            _LOGGER.warning("Error fetching some data components: %s", err)
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
        """Determine if GGUF export is needed."""
        if not self._last_update_data:
            return True
        
        # Check for significant changes
        current_metrics = data.get("metrics", {})
        last_metrics = self._last_update_data.get("metrics", {})
        
        # Export if agent count or device count changed
        if (current_metrics.get("total_agents") != last_metrics.get("total_agents") or
            current_metrics.get("total_devices") != last_metrics.get("total_devices")):
            return True
        
        # Export if any agent state changed significantly
        current_agents = data.get("active_agents", {})
        last_agents = self._last_update_data.get("active_agents", {})
        
        for agent_id, agent_data in current_agents.items():
            last_agent = last_agents.get(agent_id, {})
            if agent_data.get("state") != last_agent.get("state"):
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
        """Get a summary of the current lab state."""
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
            "last_update": self.data.get("timestamp"),
        }