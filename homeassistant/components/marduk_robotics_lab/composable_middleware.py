"""Refactored composable robotics middleware using workbench modules."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
import websockets
import json

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError

from .const import (
    CONF_WORKBENCH_HOST,
    CONF_WORKBENCH_PORT,
    DEFAULT_WORKBENCH_PORT,
)
from .workbench import WorkbenchRegistry
from .workbench.devices import DeviceConfig, StandardDeviceConfigs
from .workbench.tensors import StandardTensorSpecs
from .models import AgentKernel

_LOGGER = logging.getLogger(__name__)


class ComposableRoboticsMiddleware:
    """Composable robotics middleware using modular workbench components."""
    
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the composable robotics middleware."""
        self.hass = hass
        self.config_entry = config_entry
        self._host = config_entry.data.get(CONF_WORKBENCH_HOST, "localhost")
        self._port = config_entry.data.get(CONF_WORKBENCH_PORT, DEFAULT_WORKBENCH_PORT)
        
        # Core workbench registry
        self.workbench = WorkbenchRegistry()
        
        # Connection management
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connected = False
        self._event_listeners: Set[callable] = set()
        
        # Initialize with standard components
        self._initialize_standard_components()
        
    def _initialize_standard_components(self) -> None:
        """Initialize standard robotics components."""
        # Load standard tensor specifications
        for spec in StandardTensorSpecs.get_all_specs():
            self.workbench.tensor_manager.register_tensor_spec(spec)
        
        # Load standard device configurations
        robot_arm = StandardDeviceConfigs.create_robot_arm_7dof()
        mobile_robot = StandardDeviceConfigs.create_mobile_robot()
        
        self.workbench.device_manager.register_device_config(robot_arm)
        self.workbench.device_manager.register_device_config(mobile_robot)
        
        # Register sensor and actuator arrays
        self.workbench.sensor_manager.register_sensor_array(
            robot_arm.device_id, robot_arm.sensor_specs
        )
        self.workbench.actuator_manager.register_actuator_array(
            robot_arm.device_id, robot_arm.actuator_specs
        )
        
        self.workbench.sensor_manager.register_sensor_array(
            mobile_robot.device_id, mobile_robot.sensor_specs
        )
        self.workbench.actuator_manager.register_actuator_array(
            mobile_robot.device_id, mobile_robot.actuator_specs
        )
        
        _LOGGER.info("Initialized standard robotics components")
    
    @property
    def connected(self) -> bool:
        """Return if middleware is connected to workbench."""
        return self._connected
    
    async def async_connect(self) -> bool:
        """Establish connection to robotics workbench."""
        try:
            uri = f"ws://{self._host}:{self._port}/ws"
            _LOGGER.debug("Connecting to robotics workbench at %s", uri)
            
            self._websocket = await websockets.connect(uri)
            self._connected = True
            
            # Start listening for events
            asyncio.create_task(self._listen_for_events())
            
            # Send workbench initialization data
            await self._send_initialization_data()
            
            _LOGGER.info("Successfully connected to Marduk's Robotics Workbench")
            return True
            
        except Exception as err:
            _LOGGER.error("Failed to connect to workbench: %s", err)
            self._connected = False
            return False
    
    async def _send_initialization_data(self) -> None:
        """Send workbench initialization data."""
        init_data = {
            "type": "workbench_init",
            "data": self.workbench.export_complete_workbench()
        }
        
        try:
            await self.send_command(init_data)
            _LOGGER.debug("Sent workbench initialization data")
        except Exception as err:
            _LOGGER.error("Failed to send initialization data: %s", err)
    
    async def async_disconnect(self) -> None:
        """Disconnect from robotics workbench."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        self._connected = False
        _LOGGER.debug("Disconnected from robotics workbench")
    
    async def _listen_for_events(self) -> None:
        """Listen for events from the workbench."""
        if not self._websocket:
            return
            
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    await self._handle_workbench_event(data)
                except json.JSONDecodeError as err:
                    _LOGGER.error("Invalid JSON received: %s", err)
                except Exception as err:
                    _LOGGER.error("Error handling workbench event: %s", err)
                    
        except websockets.exceptions.ConnectionClosed:
            _LOGGER.debug("WebSocket connection closed")
            self._connected = False
        except Exception as err:
            _LOGGER.error("Error in event listener: %s", err)
            self._connected = False
    
    async def _handle_workbench_event(self, data: Dict[str, Any]) -> None:
        """Handle events from the robotics workbench."""
        event_type = data.get("type")
        
        if event_type == "device_update":
            await self._handle_device_update(data)
        elif event_type == "tensor_update":
            await self._handle_tensor_update(data)
        elif event_type == "agent_state_change":
            await self._handle_agent_state_change(data)
        elif event_type == "hypergraph_modification":
            await self._handle_hypergraph_modification(data)
        else:
            _LOGGER.debug("Unknown event type: %s", event_type)
    
    async def _handle_device_update(self, data: Dict[str, Any]) -> None:
        """Handle device update events."""
        device_id = data.get("device_id")
        if not device_id:
            return
        
        # Update device component if it exists
        device_component = self.workbench.device_manager.get_device_component(device_id)
        if device_component:
            await device_component.async_update()
        
        _LOGGER.debug("Handled device update for %s", device_id)
        await self._notify_listeners("device_update", data)
    
    async def _handle_tensor_update(self, data: Dict[str, Any]) -> None:
        """Handle tensor field update events."""
        field_name = data.get("field_name")
        tensor_data = data.get("tensor_data")
        
        if field_name and tensor_data is not None:
            self.workbench.tensor_manager.update_tensor_field(field_name, tensor_data)
        
        _LOGGER.debug("Updated tensor field %s", field_name)
        await self._notify_listeners("tensor_update", data)
    
    async def _handle_agent_state_change(self, data: Dict[str, Any]) -> None:
        """Handle agent state change events."""
        agent_id = data.get("agent_id")
        new_state = data.get("state")
        
        if agent_id and new_state:
            self.workbench.agent_manager.update_agent_state(agent_id, new_state)
        
        _LOGGER.debug("Agent %s state changed to %s", agent_id, new_state)
        await self._notify_listeners("agent_state_change", data)
    
    async def _handle_hypergraph_modification(self, data: Dict[str, Any]) -> None:
        """Handle hypergraph structure modifications."""
        modification_type = data.get("modification_type")
        
        if modification_type == "node_added":
            await self._handle_node_added(data)
        elif modification_type == "node_removed":
            await self._handle_node_removed(data)
        elif modification_type == "edge_added":
            await self._handle_edge_added(data)
        elif modification_type == "edge_removed":
            await self._handle_edge_removed(data)
        
        await self._notify_listeners("hypergraph_modification", data)
    
    async def _handle_node_added(self, data: Dict[str, Any]) -> None:
        """Handle node addition to hypergraph."""
        node_data = data.get("node_data", {})
        # The hypergraph manager handles this internally
        _LOGGER.debug("Node added: %s", node_data.get("node_id"))
    
    async def _handle_node_removed(self, data: Dict[str, Any]) -> None:
        """Handle node removal from hypergraph."""
        node_id = data.get("node_id")
        # The hypergraph manager handles this internally
        _LOGGER.debug("Node removed: %s", node_id)
    
    async def _handle_edge_added(self, data: Dict[str, Any]) -> None:
        """Handle edge addition to hypergraph."""
        edge_data = data.get("edge_data", {})
        # The hypergraph manager handles this internally
        _LOGGER.debug("Edge added: %s", edge_data.get("edge_id"))
    
    async def _handle_edge_removed(self, data: Dict[str, Any]) -> None:
        """Handle edge removal from hypergraph."""
        edge_id = data.get("edge_id")
        # The hypergraph manager handles this internally
        _LOGGER.debug("Edge removed: %s", edge_id)
    
    async def _notify_listeners(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify all registered event listeners."""
        for listener in self._event_listeners:
            try:
                await listener(event_type, data)
            except Exception as err:
                _LOGGER.error("Error in event listener: %s", err)
    
    def register_event_listener(self, listener: callable) -> None:
        """Register an event listener."""
        self._event_listeners.add(listener)
    
    def unregister_event_listener(self, listener: callable) -> None:
        """Unregister an event listener."""
        self._event_listeners.discard(listener)
    
    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to the robotics workbench."""
        if not self._websocket or not self._connected:
            raise HomeAssistantError("Not connected to robotics workbench")
        
        try:
            message = json.dumps(command)
            await self._websocket.send(message)
            
            # Wait for response
            response = await self._websocket.recv()
            return json.loads(response)
            
        except Exception as err:
            _LOGGER.error("Failed to send command: %s", err)
            raise HomeAssistantError(f"Command failed: {err}") from err
    
    async def register_device(self, device_config: DeviceConfig) -> bool:
        """Register a new robotics device."""
        # Register with device manager
        success = self.workbench.device_manager.register_device_config(device_config)
        if not success:
            return False
        
        # Register sensor and actuator arrays
        if device_config.sensor_specs:
            self.workbench.sensor_manager.register_sensor_array(
                device_config.device_id, device_config.sensor_specs
            )
        
        if device_config.actuator_specs:
            self.workbench.actuator_manager.register_actuator_array(
                device_config.device_id, device_config.actuator_specs
            )
        
        # Send to workbench if connected
        if self._connected:
            command = {
                "type": "register_device",
                "device_config": device_config.to_dict()
            }
            
            try:
                response = await self.send_command(command)
                if response.get("status") == "success":
                    _LOGGER.info("Successfully registered device %s", device_config.device_id)
                    return True
                else:
                    _LOGGER.error("Failed to register device: %s", response.get("error"))
                    return False
            except Exception as err:
                _LOGGER.error("Error registering device: %s", err)
                return False
        
        return True
    
    async def spawn_agent(self, agent_kernel: AgentKernel) -> bool:
        """Spawn a new agent kernel."""
        # Register with agent manager
        success = self.workbench.agent_manager.register_agent_kernel(agent_kernel)
        if not success:
            return False
        
        # Send to workbench if connected
        if self._connected:
            command = {
                "type": "spawn_agent",
                "agent_data": agent_kernel.to_gguf_dict()
            }
            
            try:
                response = await self.send_command(command)
                if response.get("status") == "success":
                    _LOGGER.info("Successfully spawned agent %s", agent_kernel.kernel_id)
                    return True
                else:
                    _LOGGER.error("Failed to spawn agent: %s", response.get("error"))
                    return False
            except Exception as err:
                _LOGGER.error("Error spawning agent: %s", err)
                return False
        
        return True
    
    def get_workbench_stats(self) -> Dict[str, Any]:
        """Get comprehensive workbench statistics."""
        return self.workbench.get_complete_stats()
    
    def export_workbench_state(self) -> Dict[str, Any]:
        """Export complete workbench state."""
        return self.workbench.export_complete_workbench()
    
    def validate_workbench_consistency(self) -> List[str]:
        """Validate workbench consistency."""
        return self.workbench.validate_consistency()