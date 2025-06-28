"""Robotics Middleware Abstraction Layer for Marduk's Lab."""

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
    NODE_TYPE_DEVICE,
    NODE_TYPE_SENSOR,
    NODE_TYPE_ACTUATOR,
    NODE_TYPE_AGENT,
    API_ENDPOINT_AGENTS,
    API_ENDPOINT_DEVICES,
)
from .models import HypergraphNode, DeviceConfiguration, AgentKernel, TensorField

_LOGGER = logging.getLogger(__name__)


class RoboticsMiddleware:
    """Abstraction layer for robotics device communication and control."""
    
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the robotics middleware."""
        self.hass = hass
        self.config_entry = config_entry
        self._host = config_entry.data.get(CONF_WORKBENCH_HOST, "localhost")
        self._port = config_entry.data.get(CONF_WORKBENCH_PORT, DEFAULT_WORKBENCH_PORT)
        
        # Core components
        self._hypergraph_nodes: Dict[str, HypergraphNode] = {}
        self._device_registry: Dict[str, DeviceConfiguration] = {}
        self._agent_kernels: Dict[str, AgentKernel] = {}
        
        # Connection management
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connected = False
        self._event_listeners: Set[callable] = set()
        
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
            
            _LOGGER.info("Successfully connected to Marduk's Robotics Workbench")
            return True
            
        except Exception as err:
            _LOGGER.error("Failed to connect to workbench: %s", err)
            self._connected = False
            return False
    
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
        elif event_type == "agent_state_change":
            await self._handle_agent_state_change(data)
        elif event_type == "tensor_field_update":
            await self._handle_tensor_field_update(data)
        elif event_type == "hypergraph_modification":
            await self._handle_hypergraph_modification(data)
        else:
            _LOGGER.debug("Unknown event type: %s", event_type)
    
    async def _handle_device_update(self, data: Dict[str, Any]) -> None:
        """Handle device update events."""
        device_id = data.get("device_id")
        if not device_id:
            return
            
        # Update device configuration
        device_data = data.get("device_data", {})
        if device_id in self._device_registry:
            device = self._device_registry[device_id]
            # Update device properties based on received data
            _LOGGER.debug("Updated device %s", device_id)
        
        # Notify listeners
        await self._notify_listeners("device_update", data)
    
    async def _handle_agent_state_change(self, data: Dict[str, Any]) -> None:
        """Handle agent state change events."""
        agent_id = data.get("agent_id")
        new_state = data.get("state")
        
        if agent_id in self._agent_kernels:
            self._agent_kernels[agent_id].state = new_state
            _LOGGER.debug("Agent %s state changed to %s", agent_id, new_state)
        
        await self._notify_listeners("agent_state_change", data)
    
    async def _handle_tensor_field_update(self, data: Dict[str, Any]) -> None:
        """Handle tensor field update events."""
        field_name = data.get("field_name")
        tensor_data = data.get("tensor_data")
        
        # Update tensor field in hypergraph
        _LOGGER.debug("Updated tensor field %s", field_name)
        await self._notify_listeners("tensor_field_update", data)
    
    async def _handle_hypergraph_modification(self, data: Dict[str, Any]) -> None:
        """Handle hypergraph structure modifications."""
        modification_type = data.get("modification_type")
        node_id = data.get("node_id")
        
        if modification_type == "node_added":
            await self._add_hypergraph_node(data)
        elif modification_type == "node_removed":
            await self._remove_hypergraph_node(node_id)
        elif modification_type == "edge_added":
            await self._add_hypergraph_edge(data)
        elif modification_type == "edge_removed":
            await self._remove_hypergraph_edge(data)
        
        await self._notify_listeners("hypergraph_modification", data)
    
    async def _add_hypergraph_node(self, data: Dict[str, Any]) -> None:
        """Add a node to the hypergraph."""
        node_data = data.get("node_data", {})
        node_id = node_data.get("node_id")
        node_type = node_data.get("node_type", NODE_TYPE_DEVICE)
        
        if node_id:
            node = HypergraphNode(
                node_id=node_id,
                node_type=node_type,
                properties=node_data.get("properties", {}),
                connections=node_data.get("connections", [])
            )
            self._hypergraph_nodes[node_id] = node
            _LOGGER.debug("Added hypergraph node %s of type %s", node_id, node_type)
    
    async def _remove_hypergraph_node(self, node_id: str) -> None:
        """Remove a node from the hypergraph."""
        if node_id in self._hypergraph_nodes:
            del self._hypergraph_nodes[node_id]
            _LOGGER.debug("Removed hypergraph node %s", node_id)
    
    async def _add_hypergraph_edge(self, data: Dict[str, Any]) -> None:
        """Add an edge between hypergraph nodes."""
        source_id = data.get("source_id")
        target_id = data.get("target_id")
        
        if source_id in self._hypergraph_nodes and target_id not in self._hypergraph_nodes[source_id].connections:
            self._hypergraph_nodes[source_id].connections.append(target_id)
            _LOGGER.debug("Added edge from %s to %s", source_id, target_id)
    
    async def _remove_hypergraph_edge(self, data: Dict[str, Any]) -> None:
        """Remove an edge between hypergraph nodes."""
        source_id = data.get("source_id")
        target_id = data.get("target_id")
        
        if source_id in self._hypergraph_nodes and target_id in self._hypergraph_nodes[source_id].connections:
            self._hypergraph_nodes[source_id].connections.remove(target_id)
            _LOGGER.debug("Removed edge from %s to %s", source_id, target_id)
    
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
    
    async def register_device(self, device_config: DeviceConfiguration) -> bool:
        """Register a new robotics device."""
        command = {
            "type": "register_device",
            "device_config": {
                "device_id": device_config.device_id,
                "device_type": device_config.device_type,
                "degrees_of_freedom": device_config.degrees_of_freedom,
                "sensor_channels": device_config.sensor_channels,
                "actuator_channels": device_config.actuator_channels,
                "tensor_dimensions": device_config.tensor_dimensions,
                "modalities": device_config.modalities
            }
        }
        
        try:
            response = await self.send_command(command)
            if response.get("status") == "success":
                self._device_registry[device_config.device_id] = device_config
                _LOGGER.info("Successfully registered device %s", device_config.device_id)
                return True
            else:
                _LOGGER.error("Failed to register device: %s", response.get("error"))
                return False
        except Exception as err:
            _LOGGER.error("Error registering device: %s", err)
            return False
    
    async def spawn_agent(self, agent_kernel: AgentKernel) -> bool:
        """Spawn a new agent kernel."""
        command = {
            "type": "spawn_agent",
            "agent_data": agent_kernel.to_gguf_dict()
        }
        
        try:
            response = await self.send_command(command)
            if response.get("status") == "success":
                self._agent_kernels[agent_kernel.kernel_id] = agent_kernel
                _LOGGER.info("Successfully spawned agent %s", agent_kernel.kernel_id)
                return True
            else:
                _LOGGER.error("Failed to spawn agent: %s", response.get("error"))
                return False
        except Exception as err:
            _LOGGER.error("Error spawning agent: %s", err)
            return False
    
    def get_hypergraph_nodes(self) -> Dict[str, HypergraphNode]:
        """Get all hypergraph nodes."""
        return self._hypergraph_nodes.copy()
    
    def get_registered_devices(self) -> Dict[str, DeviceConfiguration]:
        """Get all registered devices."""
        return self._device_registry.copy()
    
    def get_active_agents(self) -> Dict[str, AgentKernel]:
        """Get all active agent kernels."""
        return self._agent_kernels.copy()