"""Device tracker platform for Marduk's Robotics Lab."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from homeassistant.components.device_tracker import TrackerEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, NODE_TYPE_DEVICE, NODE_TYPE_AGENT
from .coordinator import MardukCoordinator
from .entity import MardukCoordinatedEntity, MardukDeviceEntity, MardukAgentEntity
from .models import MardukRoboticsData

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Marduk's Robotics Lab device tracker platform."""
    data: MardukRoboticsData = hass.data[DOMAIN][config_entry.entry_id]
    coordinator = data.coordinator
    
    entities: list[TrackerEntity] = []
    
    # Add device trackers for mobile devices
    for device_id, device_config in data.device_configs.items():
        if device_config.device_type in ["mobile_base", "autonomous_vehicle", "drone"]:
            entities.append(
                MardukDeviceTracker(coordinator, data, device_id)
            )
    
    # Add agent trackers for mobile agents
    for agent_id in data.active_agents:
        entities.append(
            MardukAgentTracker(coordinator, data, agent_id)
        )
    
    async_add_entities(entities)


class MardukDeviceTracker(MardukDeviceEntity, TrackerEntity):
    """Device tracker for mobile robotics devices."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        device_id: str,
    ) -> None:
        """Initialize the device tracker."""
        super().__init__(
            coordinator,
            data,
            device_id,
            "location",
            f"Device {device_id} location",
        )
        self._attr_icon = "mdi:robot"

    @property
    def latitude(self) -> Optional[float]:
        """Return latitude value of the device."""
        if not self.coordinator.data:
            return None
        
        # Try to get location from hypergraph nodes
        nodes = self.coordinator.data.get("hypergraph_nodes", {})
        device_node = nodes.get(self._device_config_id)
        
        if device_node:
            properties = device_node.get("properties", {})
            location = properties.get("location", {})
            return location.get("latitude")
        
        return None

    @property
    def longitude(self) -> Optional[float]:
        """Return longitude value of the device."""
        if not self.coordinator.data:
            return None
        
        # Try to get location from hypergraph nodes
        nodes = self.coordinator.data.get("hypergraph_nodes", {})
        device_node = nodes.get(self._device_config_id)
        
        if device_node:
            properties = device_node.get("properties", {})
            location = properties.get("location", {})
            return location.get("longitude")
        
        return None

    @property
    def location_accuracy(self) -> int:
        """Return the location accuracy of the device."""
        if not self.coordinator.data:
            return 0
        
        # Try to get accuracy from hypergraph nodes
        nodes = self.coordinator.data.get("hypergraph_nodes", {})
        device_node = nodes.get(self._device_config_id)
        
        if device_node:
            properties = device_node.get("properties", {})
            location = properties.get("location", {})
            return location.get("accuracy", 5)  # Default 5m accuracy
        
        return 0

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extra state attributes."""
        if not self.coordinator.data:
            return {}
        
        attributes = {}
        
        # Get device-specific attributes
        devices = self.coordinator.data.get("registered_devices", {})
        device_data = devices.get(self._device_config_id, {})
        
        attributes.update({
            "device_type": device_data.get("device_type"),
            "degrees_of_freedom": device_data.get("degrees_of_freedom", 0),
            "sensor_channels": device_data.get("sensor_channels", 0),
            "actuator_channels": device_data.get("actuator_channels", 0),
        })
        
        # Get location-specific attributes from hypergraph
        nodes = self.coordinator.data.get("hypergraph_nodes", {})
        device_node = nodes.get(self._device_config_id)
        
        if device_node:
            properties = device_node.get("properties", {})
            location = properties.get("location", {})
            
            attributes.update({
                "altitude": location.get("altitude"),
                "heading": location.get("heading"),
                "speed": location.get("speed"),
                "last_seen": location.get("last_update"),
                "battery_level": properties.get("battery_level"),
                "signal_strength": properties.get("signal_strength"),
            })
        
        return attributes


class MardukAgentTracker(MardukAgentEntity, TrackerEntity):
    """Tracker for mobile agents in the lab."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        agent_id: str,
    ) -> None:
        """Initialize the agent tracker."""
        super().__init__(
            coordinator,
            data,
            agent_id,
            "location",
            f"Agent {agent_id} location",
        )
        self._attr_icon = "mdi:robot-excited"

    @property
    def latitude(self) -> Optional[float]:
        """Return latitude value of the agent."""
        if not self.coordinator.data:
            return None
        
        # Agents might be virtual or associated with physical devices
        # Try to get location from agent's current context
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        
        # Check if agent has location data
        location = agent_data.get("location", {})
        return location.get("latitude")

    @property
    def longitude(self) -> Optional[float]:
        """Return longitude value of the agent."""
        if not self.coordinator.data:
            return None
        
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        
        location = agent_data.get("location", {})
        return location.get("longitude")

    @property
    def location_accuracy(self) -> int:
        """Return the location accuracy of the agent."""
        if not self.coordinator.data:
            return 0
        
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        
        location = agent_data.get("location", {})
        return location.get("accuracy", 10)  # Default 10m accuracy for agents

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extra state attributes."""
        if not self.coordinator.data:
            return {}
        
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        
        attributes = {
            "agent_type": agent_data.get("agent_type"),
            "state": agent_data.get("state"),
            "cognitive_grammar": agent_data.get("cognitive_grammar"),
            "learning_rate": agent_data.get("learning_rate"),
            "exploration_factor": agent_data.get("exploration_factor"),
            "memory_banks": agent_data.get("memory_bank_count", 0),
            "functions": agent_data.get("function_count", 0),
        }
        
        # Add location-specific attributes
        location = agent_data.get("location", {})
        if location:
            attributes.update({
                "altitude": location.get("altitude"),
                "heading": location.get("heading"),
                "speed": location.get("speed"),
                "last_seen": location.get("last_update"),
                "zone": location.get("zone"),
                "context": location.get("context"),
            })
        
        return attributes