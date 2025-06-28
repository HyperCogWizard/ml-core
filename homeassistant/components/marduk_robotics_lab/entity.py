"""Base entity for Marduk's Robotics Lab."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

from homeassistant.const import ATTR_CONNECTIONS
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import MardukCoordinator
from .models import MardukRoboticsData


class MardukEntity(Entity):
    """Base class for Marduk's Robotics Lab entities."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(
        self,
        data: MardukRoboticsData,
        entity_id: str,
        name: str,
        device_id: Optional[str] = None,
    ) -> None:
        """Initialize the Marduk entity."""
        self._data = data
        self._entity_id = entity_id
        self._device_id = device_id or f"marduk_lab_{data.config_entry.entry_id}"
        
        self._attr_unique_id = f"{DOMAIN}_{data.config_entry.entry_id}_{entity_id}"
        self._attr_name = name
        
        # Set up device info
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._device_id)},
            name=f"Marduk's Robotics Lab ({self._data.config_entry.data.get('host', 'localhost')})",
            manufacturer="HyperCogWizard",
            model="Robotics Engineering Workbench",
            sw_version="1.0.0",
            configuration_url=f"http://{self._data.config_entry.data.get('host', 'localhost')}:{self._data.config_entry.data.get('port', 8765)}",
        )

    @property
    def coordinator(self) -> MardukCoordinator:
        """Return the coordinator."""
        return self._data.coordinator

    @property
    def lab_data(self) -> MardukRoboticsData:
        """Return the lab data."""
        return self._data

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        # Register for middleware events if needed
        self._data.middleware.register_event_listener(self._handle_middleware_event)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from hass."""
        await super().async_will_remove_from_hass()
        # Unregister from middleware events
        self._data.middleware.unregister_event_listener(self._handle_middleware_event)

    async def _handle_middleware_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle middleware events."""
        # Override in subclasses if needed
        pass


class MardukCoordinatedEntity(MardukEntity, CoordinatorEntity):
    """Base class for coordinated Marduk entities."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        entity_id: str,
        name: str,
        device_id: Optional[str] = None,
    ) -> None:
        """Initialize the coordinated entity."""
        CoordinatorEntity.__init__(self, coordinator)
        MardukEntity.__init__(self, data, entity_id, name, device_id)

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return super().available and self._data.middleware.connected


class MardukDeviceEntity(MardukCoordinatedEntity):
    """Base class for device-specific entities in Marduk's Lab."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        device_id: str,
        entity_suffix: str,
        name: str,
    ) -> None:
        """Initialize the device entity."""
        entity_id = f"{device_id}_{entity_suffix}"
        super().__init__(coordinator, data, entity_id, name, device_id)
        
        self._device_config_id = device_id
        
        # Update device info with device-specific details
        device_config = self._data.get_device(device_id)
        if device_config:
            self._attr_device_info = DeviceInfo(
                identifiers={(DOMAIN, device_id)},
                name=f"Device {device_id} ({device_config.device_type})",
                manufacturer="HyperCogWizard",
                model=device_config.device_type,
                sw_version="1.0.0",
                via_device=(DOMAIN, self._data.config_entry.entry_id),
            )

    @property
    def device_config(self):
        """Return the device configuration."""
        return self._data.get_device(self._device_config_id)

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return (
            super().available 
            and self.device_config is not None
            and self._device_config_id in self.coordinator.data.get("registered_devices", {})
        )


class MardukAgentEntity(MardukCoordinatedEntity):
    """Base class for agent-specific entities in Marduk's Lab."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        agent_id: str,
        entity_suffix: str,
        name: str,
    ) -> None:
        """Initialize the agent entity."""
        entity_id = f"{agent_id}_{entity_suffix}"
        super().__init__(coordinator, data, entity_id, name, f"agent_{agent_id}")
        
        self._agent_id = agent_id
        
        # Update device info with agent-specific details
        agent = self._data.get_agent(agent_id)
        if agent:
            self._attr_device_info = DeviceInfo(
                identifiers={(DOMAIN, f"agent_{agent_id}")},
                name=f"Agent {agent_id} ({agent.agent_type})",
                manufacturer="HyperCogWizard",
                model=f"Agent Kernel ({agent.cognitive_grammar})",
                sw_version="1.0.0",
                via_device=(DOMAIN, self._data.config_entry.entry_id),
            )

    @property
    def agent_kernel(self):
        """Return the agent kernel."""
        return self._data.get_agent(self._agent_id)

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return (
            super().available 
            and self.agent_kernel is not None
            and self._agent_id in self.coordinator.data.get("active_agents", {})
        )

    async def _handle_middleware_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle middleware events for agents."""
        if (event_type == "agent_state_change" 
            and event_data.get("agent_id") == self._agent_id):
            # Agent state changed, update entity
            self.async_write_ha_state()


class MardukHypergraphNodeEntity(MardukCoordinatedEntity):
    """Base class for hypergraph node entities in Marduk's Lab."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        node_id: str,
        entity_suffix: str,
        name: str,
    ) -> None:
        """Initialize the hypergraph node entity."""
        entity_id = f"node_{node_id}_{entity_suffix}"
        super().__init__(coordinator, data, entity_id, name, f"hypergraph_node_{node_id}")
        
        self._node_id = node_id
        
        # Update device info with node-specific details
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"hypergraph_node_{node_id}")},
            name=f"Hypergraph Node {node_id}",
            manufacturer="HyperCogWizard",
            model="Hypergraph Node",
            sw_version="1.0.0",
            via_device=(DOMAIN, self._data.config_entry.entry_id),
        )

    @property
    def hypergraph_node(self):
        """Return the hypergraph node."""
        return self._data.hypergraph_nodes.get(self._node_id)

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return (
            super().available 
            and self.hypergraph_node is not None
            and self._node_id in self.coordinator.data.get("hypergraph_nodes", {})
        )

    async def _handle_middleware_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle middleware events for hypergraph nodes."""
        if (event_type == "hypergraph_modification" 
            and event_data.get("node_id") == self._node_id):
            # Node modified, update entity
            self.async_write_ha_state()