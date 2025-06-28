"""Sensor platform for Marduk's Robotics Lab."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType

from .const import DOMAIN, AGENT_STATE_IDLE, AGENT_STATE_ERROR
from .coordinator import MardukCoordinator
from .entity import MardukCoordinatedEntity, MardukAgentEntity, MardukDeviceEntity
from .models import MardukRoboticsData

_LOGGER = logging.getLogger(__name__)

# Lab system sensors
LAB_SENSORS = [
    SensorEntityDescription(
        key="total_agents",
        name="Total agents",
        icon="mdi:robot",
        entity_category=EntityCategory.DIAGNOSTIC,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="active_agents",
        name="Active agents",
        icon="mdi:robot-excited",
        entity_category=EntityCategory.DIAGNOSTIC,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="total_devices",
        name="Total devices",
        icon="mdi:devices",
        entity_category=EntityCategory.DIAGNOSTIC,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="total_nodes",
        name="Hypergraph nodes",
        icon="mdi:graph",
        entity_category=EntityCategory.DIAGNOSTIC,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="total_complexity",
        name="System complexity",
        icon="mdi:brain",
        entity_category=EntityCategory.DIAGNOSTIC,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    SensorEntityDescription(
        key="system_status",
        name="System status",
        icon="mdi:state-machine",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Marduk's Robotics Lab sensor platform."""
    data: MardukRoboticsData = hass.data[DOMAIN][config_entry.entry_id]
    coordinator = data.coordinator
    
    entities: list[SensorEntity] = []
    
    # Add lab system sensors
    for description in LAB_SENSORS:
        entities.append(
            MardukLabSensor(coordinator, data, description)
        )
    
    # Add agent-specific sensors
    for agent_id in data.active_agents:
        entities.extend([
            MardukAgentStateSensor(coordinator, data, agent_id),
            MardukAgentLearningRateSensor(coordinator, data, agent_id),
            MardukAgentMemoryBanksSensor(coordinator, data, agent_id),
        ])
    
    # Add device-specific sensors
    for device_id in data.device_configs:
        entities.extend([
            MardukDeviceComplexitySensor(coordinator, data, device_id),
            MardukDeviceChannelsSensor(coordinator, data, device_id),
        ])
    
    async_add_entities(entities)


class MardukLabSensor(MardukCoordinatedEntity, SensorEntity):
    """Sensor for lab-wide metrics."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        description: SensorEntityDescription,
    ) -> None:
        """Initialize the lab sensor."""
        super().__init__(
            coordinator,
            data,
            f"lab_{description.key}",
            description.name,
        )
        self.entity_description = description
        self._attr_translation_key = description.key

    @property
    def native_value(self) -> StateType:
        """Return the state of the sensor."""
        if not self.coordinator.data:
            return None
        
        metrics = self.coordinator.data.get("metrics", {})
        
        if self.entity_description.key == "system_status":
            return self.coordinator.data.get("system_status", "unknown")
        
        return metrics.get(self.entity_description.key)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extra state attributes."""
        if not self.coordinator.data:
            return {}
        
        attributes = {
            "last_update": self.coordinator.data.get("timestamp"),
            "lab_id": self.lab_data.config_entry.entry_id,
        }
        
        if self.entity_description.key == "system_status":
            metrics = self.coordinator.data.get("metrics", {})
            attributes.update({
                "error_agents": metrics.get("error_agents", 0),
                "middleware_connected": self.lab_data.middleware.connected,
            })
        
        return attributes


class MardukAgentStateSensor(MardukAgentEntity, SensorEntity):
    """Sensor for agent state."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        agent_id: str,
    ) -> None:
        """Initialize the agent state sensor."""
        super().__init__(
            coordinator,
            data,
            agent_id,
            "state",
            f"Agent {agent_id} state",
        )
        self._attr_icon = "mdi:robot"
        self._attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> StateType:
        """Return the state of the agent."""
        if not self.coordinator.data:
            return None
        
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        return agent_data.get("state", "unknown")

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extra state attributes."""
        if not self.coordinator.data:
            return {}
        
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        
        return {
            "agent_type": agent_data.get("agent_type"),
            "cognitive_grammar": agent_data.get("cognitive_grammar"),
            "memory_banks": agent_data.get("memory_bank_count", 0),
            "functions": agent_data.get("function_count", 0),
            "learning_rate": agent_data.get("learning_rate"),
            "exploration_factor": agent_data.get("exploration_factor"),
        }

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend."""
        state = self.native_value
        if state == AGENT_STATE_ERROR:
            return "mdi:robot-angry"
        elif state == AGENT_STATE_IDLE:
            return "mdi:robot"
        else:
            return "mdi:robot-excited"


class MardukAgentLearningRateSensor(MardukAgentEntity, SensorEntity):
    """Sensor for agent learning rate."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        agent_id: str,
    ) -> None:
        """Initialize the agent learning rate sensor."""
        super().__init__(
            coordinator,
            data,
            agent_id,
            "learning_rate",
            f"Agent {agent_id} learning rate",
        )
        self._attr_icon = "mdi:brain"
        self._attr_entity_category = EntityCategory.DIAGNOSTIC
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_suggested_display_precision = 4

    @property
    def native_value(self) -> StateType:
        """Return the learning rate of the agent."""
        if not self.coordinator.data:
            return None
        
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        return agent_data.get("learning_rate")


class MardukAgentMemoryBanksSensor(MardukAgentEntity, SensorEntity):
    """Sensor for agent memory banks count."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        agent_id: str,
    ) -> None:
        """Initialize the agent memory banks sensor."""
        super().__init__(
            coordinator,
            data,
            agent_id,
            "memory_banks",
            f"Agent {agent_id} memory banks",
        )
        self._attr_icon = "mdi:memory"
        self._attr_entity_category = EntityCategory.DIAGNOSTIC
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> StateType:
        """Return the number of memory banks."""
        if not self.coordinator.data:
            return None
        
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        return agent_data.get("memory_bank_count", 0)


class MardukDeviceComplexitySensor(MardukDeviceEntity, SensorEntity):
    """Sensor for device complexity."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        device_id: str,
    ) -> None:
        """Initialize the device complexity sensor."""
        super().__init__(
            coordinator,
            data,
            device_id,
            "complexity",
            f"Device {device_id} complexity",
        )
        self._attr_icon = "mdi:complexity"
        self._attr_entity_category = EntityCategory.DIAGNOSTIC
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> StateType:
        """Return the complexity of the device."""
        if not self.coordinator.data:
            return None
        
        devices = self.coordinator.data.get("registered_devices", {})
        device_data = devices.get(self._device_config_id, {})
        return device_data.get("complexity", 0)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extra state attributes."""
        if not self.coordinator.data:
            return {}
        
        devices = self.coordinator.data.get("registered_devices", {})
        device_data = devices.get(self._device_config_id, {})
        
        return {
            "device_type": device_data.get("device_type"),
            "degrees_of_freedom": device_data.get("degrees_of_freedom", 0),
            "sensor_channels": device_data.get("sensor_channels", 0),
            "actuator_channels": device_data.get("actuator_channels", 0),
            "modalities": device_data.get("modalities", []),
        }


class MardukDeviceChannelsSensor(MardukDeviceEntity, SensorEntity):
    """Sensor for device total channels."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        device_id: str,
    ) -> None:
        """Initialize the device channels sensor."""
        super().__init__(
            coordinator,
            data,
            device_id,
            "channels",
            f"Device {device_id} channels",
        )
        self._attr_icon = "mdi:connection"
        self._attr_entity_category = EntityCategory.DIAGNOSTIC
        self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def native_value(self) -> StateType:
        """Return the total channels of the device."""
        if not self.coordinator.data:
            return None
        
        devices = self.coordinator.data.get("registered_devices", {})
        device_data = devices.get(self._device_config_id, {})
        
        sensor_channels = device_data.get("sensor_channels", 0)
        actuator_channels = device_data.get("actuator_channels", 0)
        
        return sensor_channels + actuator_channels