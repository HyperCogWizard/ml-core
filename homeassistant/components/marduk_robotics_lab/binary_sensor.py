"""Binary sensor platform for Marduk's Robotics Lab."""

from __future__ import annotations

import logging
from typing import Any, Dict

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorEntityDescription,
    BinarySensorDeviceClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, AGENT_STATE_IDLE, AGENT_STATE_ERROR
from .coordinator import MardukCoordinator
from .entity import MardukCoordinatedEntity
from .models import MardukRoboticsData

_LOGGER = logging.getLogger(__name__)

# Lab system binary sensors
LAB_BINARY_SENSORS = [
    BinarySensorEntityDescription(
        key="middleware_connected",
        name="Middleware connected",
        device_class=BinarySensorDeviceClass.CONNECTIVITY,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    BinarySensorEntityDescription(
        key="gguf_enabled",
        name="GGUF enabled",
        icon="mdi:file-export",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    BinarySensorEntityDescription(
        key="agents_active",
        name="Agents active",
        icon="mdi:robot-excited",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Marduk's Robotics Lab binary sensor platform."""
    data: MardukRoboticsData = hass.data[DOMAIN][config_entry.entry_id]
    coordinator = data.coordinator
    
    entities: list[BinarySensorEntity] = []
    
    # Add lab system binary sensors
    for description in LAB_BINARY_SENSORS:
        entities.append(
            MardukLabBinarySensor(coordinator, data, description)
        )
    
    async_add_entities(entities)


class MardukLabBinarySensor(MardukCoordinatedEntity, BinarySensorEntity):
    """Binary sensor for lab status indicators."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        description: BinarySensorEntityDescription,
    ) -> None:
        """Initialize the lab binary sensor."""
        super().__init__(
            coordinator,
            data,
            f"lab_{description.key}",
            description.name,
        )
        self.entity_description = description
        self._attr_translation_key = description.key

    @property
    def is_on(self) -> bool:
        """Return the state of the binary sensor."""
        if self.entity_description.key == "middleware_connected":
            return self.lab_data.middleware.connected
        
        elif self.entity_description.key == "gguf_enabled":
            return self.coordinator.gguf_handler is not None
        
        elif self.entity_description.key == "agents_active":
            if not self.coordinator.data:
                return False
            metrics = self.coordinator.data.get("metrics", {})
            return metrics.get("active_agents", 0) > 0
        
        return False

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extra state attributes."""
        if not self.coordinator.data:
            return {}
        
        attributes = {
            "last_update": self.coordinator.data.get("timestamp"),
            "lab_id": self.lab_data.config_entry.entry_id,
        }
        
        if self.entity_description.key == "agents_active":
            metrics = self.coordinator.data.get("metrics", {})
            attributes.update({
                "total_agents": metrics.get("total_agents", 0),
                "active_agents": metrics.get("active_agents", 0),
                "error_agents": metrics.get("error_agents", 0),
            })
        
        return attributes