"""Button platform for Marduk's Robotics Lab."""

from __future__ import annotations

import logging
from typing import Any, Dict

from homeassistant.components.button import ButtonEntity, ButtonEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import MardukCoordinator
from .entity import MardukCoordinatedEntity
from .models import MardukRoboticsData

_LOGGER = logging.getLogger(__name__)

# Lab control buttons
LAB_BUTTONS = [
    ButtonEntityDescription(
        key="spawn_agent",
        name="Spawn agent",
        icon="mdi:robot-excited",
        entity_category=EntityCategory.CONFIG,
    ),
    ButtonEntityDescription(
        key="export_gguf",
        name="Export GGUF",
        icon="mdi:export",
        entity_category=EntityCategory.CONFIG,
    ),
    ButtonEntityDescription(
        key="refresh_hypergraph",
        name="Refresh hypergraph",
        icon="mdi:graph",
        entity_category=EntityCategory.CONFIG,
    ),
    ButtonEntityDescription(
        key="emergency_stop",
        name="Emergency stop",
        icon="mdi:stop-circle",
        entity_category=EntityCategory.CONFIG,
    ),
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Marduk's Robotics Lab button platform."""
    data: MardukRoboticsData = hass.data[DOMAIN][config_entry.entry_id]
    coordinator = data.coordinator
    
    entities: list[ButtonEntity] = []
    
    # Add lab control buttons
    for description in LAB_BUTTONS:
        entities.append(
            MardukLabButton(coordinator, data, description)
        )
    
    async_add_entities(entities)


class MardukLabButton(MardukCoordinatedEntity, ButtonEntity):
    """Button for lab control actions."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        description: ButtonEntityDescription,
    ) -> None:
        """Initialize the lab button."""
        super().__init__(
            coordinator,
            data,
            f"lab_{description.key}",
            description.name,
        )
        self.entity_description = description
        self._attr_translation_key = description.key

    async def async_press(self) -> None:
        """Press the button."""
        if self.entity_description.key == "spawn_agent":
            await self._spawn_default_agent()
        elif self.entity_description.key == "export_gguf":
            await self._export_gguf()
        elif self.entity_description.key == "refresh_hypergraph":
            await self._refresh_hypergraph()
        elif self.entity_description.key == "emergency_stop":
            await self._emergency_stop()

    async def _spawn_default_agent(self) -> None:
        """Spawn a default exploration agent."""
        try:
            agent_id = await self.coordinator.async_spawn_agent("exploration")
            _LOGGER.info("Spawned new exploration agent: %s", agent_id)
        except Exception as err:
            _LOGGER.error("Failed to spawn agent: %s", err)
            raise HomeAssistantError(f"Failed to spawn agent: {err}") from err

    async def _export_gguf(self) -> None:
        """Export current lab state to GGUF."""
        try:
            if not self.coordinator.data:
                raise HomeAssistantError("No lab data available for export")
            
            gguf_path = await self.coordinator.gguf_handler.export_lab_state(
                self.coordinator.data
            )
            _LOGGER.info("Exported lab state to GGUF: %s", gguf_path)
        except Exception as err:
            _LOGGER.error("Failed to export GGUF: %s", err)
            raise HomeAssistantError(f"Failed to export GGUF: {err}") from err

    async def _refresh_hypergraph(self) -> None:
        """Refresh hypergraph data."""
        try:
            # Send refresh command to middleware
            command = {"type": "refresh_hypergraph"}
            await self.lab_data.middleware.send_command(command)
            
            # Trigger coordinator update
            await self.coordinator.async_request_refresh()
            
            _LOGGER.info("Refreshed hypergraph data")
        except Exception as err:
            _LOGGER.error("Failed to refresh hypergraph: %s", err)
            raise HomeAssistantError(f"Failed to refresh hypergraph: {err}") from err

    async def _emergency_stop(self) -> None:
        """Emergency stop all agents and devices."""
        try:
            # Send emergency stop command
            command = {"type": "emergency_stop"}
            await self.lab_data.middleware.send_command(command)
            
            _LOGGER.warning("Emergency stop activated for all lab components")
        except Exception as err:
            _LOGGER.error("Failed to execute emergency stop: %s", err)
            raise HomeAssistantError(f"Failed to execute emergency stop: {err}") from err