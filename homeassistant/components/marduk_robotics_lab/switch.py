"""Switch platform for Marduk's Robotics Lab."""

from __future__ import annotations

import logging
from typing import Any, Dict

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import MardukCoordinator
from .entity import MardukCoordinatedEntity, MardukAgentEntity
from .models import MardukRoboticsData

_LOGGER = logging.getLogger(__name__)

# Lab system switches
LAB_SWITCHES = [
    SwitchEntityDescription(
        key="tensor_visualization",
        name="Tensor visualization",
        icon="mdi:chart-scatter-plot",
        entity_category=EntityCategory.CONFIG,
    ),
    SwitchEntityDescription(
        key="recursive_modification",
        name="Recursive modification",
        icon="mdi:recycle-variant",
        entity_category=EntityCategory.CONFIG,
    ),
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Marduk's Robotics Lab switch platform."""
    data: MardukRoboticsData = hass.data[DOMAIN][config_entry.entry_id]
    coordinator = data.coordinator
    
    entities: list[SwitchEntity] = []
    
    # Add lab system switches
    for description in LAB_SWITCHES:
        entities.append(
            MardukLabSwitch(coordinator, data, description)
        )
    
    # Add agent-specific learning switches
    for agent_id in data.active_agents:
        entities.append(
            MardukAgentLearningSwitch(coordinator, data, agent_id)
        )
    
    async_add_entities(entities)


class MardukLabSwitch(MardukCoordinatedEntity, SwitchEntity):
    """Switch for lab system controls."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        description: SwitchEntityDescription,
    ) -> None:
        """Initialize the lab switch."""
        super().__init__(
            coordinator,
            data,
            f"lab_{description.key}",
            description.name,
        )
        self.entity_description = description
        self._attr_translation_key = description.key
        self._is_on = False

    @property
    def is_on(self) -> bool:
        """Return the state of the switch."""
        return self._is_on

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on."""
        try:
            command = {
                "type": "enable_feature",
                "feature": self.entity_description.key,
                "enabled": True
            }
            response = await self.lab_data.middleware.send_command(command)
            
            if response.get("status") == "success":
                self._is_on = True
                self.async_write_ha_state()
                _LOGGER.info("Enabled %s", self.entity_description.key)
            else:
                raise HomeAssistantError(f"Failed to enable feature: {response.get('error')}")
                
        except Exception as err:
            _LOGGER.error("Failed to turn on %s: %s", self.entity_description.key, err)
            raise HomeAssistantError(f"Failed to turn on {self.entity_description.key}: {err}") from err

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off."""
        try:
            command = {
                "type": "enable_feature",
                "feature": self.entity_description.key,
                "enabled": False
            }
            response = await self.lab_data.middleware.send_command(command)
            
            if response.get("status") == "success":
                self._is_on = False
                self.async_write_ha_state()
                _LOGGER.info("Disabled %s", self.entity_description.key)
            else:
                raise HomeAssistantError(f"Failed to disable feature: {response.get('error')}")
                
        except Exception as err:
            _LOGGER.error("Failed to turn off %s: %s", self.entity_description.key, err)
            raise HomeAssistantError(f"Failed to turn off {self.entity_description.key}: {err}") from err


class MardukAgentLearningSwitch(MardukAgentEntity, SwitchEntity):
    """Switch for agent learning control."""

    def __init__(
        self,
        coordinator: MardukCoordinator,
        data: MardukRoboticsData,
        agent_id: str,
    ) -> None:
        """Initialize the agent learning switch."""
        super().__init__(
            coordinator,
            data,
            agent_id,
            "learning",
            f"Agent {agent_id} learning",
        )
        self._attr_icon = "mdi:brain"
        self._attr_entity_category = EntityCategory.CONFIG
        self._attr_translation_key = "agent_learning"

    @property
    def is_on(self) -> bool:
        """Return if agent learning is enabled."""
        if not self.coordinator.data:
            return False
        
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        
        # Consider learning enabled if learning rate > 0
        return agent_data.get("learning_rate", 0) > 0

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Enable agent learning."""
        try:
            command = {
                "type": "modify_agent",
                "agent_id": self._agent_id,
                "parameters": {
                    "learning_rate": 0.01,  # Default learning rate
                }
            }
            response = await self.lab_data.middleware.send_command(command)
            
            if response.get("status") == "success":
                await self.coordinator.async_request_refresh()
                _LOGGER.info("Enabled learning for agent %s", self._agent_id)
            else:
                raise HomeAssistantError(f"Failed to enable learning: {response.get('error')}")
                
        except Exception as err:
            _LOGGER.error("Failed to enable learning for agent %s: %s", self._agent_id, err)
            raise HomeAssistantError(f"Failed to enable learning: {err}") from err

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Disable agent learning."""
        try:
            command = {
                "type": "modify_agent",
                "agent_id": self._agent_id,
                "parameters": {
                    "learning_rate": 0.0,  # Disable learning
                }
            }
            response = await self.lab_data.middleware.send_command(command)
            
            if response.get("status") == "success":
                await self.coordinator.async_request_refresh()
                _LOGGER.info("Disabled learning for agent %s", self._agent_id)
            else:
                raise HomeAssistantError(f"Failed to disable learning: {response.get('error')}")
                
        except Exception as err:
            _LOGGER.error("Failed to disable learning for agent %s: %s", self._agent_id, err)
            raise HomeAssistantError(f"Failed to disable learning: {err}") from err

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return extra state attributes."""
        if not self.coordinator.data:
            return {}
        
        agents = self.coordinator.data.get("active_agents", {})
        agent_data = agents.get(self._agent_id, {})
        
        return {
            "learning_rate": agent_data.get("learning_rate", 0),
            "exploration_factor": agent_data.get("exploration_factor", 0),
            "agent_state": agent_data.get("state", "unknown"),
            "cognitive_grammar": agent_data.get("cognitive_grammar", "unknown"),
        }