"""Marduk's Robotics Lab - Engineering Workbench with GGUF Integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_NAME, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN, PLATFORMS
from .coordinator import MardukCoordinator
from .middleware import RoboticsMiddleware
from .models import MardukRoboticsData

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Set up Marduk's Robotics Lab from a config entry."""
    
    # Initialize robotics middleware abstraction
    middleware = RoboticsMiddleware(hass, config_entry)
    
    # Create coordinator for distributed cognition
    coordinator = MardukCoordinator(hass, middleware, config_entry)
    
    try:
        # Initialize GGUF serialization capabilities
        await coordinator.async_setup()
        
        # Test connection and verify agent kernels
        await coordinator.async_validate_connection()
        
    except Exception as err:
        _LOGGER.error("Failed to setup Marduk Robotics Lab: %s", err)
        raise ConfigEntryNotReady from err

    # Store domain data for access across platforms
    domain_data = MardukRoboticsData(
        middleware=middleware,
        coordinator=coordinator,
        config_entry=config_entry
    )
    hass.data.setdefault(DOMAIN, {})[config_entry.entry_id] = domain_data

    # Setup workbench platforms
    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)

    # Register shutdown cleanup
    @callback
    def _async_shutdown(event):
        """Handle Home Assistant shutdown."""
        asyncio.create_task(coordinator.async_shutdown())

    config_entry.async_on_unload(
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _async_shutdown)
    )

    return True


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(
        config_entry, PLATFORMS
    )
    
    if unload_ok:
        domain_data: MardukRoboticsData = hass.data[DOMAIN][config_entry.entry_id]
        await domain_data.coordinator.async_shutdown()
        hass.data[DOMAIN].pop(config_entry.entry_id)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, config_entry)
    await async_setup_entry(hass, config_entry)