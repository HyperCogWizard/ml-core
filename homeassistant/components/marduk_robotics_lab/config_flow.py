"""Config flow for Marduk's Robotics Lab integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    CONF_WORKBENCH_HOST,
    CONF_WORKBENCH_PORT,
    CONF_AGENT_ID,
    CONF_GGUF_PATH,
    CONF_KERNEL_MODE,
    CONF_COGNITIVE_GRAMMAR,
    DEFAULT_WORKBENCH_PORT,
    DEFAULT_KERNEL_MODE,
    DEFAULT_COGNITIVE_GRAMMAR,
)

_LOGGER = logging.getLogger(__name__)


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Marduk's Robotics Lab."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._discovered_info: Optional[Dict[str, Any]] = None

    async def async_step_user(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            try:
                # Validate the connection
                info = await validate_input(self.hass, user_input)
                
                # Create unique ID based on host and port
                unique_id = f"{user_input[CONF_HOST]}:{user_input[CONF_PORT]}"
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()

                return self.async_create_entry(title=info["title"], data=user_input)
                
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        data_schema = vol.Schema(
            {
                vol.Required(CONF_HOST, default="localhost"): cv.string,
                vol.Required(CONF_PORT, default=DEFAULT_WORKBENCH_PORT): cv.port,
                vol.Optional(CONF_NAME, default="Marduk's Robotics Lab"): cv.string,
                vol.Optional(CONF_GGUF_PATH): cv.string,
                vol.Optional(CONF_KERNEL_MODE, default=DEFAULT_KERNEL_MODE): vol.In([
                    "agentic", "reactive", "hybrid"
                ]),
                vol.Optional(CONF_COGNITIVE_GRAMMAR, default=DEFAULT_COGNITIVE_GRAMMAR): vol.In([
                    "scheme", "prolog", "javascript", "python"
                ]),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_zeroconf(
        self, discovery_info: Dict[str, Any]
    ) -> FlowResult:
        """Handle zeroconf discovery."""
        self._host = discovery_info.get("host")
        self._port = discovery_info.get("port", DEFAULT_WORKBENCH_PORT)
        
        # Set unique ID based on discovered info
        unique_id = f"{self._host}:{self._port}"
        await self.async_set_unique_id(unique_id)
        self._abort_if_unique_id_configured()
        
        self._discovered_info = discovery_info
        
        # Try to connect to validate discovery
        try:
            test_input = {
                CONF_HOST: self._host,
                CONF_PORT: self._port,
                CONF_NAME: f"Marduk Lab ({self._host})",
            }
            await validate_input(self.hass, test_input)
            
            return await self.async_step_discovery_confirm()
            
        except Exception:
            # Discovery validation failed, abort
            return self.async_abort(reason="cannot_connect")

    async def async_step_discovery_confirm(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Confirm discovery."""
        if user_input is not None:
            # Complete the setup with discovered info
            data = {
                CONF_HOST: self._host,
                CONF_PORT: self._port,
                CONF_NAME: user_input.get(CONF_NAME, f"Marduk Lab ({self._host})"),
                CONF_KERNEL_MODE: user_input.get(CONF_KERNEL_MODE, DEFAULT_KERNEL_MODE),
                CONF_COGNITIVE_GRAMMAR: user_input.get(CONF_COGNITIVE_GRAMMAR, DEFAULT_COGNITIVE_GRAMMAR),
            }
            
            if CONF_GGUF_PATH in user_input:
                data[CONF_GGUF_PATH] = user_input[CONF_GGUF_PATH]
            
            return self.async_create_entry(title=data[CONF_NAME], data=data)

        data_schema = vol.Schema(
            {
                vol.Required(CONF_NAME, default=f"Marduk Lab ({self._host})"): cv.string,
                vol.Optional(CONF_GGUF_PATH): cv.string,
                vol.Optional(CONF_KERNEL_MODE, default=DEFAULT_KERNEL_MODE): vol.In([
                    "agentic", "reactive", "hybrid"
                ]),
                vol.Optional(CONF_COGNITIVE_GRAMMAR, default=DEFAULT_COGNITIVE_GRAMMAR): vol.In([
                    "scheme", "prolog", "javascript", "python"
                ]),
            }
        )

        return self.async_show_form(
            step_id="discovery_confirm",
            data_schema=data_schema,
            description_placeholders={
                "host": self._host,
                "port": str(self._port),
            },
        )

    async def async_step_dhcp(
        self, discovery_info: Dict[str, Any]
    ) -> FlowResult:
        """Handle DHCP discovery."""
        return await self.async_step_zeroconf(discovery_info)

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> OptionsFlowHandler:
        """Create the options flow."""
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for Marduk's Robotics Lab."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        data_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_GGUF_PATH,
                    default=self.config_entry.options.get(
                        CONF_GGUF_PATH, self.config_entry.data.get(CONF_GGUF_PATH, "")
                    ),
                ): cv.string,
                vol.Optional(
                    CONF_KERNEL_MODE,
                    default=self.config_entry.options.get(
                        CONF_KERNEL_MODE, self.config_entry.data.get(CONF_KERNEL_MODE, DEFAULT_KERNEL_MODE)
                    ),
                ): vol.In(["agentic", "reactive", "hybrid"]),
                vol.Optional(
                    CONF_COGNITIVE_GRAMMAR,
                    default=self.config_entry.options.get(
                        CONF_COGNITIVE_GRAMMAR, self.config_entry.data.get(CONF_COGNITIVE_GRAMMAR, DEFAULT_COGNITIVE_GRAMMAR)
                    ),
                ): vol.In(["scheme", "prolog", "javascript", "python"]),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
        )


async def validate_input(hass: HomeAssistant, data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the user input allows us to connect."""
    
    host = data[CONF_HOST]
    port = data[CONF_PORT]
    
    # Create a simple test connection
    try:
        # Test basic connectivity using asyncio
        future = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(future, timeout=5.0)
        
        # Close the test connection
        writer.close()
        await writer.wait_closed()
        
    except asyncio.TimeoutError as err:
        _LOGGER.error("Timeout connecting to %s:%s", host, port)
        raise CannotConnect from err
    except Exception as err:
        _LOGGER.error("Error connecting to %s:%s: %s", host, port, err)
        raise CannotConnect from err

    # Return info that you want to store in the config entry
    return {
        "title": data.get(CONF_NAME, f"Marduk Lab ({host}:{port})"),
        "host": host,
        "port": port,
    }


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class InvalidAuth(HomeAssistantError):
    """Error to indicate there is invalid auth."""