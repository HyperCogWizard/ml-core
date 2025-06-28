"""Test Marduk's Robotics Lab integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

from homeassistant.components.marduk_robotics_lab.const import (
    DOMAIN,
    CONF_WORKBENCH_HOST,
    CONF_WORKBENCH_PORT,
    CONF_KERNEL_MODE,
    CONF_COGNITIVE_GRAMMAR,
    DEFAULT_WORKBENCH_PORT,
    DEFAULT_KERNEL_MODE,
    DEFAULT_COGNITIVE_GRAMMAR,
)


@pytest.fixture
def mock_config_entry() -> ConfigEntry:
    """Return a mock config entry."""
    return ConfigEntry(
        version=1,
        minor_version=1,
        domain=DOMAIN,
        title="Test Marduk Lab",
        data={
            CONF_HOST: "localhost",
            CONF_PORT: 8765,
            CONF_NAME: "Test Marduk Lab",
            CONF_KERNEL_MODE: DEFAULT_KERNEL_MODE,
            CONF_COGNITIVE_GRAMMAR: DEFAULT_COGNITIVE_GRAMMAR,
        },
        source="user",
        entry_id="test_entry_id",
        unique_id="localhost:8765",
    )


@pytest.fixture
def mock_middleware():
    """Return a mock middleware."""
    middleware = MagicMock()
    middleware.connected = True
    middleware.async_connect = AsyncMock(return_value=True)
    middleware.async_disconnect = AsyncMock()
    middleware.send_command = AsyncMock(return_value={"type": "pong"})
    middleware.register_event_listener = MagicMock()
    middleware.unregister_event_listener = MagicMock()
    middleware.get_hypergraph_nodes = MagicMock(return_value={})
    middleware.get_registered_devices = MagicMock(return_value={})
    middleware.get_active_agents = MagicMock(return_value={})
    return middleware


@pytest.fixture
def mock_coordinator():
    """Return a mock coordinator."""
    coordinator = MagicMock()
    coordinator.async_setup = AsyncMock()
    coordinator.async_validate_connection = AsyncMock()
    coordinator.async_shutdown = AsyncMock()
    coordinator.data = {
        "timestamp": 123456789.0,
        "hypergraph_nodes": {},
        "registered_devices": {},
        "active_agents": {},
        "metrics": {
            "total_nodes": 0,
            "total_devices": 0,
            "total_agents": 0,
            "active_agents": 0,
            "error_agents": 0,
            "total_complexity": 0,
        },
        "system_status": "operational",
    }
    return coordinator


async def test_async_setup_entry(
    hass: HomeAssistant,
    mock_config_entry: ConfigEntry,
    mock_middleware,
    mock_coordinator,
) -> None:
    """Test successful setup of config entry."""
    
    with patch(
        "homeassistant.components.marduk_robotics_lab.RoboticsMiddleware",
        return_value=mock_middleware,
    ), patch(
        "homeassistant.components.marduk_robotics_lab.MardukCoordinator",
        return_value=mock_coordinator,
    ):
        # Setup the integration
        mock_config_entry.add_to_hass(hass)
        assert await async_setup_component(hass, DOMAIN, {})
        await hass.async_block_till_done()
        
        # Verify the entry was setup
        assert mock_config_entry.state is ConfigEntryState.LOADED
        
        # Verify middleware and coordinator were called
        mock_coordinator.async_setup.assert_called_once()
        mock_coordinator.async_validate_connection.assert_called_once()
        
        # Verify domain data was stored
        assert DOMAIN in hass.data
        assert mock_config_entry.entry_id in hass.data[DOMAIN]


async def test_async_setup_entry_connection_failure(
    hass: HomeAssistant,
    mock_config_entry: ConfigEntry,
    mock_middleware,
    mock_coordinator,
) -> None:
    """Test setup failure when connection fails."""
    
    # Make connection fail
    mock_middleware.async_connect.return_value = False
    
    with patch(
        "homeassistant.components.marduk_robotics_lab.RoboticsMiddleware",
        return_value=mock_middleware,
    ), patch(
        "homeassistant.components.marduk_robotics_lab.MardukCoordinator",
        return_value=mock_coordinator,
    ):
        # Setup should fail
        mock_config_entry.add_to_hass(hass)
        assert not await async_setup_component(hass, DOMAIN, {})


async def test_async_unload_entry(
    hass: HomeAssistant,
    mock_config_entry: ConfigEntry,
    mock_middleware,
    mock_coordinator,
) -> None:
    """Test successful unload of config entry."""
    
    with patch(
        "homeassistant.components.marduk_robotics_lab.RoboticsMiddleware",
        return_value=mock_middleware,
    ), patch(
        "homeassistant.components.marduk_robotics_lab.MardukCoordinator",
        return_value=mock_coordinator,
    ):
        # Setup first
        mock_config_entry.add_to_hass(hass)
        assert await async_setup_component(hass, DOMAIN, {})
        await hass.async_block_till_done()
        
        # Now unload
        assert await hass.config_entries.async_unload(mock_config_entry.entry_id)
        
        # Verify cleanup was called
        mock_coordinator.async_shutdown.assert_called_once()


def test_domain_constant():
    """Test that domain constant is correctly defined."""
    assert DOMAIN == "marduk_robotics_lab"


def test_default_constants():
    """Test that default constants are reasonable."""
    assert DEFAULT_WORKBENCH_PORT == 8765
    assert DEFAULT_KERNEL_MODE == "agentic"
    assert DEFAULT_COGNITIVE_GRAMMAR == "scheme"


class TestMardukRoboticsData:
    """Test MardukRoboticsData model."""
    
    def test_initialization(self, mock_config_entry):
        """Test data model initialization."""
        from homeassistant.components.marduk_robotics_lab.models import MardukRoboticsData
        
        data = MardukRoboticsData(
            middleware=MagicMock(),
            coordinator=MagicMock(),
            config_entry=mock_config_entry
        )
        
        assert data.config_entry == mock_config_entry
        assert data.active_agents == {}
        assert data.device_configs == {}
        assert data.hypergraph_nodes == {}
        assert data.tensor_fields == {}
    
    def test_export_to_gguf(self, mock_config_entry):
        """Test GGUF export functionality."""
        from homeassistant.components.marduk_robotics_lab.models import (
            MardukRoboticsData,
            AgentKernel,
            DeviceConfiguration
        )
        
        data = MardukRoboticsData(
            middleware=MagicMock(),
            coordinator=MagicMock(),
            config_entry=mock_config_entry
        )
        
        # Add test agent
        agent = AgentKernel(
            kernel_id="test_agent",
            agent_type="exploration",
            state="idle"
        )
        data.add_agent(agent)
        
        # Add test device
        device = DeviceConfiguration(
            device_id="test_device",
            device_type="robotic_arm",
            degrees_of_freedom=6,
            sensor_channels=10,
            actuator_channels=6
        )
        data.add_device(device)
        
        # Export to GGUF
        gguf_data = data.export_to_gguf()
        
        assert "version" in gguf_data
        assert "lab_id" in gguf_data
        assert "agents" in gguf_data
        assert "devices" in gguf_data
        assert "test_agent" in gguf_data["agents"]
        assert "test_device" in gguf_data["devices"]


class TestTensorField:
    """Test TensorField model."""
    
    def test_tensor_field_creation(self):
        """Test tensor field creation."""
        from homeassistant.components.marduk_robotics_lab.models import TensorField
        import numpy as np
        
        field = TensorField(
            name="test_tensor",
            dimensions=(3, 4),
            dtype="float32",
            data=np.zeros((3, 4))
        )
        
        assert field.name == "test_tensor"
        assert field.dimensions == (3, 4)
        assert field.dtype == "float32"
        assert field.data.shape == (3, 4)
    
    def test_to_gguf_dict(self):
        """Test tensor field GGUF conversion."""
        from homeassistant.components.marduk_robotics_lab.models import TensorField
        import numpy as np
        
        field = TensorField(
            name="test_tensor",
            dimensions=(2, 3),
            dtype="float32",
            data=np.ones((2, 3))
        )
        
        gguf_dict = field.to_gguf_dict()
        
        assert gguf_dict["name"] == "test_tensor"
        assert gguf_dict["shape"] == (2, 3)
        assert gguf_dict["dtype"] == "float32"
        assert len(gguf_dict["data"]) == 6  # 2*3


@pytest.mark.asyncio
async def test_coordinator_setup():
    """Test coordinator setup process."""
    from homeassistant.components.marduk_robotics_lab.coordinator import MardukCoordinator
    
    hass = MagicMock()
    middleware = MagicMock()
    middleware.async_connect = AsyncMock(return_value=True)
    config_entry = MagicMock()
    
    coordinator = MardukCoordinator(hass, middleware, config_entry)
    
    with patch.object(coordinator, 'gguf_handler') as mock_gguf:
        mock_gguf.async_setup = AsyncMock()
        await coordinator.async_setup()
        
        mock_gguf.async_setup.assert_called_once()
        middleware.async_connect.assert_called_once()


def test_constants_are_strings():
    """Test that all string constants are actually strings."""
    from homeassistant.components.marduk_robotics_lab.const import (
        DOMAIN,
        NODE_TYPE_DEVICE,
        NODE_TYPE_SENSOR,
        NODE_TYPE_ACTUATOR,
        NODE_TYPE_AGENT,
        AGENT_STATE_IDLE,
        AGENT_STATE_ERROR,
    )
    
    assert isinstance(DOMAIN, str)
    assert isinstance(NODE_TYPE_DEVICE, str)
    assert isinstance(NODE_TYPE_SENSOR, str)
    assert isinstance(NODE_TYPE_ACTUATOR, str)
    assert isinstance(NODE_TYPE_AGENT, str)
    assert isinstance(AGENT_STATE_IDLE, str)
    assert isinstance(AGENT_STATE_ERROR, str)