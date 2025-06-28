"""Test configuration file for Marduk's Robotics Lab tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components.marduk_robotics_lab.const import DOMAIN


@pytest.fixture
def mock_middleware():
    """Create a mock middleware for testing."""
    middleware = MagicMock()
    middleware.connected = True
    middleware.async_connect = AsyncMock(return_value=True)
    middleware.async_disconnect = AsyncMock()
    middleware.send_command = AsyncMock(return_value={"type": "pong", "status": "success"})
    middleware.register_event_listener = MagicMock()
    middleware.unregister_event_listener = MagicMock()
    middleware.get_hypergraph_nodes = MagicMock(return_value={})
    middleware.get_registered_devices = MagicMock(return_value={})
    middleware.get_active_agents = MagicMock(return_value={})
    middleware.register_device = AsyncMock(return_value=True)
    middleware.spawn_agent = AsyncMock(return_value=True)
    return middleware


@pytest.fixture
def mock_gguf_handler():
    """Create a mock GGUF handler for testing."""
    handler = MagicMock()
    handler.async_setup = AsyncMock()
    handler.async_shutdown = AsyncMock()
    handler.export_lab_state = AsyncMock(return_value="/tmp/test.gguf")
    handler.import_lab_state = AsyncMock(return_value={})
    handler.create_agent_snapshot = AsyncMock(return_value="/tmp/agent_snapshot.gguf")
    handler.restore_agent_from_snapshot = AsyncMock(return_value={})
    handler.get_gguf_path = MagicMock(return_value="/tmp/test.gguf")
    return handler


@pytest.fixture
def mock_coordinator_data():
    """Create mock coordinator data for testing."""
    return {
        "timestamp": 1234567890.0,
        "hypergraph_nodes": {
            "node_1": {
                "node_id": "node_1",
                "node_type": "device",
                "properties": {},
                "tensor_field_count": 2,
                "connection_count": 1,
            }
        },
        "registered_devices": {
            "device_1": {
                "device_id": "device_1",
                "device_type": "robotic_arm",
                "degrees_of_freedom": 6,
                "sensor_channels": 10,
                "actuator_channels": 6,
                "complexity": 22,
                "modalities": ["position", "force"],
            }
        },
        "active_agents": {
            "agent_1": {
                "kernel_id": "agent_1",
                "agent_type": "exploration",
                "state": "idle",
                "cognitive_grammar": "scheme",
                "memory_bank_count": 3,
                "function_count": 5,
                "learning_rate": 0.01,
                "exploration_factor": 0.1,
            }
        },
        "metrics": {
            "total_nodes": 1,
            "total_devices": 1,
            "total_agents": 1,
            "active_agents": 0,
            "error_agents": 0,
            "total_complexity": 22,
        },
        "system_status": "operational",
    }