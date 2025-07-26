"""Test comprehensive GGUF support for Marduk's Robotics Lab."""

from __future__ import annotations

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

from homeassistant.components.marduk_robotics_lab.models import (
    AgentKernel,
    DeviceConfiguration,
    TensorField,
    HypergraphNode,
    MardukRoboticsData,
)
from homeassistant.components.marduk_robotics_lab.gguf_handler import GGUFHandler
from homeassistant.components.marduk_robotics_lab.const import DOMAIN


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry for testing."""
    return MagicMock(spec=ConfigEntry)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock(spec=HomeAssistant)
    hass.async_add_executor_job = AsyncMock(side_effect=lambda func, *args: func(*args))
    return hass


@pytest.fixture
def temp_gguf_path():
    """Create a temporary path for GGUF files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield str(Path(temp_dir) / "test.gguf")


@pytest.fixture
def sample_agent_kernel():
    """Create a sample agent kernel with comprehensive data."""
    agent = AgentKernel(
        kernel_id="test_agent_1",
        agent_type="exploration",
        state="executing",
        cognitive_grammar="scheme",
        learning_rate=0.05,
        exploration_factor=0.15
    )
    
    # Add comprehensive state
    agent.state_history = ["idle", "exploring", "executing"]
    agent.cognitive_parameters = {"attention_span": 0.8, "curiosity": 0.9}
    agent.execution_context = {"current_task": "navigation", "priority": "high"}
    agent.communication_channels = ["sensor_input", "motor_output"]
    agent.functions = ["navigate", "explore", "learn"]
    
    # Add P-System membrane structure
    agent.set_membrane_hierarchy("agent_membrane_1", "lab_root")
    agent.add_child_membrane("sub_agent_1")
    
    # Add memory banks
    memory_tensor = TensorField(
        name="spatial_memory",
        dimensions=(10, 10),
        dtype="float32",
        data=np.random.rand(10, 10)
    )
    memory_tensor.set_tensor_identity("memory_1", "test_agent_1", "agent_memory")
    agent.memory_banks.append(memory_tensor)
    
    return agent


@pytest.fixture
def sample_device_config():
    """Create a sample device configuration with comprehensive data."""
    device = DeviceConfiguration(
        device_id="test_device_1",
        device_type="robotic_arm",
        degrees_of_freedom=6,
        sensor_channels=12,
        actuator_channels=6,
        tensor_dimensions=(6, 12),
        modalities=["position", "force", "torque"]
    )
    
    # Add comprehensive configuration
    device.device_parameters = {"max_velocity": 1.5, "max_acceleration": 2.0}
    device.firmware_version = "2.1.3"
    device.device_status = "operational"
    device.communication_protocols = ["CAN", "Ethernet"]
    
    # Add sensor specifications
    device.add_sensor_specification({
        "sensor_type": "position_encoder",
        "resolution": 0.001,
        "range": 360.0
    })
    device.add_sensor_specification({
        "sensor_type": "force_sensor",
        "max_force": 100.0,
        "sensitivity": 0.01
    })
    
    # Add actuator specifications
    device.add_actuator_specification({
        "actuator_type": "servo_motor",
        "max_torque": 50.0,
        "speed": 3000
    })
    
    # Add P-System membrane configuration
    device.set_membrane_configuration(
        "device_membrane_1",
        ["sensor_data", "commands"],
        ["status", "telemetry"]
    )
    
    return device


@pytest.fixture
def sample_hypergraph_node():
    """Create a sample hypergraph node with tensor fields."""
    node = HypergraphNode(
        node_id="test_node_1",
        node_type="sensor",
        properties={"location": "workspace_1", "active": True}
    )
    
    # Add comprehensive state
    node.node_state = {"operational": True, "last_update": 1234567890}
    node.processing_load = 0.65
    node.communication_latency = {"node_2": 0.05, "node_3": 0.12}
    node.membrane_bindings = ["device_membrane_1"]
    
    # Add tensor fields
    sensor_tensor = TensorField(
        name="sensor_readings",
        dimensions=(5, 8),
        dtype="float32",
        data=np.random.rand(5, 8)
    )
    node.add_tensor_field(sensor_tensor)
    
    return node


@pytest.fixture
def comprehensive_lab_data(sample_agent_kernel, sample_device_config, sample_hypergraph_node):
    """Create comprehensive lab data for testing."""
    return {
        "timestamp": 1234567890.0,
        "hypergraph_nodes": {
            "test_node_1": {
                "node_id": "test_node_1",
                "node_type": "sensor",
                "properties": {"location": "workspace_1"},
                "tensor_fields": [sample_hypergraph_node.tensor_fields[0].to_gguf_dict()],
                "connections": ["node_2", "node_3"],
                "node_state": {"operational": True},
                "processing_load": 0.65,
                "communication_latency": {"node_2": 0.05}
            }
        },
        "registered_devices": {
            "test_device_1": sample_device_config.to_gguf_dict()
        },
        "active_agents": {
            "test_agent_1": sample_agent_kernel.to_gguf_dict()
        },
        "metrics": {
            "total_nodes": 1,
            "total_devices": 1,
            "total_agents": 1,
            "active_agents": 1,
            "error_agents": 0,
            "total_complexity": 24
        },
        "system_status": "operational"
    }


class TestComprehensiveGGUFSupport:
    """Test comprehensive GGUF serialization support."""
    
    @pytest.mark.asyncio
    async def test_enhanced_agent_kernel_serialization(self, sample_agent_kernel):
        """Test enhanced agent kernel GGUF serialization."""
        gguf_dict = sample_agent_kernel.to_gguf_dict()
        
        # Test comprehensive fields are included
        assert "state_history" in gguf_dict
        assert "cognitive_parameters" in gguf_dict
        assert "execution_context" in gguf_dict
        assert "communication_channels" in gguf_dict
        assert "membrane_structure" in gguf_dict
        
        # Test membrane structure
        membrane_struct = gguf_dict["membrane_structure"]
        assert membrane_struct["membrane_id"] == "agent_membrane_1"
        assert membrane_struct["parent_membrane"] == "lab_root"
        assert "sub_agent_1" in membrane_struct["child_membranes"]
        
        # Test memory banks
        assert len(gguf_dict["memory_banks"]) == 1
        memory_bank = gguf_dict["memory_banks"][0]
        assert memory_bank["name"] == "spatial_memory"
        assert memory_bank["tensor_type"] == "agent_memory"
    
    @pytest.mark.asyncio
    async def test_enhanced_device_configuration_serialization(self, sample_device_config):
        """Test enhanced device configuration GGUF serialization."""
        gguf_dict = sample_device_config.to_gguf_dict()
        
        # Test comprehensive fields are included
        assert "device_parameters" in gguf_dict
        assert "sensor_specifications" in gguf_dict
        assert "actuator_specifications" in gguf_dict
        assert "firmware_version" in gguf_dict
        assert "device_status" in gguf_dict
        assert "communication_protocols" in gguf_dict
        assert "membrane_structure" in gguf_dict
        
        # Test specifications
        assert len(gguf_dict["sensor_specifications"]) == 2
        assert len(gguf_dict["actuator_specifications"]) == 1
        
        # Test membrane structure
        membrane_struct = gguf_dict["membrane_structure"]
        assert membrane_struct["membrane_id"] == "device_membrane_1"
        assert "sensor_data" in membrane_struct["input_channels"]
        assert "status" in membrane_struct["output_channels"]
    
    @pytest.mark.asyncio
    async def test_comprehensive_gguf_export(
        self, 
        mock_hass, 
        mock_config_entry, 
        temp_gguf_path, 
        comprehensive_lab_data
    ):
        """Test comprehensive GGUF export with all features."""
        mock_config_entry.entry_id = "test_lab_id"
        mock_config_entry.data = {"gguf_path": temp_gguf_path}
        
        handler = GGUFHandler(mock_hass, mock_config_entry)
        await handler.async_setup()
        
        # Export comprehensive lab state
        gguf_path = await handler.export_lab_state(comprehensive_lab_data)
        
        assert Path(gguf_path).exists()
        
        # Verify export was called
        mock_hass.async_add_executor_job.assert_called()
    
    @pytest.mark.asyncio
    async def test_comprehensive_gguf_import(
        self, 
        mock_hass, 
        mock_config_entry, 
        temp_gguf_path, 
        comprehensive_lab_data
    ):
        """Test comprehensive GGUF import and data reconstruction."""
        mock_config_entry.entry_id = "test_lab_id"
        mock_config_entry.data = {"gguf_path": temp_gguf_path}
        
        handler = GGUFHandler(mock_hass, mock_config_entry)
        await handler.async_setup()
        
        # Export first
        gguf_path = await handler.export_lab_state(comprehensive_lab_data)
        
        # Then import
        imported_data = await handler.import_lab_state(gguf_path)
        
        # Verify imported data structure
        assert "active_agents" in imported_data
        assert "registered_devices" in imported_data
        assert "hypergraph_nodes" in imported_data
        assert "membrane_system" in imported_data
        
        # Verify agent data reconstruction
        if "test_agent_1" in imported_data["active_agents"]:
            agent_data = imported_data["active_agents"]["test_agent_1"]
            assert "memory_banks" in agent_data
            assert "state_history" in agent_data
            assert "cognitive_parameters" in agent_data
    
    @pytest.mark.asyncio
    async def test_p_system_membrane_export(
        self, 
        mock_hass, 
        mock_config_entry, 
        temp_gguf_path
    ):
        """Test P-System membrane hierarchy export."""
        mock_config_entry.entry_id = "test_lab_id"
        mock_config_entry.data = {"gguf_path": temp_gguf_path}
        
        handler = GGUFHandler(mock_hass, mock_config_entry)
        await handler.async_setup()
        
        # Create sample membrane hierarchy
        membrane_hierarchy = {
            "agent_membrane_1": {
                "membrane_type": "agent_membrane",
                "entity_id": "test_agent_1",
                "entity_type": "agent",
                "communication_channels": ["sensor_input", "motor_output"]
            },
            "device_membrane_1": {
                "membrane_type": "device_membrane",
                "entity_id": "test_device_1",
                "entity_type": "device",
                "input_channels": ["commands"],
                "output_channels": ["status"]
            },
            "lab_root": {
                "membrane_type": "lab_membrane",
                "entity_id": "test_lab_id",
                "entity_type": "laboratory",
                "child_membranes": ["agent_membrane_1", "device_membrane_1"]
            }
        }
        
        # Export P-System membranes
        p_system_path = await handler.export_p_system_membranes(membrane_hierarchy)
        
        assert Path(p_system_path).exists()
        assert "p_system_membranes.gguf" in p_system_path
    
    @pytest.mark.asyncio
    async def test_comprehensive_snapshot_creation(
        self, 
        mock_hass, 
        mock_config_entry, 
        temp_gguf_path, 
        comprehensive_lab_data
    ):
        """Test comprehensive snapshot creation."""
        mock_config_entry.entry_id = "test_lab_id"
        mock_config_entry.data = {"gguf_path": temp_gguf_path}
        
        handler = GGUFHandler(mock_hass, mock_config_entry)
        await handler.async_setup()
        
        # Create comprehensive snapshot
        snapshot_path = await handler.create_comprehensive_snapshot(comprehensive_lab_data)
        
        assert Path(snapshot_path).exists()
        assert "comprehensive_snapshot_" in snapshot_path
    
    @pytest.mark.asyncio
    async def test_gguf_integrity_validation(
        self, 
        mock_hass, 
        mock_config_entry, 
        temp_gguf_path, 
        comprehensive_lab_data
    ):
        """Test GGUF file integrity validation."""
        mock_config_entry.entry_id = "test_lab_id"
        mock_config_entry.data = {"gguf_path": temp_gguf_path}
        
        handler = GGUFHandler(mock_hass, mock_config_entry)
        await handler.async_setup()
        
        # Export and validate
        gguf_path = await handler.export_lab_state(comprehensive_lab_data)
        is_valid = await handler.validate_gguf_integrity(gguf_path)
        
        assert is_valid
    
    def test_tensor_field_comprehensive_features(self):
        """Test enhanced tensor field features."""
        tensor = TensorField(
            name="test_tensor",
            dimensions=(4, 5),
            dtype="float32",
            data=np.ones((4, 5))
        )
        
        # Test tensor identity setting
        tensor.set_tensor_identity("tensor_123", "parent_entity", "test_type")
        
        assert tensor.tensor_id == "tensor_123"
        assert tensor.parent_entity == "parent_entity"
        assert tensor.tensor_type == "test_type"
        
        # Test GGUF dict includes new fields
        gguf_dict = tensor.to_gguf_dict()
        assert "tensor_id" in gguf_dict
        assert "parent_entity" in gguf_dict
        assert "tensor_type" in gguf_dict
        assert "update_frequency" in gguf_dict
    
    def test_hypergraph_node_comprehensive_features(self, sample_hypergraph_node):
        """Test enhanced hypergraph node features."""
        gguf_dict = sample_hypergraph_node.to_gguf_dict()
        
        # Test comprehensive fields
        assert "node_state" in gguf_dict
        assert "processing_load" in gguf_dict
        assert "communication_latency" in gguf_dict
        assert "membrane_bindings" in gguf_dict
        
        # Test tensor field identity is set correctly
        tensor_field = sample_hypergraph_node.tensor_fields[0]
        assert tensor_field.tensor_id == "test_node_1_sensor_readings"
        assert tensor_field.parent_entity == "test_node_1"
        assert tensor_field.tensor_type == "hypergraph_sensor"
    
    def test_marduk_robotics_data_comprehensive_export(
        self, 
        mock_config_entry, 
        sample_agent_kernel, 
        sample_device_config, 
        sample_hypergraph_node
    ):
        """Test comprehensive export from MardukRoboticsData."""
        data = MardukRoboticsData(
            middleware=MagicMock(),
            coordinator=MagicMock(),
            config_entry=mock_config_entry
        )
        
        # Add entities
        data.add_agent(sample_agent_kernel)
        data.add_device(sample_device_config)
        data.hypergraph_nodes["test_node_1"] = sample_hypergraph_node
        
        # Export to GGUF
        gguf_data = data.export_to_gguf()
        
        # Test comprehensive structure
        assert gguf_data["version"] == "2.0.0"
        assert gguf_data["p_system_compatible"] is True
        assert "membrane_system" in gguf_data
        assert "metadata" in gguf_data
        
        # Test membrane hierarchy generation
        membrane_system = gguf_data["membrane_system"]
        assert "lab_root" in membrane_system
        assert membrane_system["lab_root"]["membrane_type"] == "lab_membrane"


class TestGGUFMembraneIntegration:
    """Test P-System membrane integration in GGUF."""
    
    def test_agent_membrane_hierarchy(self, sample_agent_kernel):
        """Test agent membrane hierarchy creation."""
        sample_agent_kernel.set_membrane_hierarchy("agent_mem_1", "lab_root")
        sample_agent_kernel.add_child_membrane("sub_mem_1")
        sample_agent_kernel.add_child_membrane("sub_mem_2")
        
        gguf_dict = sample_agent_kernel.to_gguf_dict()
        membrane_struct = gguf_dict["membrane_structure"]
        
        assert membrane_struct["membrane_id"] == "agent_mem_1"
        assert membrane_struct["parent_membrane"] == "lab_root"
        assert len(membrane_struct["child_membranes"]) == 2
        assert "sub_mem_1" in membrane_struct["child_membranes"]
        assert "sub_mem_2" in membrane_struct["child_membranes"]
    
    def test_device_membrane_configuration(self, sample_device_config):
        """Test device membrane configuration."""
        sample_device_config.set_membrane_configuration(
            "device_mem_1",
            ["input_1", "input_2"],
            ["output_1", "output_2", "output_3"]
        )
        
        gguf_dict = sample_device_config.to_gguf_dict()
        membrane_struct = gguf_dict["membrane_structure"]
        
        assert membrane_struct["membrane_id"] == "device_mem_1"
        assert len(membrane_struct["input_channels"]) == 2
        assert len(membrane_struct["output_channels"]) == 3
        assert membrane_struct["membrane_type"] == "device_membrane"
    
    @pytest.mark.asyncio
    async def test_membrane_tensor_generation(self, mock_hass, mock_config_entry, temp_gguf_path):
        """Test membrane communication tensor generation."""
        mock_config_entry.data = {"gguf_path": temp_gguf_path}
        
        handler = GGUFHandler(mock_hass, mock_config_entry)
        
        # Create membrane hierarchy with communication channels
        membrane_hierarchy = {
            "agent_membrane_1": {
                "membrane_type": "agent_membrane",
                "entity_id": "agent_1",
                "input_channels": ["sensor_data", "commands"],
                "output_channels": ["status", "results"],
                "communication_channels": ["sensor_data", "commands", "status", "results"]
            }
        }
        
        # Generate membrane tensors
        membrane_tensors = handler._generate_membrane_tensors(membrane_hierarchy)
        
        # Verify communication matrix tensor
        comm_tensor_key = "membrane_agent_membrane_1_comm_matrix"
        assert comm_tensor_key in membrane_tensors
        
        comm_tensor = membrane_tensors[comm_tensor_key]
        assert comm_tensor["tensor_type"] == "membrane_communication"
        assert "input_channels" in comm_tensor["metadata"]
        assert "output_channels" in comm_tensor["metadata"]
        
        # Verify state tensor
        state_tensor_key = "membrane_agent_membrane_1_state"
        assert state_tensor_key in membrane_tensors
        
        state_tensor = membrane_tensors[state_tensor_key]
        assert state_tensor["tensor_type"] == "membrane_state"
        assert state_tensor["membrane_id"] == "agent_membrane_1"


@pytest.mark.asyncio
async def test_round_trip_data_integrity(
    mock_hass, 
    mock_config_entry, 
    temp_gguf_path, 
    comprehensive_lab_data
):
    """Test round-trip data integrity (export then import)."""
    mock_config_entry.entry_id = "integrity_test_lab"
    mock_config_entry.data = {"gguf_path": temp_gguf_path}
    
    handler = GGUFHandler(mock_hass, mock_config_entry)
    await handler.async_setup()
    
    # Export comprehensive data
    gguf_path = await handler.export_lab_state(comprehensive_lab_data)
    
    # Import the data back
    imported_data = await handler.import_lab_state(gguf_path)
    
    # Verify key data integrity
    assert imported_data["system_status"] == "imported_from_gguf"
    assert "active_agents" in imported_data
    assert "registered_devices" in imported_data
    assert "hypergraph_nodes" in imported_data
    
    # Verify metrics are preserved
    if "metrics" in imported_data:
        original_metrics = comprehensive_lab_data.get("metrics", {})
        imported_metrics = imported_data["metrics"]
        assert imported_metrics.get("total_agents") == original_metrics.get("total_agents")
        assert imported_metrics.get("total_devices") == original_metrics.get("total_devices")