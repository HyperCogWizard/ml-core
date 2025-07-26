"""Test composable workbench modules."""

from __future__ import annotations

import pytest
import numpy as np

from homeassistant.components.marduk_robotics_lab.workbench import (
    WorkbenchRegistry,
    HypergraphManager,
    TensorManager,
    DeviceManager,
)
from homeassistant.components.marduk_robotics_lab.workbench.tensors import (
    TensorFieldSpec,
    TensorModalityType,
    TensorDataType,
    StandardTensorSpecs,
)
from homeassistant.components.marduk_robotics_lab.workbench.devices import (
    DeviceConfig,
    DeviceType,
    ConnectionType,
    StandardDeviceConfigs,
)
from homeassistant.components.marduk_robotics_lab.workbench.hypergraph import (
    WorkbenchNode,
    WorkbenchEdge,
    EdgeType,
)


def test_tensor_field_spec_creation():
    """Test tensor field specification creation."""
    spec = TensorFieldSpec(
        name="test_tensor",
        dimensions=(3, 3),
        dtype=TensorDataType.FLOAT32,
        modality=TensorModalityType.POSITION,
        unit="m",
        description="Test tensor field",
    )
    
    assert spec.name == "test_tensor"
    assert spec.dimensions == (3, 3)
    assert spec.get_total_elements() == 9
    assert spec.get_memory_size_bytes() == 36  # 9 * 4 bytes


def test_standard_tensor_specs():
    """Test standard tensor specifications."""
    specs = StandardTensorSpecs.get_all_specs()
    assert len(specs) >= 5  # Should have multiple standard specs
    
    # Check specific specs exist
    spec_names = [spec.name for spec in specs]
    assert "pose_6dof" in spec_names
    assert "joint_positions" in spec_names


def test_tensor_manager():
    """Test tensor manager functionality."""
    manager = TensorManager()
    
    # Should have standard specs loaded
    stats = manager.get_tensor_stats()
    assert stats["modality_counts"]
    
    # Test creating a tensor field
    field = manager.create_tensor_field("pose_6dof", data=[1, 2, 3, 4, 5, 6])
    assert field is not None
    assert field.spec.name == "pose_6dof"
    assert field.is_valid()


def test_device_config_creation():
    """Test device configuration creation."""
    config = DeviceConfig(
        device_id="test_robot",
        device_type=DeviceType.MANIPULATOR,
        name="Test Robot",
        degrees_of_freedom=6,
        connection_type=ConnectionType.ETHERNET,
    )
    
    # Add tensor specs
    position_spec = TensorFieldSpec(
        name="joint_positions",
        dimensions=(6,),
        dtype=TensorDataType.FLOAT32,
        modality=TensorModalityType.MOTOR_POSITION,
    )
    config.add_sensor_spec(position_spec)
    
    assert config.device_id == "test_robot"
    assert config.get_sensor_channel_count() == 6
    assert len(config.validate_configuration()) == 0  # Should be valid


def test_standard_device_configs():
    """Test standard device configurations."""
    robot_arm = StandardDeviceConfigs.create_robot_arm_7dof()
    mobile_robot = StandardDeviceConfigs.create_mobile_robot()
    
    assert robot_arm.device_type == DeviceType.MANIPULATOR
    assert robot_arm.degrees_of_freedom == 7
    assert len(robot_arm.sensor_specs) > 0
    assert len(robot_arm.actuator_specs) > 0
    
    assert mobile_robot.device_type == DeviceType.MOBILE_ROBOT
    assert mobile_robot.degrees_of_freedom == 3


def test_hypergraph_manager():
    """Test hypergraph manager functionality."""
    manager = HypergraphManager()
    
    # Create and add a node
    node = WorkbenchNode(
        node_id="test_node",
        node_type="device",
        component_class="TestDevice",
    )
    
    assert manager.add_node(node)
    assert manager.get_node("test_node") is not None
    
    # Create and add another node
    node2 = WorkbenchNode(
        node_id="test_node2",
        node_type="sensor",
        component_class="TestSensor",
    )
    manager.add_node(node2)
    
    # Create an edge between nodes
    edge = WorkbenchEdge(
        edge_id="test_edge",
        source_node_id="test_node",
        target_node_id="test_node2",
        edge_type=EdgeType.DATA_FLOW,
    )
    
    assert manager.add_edge(edge)
    assert manager.get_edge("test_edge") is not None
    
    # Test path finding
    path = manager.find_path("test_node", "test_node2")
    assert path == ["test_node", "test_node2"]


def test_device_manager():
    """Test device manager functionality."""
    hypergraph_manager = HypergraphManager()
    device_manager = DeviceManager(hypergraph_manager)
    
    # Register a device config
    config = StandardDeviceConfigs.create_robot_arm_7dof()
    assert device_manager.register_device_config(config)
    
    # Check it was added to hypergraph
    node = hypergraph_manager.get_node(config.device_id)
    assert node is not None
    assert node.node_type == "device"
    
    # Get device stats
    stats = device_manager.get_device_stats()
    assert stats["total_devices"] == 1
    assert stats["total_degrees_of_freedom"] == 7


def test_workbench_registry_integration():
    """Test complete workbench registry integration."""
    registry = WorkbenchRegistry()
    
    # Register a device
    robot_arm = StandardDeviceConfigs.create_robot_arm_7dof()
    assert registry.device_manager.register_device_config(robot_arm)
    
    # Register sensor and actuator arrays
    assert registry.sensor_manager.register_sensor_array(
        robot_arm.device_id, robot_arm.sensor_specs
    )
    assert registry.actuator_manager.register_actuator_array(
        robot_arm.device_id, robot_arm.actuator_specs
    )
    
    # Get complete stats
    stats = registry.get_complete_stats()
    assert "hypergraph" in stats
    assert "tensors" in stats
    assert "devices" in stats
    assert "sensors" in stats
    assert "actuators" in stats
    
    # Export complete workbench
    export_data = registry.export_complete_workbench()
    assert "hypergraph" in export_data
    assert "tensors" in export_data
    assert "devices" in export_data
    
    # Validate consistency
    issues = registry.validate_consistency()
    # Should have no consistency issues with proper setup
    assert len(issues) == 0


def test_tensor_dimension_calculations():
    """Test tensor dimension calculations across components."""
    registry = WorkbenchRegistry()
    
    # Add multiple devices with different tensor dimensions
    robot_arm = StandardDeviceConfigs.create_robot_arm_7dof()
    mobile_robot = StandardDeviceConfigs.create_mobile_robot()
    
    registry.device_manager.register_device_config(robot_arm)
    registry.device_manager.register_device_config(mobile_robot)
    
    # Register arrays
    registry.sensor_manager.register_sensor_array(robot_arm.device_id, robot_arm.sensor_specs)
    registry.actuator_manager.register_actuator_array(robot_arm.device_id, robot_arm.actuator_specs)
    registry.sensor_manager.register_sensor_array(mobile_robot.device_id, mobile_robot.sensor_specs)
    registry.actuator_manager.register_actuator_array(mobile_robot.device_id, mobile_robot.actuator_specs)
    
    # Check total dimensions
    total_dimensions = registry.device_manager.get_total_tensor_dimensions()
    assert total_dimensions > 0
    
    # Check individual device dimensions
    arm_dimensions = robot_arm.get_total_tensor_dimensions()
    mobile_dimensions = mobile_robot.get_total_tensor_dimensions()
    
    assert arm_dimensions > 0
    assert mobile_dimensions > 0
    assert total_dimensions == arm_dimensions + mobile_dimensions


def test_hypergraph_node_connections():
    """Test hypergraph node connections and tensor flow."""
    manager = HypergraphManager()
    
    # Create device nodes
    device_node = WorkbenchNode(
        node_id="robot_arm",
        node_type="device", 
        component_class="RobotArm",
    )
    
    sensor_node = WorkbenchNode(
        node_id="robot_arm_sensors",
        node_type="sensor_array",
        component_class="SensorArray",
    )
    
    actuator_node = WorkbenchNode(
        node_id="robot_arm_actuators", 
        node_type="actuator_array",
        component_class="ActuatorArray",
    )
    
    manager.add_node(device_node)
    manager.add_node(sensor_node)
    manager.add_node(actuator_node)
    
    # Create tensor coupling edges
    sensor_edge = WorkbenchEdge(
        edge_id="sensor_to_device",
        source_node_id="robot_arm_sensors",
        target_node_id="robot_arm",
        edge_type=EdgeType.TENSOR_COUPLING,
    )
    sensor_edge.add_tensor_mapping("joint_positions", "current_positions")
    
    actuator_edge = WorkbenchEdge(
        edge_id="device_to_actuator",
        source_node_id="robot_arm",
        target_node_id="robot_arm_actuators",
        edge_type=EdgeType.TENSOR_COUPLING,
    )
    actuator_edge.add_tensor_mapping("target_positions", "joint_commands")
    
    manager.add_edge(sensor_edge)
    manager.add_edge(actuator_edge)
    
    # Test tensor flow paths
    tensor_edges = manager.get_tensor_flow_paths("robot_arm_sensors", "robot_arm_actuators")
    assert len(tensor_edges) == 2
    
    # Test connections
    connected_nodes = manager.get_connected_nodes("robot_arm")
    assert len(connected_nodes) == 2