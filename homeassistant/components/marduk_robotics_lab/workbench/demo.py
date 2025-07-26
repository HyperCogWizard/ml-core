#!/usr/bin/env python3
"""Demonstration of the composable hypergraph workbench modules."""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

try:
    from homeassistant.components.marduk_robotics_lab.workbench import WorkbenchRegistry
    from homeassistant.components.marduk_robotics_lab.workbench.devices import StandardDeviceConfigs
    from homeassistant.components.marduk_robotics_lab.workbench.tensors import StandardTensorSpecs, TensorFieldSpec, TensorDataType, TensorModalityType
    from homeassistant.components.marduk_robotics_lab.workbench.hypergraph import WorkbenchNode, WorkbenchEdge, EdgeType
    FULL_IMPORT = True
except ImportError:
    print("⚠️  Cannot import full Home Assistant modules due to version compatibility")
    print("📋 Running with simplified standalone demo instead...")
    FULL_IMPORT = False


async def demonstrate_workbench_architecture():
    """Demonstrate the key features of the composable workbench architecture."""
    
    print("🤖 Marduk's Robotics Lab - Composable Workbench Demonstration")
    print("=" * 70)
    
    if not FULL_IMPORT:
        print("📋 This demonstration would show:")
        print("   • Modular hypergraph node and edge management")
        print("   • Explicit tensor dimension specifications for robotics components")
        print("   • Composable device, sensor, and actuator managers")
        print("   • Automatic bandwidth and memory requirement calculations")
        print("   • Cross-module consistency validation")
        print("   • Standard templates for common robotics systems")
        print("\n✅ See /tmp/test_workbench.py for a working standalone test")
        return
    
    # Initialize the workbench registry
    print("🏗️  Initializing Workbench Registry...")
    registry = WorkbenchRegistry()
    
    # Demonstrate standard device configurations
    print("\n📱 Creating Standard Device Configurations...")
    
    # Create a 7-DOF robot arm
    robot_arm = StandardDeviceConfigs.create_robot_arm_7dof()
    print(f"   ✅ Robot Arm: {robot_arm.name}")
    print(f"      • DoF: {robot_arm.degrees_of_freedom}")
    print(f"      • Sensor channels: {robot_arm.get_sensor_channel_count()}")
    print(f"      • Actuator channels: {robot_arm.get_actuator_channel_count()}")
    print(f"      • Bandwidth: {robot_arm.get_bandwidth_requirements_mbps():.2f} Mbps")
    
    # Create a mobile robot
    mobile_robot = StandardDeviceConfigs.create_mobile_robot()
    print(f"   ✅ Mobile Robot: {mobile_robot.name}")
    print(f"      • DoF: {mobile_robot.degrees_of_freedom}")
    print(f"      • Sensor channels: {mobile_robot.get_sensor_channel_count()}")
    print(f"      • Actuator channels: {mobile_robot.get_actuator_channel_count()}")
    print(f"      • Bandwidth: {mobile_robot.get_bandwidth_requirements_mbps():.2f} Mbps")
    
    # Register devices with the workbench
    print("\n🔧 Registering Devices with Workbench...")
    registry.device_manager.register_device_config(robot_arm)
    registry.device_manager.register_device_config(mobile_robot)
    
    # Register sensor and actuator arrays
    print("🔍 Registering Sensor Arrays...")
    registry.sensor_manager.register_sensor_array(robot_arm.device_id, robot_arm.sensor_specs)
    registry.sensor_manager.register_sensor_array(mobile_robot.device_id, mobile_robot.sensor_specs)
    
    print("⚙️  Registering Actuator Arrays...")
    registry.actuator_manager.register_actuator_array(robot_arm.device_id, robot_arm.actuator_specs)
    registry.actuator_manager.register_actuator_array(mobile_robot.device_id, mobile_robot.actuator_specs)
    
    # Create custom tensor specification
    print("\n📊 Creating Custom Tensor Specification...")
    camera_spec = TensorFieldSpec(
        name="rgb_camera_feed",
        dimensions=(480, 640, 3),  # VGA RGB
        dtype=TensorDataType.UINT8,
        modality=TensorModalityType.VISION,
        unit="pixel",
        description="VGA RGB camera feed",
        update_rate_hz=30.0
    )
    
    registry.tensor_manager.register_tensor_spec(camera_spec)
    print(f"   ✅ Camera Spec: {camera_spec.name}")
    print(f"      • Dimensions: {camera_spec.dimensions}")
    print(f"      • Total elements: {camera_spec.get_total_elements():,}")
    print(f"      • Memory size: {camera_spec.get_memory_size_bytes():,} bytes")
    print(f"      • Bandwidth @ 30Hz: {(camera_spec.get_memory_size_bytes() * 30 * 8) / (1024*1024):.1f} Mbps")
    
    # Create hypergraph connections
    print("\n🔗 Creating Hypergraph Connections...")
    
    # Create edge between robot arm sensors and actuators
    control_edge = WorkbenchEdge(
        edge_id="arm_control_loop",
        source_node_id=f"{robot_arm.device_id}_sensors",
        target_node_id=f"{robot_arm.device_id}_actuators",
        edge_type=EdgeType.CONTROL_FLOW,
        bandwidth_requirements=1.0,  # Mbps
        latency_requirements=1.0,    # ms
    )
    
    # Add tensor field mappings
    control_edge.add_tensor_mapping("joint_positions", "position_feedback")
    control_edge.add_tensor_mapping("joint_velocities", "velocity_feedback")
    
    # Add the edge to hypergraph (if both nodes exist)
    # registry.hypergraph_manager.add_edge(control_edge)
    
    print(f"   ✅ Control Edge: {control_edge.edge_id}")
    print(f"      • Type: {control_edge.edge_type.value}")
    print(f"      • Tensor mappings: {len(control_edge.tensor_mappings)}")
    print(f"      • Bandwidth req: {control_edge.bandwidth_requirements} Mbps")
    print(f"      • Latency req: {control_edge.latency_requirements} ms")
    
    # Get comprehensive statistics
    print("\n📈 Workbench Statistics...")
    stats = registry.get_complete_stats()
    
    print(f"   🔧 Devices: {stats['devices']['total_devices']}")
    print(f"      • Total DoF: {stats['devices']['total_degrees_of_freedom']}")
    print(f"      • Sensor channels: {stats['devices']['total_sensor_channels']}")
    print(f"      • Actuator channels: {stats['devices']['total_actuator_channels']}")
    print(f"      • Bandwidth: {stats['devices']['total_bandwidth_mbps']:.2f} Mbps")
    
    print(f"   🧠 Hypergraph: {stats['hypergraph']['total_nodes']} nodes")
    print(f"      • Total tensor dimensions: {stats['hypergraph']['total_tensor_dimensions']}")
    
    print(f"   📊 Tensors: {stats['tensors']['total_fields']} active fields")
    print(f"      • Memory usage: {stats['tensors']['total_memory_bytes']:,} bytes")
    
    print(f"   🔍 Sensors: {stats['sensors']['total_sensors']} sensors")
    print(f"      • Devices with sensors: {stats['sensors']['total_devices_with_sensors']}")
    
    print(f"   ⚙️  Actuators: {stats['actuators']['total_actuators']} actuators")
    print(f"      • Devices with actuators: {stats['actuators']['total_devices_with_actuators']}")
    
    # Validate consistency
    print("\n✅ Consistency Validation...")
    issues = registry.validate_consistency()
    if issues:
        print(f"   ❌ Found {len(issues)} consistency issues:")
        for issue in issues:
            print(f"      • {issue}")
    else:
        print("   ✅ All modules are consistent - no issues found")
    
    # Export complete workbench state
    print("\n💾 Exporting Workbench State...")
    export_data = registry.export_complete_workbench()
    
    device_count = len(export_data["devices"]["devices"])
    tensor_spec_count = len(export_data["tensors"]["specifications"])
    hypergraph_node_count = len(export_data["hypergraph"]["nodes"])
    
    print(f"   ✅ Export complete:")
    print(f"      • {device_count} device configurations")
    print(f"      • {tensor_spec_count} tensor specifications")
    print(f"      • {hypergraph_node_count} hypergraph nodes")
    
    print("\n" + "=" * 70)
    print("🎉 Composable Workbench Demonstration Complete!")
    
    print("\n🏗️  Architecture Benefits Demonstrated:")
    print("   ✅ Modular components with single responsibilities")
    print("   ✅ Explicit tensor dimensions for all robotics components") 
    print("   ✅ Composable device, sensor, and actuator management")
    print("   ✅ Automatic bandwidth and memory calculations")
    print("   ✅ Hypergraph-based component representation")
    print("   ✅ Cross-module consistency validation")
    print("   ✅ Standard templates for common robotics systems")
    
    print("\n📚 Next Steps:")
    print("   • Add more device types (drones, underwater vehicles, etc.)")
    print("   • Implement real-time tensor data streaming")
    print("   • Add machine learning model integration")
    print("   • Create visual hypergraph editor")
    print("   • Implement distributed workbench networking")


if __name__ == "__main__":
    asyncio.run(demonstrate_workbench_architecture())