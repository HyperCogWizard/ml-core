# Composable Hypergraph Workbench Modules

This directory contains the refactored robotics middleware, redesigned as a collection of composable, modular components built around a hypergraph architecture.

## Architecture Overview

### Before: Monolithic Middleware
The original `middleware.py` contained a single `RoboticsMiddleware` class that handled:
- Device communication
- Hypergraph management  
- Agent kernel operations
- Tensor field updates
- Event handling

### After: Composable Workbench Modules
The new architecture separates concerns into focused, reusable modules:

```
workbench/
├── hypergraph/         # Core graph operations
│   ├── manager.py      # Hypergraph management
│   ├── node.py         # Enhanced workbench nodes
│   └── edge.py         # Tensor flow edges
├── tensors/            # Tensor field system
│   ├── spec.py         # Tensor specifications
│   ├── field.py        # Tensor field implementation
│   └── manager.py      # Tensor management
├── devices/            # Device abstraction
│   ├── config.py       # Device configurations
│   ├── component.py    # Base component class
│   └── manager.py      # Device management
├── sensors/            # Sensor management
│   └── manager.py      # Sensor array management
├── actuators/          # Actuator management
│   └── manager.py      # Actuator array management
├── agents/             # Agent kernels
│   └── manager.py      # Agent management
└── registry.py         # Central module registry
```

## Key Features

### 1. Explicit Tensor Dimensions
Every component defines explicit tensor dimensions for:
- **Degrees of Freedom (DoF)**: Physical movement capabilities
- **Sensor Channels**: Input data streams  
- **Actuator Channels**: Output control signals
- **Modalities**: Type of data (position, velocity, force, etc.)

### 2. Standard Component Templates

#### Robot Arm (7-DOF Manipulator)
```python
config = StandardDeviceConfigs.create_robot_arm_7dof()
# DoF: 7, Sensor channels: 14, Actuator channels: 7
# Total tensor dimensions: 21
# Bandwidth: 0.64 Mbps @ 1kHz
```

#### Mobile Robot Platform  
```python
config = StandardDeviceConfigs.create_mobile_robot()
# DoF: 3 (x, y, theta), Sensor channels: 3, Actuator channels: 3
# Total tensor dimensions: 6
# Bandwidth: 0.01 Mbps @ 50Hz
```

### 3. Hypergraph-Encoded Components
Each component is represented as a hypergraph node with:
- **Tensor specifications** defining data flow
- **Connection mappings** between components
- **Edge types** (data flow, control flow, dependencies)
- **Bandwidth and latency requirements**

### 4. Composable Architecture
Modules can be mixed and matched:
```python
# Initialize workbench
registry = WorkbenchRegistry()

# Add devices
registry.device_manager.register_device_config(robot_arm_config)
registry.device_manager.register_device_config(mobile_robot_config)

# Register sensor/actuator arrays
registry.sensor_manager.register_sensor_array(device_id, sensor_specs)
registry.actuator_manager.register_actuator_array(device_id, actuator_specs)

# Spawn agent kernels
registry.agent_manager.register_agent_kernel(agent_kernel)
```

### 5. Automatic Calculations
The system automatically computes:
- **Memory requirements** (bytes per tensor field)
- **Bandwidth requirements** (Mbps based on update rates)
- **Total system dimensions** (sum across all components)
- **Consistency validation** (cross-module verification)

## Tensor Field Specifications

### Data Types
- `FLOAT32`, `FLOAT64`, `INT32`, `INT64`, `UINT8`, `BOOL`

### Modalities 
#### Sensors
- `POSITION`, `VELOCITY`, `ACCELERATION`, `ORIENTATION`
- `FORCE`, `TORQUE`, `TEMPERATURE`, `PRESSURE`
- `VISION`, `AUDIO`, `IMU`, `GPS`, `LIDAR`, `ULTRASONIC`

#### Actuators  
- `MOTOR_POSITION`, `MOTOR_VELOCITY`, `MOTOR_TORQUE`
- `SERVO_ANGLE`, `PNEUMATIC_PRESSURE`, `HYDRAULIC_PRESSURE`
- `LED_BRIGHTNESS`, `SPEAKER_VOLUME`, `HEATER_POWER`

#### States
- `STATE_VECTOR`, `COMMAND_VECTOR`, `ERROR_VECTOR`

## Usage Examples

### Define Custom Tensor Specification
```python
imu_spec = TensorFieldSpec(
    name="imu_9dof",
    dimensions=(9,),  # [accel_xyz, gyro_xyz, mag_xyz]
    dtype=TensorDataType.FLOAT32,
    modality=TensorModalityType.IMU,
    unit="m/s^2, rad/s, T",
    update_rate_hz=1000.0
)
```

### Create Custom Device Configuration
```python
drone_config = DeviceConfig(
    device_id="quadcopter_x4",
    device_type=DeviceType.HYBRID_SYSTEM,
    name="Quadcopter X4",
    degrees_of_freedom=6,  # x, y, z, roll, pitch, yaw
    max_payload=2.0,  # kg
    workspace_dimensions=(100.0, 100.0, 50.0),  # meters
    update_rate_hz=500.0
)

# Add sensor specs
drone_config.add_sensor_spec(imu_spec)
drone_config.add_sensor_spec(gps_spec)

# Add actuator specs  
drone_config.add_actuator_spec(motor_thrust_spec)
```

### Build Hypergraph Connections
```python
# Create tensor coupling edge
sensor_to_controller = WorkbenchEdge(
    edge_id="imu_to_flight_controller",
    source_node_id="drone_sensors",
    target_node_id="flight_controller",
    edge_type=EdgeType.TENSOR_COUPLING,
    bandwidth_requirements=4.0,  # Mbps
    latency_requirements=1.0,    # ms
)

# Add tensor field mappings
sensor_to_controller.add_tensor_mapping("imu_data", "sensor_input")
sensor_to_controller.add_tensor_mapping("gps_position", "position_feedback")
```

## Migration Guide

### From Old Middleware
```python
# Old approach
middleware = RoboticsMiddleware(hass, config_entry)
await middleware.register_device(device_config)
```

### To New Composable Architecture
```python
# New approach
middleware = ComposableRoboticsMiddleware(hass, config_entry)
await middleware.register_device(device_config)  # Same interface

# Or use modules directly
registry = WorkbenchRegistry()
registry.device_manager.register_device_config(device_config)
registry.sensor_manager.register_sensor_array(device_id, sensor_specs)
```

## Benefits

1. **Modularity**: Each component has a single responsibility
2. **Reusability**: Standard templates for common robotics systems
3. **Composability**: Mix and match components for different applications
4. **Explicit Dimensions**: Clear tensor specifications for each component
5. **Scalability**: Easy to add new device types and sensor modalities
6. **Validation**: Automatic consistency checking across modules
7. **Performance**: Bandwidth and memory requirements calculated automatically
8. **Maintainability**: Smaller, focused modules easier to understand and modify

## Testing

Run the workbench module tests:
```bash
python /tmp/test_workbench.py
```

Expected output shows successful creation and validation of:
- Tensor field specifications with explicit dimensions
- Device configurations with DoF and channel definitions  
- Hypergraph nodes with tensor specifications
- Cross-module consistency validation
- Bandwidth and memory requirement calculations