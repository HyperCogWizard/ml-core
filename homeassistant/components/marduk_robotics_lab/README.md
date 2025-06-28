# Marduk's Robotics Lab Integration

## Overview

Marduk's Robotics Lab is a comprehensive Home Assistant integration that extends robotics middleware abstraction into a practical robotics engineering workbench using GGUF (GPT-Generated Unified Format) for distributed agentic cognition and neural-symbolic middleware.

## Architecture

```
Robotics Middleware → Engineering Workbench → GGUF Integration → HomeAssistant → Marduk's Robotics Lab
```

### Core Components

1. **Robotics Middleware Abstraction Layer** (`middleware.py`)
   - WebSocket-based communication with robotics workbench
   - Hypergraph node and edge management
   - Device registry and agent kernel management
   - Real-time event handling

2. **GGUF Integration Layer** (`gguf_handler.py`)
   - Tensor serialization for agent states and device configurations
   - Neural membrane export/import for P-System compatibility
   - Agent snapshot and restore functionality
   - Distributed cognition state management

3. **Engineering Workbench Components** (`models.py`)
   - Hypergraph-encoded device configurations
   - Tensor field representations
   - Agent kernel cognitive structures
   - Complex system modeling

4. **Agentic Control Loops** (`coordinator.py`)
   - Event-driven neural-symbolic automation
   - Recursive agent self-modification
   - Distributed cognition coordination
   - Real-time system monitoring

## Features

### Hypergraph-Encoded Workbench
- **Dynamic Node Management**: Add/remove devices, sensors, actuators, and agents
- **Edge Connectivity**: Real-time hypergraph modifications and relationships
- **Tensor Fields**: Multi-dimensional data representation for each component
- **Complex System Modeling**: Degrees of freedom, sensor channels, and modalities

### GGUF Tensor Serialization
- **Agent State Export**: Complete cognitive state preservation
- **Device Configuration Storage**: Hardware specifications and capabilities
- **Memory Bank Serialization**: Agent learning and experience data
- **P-System Membranes**: Nested structure export for distributed processing

### Agentic Control Loops
- **Neural-Symbolic Grammar**: Support for Scheme, Prolog, JavaScript, Python
- **Recursive Self-Modification**: Agents can modify their own parameters
- **Distributed Cognition**: Multi-agent coordination and communication
- **Event-Driven Automation**: Replace static rules with adaptive behaviors

### Real-Time Visualization
- **Tensor Field Monitoring**: Live updates of multi-dimensional data
- **Agent State Tracking**: Cognitive state visualization and metrics
- **System Complexity Metrics**: Comprehensive system analysis
- **Hypergraph Structure**: Dynamic topology visualization

## Installation

1. **Add Integration Files**: Copy the `marduk_robotics_lab` directory to your Home Assistant `custom_components` folder.

2. **Install Dependencies**:
   ```bash
   pip install numpy>=1.21.0 msgpack>=1.0.0 websockets>=10.0
   ```

3. **Configure Integration**: Use the Home Assistant UI to add the integration via Settings → Integrations.

## Configuration

### Basic Setup
- **Workbench Host**: IP address or hostname of the robotics workbench
- **Workbench Port**: WebSocket port (default: 8765)
- **Lab Name**: Custom name for your robotics lab
- **GGUF Path**: Optional path for GGUF file storage
- **Kernel Mode**: Agentic, reactive, or hybrid operation
- **Cognitive Grammar**: Scheme, Prolog, JavaScript, or Python

### Discovery Support
The integration supports automatic discovery via:
- **Zeroconf/mDNS**: `_marduk-robotics._tcp.local.`
- **DHCP**: Hostname pattern `marduk-*`

## Entities

### Sensors
- **Lab Metrics**: Total agents, devices, nodes, complexity
- **Agent States**: Individual agent cognitive states and parameters
- **Device Complexity**: Hardware specifications and capabilities
- **System Status**: Overall lab operational status

### Binary Sensors
- **Middleware Connected**: WebSocket connectivity status
- **GGUF Enabled**: Tensor serialization availability
- **Agents Active**: Active agent presence indicator

### Buttons
- **Spawn Agent**: Create new exploration/manipulation agents
- **Export GGUF**: Save current lab state to GGUF format
- **Refresh Hypergraph**: Update topology and connections
- **Emergency Stop**: Halt all agents and devices

### Switches
- **Tensor Visualization**: Enable/disable real-time tensor display
- **Recursive Modification**: Allow/prevent agent self-modification
- **Agent Learning**: Individual agent learning enable/disable

### Device Trackers
- **Mobile Devices**: Track robotic arms, mobile bases, drones
- **Agent Location**: Virtual and physical agent position tracking

## Services

### `marduk_robotics_lab.spawn_agent`
Create a new agent kernel with specified capabilities.

**Parameters:**
- `agent_type`: exploration, manipulation, planning, perception, learning, coordination
- `agent_id`: Optional custom identifier
- `cognitive_grammar`: Scheme, Prolog, JavaScript, Python

### `marduk_robotics_lab.register_device`
Register a new robotics device in the lab.

**Parameters:**
- `device_id`: Unique device identifier
- `device_type`: robotic_arm, mobile_base, sensor_array, etc.
- `degrees_of_freedom`: Number of controllable DOF
- `sensor_channels`: Input channel count
- `actuator_channels`: Output channel count

### `marduk_robotics_lab.export_gguf`
Export current lab state to GGUF format.

**Parameters:**
- `file_path`: Optional custom export path
- `include_agents`: Include agent states (default: true)
- `include_devices`: Include device configurations (default: true)

### `marduk_robotics_lab.import_gguf`
Import lab state from GGUF format.

**Parameters:**
- `file_path`: Path to GGUF file
- `merge_mode`: replace, merge, or skip conflicts

### `marduk_robotics_lab.modify_agent`
Modify existing agent parameters.

**Parameters:**
- `agent_id`: Target agent identifier
- `learning_rate`: New learning rate (0.0001-1.0)
- `exploration_factor`: New exploration factor (0.0-1.0)

## API Integration

### WebSocket Protocol
The integration communicates with the robotics workbench via WebSocket using a JSON protocol:

```json
{
  "type": "command_type",
  "data": { /* command-specific data */ },
  "timestamp": 1234567890.0
}
```

### Event Types
- `device_update`: Device state changes
- `agent_state_change`: Agent cognitive state modifications
- `tensor_field_update`: Multi-dimensional data updates
- `hypergraph_modification`: Topology changes

## GGUF Format

### Structure
```json
{
  "metadata": {
    "version": "1.0.0",
    "lab_id": "unique_lab_identifier",
    "export_timestamp": "2024-01-01T00:00:00Z",
    "format": "marduk_robotics_lab_gguf"
  },
  "tensors": {
    "agent_memory_banks": { /* tensor data */ },
    "device_state_vectors": { /* tensor data */ }
  },
  "parameters": {
    "learning_rates": { /* agent parameters */ },
    "device_configurations": { /* hardware specs */ }
  },
  "hypergraph": {
    "nodes": { /* node definitions */ },
    "edges": { /* connection topology */ }
  }
}
```

### Binary Format
- **Magic Bytes**: `GGUF` (4 bytes)
- **Version**: 3 (4 bytes, little-endian)
- **Data Length**: (8 bytes, little-endian)
- **Data**: MessagePack-encoded structure

## Advanced Usage

### Cognitive Grammar Examples

#### Scheme Agent
```scheme
(define (explore-environment sensors)
  (let ((obstacles (filter obstacle? sensors)))
    (if (null? obstacles)
        (move-forward)
        (turn-random))))
```

#### Prolog Agent
```prolog
action(move_forward) :- 
    sensor(distance, D), 
    D > threshold.
action(turn_left) :- 
    sensor(distance, D), 
    D =< threshold.
```

### Tensor Field Definitions
```python
position_field = TensorField(
    name="robot_position",
    dimensions=(3,),  # x, y, z
    dtype="float32"
)

force_field = TensorField(
    name="gripper_force",
    dimensions=(6,),  # 6-DOF force/torque
    dtype="float32"
)
```

### Hypergraph Operations
```python
# Add device node
node = HypergraphNode(
    node_id="arm_001",
    node_type="device",
    properties={"joints": 6, "payload": 5.0}
)

# Connect to sensor array
node.connections.append("sensor_array_001")
```

## Quality Scale: Silver

This integration implements Silver-level quality standards including:
- ✅ Configuration flow with UI setup
- ✅ Entity unavailability handling
- ✅ Parallel updates support
- ✅ Authentication flows
- ✅ Device management
- ✅ Diagnostic information
- ✅ Translations support

## Troubleshooting

### Connection Issues
1. Verify workbench host and port settings
2. Check WebSocket connectivity
3. Ensure middleware service is running
4. Review Home Assistant logs for connection errors

### GGUF Export Failures
1. Check GGUF storage path permissions
2. Verify available disk space
3. Ensure agent/device data is valid
4. Review GGUF handler logs

### Agent Spawning Problems
1. Verify cognitive grammar support
2. Check workbench capacity limits
3. Review agent kernel configurations
4. Ensure middleware connectivity

## Contributing

Marduk's Robotics Lab is designed for extensibility:

1. **New Cognitive Grammars**: Add support for additional programming languages
2. **Device Types**: Extend device registry with new robotics hardware
3. **Agent Behaviors**: Implement specialized agent types
4. **Visualization**: Enhance tensor field and hypergraph displays

## License

This integration is part of the Home Assistant ecosystem and follows the Apache 2.0 license.

---

*For technical support and advanced configurations, refer to the Marduk's Robotics Lab documentation and community forums.*