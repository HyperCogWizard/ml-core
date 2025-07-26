"""Enhanced device configuration for workbench components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from ..tensors import TensorFieldSpec, TensorModalityType, TensorDataType


class DeviceType(Enum):
    """Types of robotics devices."""
    
    MANIPULATOR = "manipulator"
    MOBILE_ROBOT = "mobile_robot"
    SENSOR_ARRAY = "sensor_array"
    ACTUATOR_ARRAY = "actuator_array"
    HYBRID_SYSTEM = "hybrid_system"
    COMMUNICATION_HUB = "communication_hub"


class ConnectionType(Enum):
    """Device connection types."""
    
    SERIAL = "serial"
    USB = "usb"
    ETHERNET = "ethernet"
    WIFI = "wifi"
    BLUETOOTH = "bluetooth"
    CAN_BUS = "can_bus"
    I2C = "i2c"
    SPI = "spi"
    MODBUS = "modbus"


@dataclass
class DeviceConfig:
    """Enhanced configuration for robotics devices."""
    
    device_id: str
    device_type: DeviceType
    name: str
    manufacturer: str = ""
    model: str = ""
    firmware_version: str = ""
    
    # Physical characteristics
    degrees_of_freedom: int = 0
    workspace_dimensions: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # meters
    max_payload: float = 0.0  # kg
    max_velocity: float = 0.0  # m/s or rad/s
    max_acceleration: float = 0.0  # m/s^2 or rad/s^2
    
    # Connectivity
    connection_type: ConnectionType = ConnectionType.ETHERNET
    connection_params: Dict[str, Any] = field(default_factory=dict)
    
    # Tensor specifications
    sensor_specs: List[TensorFieldSpec] = field(default_factory=list)
    actuator_specs: List[TensorFieldSpec] = field(default_factory=list)
    state_specs: List[TensorFieldSpec] = field(default_factory=list)
    
    # Operational parameters
    update_rate_hz: float = 100.0
    control_loop_hz: float = 1000.0
    safety_limits: Dict[str, Any] = field(default_factory=dict)
    calibration_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_sensor_spec(self, spec: TensorFieldSpec) -> None:
        """Add a sensor tensor specification."""
        # Remove existing spec with same name
        self.sensor_specs = [s for s in self.sensor_specs if s.name != spec.name]
        self.sensor_specs.append(spec)
    
    def add_actuator_spec(self, spec: TensorFieldSpec) -> None:
        """Add an actuator tensor specification."""
        # Remove existing spec with same name
        self.actuator_specs = [s for s in self.actuator_specs if s.name != spec.name]
        self.actuator_specs.append(spec)
    
    def add_state_spec(self, spec: TensorFieldSpec) -> None:
        """Add a state tensor specification."""
        # Remove existing spec with same name
        self.state_specs = [s for s in self.state_specs if s.name != spec.name]
        self.state_specs.append(spec)
    
    def get_all_tensor_specs(self) -> List[TensorFieldSpec]:
        """Get all tensor specifications for this device."""
        return self.sensor_specs + self.actuator_specs + self.state_specs
    
    def get_total_tensor_dimensions(self) -> int:
        """Calculate total tensor dimensions for this device."""
        total = 0
        for spec in self.get_all_tensor_specs():
            total += spec.get_total_elements()
        return total
    
    def get_sensor_channel_count(self) -> int:
        """Get total number of sensor channels."""
        return sum(spec.get_total_elements() for spec in self.sensor_specs)
    
    def get_actuator_channel_count(self) -> int:
        """Get total number of actuator channels."""
        return sum(spec.get_total_elements() for spec in self.actuator_specs)
    
    def get_memory_requirements_bytes(self) -> int:
        """Calculate memory requirements for all tensor fields."""
        return sum(spec.get_memory_size_bytes() for spec in self.get_all_tensor_specs())
    
    def get_bandwidth_requirements_mbps(self) -> float:
        """Estimate bandwidth requirements in Mbps."""
        total_bytes_per_update = self.get_memory_requirements_bytes()
        updates_per_second = self.update_rate_hz
        bytes_per_second = total_bytes_per_update * updates_per_second
        return (bytes_per_second * 8) / (1024 * 1024)  # Convert to Mbps
    
    def validate_configuration(self) -> List[str]:
        """Validate device configuration and return list of issues."""
        issues = []
        
        if not self.device_id:
            issues.append("Device ID is required")
        
        if self.degrees_of_freedom < 0:
            issues.append("Degrees of freedom cannot be negative")
        
        if self.update_rate_hz <= 0:
            issues.append("Update rate must be positive")
        
        if self.control_loop_hz < self.update_rate_hz:
            issues.append("Control loop frequency should be >= update rate")
        
        # Check for duplicate tensor spec names
        all_specs = self.get_all_tensor_specs()
        spec_names = [spec.name for spec in all_specs]
        if len(spec_names) != len(set(spec_names)):
            issues.append("Duplicate tensor specification names found")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert device config to dictionary representation."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "name": self.name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "firmware_version": self.firmware_version,
            "degrees_of_freedom": self.degrees_of_freedom,
            "workspace_dimensions": self.workspace_dimensions,
            "max_payload": self.max_payload,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "connection_type": self.connection_type.value,
            "connection_params": self.connection_params,
            "sensor_specs": [spec.to_dict() for spec in self.sensor_specs],
            "actuator_specs": [spec.to_dict() for spec in self.actuator_specs],
            "state_specs": [spec.to_dict() for spec in self.state_specs],
            "update_rate_hz": self.update_rate_hz,
            "control_loop_hz": self.control_loop_hz,
            "safety_limits": self.safety_limits,
            "calibration_data": self.calibration_data,
            "metadata": self.metadata,
            "total_tensor_dimensions": self.get_total_tensor_dimensions(),
            "sensor_channels": self.get_sensor_channel_count(),
            "actuator_channels": self.get_actuator_channel_count(),
            "memory_requirements_bytes": self.get_memory_requirements_bytes(),
            "bandwidth_requirements_mbps": self.get_bandwidth_requirements_mbps(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeviceConfig:
        """Create device config from dictionary representation."""
        from ..tensors import TensorFieldSpec
        
        config = cls(
            device_id=data["device_id"],
            device_type=DeviceType(data["device_type"]),
            name=data["name"],
            manufacturer=data.get("manufacturer", ""),
            model=data.get("model", ""),
            firmware_version=data.get("firmware_version", ""),
            degrees_of_freedom=data.get("degrees_of_freedom", 0),
            workspace_dimensions=tuple(data.get("workspace_dimensions", (0.0, 0.0, 0.0))),
            max_payload=data.get("max_payload", 0.0),
            max_velocity=data.get("max_velocity", 0.0),
            max_acceleration=data.get("max_acceleration", 0.0),
            connection_type=ConnectionType(data.get("connection_type", "ethernet")),
            connection_params=data.get("connection_params", {}),
            update_rate_hz=data.get("update_rate_hz", 100.0),
            control_loop_hz=data.get("control_loop_hz", 1000.0),
            safety_limits=data.get("safety_limits", {}),
            calibration_data=data.get("calibration_data", {}),
            metadata=data.get("metadata", {}),
        )
        
        # Reconstruct tensor specs
        for spec_data in data.get("sensor_specs", []):
            spec = TensorFieldSpec.from_dict(spec_data)
            config.sensor_specs.append(spec)
        
        for spec_data in data.get("actuator_specs", []):
            spec = TensorFieldSpec.from_dict(spec_data)
            config.actuator_specs.append(spec)
            
        for spec_data in data.get("state_specs", []):
            spec = TensorFieldSpec.from_dict(spec_data)
            config.state_specs.append(spec)
        
        return config


# Predefined device configurations for common robotics systems
class StandardDeviceConfigs:
    """Standard device configurations for common robotics systems."""
    
    @staticmethod
    def create_robot_arm_7dof() -> DeviceConfig:
        """Create configuration for a 7-DOF robot arm."""
        config = DeviceConfig(
            device_id="robot_arm_7dof",
            device_type=DeviceType.MANIPULATOR,
            name="7-DOF Robot Arm",
            degrees_of_freedom=7,
            workspace_dimensions=(1.0, 1.0, 1.0),
            max_payload=5.0,
            max_velocity=2.0,
            update_rate_hz=1000.0,
        )
        
        # Joint position sensors
        config.add_sensor_spec(TensorFieldSpec(
            name="joint_positions",
            dimensions=(7,),
            dtype=TensorDataType.FLOAT32,
            modality=TensorModalityType.POSITION,
            unit="rad",
            min_value=-3.14159,
            max_value=3.14159,
        ))
        
        # Joint velocity sensors  
        config.add_sensor_spec(TensorFieldSpec(
            name="joint_velocities",
            dimensions=(7,),
            dtype=TensorDataType.FLOAT32,
            modality=TensorModalityType.VELOCITY,
            unit="rad/s",
        ))
        
        # Joint torque actuators
        config.add_actuator_spec(TensorFieldSpec(
            name="joint_torques",
            dimensions=(7,),
            dtype=TensorDataType.FLOAT32,
            modality=TensorModalityType.MOTOR_TORQUE,
            unit="Nm",
        ))
        
        return config
    
    @staticmethod  
    def create_mobile_robot() -> DeviceConfig:
        """Create configuration for a mobile robot."""
        config = DeviceConfig(
            device_id="mobile_robot",
            device_type=DeviceType.MOBILE_ROBOT,
            name="Mobile Robot Platform",
            degrees_of_freedom=3,  # x, y, theta
            max_velocity=2.0,
            max_acceleration=1.0,
            update_rate_hz=50.0,
        )
        
        # Odometry sensors
        config.add_sensor_spec(TensorFieldSpec(
            name="odometry",
            dimensions=(3,),
            dtype=TensorDataType.FLOAT32,
            modality=TensorModalityType.POSITION,
            unit="m, rad",
        ))
        
        # Velocity sensors
        config.add_sensor_spec(TensorFieldSpec(
            name="velocity",
            dimensions=(3,),
            dtype=TensorDataType.FLOAT32,
            modality=TensorModalityType.VELOCITY,
            unit="m/s, rad/s",
        ))
        
        # Velocity commands
        config.add_actuator_spec(TensorFieldSpec(
            name="velocity_cmd",
            dimensions=(3,),
            dtype=TensorDataType.FLOAT32,
            modality=TensorModalityType.COMMAND_VECTOR,
            unit="m/s, rad/s",
        ))
        
        return config