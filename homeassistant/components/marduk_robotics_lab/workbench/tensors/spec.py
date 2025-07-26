"""Tensor field specification for robotics components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class TensorDataType(Enum):
    """Supported tensor data types."""
    
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    BOOL = "bool"


class TensorModalityType(Enum):
    """Types of sensor and actuator modalities."""
    
    # Sensor modalities
    POSITION = "position"
    VELOCITY = "velocity"  
    ACCELERATION = "acceleration"
    ORIENTATION = "orientation"
    ANGULAR_VELOCITY = "angular_velocity"
    FORCE = "force"
    TORQUE = "torque"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    PROXIMITY = "proximity"
    VISION = "vision"
    AUDIO = "audio"
    IMU = "imu"
    GPS = "gps"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"
    
    # Actuator modalities
    MOTOR_POSITION = "motor_position"
    MOTOR_VELOCITY = "motor_velocity"
    MOTOR_TORQUE = "motor_torque"
    SERVO_ANGLE = "servo_angle"
    PNEUMATIC_PRESSURE = "pneumatic_pressure"
    HYDRAULIC_PRESSURE = "hydraulic_pressure"
    LED_BRIGHTNESS = "led_brightness"
    SPEAKER_VOLUME = "speaker_volume"
    HEATER_POWER = "heater_power"
    
    # State representations
    STATE_VECTOR = "state_vector"
    COMMAND_VECTOR = "command_vector"
    ERROR_VECTOR = "error_vector"


@dataclass
class TensorFieldSpec:
    """Specification for tensor fields in robotics components."""
    
    name: str
    dimensions: Tuple[int, ...]
    dtype: TensorDataType
    modality: TensorModalityType
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    update_rate_hz: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_elements(self) -> int:
        """Calculate total number of elements in the tensor."""
        total = 1
        for dim in self.dimensions:
            total *= dim
        return total
    
    def get_memory_size_bytes(self) -> int:
        """Calculate memory size in bytes for this tensor."""
        element_sizes = {
            TensorDataType.FLOAT32: 4,
            TensorDataType.FLOAT64: 8,
            TensorDataType.INT32: 4,
            TensorDataType.INT64: 8,
            TensorDataType.UINT8: 1,
            TensorDataType.BOOL: 1,
        }
        
        element_size = element_sizes.get(self.dtype, 4)
        return self.get_total_elements() * element_size
    
    def is_compatible_with(self, other: TensorFieldSpec) -> bool:
        """Check if this tensor spec is compatible with another."""
        return (
            self.dimensions == other.dimensions and
            self.dtype == other.dtype and
            self.modality == other.modality
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary representation."""
        return {
            "name": self.name,
            "dimensions": self.dimensions,
            "dtype": self.dtype.value,
            "modality": self.modality.value,
            "unit": self.unit,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "description": self.description,
            "update_rate_hz": self.update_rate_hz,
            "metadata": self.metadata,
            "total_elements": self.get_total_elements(),
            "memory_size_bytes": self.get_memory_size_bytes(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TensorFieldSpec:
        """Create spec from dictionary representation."""
        return cls(
            name=data["name"],
            dimensions=tuple(data["dimensions"]),
            dtype=TensorDataType(data["dtype"]),
            modality=TensorModalityType(data["modality"]),
            unit=data.get("unit"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            description=data.get("description", ""),
            update_rate_hz=data.get("update_rate_hz"),
            metadata=data.get("metadata", {}),
        )


# Predefined tensor specifications for common robotics components
class StandardTensorSpecs:
    """Standard tensor specifications for common robotics use cases."""
    
    # Device position and orientation (6-DOF)
    DEVICE_POSE_6DOF = TensorFieldSpec(
        name="pose_6dof",
        dimensions=(6,),
        dtype=TensorDataType.FLOAT32,
        modality=TensorModalityType.POSITION,
        unit="m, rad",
        description="6-DOF pose: [x, y, z, roll, pitch, yaw]",
        update_rate_hz=100.0,
    )
    
    # Joint positions for typical robot arm
    ROBOT_ARM_7_JOINTS = TensorFieldSpec(
        name="joint_positions",
        dimensions=(7,),
        dtype=TensorDataType.FLOAT32,
        modality=TensorModalityType.MOTOR_POSITION,
        unit="rad",
        min_value=-3.14159,
        max_value=3.14159,
        description="7-DOF robot arm joint positions",
        update_rate_hz=1000.0,
    )
    
    # RGB camera frame
    RGB_CAMERA_HD = TensorFieldSpec(
        name="rgb_frame",
        dimensions=(720, 1280, 3),
        dtype=TensorDataType.UINT8,
        modality=TensorModalityType.VISION,
        unit="pixel",
        min_value=0,
        max_value=255,
        description="HD RGB camera frame",
        update_rate_hz=30.0,
    )
    
    # IMU sensor data
    IMU_9DOF = TensorFieldSpec(
        name="imu_data",
        dimensions=(9,),
        dtype=TensorDataType.FLOAT32,
        modality=TensorModalityType.IMU,
        unit="m/s^2, rad/s, T",
        description="9-DOF IMU: [accel_xyz, gyro_xyz, mag_xyz]",
        update_rate_hz=1000.0,
    )
    
    # Force/torque sensor
    FORCE_TORQUE_6D = TensorFieldSpec(
        name="force_torque",
        dimensions=(6,),
        dtype=TensorDataType.FLOAT32,
        modality=TensorModalityType.FORCE,
        unit="N, Nm",
        description="6-axis force/torque sensor",
        update_rate_hz=1000.0,
    )
    
    # Mobile robot velocity commands
    MOBILE_ROBOT_CMD = TensorFieldSpec(
        name="velocity_cmd",
        dimensions=(3,),
        dtype=TensorDataType.FLOAT32,
        modality=TensorModalityType.COMMAND_VECTOR,
        unit="m/s, rad/s",
        description="Mobile robot velocity command: [v_x, v_y, omega_z]",
        update_rate_hz=50.0,
    )
    
    @classmethod
    def get_all_specs(cls) -> List[TensorFieldSpec]:
        """Get all predefined tensor specifications."""
        return [
            cls.DEVICE_POSE_6DOF,
            cls.ROBOT_ARM_7_JOINTS,
            cls.RGB_CAMERA_HD,
            cls.IMU_9DOF,
            cls.FORCE_TORQUE_6D,
            cls.MOBILE_ROBOT_CMD,
        ]