"""Constants for Marduk's Robotics Lab integration."""

from __future__ import annotations

from typing import Final

# Domain
DOMAIN: Final = "marduk_robotics_lab"

# Platforms
PLATFORMS: Final = [
    "sensor",
    "binary_sensor", 
    "button",
    "switch",
    "device_tracker",
]

# Configuration keys
CONF_WORKBENCH_HOST: Final = "workbench_host"
CONF_WORKBENCH_PORT: Final = "workbench_port"
CONF_AGENT_ID: Final = "agent_id"
CONF_GGUF_PATH: Final = "gguf_path"
CONF_TENSOR_DIMENSIONS: Final = "tensor_dimensions"
CONF_KERNEL_MODE: Final = "kernel_mode"
CONF_COGNITIVE_GRAMMAR: Final = "cognitive_grammar"

# Default values
DEFAULT_WORKBENCH_PORT: Final = 8765
DEFAULT_KERNEL_MODE: Final = "agentic"
DEFAULT_COGNITIVE_GRAMMAR: Final = "scheme"

# GGUF configuration
GGUF_MAGIC: Final = b"GGUF"
GGUF_VERSION: Final = 3

# Hypergraph node types
NODE_TYPE_DEVICE: Final = "device"
NODE_TYPE_SENSOR: Final = "sensor"
NODE_TYPE_ACTUATOR: Final = "actuator"
NODE_TYPE_AGENT: Final = "agent"
NODE_TYPE_KERNEL: Final = "kernel"

# Tensor field types
TENSOR_FIELD_POSITION: Final = "position"
TENSOR_FIELD_VELOCITY: Final = "velocity"
TENSOR_FIELD_ORIENTATION: Final = "orientation"
TENSOR_FIELD_STATE: Final = "state"
TENSOR_FIELD_COMMAND: Final = "command"

# Agent cognitive states
AGENT_STATE_IDLE: Final = "idle"
AGENT_STATE_EXPLORING: Final = "exploring"
AGENT_STATE_EXECUTING: Final = "executing"
AGENT_STATE_LEARNING: Final = "learning"
AGENT_STATE_ERROR: Final = "error"

# Workbench API endpoints
API_ENDPOINT_AGENTS: Final = "/api/agents"
API_ENDPOINT_DEVICES: Final = "/api/devices"
API_ENDPOINT_TENSORS: Final = "/api/tensors"
API_ENDPOINT_KERNELS: Final = "/api/kernels"
API_ENDPOINT_GGUF: Final = "/api/gguf"

# Update intervals
UPDATE_INTERVAL_FAST: Final = 1  # seconds
UPDATE_INTERVAL_NORMAL: Final = 5  # seconds  
UPDATE_INTERVAL_SLOW: Final = 30  # seconds