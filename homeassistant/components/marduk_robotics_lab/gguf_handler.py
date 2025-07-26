"""GGUF serialization handler for Marduk's Robotics Lab."""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import msgpack

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import (
    CONF_GGUF_PATH,
    GGUF_MAGIC,
    GGUF_VERSION,
)

_LOGGER = logging.getLogger(__name__)


class GGUFHandler:
    """Handler for GGUF serialization of agent states and device configurations."""
    
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize GGUF handler."""
        self.hass = hass
        self.config_entry = config_entry
        self._gguf_path = config_entry.data.get(CONF_GGUF_PATH)
        self._temp_dir: Optional[Path] = None
        
    async def async_setup(self) -> None:
        """Set up GGUF handler."""
        if not self._gguf_path:
            # Create temporary directory for GGUF files
            self._temp_dir = Path(tempfile.mkdtemp(prefix="marduk_gguf_"))
            self._gguf_path = str(self._temp_dir / "lab_state.gguf")
            _LOGGER.info("Using temporary GGUF path: %s", self._gguf_path)
        else:
            # Ensure the directory exists
            gguf_path = Path(self._gguf_path)
            gguf_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def async_shutdown(self) -> None:
        """Shutdown GGUF handler and cleanup."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self._temp_dir)
                _LOGGER.debug("Cleaned up temporary GGUF directory")
            except Exception as err:
                _LOGGER.warning("Failed to cleanup GGUF temp directory: %s", err)
    
    async def export_lab_state(self, lab_data: Dict[str, Any]) -> str:
        """Export lab state to GGUF format."""
        try:
            # Prepare GGUF data structure
            gguf_data = self._prepare_gguf_data(lab_data)
            
            # Write GGUF file
            gguf_path = await self._write_gguf_file(gguf_data)
            
            _LOGGER.info("Exported lab state to GGUF: %s", gguf_path)
            return gguf_path
            
        except Exception as err:
            _LOGGER.error("Failed to export GGUF: %s", err)
            raise
    
    async def import_lab_state(self, gguf_path: str) -> Dict[str, Any]:
        """Import lab state from GGUF format."""
        try:
            # Read GGUF file
            gguf_data = await self._read_gguf_file(gguf_path)
            
            # Convert back to lab data structure
            lab_data = self._parse_gguf_data(gguf_data)
            
            _LOGGER.info("Imported lab state from GGUF: %s", gguf_path)
            return lab_data
            
        except Exception as err:
            _LOGGER.error("Failed to import GGUF: %s", err)
            raise
    
    def _prepare_gguf_data(self, lab_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare lab data for comprehensive GGUF serialization with P-System membranes."""
        
        # Create comprehensive GGUF-compatible structure
        gguf_structure = {
            "metadata": {
                "version": "2.0.0",  # Updated for comprehensive support
                "lab_id": self.config_entry.entry_id,
                "export_timestamp": dt_util.now().isoformat(),
                "format": "marduk_comprehensive_gguf",
                "p_system_compatible": True
            },
            "tensors": {},
            "parameters": {},
            "hypergraph": lab_data.get("hypergraph_nodes", {}),
            "devices": lab_data.get("registered_devices", {}),
            "agents": lab_data.get("active_agents", {}),
            "metrics": lab_data.get("metrics", {}),
            "membrane_system": {},  # P-System membrane hierarchy
            "environment_tensors": {},  # System-wide tensor fields
        }
        
        # Convert ALL agent state to tensors with comprehensive coverage
        for agent_id, agent_data in gguf_structure["agents"].items():
            self._process_agent_comprehensive_state(agent_id, agent_data, gguf_structure)
        
        # Convert ALL device configuration to tensors
        for device_id, device_data in gguf_structure["devices"].items():
            self._process_device_comprehensive_config(device_id, device_data, gguf_structure)
        
        # Process hypergraph tensor fields
        for node_id, node_data in gguf_structure["hypergraph"].items():
            self._process_hypergraph_tensors(node_id, node_data, gguf_structure)
        
        # Generate P-System membrane structure
        self._generate_p_system_membranes(gguf_structure)
        
        return gguf_structure
    
    def _process_agent_comprehensive_state(self, agent_id: str, agent_data: Dict[str, Any], gguf_structure: Dict[str, Any]) -> None:
        """Process comprehensive agent state for GGUF export."""
        # Convert memory banks to tensors (existing functionality enhanced)
        if "memory_banks" in agent_data:
            for i, memory_bank in enumerate(agent_data["memory_banks"]):
                tensor_name = f"agent_{agent_id}_memory_{i}"
                gguf_structure["tensors"][tensor_name] = {
                    "name": memory_bank.get("name", f"memory_{i}"),
                    "shape": memory_bank.get("dimensions", []),
                    "dtype": memory_bank.get("dtype", "float32"),
                    "data": memory_bank.get("data", []),
                    "agent_id": agent_id,
                    "memory_bank_index": i,
                    "tensor_type": "agent_memory",
                    "parent_entity": agent_id
                }
        
        # Convert agent state history to tensor
        if "state_history" in agent_data and agent_data["state_history"]:
            tensor_name = f"agent_{agent_id}_state_history"
            # Encode state history as categorical tensor
            state_encoded = [hash(state) % 1000 for state in agent_data["state_history"]]
            gguf_structure["tensors"][tensor_name] = {
                "name": "state_history",
                "shape": [len(state_encoded)],
                "dtype": "int32",
                "data": state_encoded,
                "agent_id": agent_id,
                "tensor_type": "agent_state_history",
                "metadata": {"original_states": agent_data["state_history"]}
            }
        
        # Convert cognitive parameters to tensor
        if "cognitive_parameters" in agent_data:
            tensor_name = f"agent_{agent_id}_cognitive_params"
            # Flatten cognitive parameters to numeric tensor
            param_values = []
            param_keys = []
            for key, value in agent_data["cognitive_parameters"].items():
                if isinstance(value, (int, float)):
                    param_values.append(float(value))
                    param_keys.append(key)
            
            if param_values:
                gguf_structure["tensors"][tensor_name] = {
                    "name": "cognitive_parameters",
                    "shape": [len(param_values)],
                    "dtype": "float32",
                    "data": param_values,
                    "agent_id": agent_id,
                    "tensor_type": "agent_cognitive_params",
                    "metadata": {"parameter_keys": param_keys}
                }
        
        # Store ALL agent parameters
        param_prefix = f"agent_{agent_id}"
        gguf_structure["parameters"][f"{param_prefix}.learning_rate"] = agent_data.get("learning_rate", 0.01)
        gguf_structure["parameters"][f"{param_prefix}.exploration_factor"] = agent_data.get("exploration_factor", 0.1)
        gguf_structure["parameters"][f"{param_prefix}.cognitive_grammar"] = agent_data.get("cognitive_grammar", "scheme")
        gguf_structure["parameters"][f"{param_prefix}.agent_type"] = agent_data.get("agent_type", "unknown")
        gguf_structure["parameters"][f"{param_prefix}.current_state"] = agent_data.get("state", "idle")
        
        # Add function list as parameter
        if "functions" in agent_data:
            gguf_structure["parameters"][f"{param_prefix}.functions"] = agent_data["functions"]
        
        # Add communication channels
        if "communication_channels" in agent_data:
            gguf_structure["parameters"][f"{param_prefix}.communication_channels"] = agent_data["communication_channels"]
    
    def _process_device_comprehensive_config(self, device_id: str, device_data: Dict[str, Any], gguf_structure: Dict[str, Any]) -> None:
        """Process comprehensive device configuration for GGUF export."""
        # Enhanced device state tensor
        if "tensor_dimensions" in device_data and device_data["tensor_dimensions"]:
            tensor_name = f"device_{device_id}_state"
            # Create more comprehensive device state tensor
            total_channels = device_data.get("sensor_channels", 0) + device_data.get("actuator_channels", 0)
            state_data = [0.0] * total_channels
            
            gguf_structure["tensors"][tensor_name] = {
                "name": f"{device_id}_state",
                "shape": device_data["tensor_dimensions"],
                "dtype": "float32",
                "data": state_data,
                "device_id": device_id,
                "tensor_type": "device_state",
                "parent_entity": device_id
            }
        
        # Convert sensor specifications to tensor
        if "sensor_specifications" in device_data and device_data["sensor_specifications"]:
            tensor_name = f"device_{device_id}_sensor_specs"
            # Create tensor from sensor specifications
            sensor_data = []
            for spec in device_data["sensor_specifications"]:
                # Extract numeric values from specifications
                for key, value in spec.items():
                    if isinstance(value, (int, float)):
                        sensor_data.append(float(value))
            
            if sensor_data:
                gguf_structure["tensors"][tensor_name] = {
                    "name": "sensor_specifications",
                    "shape": [len(sensor_data)],
                    "dtype": "float32",
                    "data": sensor_data,
                    "device_id": device_id,
                    "tensor_type": "device_sensor_specs",
                    "metadata": {"original_specs": device_data["sensor_specifications"]}
                }
        
        # Convert actuator specifications to tensor
        if "actuator_specifications" in device_data and device_data["actuator_specifications"]:
            tensor_name = f"device_{device_id}_actuator_specs"
            actuator_data = []
            for spec in device_data["actuator_specifications"]:
                for key, value in spec.items():
                    if isinstance(value, (int, float)):
                        actuator_data.append(float(value))
            
            if actuator_data:
                gguf_structure["tensors"][tensor_name] = {
                    "name": "actuator_specifications",
                    "shape": [len(actuator_data)],
                    "dtype": "float32",
                    "data": actuator_data,
                    "device_id": device_id,
                    "tensor_type": "device_actuator_specs",
                    "metadata": {"original_specs": device_data["actuator_specifications"]}
                }
        
        # Store ALL device parameters
        param_prefix = f"device_{device_id}"
        gguf_structure["parameters"][f"{param_prefix}.device_type"] = device_data.get("device_type", "unknown")
        gguf_structure["parameters"][f"{param_prefix}.degrees_of_freedom"] = device_data.get("degrees_of_freedom", 0)
        gguf_structure["parameters"][f"{param_prefix}.sensor_channels"] = device_data.get("sensor_channels", 0)
        gguf_structure["parameters"][f"{param_prefix}.actuator_channels"] = device_data.get("actuator_channels", 0)
        gguf_structure["parameters"][f"{param_prefix}.complexity"] = device_data.get("complexity", 0)
        gguf_structure["parameters"][f"{param_prefix}.modalities"] = device_data.get("modalities", [])
        gguf_structure["parameters"][f"{param_prefix}.firmware_version"] = device_data.get("firmware_version", "unknown")
        gguf_structure["parameters"][f"{param_prefix}.device_status"] = device_data.get("device_status", "operational")
        
        if "communication_protocols" in device_data:
            gguf_structure["parameters"][f"{param_prefix}.communication_protocols"] = device_data["communication_protocols"]
    
    def _process_hypergraph_tensors(self, node_id: str, node_data: Dict[str, Any], gguf_structure: Dict[str, Any]) -> None:
        """Process hypergraph node tensor fields for GGUF export."""
        if "tensor_fields" in node_data:
            for i, tensor_field in enumerate(node_data["tensor_fields"]):
                tensor_name = f"hypergraph_{node_id}_tensor_{i}"
                gguf_structure["tensors"][tensor_name] = {
                    "name": tensor_field.get("name", f"tensor_{i}"),
                    "shape": tensor_field.get("shape", []),
                    "dtype": tensor_field.get("dtype", "float32"),
                    "data": tensor_field.get("data", []),
                    "hypergraph_node_id": node_id,
                    "tensor_field_index": i,
                    "tensor_type": "hypergraph_tensor",
                    "parent_entity": node_id,
                    "metadata": tensor_field.get("metadata", {})
                }
        
        # Store hypergraph node parameters
        param_prefix = f"hypergraph_{node_id}"
        gguf_structure["parameters"][f"{param_prefix}.node_type"] = node_data.get("node_type", "unknown")
        gguf_structure["parameters"][f"{param_prefix}.processing_load"] = node_data.get("processing_load", 0.0)
        gguf_structure["parameters"][f"{param_prefix}.connections"] = node_data.get("connections", [])
        
        if "communication_latency" in node_data:
            gguf_structure["parameters"][f"{param_prefix}.communication_latency"] = node_data["communication_latency"]
    
    def _generate_p_system_membranes(self, gguf_structure: Dict[str, Any]) -> None:
        """Generate P-System membrane hierarchy for GGUF export."""
        membranes = {}
        
        # Process agent membranes
        for agent_id, agent_data in gguf_structure["agents"].items():
            if "membrane_structure" in agent_data:
                membrane_data = agent_data["membrane_structure"]
                if membrane_data.get("membrane_id"):
                    membranes[membrane_data["membrane_id"]] = {
                        "membrane_type": "agent_membrane",
                        "entity_id": agent_id,
                        "entity_type": "agent",
                        "parent_membrane": membrane_data.get("parent_membrane"),
                        "child_membranes": membrane_data.get("child_membranes", []),
                        "communication_channels": agent_data.get("communication_channels", [])
                    }
        
        # Process device membranes
        for device_id, device_data in gguf_structure["devices"].items():
            if "membrane_structure" in device_data:
                membrane_data = device_data["membrane_structure"]
                if membrane_data.get("membrane_id"):
                    membranes[membrane_data["membrane_id"]] = {
                        "membrane_type": membrane_data.get("membrane_type", "device_membrane"),
                        "entity_id": device_id,
                        "entity_type": "device",
                        "input_channels": membrane_data.get("input_channels", []),
                        "output_channels": membrane_data.get("output_channels", [])
                    }
        
        # Create lab-level root membrane
        membranes["lab_root"] = {
            "membrane_type": "lab_membrane",
            "entity_id": gguf_structure["metadata"]["lab_id"],
            "entity_type": "laboratory",
            "child_membranes": list(membranes.keys()),
            "total_agents": len(gguf_structure["agents"]),
            "total_devices": len(gguf_structure["devices"]),
            "total_tensors": len(gguf_structure["tensors"])
        }
        
        gguf_structure["membrane_system"] = membranes
    
    async def _write_gguf_file(self, gguf_data: Dict[str, Any]) -> str:
        """Write GGUF data to file."""
        gguf_path = self._gguf_path
        
        def _write_file():
            with open(gguf_path, "wb") as f:
                # Write GGUF header
                f.write(GGUF_MAGIC)  # Magic bytes
                f.write(struct.pack("<I", GGUF_VERSION))  # Version
                
                # Serialize data using msgpack
                data_bytes = msgpack.packb(gguf_data, use_bin_type=True)
                
                # Write data length and data
                f.write(struct.pack("<Q", len(data_bytes)))  # Data length (8 bytes)
                f.write(data_bytes)
        
        # Use executor to avoid blocking
        await self.hass.async_add_executor_job(_write_file)
        return gguf_path
    
    async def _read_gguf_file(self, gguf_path: str) -> Dict[str, Any]:
        """Read GGUF data from file."""
        
        def _read_file():
            with open(gguf_path, "rb") as f:
                # Read and verify header
                magic = f.read(4)
                if magic != GGUF_MAGIC:
                    raise ValueError(f"Invalid GGUF magic bytes: {magic}")
                
                version = struct.unpack("<I", f.read(4))[0]
                if version != GGUF_VERSION:
                    raise ValueError(f"Unsupported GGUF version: {version}")
                
                # Read data length and data
                data_length = struct.unpack("<Q", f.read(8))[0]
                data_bytes = f.read(data_length)
                
                # Deserialize using msgpack
                return msgpack.unpackb(data_bytes, raw=False)
        
        # Use executor to avoid blocking
        return await self.hass.async_add_executor_job(_read_file)
    
    def _parse_gguf_data(self, gguf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse comprehensive GGUF data back to lab data structure."""
        
        lab_data = {
            "timestamp": dt_util.now().timestamp(),
            "hypergraph_nodes": gguf_data.get("hypergraph", {}),
            "registered_devices": gguf_data.get("devices", {}),
            "active_agents": gguf_data.get("agents", {}),
            "tensor_fields": gguf_data.get("tensors", {}),
            "metrics": gguf_data.get("metrics", {}),
            "system_status": "imported_from_gguf",
            "metadata": gguf_data.get("metadata", {}),
            "membrane_system": gguf_data.get("membrane_system", {}),
            "environment_tensors": gguf_data.get("environment_tensors", {})
        }
        
        # Reconstruct agent memory banks and comprehensive state from tensors
        self._reconstruct_agent_comprehensive_state(gguf_data, lab_data)
        
        # Reconstruct device specifications from tensors
        self._reconstruct_device_comprehensive_config(gguf_data, lab_data)
        
        # Reconstruct hypergraph tensor fields
        self._reconstruct_hypergraph_tensors(gguf_data, lab_data)
        
        # Restore parameters to entities
        self._restore_comprehensive_parameters(gguf_data, lab_data)
        
        return lab_data
    
    def _reconstruct_agent_comprehensive_state(self, gguf_data: Dict[str, Any], lab_data: Dict[str, Any]) -> None:
        """Reconstruct comprehensive agent state from GGUF tensors."""
        for tensor_name, tensor_data in gguf_data.get("tensors", {}).items():
            if "agent_id" in tensor_data:
                agent_id = tensor_data["agent_id"]
                tensor_type = tensor_data.get("tensor_type", "unknown")
                
                if agent_id in lab_data["active_agents"]:
                    agent_data = lab_data["active_agents"][agent_id]
                    
                    if tensor_type == "agent_memory" and "memory_bank_index" in tensor_data:
                        # Reconstruct memory banks
                        memory_index = tensor_data["memory_bank_index"]
                        if "memory_banks" not in agent_data:
                            agent_data["memory_banks"] = []
                        
                        # Ensure memory_banks list is large enough
                        while len(agent_data["memory_banks"]) <= memory_index:
                            agent_data["memory_banks"].append({})
                        
                        # Set memory bank data
                        agent_data["memory_banks"][memory_index] = {
                            "name": tensor_data.get("name"),
                            "dimensions": tensor_data.get("shape", []),
                            "dtype": tensor_data.get("dtype", "float32"),
                            "data": tensor_data.get("data", [])
                        }
                    
                    elif tensor_type == "agent_state_history":
                        # Reconstruct state history
                        if "metadata" in tensor_data and "original_states" in tensor_data["metadata"]:
                            agent_data["state_history"] = tensor_data["metadata"]["original_states"]
                        else:
                            # Decode from tensor data if metadata not available
                            encoded_states = tensor_data.get("data", [])
                            agent_data["state_history"] = [f"state_{code}" for code in encoded_states]
                    
                    elif tensor_type == "agent_cognitive_params":
                        # Reconstruct cognitive parameters
                        param_values = tensor_data.get("data", [])
                        param_keys = tensor_data.get("metadata", {}).get("parameter_keys", [])
                        
                        if len(param_values) == len(param_keys):
                            agent_data["cognitive_parameters"] = dict(zip(param_keys, param_values))
                        else:
                            agent_data["cognitive_parameters"] = {f"param_{i}": val for i, val in enumerate(param_values)}
    
    def _reconstruct_device_comprehensive_config(self, gguf_data: Dict[str, Any], lab_data: Dict[str, Any]) -> None:
        """Reconstruct comprehensive device configuration from GGUF tensors."""
        for tensor_name, tensor_data in gguf_data.get("tensors", {}).items():
            if "device_id" in tensor_data:
                device_id = tensor_data["device_id"]
                tensor_type = tensor_data.get("tensor_type", "unknown")
                
                if device_id in lab_data["registered_devices"]:
                    device_data = lab_data["registered_devices"][device_id]
                    
                    if tensor_type == "device_sensor_specs":
                        # Reconstruct sensor specifications
                        if "metadata" in tensor_data and "original_specs" in tensor_data["metadata"]:
                            device_data["sensor_specifications"] = tensor_data["metadata"]["original_specs"]
                        else:
                            # Create basic specs from tensor data
                            sensor_values = tensor_data.get("data", [])
                            device_data["sensor_specifications"] = [
                                {"spec_value": val, "spec_index": i} for i, val in enumerate(sensor_values)
                            ]
                    
                    elif tensor_type == "device_actuator_specs":
                        # Reconstruct actuator specifications
                        if "metadata" in tensor_data and "original_specs" in tensor_data["metadata"]:
                            device_data["actuator_specifications"] = tensor_data["metadata"]["original_specs"]
                        else:
                            # Create basic specs from tensor data
                            actuator_values = tensor_data.get("data", [])
                            device_data["actuator_specifications"] = [
                                {"spec_value": val, "spec_index": i} for i, val in enumerate(actuator_values)
                            ]
                    
                    elif tensor_type == "device_state":
                        # Update device state tensor information
                        device_data["current_state_tensor"] = {
                            "shape": tensor_data.get("shape", []),
                            "data": tensor_data.get("data", [])
                        }
    
    def _reconstruct_hypergraph_tensors(self, gguf_data: Dict[str, Any], lab_data: Dict[str, Any]) -> None:
        """Reconstruct hypergraph tensor fields from GGUF data."""
        for tensor_name, tensor_data in gguf_data.get("tensors", {}).items():
            if "hypergraph_node_id" in tensor_data:
                node_id = tensor_data["hypergraph_node_id"]
                tensor_index = tensor_data.get("tensor_field_index", 0)
                
                if node_id in lab_data["hypergraph_nodes"]:
                    node_data = lab_data["hypergraph_nodes"][node_id]
                    
                    if "tensor_fields" not in node_data:
                        node_data["tensor_fields"] = []
                    
                    # Ensure tensor_fields list is large enough
                    while len(node_data["tensor_fields"]) <= tensor_index:
                        node_data["tensor_fields"].append({})
                    
                    # Set tensor field data
                    node_data["tensor_fields"][tensor_index] = {
                        "name": tensor_data.get("name"),
                        "shape": tensor_data.get("shape", []),
                        "dtype": tensor_data.get("dtype", "float32"),
                        "data": tensor_data.get("data", []),
                        "metadata": tensor_data.get("metadata", {})
                    }
    
    def _restore_comprehensive_parameters(self, gguf_data: Dict[str, Any], lab_data: Dict[str, Any]) -> None:
        """Restore comprehensive parameters to agents, devices, and hypergraph nodes."""
        for param_name, param_value in gguf_data.get("parameters", {}).items():
            if param_name.startswith("agent_"):
                self._restore_agent_parameter(param_name, param_value, lab_data)
            elif param_name.startswith("device_"):
                self._restore_device_parameter(param_name, param_value, lab_data)
            elif param_name.startswith("hypergraph_"):
                self._restore_hypergraph_parameter(param_name, param_value, lab_data)
    
    def _restore_agent_parameter(self, param_name: str, param_value: Any, lab_data: Dict[str, Any]) -> None:
        """Restore agent parameter from GGUF data."""
        parts = param_name.split(".")
        if len(parts) == 2:
            agent_ref, param_key = parts
            agent_id = agent_ref.replace("agent_", "")
            
            if agent_id in lab_data["active_agents"]:
                agent_data = lab_data["active_agents"][agent_id]
                # Map parameters back to agent structure
                if param_key in ["learning_rate", "exploration_factor", "cognitive_grammar", "agent_type", "current_state"]:
                    agent_data[param_key] = param_value
                elif param_key in ["functions", "communication_channels"]:
                    agent_data[param_key] = param_value
    
    def _restore_device_parameter(self, param_name: str, param_value: Any, lab_data: Dict[str, Any]) -> None:
        """Restore device parameter from GGUF data."""
        parts = param_name.split(".")
        if len(parts) == 2:
            device_ref, param_key = parts
            device_id = device_ref.replace("device_", "")
            
            if device_id in lab_data["registered_devices"]:
                device_data = lab_data["registered_devices"][device_id]
                # Map parameters back to device structure
                if param_key in ["device_type", "degrees_of_freedom", "sensor_channels", "actuator_channels", 
                                "complexity", "firmware_version", "device_status"]:
                    device_data[param_key] = param_value
                elif param_key in ["modalities", "communication_protocols"]:
                    device_data[param_key] = param_value
    
    def _restore_hypergraph_parameter(self, param_name: str, param_value: Any, lab_data: Dict[str, Any]) -> None:
        """Restore hypergraph node parameter from GGUF data."""
        parts = param_name.split(".")
        if len(parts) == 2:
            node_ref, param_key = parts
            node_id = node_ref.replace("hypergraph_", "")
            
            if node_id in lab_data["hypergraph_nodes"]:
                node_data = lab_data["hypergraph_nodes"][node_id]
                # Map parameters back to hypergraph node structure
                if param_key in ["node_type", "processing_load"]:
                    node_data[param_key] = param_value
                elif param_key in ["connections", "communication_latency"]:
                    node_data[param_key] = param_value
    
    async def create_agent_snapshot(self, agent_id: str, agent_data: Dict[str, Any]) -> str:
        """Create a GGUF snapshot of a specific agent."""
        snapshot_data = {
            "metadata": {
                "type": "agent_snapshot",
                "agent_id": agent_id,
                "timestamp": dt_util.now().isoformat(),
                "lab_id": self.config_entry.entry_id
            },
            "agent": agent_data,
            "tensors": {},
            "parameters": {}
        }
        
        # Convert agent memory banks to tensors
        if "memory_banks" in agent_data:
            for i, memory_bank in enumerate(agent_data["memory_banks"]):
                tensor_name = f"memory_{i}"
                snapshot_data["tensors"][tensor_name] = {
                    "name": memory_bank.get("name", f"memory_{i}"),
                    "shape": memory_bank.get("dimensions", []),
                    "dtype": memory_bank.get("dtype", "float32"),
                    "data": memory_bank.get("data", [])
                }
        
        # Store agent parameters
        snapshot_data["parameters"]["learning_rate"] = agent_data.get("learning_rate", 0.01)
        snapshot_data["parameters"]["exploration_factor"] = agent_data.get("exploration_factor", 0.1)
        
        # Write snapshot file
        snapshot_path = str(Path(self._gguf_path).parent / f"agent_{agent_id}_snapshot.gguf")
        
        def _write_snapshot():
            with open(snapshot_path, "wb") as f:
                f.write(GGUF_MAGIC)
                f.write(struct.pack("<I", GGUF_VERSION))
                
                data_bytes = msgpack.packb(snapshot_data, use_bin_type=True)
                f.write(struct.pack("<Q", len(data_bytes)))
                f.write(data_bytes)
        
        await self.hass.async_add_executor_job(_write_snapshot)
        
        _LOGGER.info("Created agent snapshot: %s", snapshot_path)
        return snapshot_path
    
    async def restore_agent_from_snapshot(self, snapshot_path: str) -> Dict[str, Any]:
        """Restore an agent from a GGUF snapshot."""
        snapshot_data = await self._read_gguf_file(snapshot_path)
        
        if snapshot_data.get("metadata", {}).get("type") != "agent_snapshot":
            raise ValueError("Invalid agent snapshot format")
        
        agent_data = snapshot_data.get("agent", {})
        
        # Reconstruct memory banks from tensors
        if "tensors" in snapshot_data:
            agent_data["memory_banks"] = []
            for tensor_name, tensor_data in snapshot_data["tensors"].items():
                memory_bank = {
                    "name": tensor_data.get("name"),
                    "dimensions": tensor_data.get("shape", []),
                    "dtype": tensor_data.get("dtype", "float32"),
                    "data": tensor_data.get("data", [])
                }
                agent_data["memory_banks"].append(memory_bank)
        
        # Restore parameters
        parameters = snapshot_data.get("parameters", {})
        agent_data.update(parameters)
        
        _LOGGER.info("Restored agent from snapshot: %s", snapshot_path)
        return agent_data
    
    def get_gguf_path(self) -> str:
        """Get the current GGUF file path."""
        return self._gguf_path
    
    async def export_p_system_membranes(self, membrane_hierarchy: Dict[str, Any]) -> str:
        """Export P-System membrane hierarchy to dedicated GGUF file."""
        try:
            # Prepare P-System specific GGUF structure
            p_system_data = {
                "metadata": {
                    "type": "p_system_membranes",
                    "version": "2.0.0",
                    "lab_id": self.config_entry.entry_id,
                    "export_timestamp": dt_util.now().isoformat(),
                    "format": "marduk_p_system_gguf"
                },
                "membrane_hierarchy": membrane_hierarchy,
                "communication_rules": self._extract_communication_rules(membrane_hierarchy),
                "membrane_tensors": self._generate_membrane_tensors(membrane_hierarchy)
            }
            
            # Write P-System GGUF file
            p_system_path = str(Path(self._gguf_path).parent / "p_system_membranes.gguf")
            
            def _write_p_system():
                with open(p_system_path, "wb") as f:
                    f.write(GGUF_MAGIC)
                    f.write(struct.pack("<I", GGUF_VERSION))
                    
                    data_bytes = msgpack.packb(p_system_data, use_bin_type=True)
                    f.write(struct.pack("<Q", len(data_bytes)))
                    f.write(data_bytes)
            
            await self.hass.async_add_executor_job(_write_p_system)
            
            _LOGGER.info("Exported P-System membranes to: %s", p_system_path)
            return p_system_path
            
        except Exception as err:
            _LOGGER.error("Failed to export P-System membranes: %s", err)
            raise
    
    def _extract_communication_rules(self, membrane_hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract communication rules from membrane hierarchy."""
        rules = {}
        
        for membrane_id, membrane_data in membrane_hierarchy.items():
            membrane_type = membrane_data.get("membrane_type", "unknown")
            
            if membrane_type == "agent_membrane":
                # Agent communication rules
                rules[membrane_id] = {
                    "rule_type": "agent_communication",
                    "input_channels": membrane_data.get("communication_channels", []),
                    "output_channels": membrane_data.get("communication_channels", []),
                    "parent_membrane": membrane_data.get("parent_membrane"),
                    "child_membranes": membrane_data.get("child_membranes", [])
                }
            elif membrane_type == "device_membrane":
                # Device communication rules
                rules[membrane_id] = {
                    "rule_type": "device_communication",
                    "input_channels": membrane_data.get("input_channels", []),
                    "output_channels": membrane_data.get("output_channels", [])
                }
            elif membrane_type == "lab_membrane":
                # Lab-level communication rules
                rules[membrane_id] = {
                    "rule_type": "lab_coordination",
                    "child_membranes": membrane_data.get("child_membranes", []),
                    "total_entities": {
                        "agents": membrane_data.get("total_agents", 0),
                        "devices": membrane_data.get("total_devices", 0)
                    }
                }
        
        return rules
    
    def _generate_membrane_tensors(self, membrane_hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tensor representations for membrane communication."""
        membrane_tensors = {}
        
        for membrane_id, membrane_data in membrane_hierarchy.items():
            # Create communication matrix tensor for each membrane
            input_channels = membrane_data.get("input_channels", [])
            output_channels = membrane_data.get("output_channels", [])
            
            if input_channels and output_channels:
                # Create communication matrix
                matrix_size = max(len(input_channels), len(output_channels))
                if matrix_size > 0:
                    membrane_tensors[f"membrane_{membrane_id}_comm_matrix"] = {
                        "name": f"communication_matrix_{membrane_id}",
                        "shape": [matrix_size, matrix_size],
                        "dtype": "float32",
                        "data": [0.0] * (matrix_size * matrix_size),
                        "membrane_id": membrane_id,
                        "tensor_type": "membrane_communication",
                        "metadata": {
                            "input_channels": input_channels,
                            "output_channels": output_channels
                        }
                    }
            
            # Create membrane state tensor
            child_membranes = membrane_data.get("child_membranes", [])
            if child_membranes or membrane_data.get("entity_id"):
                state_size = max(len(child_membranes), 1)
                membrane_tensors[f"membrane_{membrane_id}_state"] = {
                    "name": f"state_{membrane_id}",
                    "shape": [state_size],
                    "dtype": "float32",
                    "data": [1.0] * state_size,  # Initialize as active
                    "membrane_id": membrane_id,
                    "tensor_type": "membrane_state",
                    "metadata": {
                        "membrane_type": membrane_data.get("membrane_type"),
                        "entity_type": membrane_data.get("entity_type")
                    }
                }
        
        return membrane_tensors
    
    async def create_comprehensive_snapshot(self, lab_data: Dict[str, Any]) -> str:
        """Create a comprehensive snapshot with all agent states, device configs, and P-System membranes."""
        try:
            # Prepare comprehensive snapshot data
            snapshot_data = {
                "metadata": {
                    "type": "comprehensive_snapshot",
                    "version": "2.0.0",
                    "lab_id": self.config_entry.entry_id,
                    "snapshot_timestamp": dt_util.now().isoformat(),
                    "format": "marduk_comprehensive_snapshot_gguf"
                },
                "complete_lab_state": self._prepare_gguf_data(lab_data),
                "integrity_checksum": self._calculate_data_checksum(lab_data),
                "export_statistics": {
                    "total_tensors": 0,
                    "total_parameters": 0,
                    "total_membranes": 0
                }
            }
            
            # Update statistics
            complete_state = snapshot_data["complete_lab_state"]
            snapshot_data["export_statistics"] = {
                "total_tensors": len(complete_state.get("tensors", {})),
                "total_parameters": len(complete_state.get("parameters", {})),
                "total_membranes": len(complete_state.get("membrane_system", {}))
            }
            
            # Write comprehensive snapshot
            snapshot_path = str(Path(self._gguf_path).parent / f"comprehensive_snapshot_{int(dt_util.now().timestamp())}.gguf")
            
            def _write_comprehensive_snapshot():
                with open(snapshot_path, "wb") as f:
                    f.write(GGUF_MAGIC)
                    f.write(struct.pack("<I", GGUF_VERSION))
                    
                    data_bytes = msgpack.packb(snapshot_data, use_bin_type=True)
                    f.write(struct.pack("<Q", len(data_bytes)))
                    f.write(data_bytes)
            
            await self.hass.async_add_executor_job(_write_comprehensive_snapshot)
            
            _LOGGER.info("Created comprehensive snapshot: %s", snapshot_path)
            return snapshot_path
            
        except Exception as err:
            _LOGGER.error("Failed to create comprehensive snapshot: %s", err)
            raise
    
    def _calculate_data_checksum(self, lab_data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity verification."""
        import hashlib
        
        # Create a deterministic representation of the data
        data_string = json.dumps(lab_data, sort_keys=True, default=str)
        return hashlib.md5(data_string.encode()).hexdigest()
    
    async def validate_gguf_integrity(self, gguf_path: str) -> bool:
        """Validate the integrity of a GGUF file."""
        try:
            # Read and parse the GGUF file
            gguf_data = await self._read_gguf_file(gguf_path)
            
            # Check for required metadata
            metadata = gguf_data.get("metadata", {})
            if not metadata.get("lab_id") or not metadata.get("version"):
                _LOGGER.error("GGUF file missing required metadata")
                return False
            
            # Validate P-System compatibility if claimed
            if metadata.get("p_system_compatible"):
                if "membrane_system" not in gguf_data:
                    _LOGGER.error("GGUF file claims P-System compatibility but lacks membrane system")
                    return False
            
            # Validate tensor consistency
            tensors = gguf_data.get("tensors", {})
            for tensor_name, tensor_data in tensors.items():
                if not all(key in tensor_data for key in ["name", "shape", "dtype"]):
                    _LOGGER.error("Tensor %s missing required fields", tensor_name)
                    return False
            
            _LOGGER.info("GGUF file validation successful: %s", gguf_path)
            return True
            
        except Exception as err:
            _LOGGER.error("GGUF validation failed: %s", err)
            return False