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
        """Prepare lab data for GGUF serialization."""
        
        # Create GGUF-compatible structure
        gguf_structure = {
            "metadata": {
                "version": "1.0.0",
                "lab_id": self.config_entry.entry_id,
                "export_timestamp": dt_util.now().isoformat(),
                "format": "marduk_robotics_lab_gguf"
            },
            "tensors": {},
            "parameters": {},
            "hypergraph": lab_data.get("hypergraph_nodes", {}),
            "devices": lab_data.get("registered_devices", {}),
            "agents": lab_data.get("active_agents", {}),
            "metrics": lab_data.get("metrics", {}),
        }
        
        # Convert agent memory banks to tensors
        for agent_id, agent_data in gguf_structure["agents"].items():
            if "memory_banks" in agent_data:
                # Convert memory banks to tensor format
                for i, memory_bank in enumerate(agent_data["memory_banks"]):
                    tensor_name = f"agent_{agent_id}_memory_{i}"
                    gguf_structure["tensors"][tensor_name] = {
                        "name": memory_bank.get("name", f"memory_{i}"),
                        "shape": memory_bank.get("dimensions", []),
                        "dtype": memory_bank.get("dtype", "float32"),
                        "data": memory_bank.get("data", []),
                        "agent_id": agent_id,
                        "memory_bank_index": i
                    }
        
        # Convert device tensor fields
        for device_id, device_data in gguf_structure["devices"].items():
            if "tensor_dimensions" in device_data and device_data["tensor_dimensions"]:
                tensor_name = f"device_{device_id}_state"
                gguf_structure["tensors"][tensor_name] = {
                    "name": f"{device_id}_state",
                    "shape": device_data["tensor_dimensions"],
                    "dtype": "float32",
                    "data": [0.0] * (device_data.get("sensor_channels", 0) + device_data.get("actuator_channels", 0)),
                    "device_id": device_id,
                    "tensor_type": "device_state"
                }
        
        # Store cognitive parameters
        for agent_id, agent_data in gguf_structure["agents"].items():
            param_prefix = f"agent_{agent_id}"
            gguf_structure["parameters"][f"{param_prefix}.learning_rate"] = agent_data.get("learning_rate", 0.01)
            gguf_structure["parameters"][f"{param_prefix}.exploration_factor"] = agent_data.get("exploration_factor", 0.1)
            gguf_structure["parameters"][f"{param_prefix}.cognitive_grammar"] = agent_data.get("cognitive_grammar", "scheme")
        
        return gguf_structure
    
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
        """Parse GGUF data back to lab data structure."""
        
        lab_data = {
            "timestamp": dt_util.now().timestamp(),
            "hypergraph_nodes": gguf_data.get("hypergraph", {}),
            "registered_devices": gguf_data.get("devices", {}),
            "active_agents": gguf_data.get("agents", {}),
            "tensor_fields": gguf_data.get("tensors", {}),
            "metrics": gguf_data.get("metrics", {}),
            "system_status": "imported_from_gguf",
            "metadata": gguf_data.get("metadata", {})
        }
        
        # Reconstruct agent memory banks from tensors
        for tensor_name, tensor_data in gguf_data.get("tensors", {}).items():
            if "agent_id" in tensor_data and "memory_bank_index" in tensor_data:
                agent_id = tensor_data["agent_id"]
                memory_index = tensor_data["memory_bank_index"]
                
                if agent_id in lab_data["active_agents"]:
                    agent_data = lab_data["active_agents"][agent_id]
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
        
        # Restore parameters to agents
        for param_name, param_value in gguf_data.get("parameters", {}).items():
            if param_name.startswith("agent_"):
                parts = param_name.split(".")
                if len(parts) == 2:
                    agent_ref, param_key = parts
                    agent_id = agent_ref.replace("agent_", "")
                    
                    if agent_id in lab_data["active_agents"]:
                        lab_data["active_agents"][agent_id][param_key] = param_value
        
        return lab_data
    
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