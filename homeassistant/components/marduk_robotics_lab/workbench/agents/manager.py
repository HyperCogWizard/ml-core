"""Agent manager for workbench components."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...models import AgentKernel
from ..hypergraph import HypergraphManager

_LOGGER = logging.getLogger(__name__)


class AgentManager:
    """Manages agent kernels in the workbench."""
    
    def __init__(self, hypergraph_manager: HypergraphManager) -> None:
        """Initialize agent manager."""
        self._hypergraph_manager = hypergraph_manager
        self._agent_kernels: Dict[str, AgentKernel] = {}
    
    def register_agent_kernel(self, agent: AgentKernel) -> bool:
        """Register an agent kernel."""
        if agent.kernel_id in self._agent_kernels:
            _LOGGER.warning("Agent kernel %s already exists, updating", agent.kernel_id)
        
        self._agent_kernels[agent.kernel_id] = agent
        
        # Create hypergraph node for the agent
        self._create_agent_node(agent)
        
        _LOGGER.info("Registered agent kernel %s (%s)", agent.kernel_id, agent.agent_type)
        return True
    
    def _create_agent_node(self, agent: AgentKernel) -> None:
        """Create hypergraph node for agent kernel."""
        from ..hypergraph import WorkbenchNode
        
        node = WorkbenchNode(
            node_id=agent.kernel_id,
            node_type="agent",
            component_class="AgentKernel",
            properties={
                "agent_type": agent.agent_type,
                "state": agent.state,
                "cognitive_grammar": agent.cognitive_grammar,
                "function_count": len(agent.functions),
                "memory_bank_count": len(agent.memory_banks),
                "learning_rate": agent.learning_rate,
                "exploration_factor": agent.exploration_factor,
            }
        )
        
        self._hypergraph_manager.add_node(node)
    
    def get_agent_kernel(self, kernel_id: str) -> Optional[AgentKernel]:
        """Get agent kernel by ID."""
        return self._agent_kernels.get(kernel_id)
    
    def get_all_agent_kernels(self) -> Dict[str, AgentKernel]:
        """Get all agent kernels."""
        return self._agent_kernels.copy()
    
    def update_agent_state(self, kernel_id: str, new_state: str) -> bool:
        """Update agent kernel state."""
        if kernel_id not in self._agent_kernels:
            _LOGGER.error("Agent kernel %s not found", kernel_id)
            return False
        
        self._agent_kernels[kernel_id].state = new_state
        
        # Update hypergraph node
        node = self._hypergraph_manager.get_node(kernel_id)
        if node:
            node.properties["state"] = new_state
        
        _LOGGER.debug("Updated agent %s state to %s", kernel_id, new_state)
        return True
    
    def remove_agent_kernel(self, kernel_id: str) -> bool:
        """Remove an agent kernel."""
        if kernel_id not in self._agent_kernels:
            _LOGGER.warning("Agent kernel %s not found", kernel_id)
            return False
        
        # Remove from hypergraph
        self._hypergraph_manager.remove_node(kernel_id)
        
        # Remove agent
        del self._agent_kernels[kernel_id]
        
        _LOGGER.info("Removed agent kernel %s", kernel_id)
        return True
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about agent kernels."""
        total_agents = len(self._agent_kernels)
        
        agent_type_counts = {}
        state_counts = {}
        total_functions = 0
        total_memory_banks = 0
        
        for agent in self._agent_kernels.values():
            # Count by type
            agent_type = agent.agent_type
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1
            
            # Count by state
            state = agent.state
            state_counts[state] = state_counts.get(state, 0) + 1
            
            # Accumulate totals
            total_functions += len(agent.functions)
            total_memory_banks += len(agent.memory_banks)
        
        return {
            "total_agents": total_agents,
            "agent_type_counts": agent_type_counts,
            "state_counts": state_counts,
            "total_functions": total_functions,
            "total_memory_banks": total_memory_banks,
        }