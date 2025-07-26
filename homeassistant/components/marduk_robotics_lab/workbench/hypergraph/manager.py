"""Hypergraph manager for workbench components."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

from .node import WorkbenchNode
from .edge import WorkbenchEdge, EdgeType

_LOGGER = logging.getLogger(__name__)


class HypergraphManager:
    """Manages the hypergraph structure of workbench components."""
    
    def __init__(self) -> None:
        """Initialize hypergraph manager."""
        self._nodes: Dict[str, WorkbenchNode] = {}
        self._edges: Dict[str, WorkbenchEdge] = {}
        self._node_edges: Dict[str, Set[str]] = defaultdict(set)  # node_id -> edge_ids
        self._component_index: Dict[str, Set[str]] = defaultdict(set)  # component_class -> node_ids
        
    def add_node(self, node: WorkbenchNode) -> bool:
        """Add a node to the hypergraph."""
        if node.node_id in self._nodes:
            _LOGGER.warning("Node %s already exists, updating", node.node_id)
        
        self._nodes[node.node_id] = node
        self._component_index[node.component_class].add(node.node_id)
        
        _LOGGER.debug("Added node %s of type %s", node.node_id, node.node_type)
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges from the hypergraph."""
        if node_id not in self._nodes:
            _LOGGER.warning("Node %s not found", node_id)
            return False
        
        node = self._nodes[node_id]
        
        # Remove all edges connected to this node
        edges_to_remove = list(self._node_edges[node_id])
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
        
        # Remove from component index
        self._component_index[node.component_class].discard(node_id)
        
        # Remove the node
        del self._nodes[node_id]
        del self._node_edges[node_id]
        
        _LOGGER.debug("Removed node %s", node_id)
        return True
    
    def add_edge(self, edge: WorkbenchEdge) -> bool:
        """Add an edge to the hypergraph."""
        # Validate that source and target nodes exist
        if edge.source_node_id not in self._nodes:
            _LOGGER.error("Source node %s not found", edge.source_node_id)
            return False
        
        if edge.target_node_id not in self._nodes:
            _LOGGER.error("Target node %s not found", edge.target_node_id)
            return False
        
        if edge.edge_id in self._edges:
            _LOGGER.warning("Edge %s already exists, updating", edge.edge_id)
        
        self._edges[edge.edge_id] = edge
        self._node_edges[edge.source_node_id].add(edge.edge_id)
        self._node_edges[edge.target_node_id].add(edge.edge_id)
        
        # Update node connections
        source_node = self._nodes[edge.source_node_id]
        target_node = self._nodes[edge.target_node_id]
        source_node.add_connection(edge.target_node_id)
        
        _LOGGER.debug("Added edge %s from %s to %s", 
                     edge.edge_id, edge.source_node_id, edge.target_node_id)
        return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the hypergraph."""
        if edge_id not in self._edges:
            _LOGGER.warning("Edge %s not found", edge_id)
            return False
        
        edge = self._edges[edge_id]
        
        # Update node connections
        if edge.source_node_id in self._nodes:
            source_node = self._nodes[edge.source_node_id]
            source_node.remove_connection(edge.target_node_id)
        
        # Remove from node edge tracking
        self._node_edges[edge.source_node_id].discard(edge_id)
        self._node_edges[edge.target_node_id].discard(edge_id)
        
        # Remove the edge
        del self._edges[edge_id]
        
        _LOGGER.debug("Removed edge %s", edge_id)
        return True
    
    def get_node(self, node_id: str) -> Optional[WorkbenchNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[WorkbenchEdge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[WorkbenchNode]:
        """Get all nodes of a specific type."""
        return [node for node in self._nodes.values() if node.node_type == node_type]
    
    def get_nodes_by_component_class(self, component_class: str) -> List[WorkbenchNode]:
        """Get all nodes of a specific component class."""
        node_ids = self._component_index.get(component_class, set())
        return [self._nodes[node_id] for node_id in node_ids if node_id in self._nodes]
    
    def get_connected_nodes(self, node_id: str) -> List[WorkbenchNode]:
        """Get all nodes connected to the given node."""
        if node_id not in self._nodes:
            return []
        
        connected_ids = self._nodes[node_id].connections
        return [self._nodes[conn_id] for conn_id in connected_ids if conn_id in self._nodes]
    
    def get_node_edges(self, node_id: str) -> List[WorkbenchEdge]:
        """Get all edges connected to a node."""
        if node_id not in self._node_edges:
            return []
        
        edge_ids = self._node_edges[node_id]
        return [self._edges[edge_id] for edge_id in edge_ids if edge_id in self._edges]
    
    def find_path(self, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between two nodes (BFS)."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return []
        
        if source_id == target_id:
            return [source_id]
        
        queue = [(source_id, [source_id])]
        visited = {source_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            for connected_id in self._nodes[current_id].connections:
                if connected_id == target_id:
                    return path + [connected_id]
                
                if connected_id not in visited:
                    visited.add(connected_id)
                    queue.append((connected_id, path + [connected_id]))
        
        return []  # No path found
    
    def get_tensor_flow_paths(self, source_id: str, target_id: str) -> List[WorkbenchEdge]:
        """Get all tensor flow edges in the path between two nodes."""
        path = self.find_path(source_id, target_id)
        if len(path) < 2:
            return []
        
        tensor_edges = []
        for i in range(len(path) - 1):
            source_node_id = path[i]
            target_node_id = path[i + 1]
            
            # Find edge between consecutive nodes that has tensor flow
            for edge in self.get_node_edges(source_node_id):
                if (edge.target_node_id == target_node_id and 
                    edge.edge_type == EdgeType.TENSOR_COUPLING):
                    tensor_edges.append(edge)
                    break
        
        return tensor_edges
    
    def get_hypergraph_stats(self) -> Dict[str, Any]:
        """Get statistics about the hypergraph."""
        total_dimensions = sum(node.get_total_tensor_dimensions() for node in self._nodes.values())
        
        component_counts = {}
        for component_class, node_ids in self._component_index.items():
            component_counts[component_class] = len(node_ids)
        
        edge_type_counts = {}
        for edge in self._edges.values():
            edge_type = edge.edge_type.value
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "total_tensor_dimensions": total_dimensions,
            "component_counts": component_counts,
            "edge_type_counts": edge_type_counts,
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the entire hypergraph structure."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "edges": {edge_id: edge.to_dict() for edge_id, edge in self._edges.items()},
            "stats": self.get_hypergraph_stats(),
        }
    
    def clear(self) -> None:
        """Clear all nodes and edges from the hypergraph."""
        self._nodes.clear()
        self._edges.clear()
        self._node_edges.clear()
        self._component_index.clear()
        _LOGGER.debug("Cleared hypergraph")