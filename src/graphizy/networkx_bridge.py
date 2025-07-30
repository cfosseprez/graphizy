"""
NetworkX bridge for graphizy - Convert igraph to NetworkX for advanced analysis
"""
import logging
from typing import Any, Dict, Optional, Union
import numpy as np

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from graphizy.exceptions import DependencyError, GraphCreationError


def to_networkx(igraph_graph: Any,
                copy_vertex_attrs: bool = True,
                copy_edge_attrs: bool = True,
                directed: Optional[bool] = None) -> 'nx.Graph':
    """
    Convert igraph Graph to NetworkX Graph.

    Args:
        igraph_graph: igraph Graph object
        copy_vertex_attrs: Whether to copy vertex attributes
        copy_edge_attrs: Whether to copy edge attributes
        directed: Force directed/undirected. If None, preserves original

    Returns:
        NetworkX Graph object

    Examples:
        >>> # Basic conversion
        >>> nx_graph = to_networkx(igraph_graph)
        >>>
        >>> # Use NetworkX algorithms
        >>> centrality = nx.betweenness_centrality(nx_graph)
        >>> communities = nx.community.greedy_modularity_communities(nx_graph)
    """
    if not NETWORKX_AVAILABLE:
        raise DependencyError(
            "NetworkX not available. Install with: pip install networkx"
        )

    if igraph_graph is None:
        raise GraphCreationError("Cannot convert None graph")

    # Determine graph type
    if directed is None:
        directed = igraph_graph.is_directed()

    # Create NetworkX graph
    if directed:
        nx_graph = nx.DiGraph()
    else:
        nx_graph = nx.Graph()

    # Add vertices with attributes
    for vertex in igraph_graph.vs:
        node_id = vertex["id"] if "id" in vertex.attributes() else vertex.index

        if copy_vertex_attrs:
            attrs = {k: v for k, v in vertex.attributes().items()}
            nx_graph.add_node(node_id, **attrs)
        else:
            nx_graph.add_node(node_id)

    # Add edges with attributes
    for edge in igraph_graph.es:
        source_id = igraph_graph.vs[edge.source]["id"] if "id" in igraph_graph.vs.attributes() else edge.source
        target_id = igraph_graph.vs[edge.target]["id"] if "id" in igraph_graph.vs.attributes() else edge.target

        if copy_edge_attrs:
            attrs = {k: v for k, v in edge.attributes().items()}
            nx_graph.add_edge(source_id, target_id, **attrs)
        else:
            nx_graph.add_edge(source_id, target_id)

    return nx_graph


def from_networkx(nx_graph: 'nx.Graph',
                  copy_vertex_attrs: bool = True,
                  copy_edge_attrs: bool = True) -> Any:
    """
    Convert NetworkX Graph to igraph Graph.

    Args:
        nx_graph: NetworkX Graph object
        copy_vertex_attrs: Whether to copy node attributes
        copy_edge_attrs: Whether to copy edge attributes

    Returns:
        igraph Graph object
    """
    if not NETWORKX_AVAILABLE:
        raise DependencyError("NetworkX not available")

    import igraph as ig

    # Create vertex list with proper ID mapping
    nodes = list(nx_graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Create edge list with index mapping
    edges = [(node_to_index[u], node_to_index[v]) for u, v in nx_graph.edges()]

    # Create igraph
    igraph_graph = ig.Graph(
        n=len(nodes),
        edges=edges,
        directed=nx_graph.is_directed()
    )

    # Set vertex attributes
    if copy_vertex_attrs:
        # Always set 'id' attribute to original node identifier
        igraph_graph.vs["id"] = nodes

        # Copy other attributes
        for attr in nx_graph.nodes[nodes[0]].keys() if nodes else []:
            values = [nx_graph.nodes[node].get(attr) for node in nodes]
            igraph_graph.vs[attr] = values
    else:
        igraph_graph.vs["id"] = nodes

    # Set edge attributes
    if copy_edge_attrs and nx_graph.edges():
        first_edge = list(nx_graph.edges(data=True))[0]
        if len(first_edge) > 2:  # Has attributes
            edge_attrs = first_edge[2].keys()
            for attr in edge_attrs:
                values = [nx_graph.edges[u, v].get(attr) for u, v in nx_graph.edges()]
                igraph_graph.es[attr] = values

    return igraph_graph


class NetworkXAnalyzer:
    """
    High-level interface for NetworkX analysis of graphizy graphs.
    """

    def __init__(self, grapher):
        """Initialize with a Graphing instance."""
        if not NETWORKX_AVAILABLE:
            raise DependencyError("NetworkX not available")
        self.grapher = grapher
        self._nx_cache = {}

    def get_networkx(self, graph_type: str = None, igraph_graph: Any = None,
                     use_cache: bool = True) -> 'nx.Graph':
        """
        Get NetworkX version of a graph.

        Args:
            graph_type: Type from current graphs, or None for manual graph
            igraph_graph: Manual igraph to convert, or None to use graph_type
            use_cache: Whether to cache conversions
        """
        if igraph_graph is not None:
            # Manual graph provided
            return to_networkx(igraph_graph)

        if graph_type is None:
            raise ValueError("Must provide either graph_type or igraph_graph")

        # Check cache
        if use_cache and graph_type in self._nx_cache:
            return self._nx_cache[graph_type]

        # Get from current graphs
        current_graphs = self.grapher.get_current_graphs()
        if graph_type not in current_graphs:
            raise ValueError(f"Graph type '{graph_type}' not in current graphs")

        igraph_graph = current_graphs[graph_type]
        if igraph_graph is None:
            raise ValueError(f"Graph type '{graph_type}' is None")

        nx_graph = to_networkx(igraph_graph)

        if use_cache:
            self._nx_cache[graph_type] = nx_graph

        return nx_graph

    def clear_cache(self):
        """Clear conversion cache."""
        self._nx_cache.clear()

    def analyze(self, graph_type: str = None, igraph_graph: Any = None) -> Dict[str, Any]:
        """
        Perform comprehensive NetworkX analysis.

        Returns:
            Dict with NetworkX-specific metrics
        """
        nx_graph = self.get_networkx(graph_type, igraph_graph)

        analysis = {}

        # Basic metrics
        analysis['nodes'] = nx_graph.number_of_nodes()
        analysis['edges'] = nx_graph.number_of_edges()
        analysis['density'] = nx.density(nx_graph)

        # Connectivity
        if nx_graph.is_directed():
            analysis['strongly_connected'] = nx.is_strongly_connected(nx_graph)
            analysis['weakly_connected'] = nx.is_weakly_connected(nx_graph)
        else:
            analysis['connected'] = nx.is_connected(nx_graph)

        # Centrality measures (sample nodes if large)
        if nx_graph.number_of_nodes() <= 1000:
            analysis['betweenness_centrality'] = nx.betweenness_centrality(nx_graph)
            analysis['closeness_centrality'] = nx.closeness_centrality(nx_graph)
            analysis['degree_centrality'] = nx.degree_centrality(nx_graph)
            if not nx_graph.is_directed():
                analysis['eigenvector_centrality'] = nx.eigenvector_centrality(nx_graph, max_iter=1000)

        # Community detection (for undirected graphs)
        if not nx_graph.is_directed() and nx_graph.number_of_nodes() >= 3:
            try:
                communities = nx.community.greedy_modularity_communities(nx_graph)
                analysis['num_communities'] = len(communities)
                analysis['modularity'] = nx.community.modularity(nx_graph, communities)
            except:
                pass

        # Path metrics (if connected and reasonable size)
        if nx_graph.number_of_nodes() <= 500:
            try:
                if not nx_graph.is_directed() and nx.is_connected(nx_graph):
                    analysis['average_shortest_path_length'] = nx.average_shortest_path_length(nx_graph)
                    analysis['diameter'] = nx.diameter(nx_graph)
                elif nx_graph.is_directed() and nx.is_strongly_connected(nx_graph):
                    analysis['average_shortest_path_length'] = nx.average_shortest_path_length(nx_graph)
                    analysis['diameter'] = nx.diameter(nx_graph)
            except:
                pass

        return analysis