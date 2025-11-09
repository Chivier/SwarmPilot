"""Graph implementation for managing execution flow.

A Graph represents a directed acyclic graph (DAG) of Nodes with routing logic
to control task flow between nodes. The Graph manages:
- Node lifecycle and relationships
- Routing configuration between nodes
- Fanout prediction via predictor client
- Optional fanout overrides for experiments

Typical usage example:

    graph = Graph(graph_id="my-graph", predictor_url="http://localhost:8001")

    # Add nodes
    node1 = Node(model_id="gpt-4", predictor_url="...", ...)
    graph.add_node(node1, entrance=True)

    node2 = Node(model_id="gpt-3.5", predictor_url="...", ...)
    graph.add_node(node2, exit=True)

    # Add routing
    graph.add_router(outbound_node=node1, inbound_node=node2)

    # Execute graph
    graph.exec()
"""

from typing import Dict, Optional, Set

from clients.predictor_client import PredictorClient
from node import Node


class Graph:
    """Directed acyclic graph of Nodes with routing logic.

    Attributes:
        graph_id: Unique identifier for this graph.
        entrance: Node ID of the entrance node (None if not set).
        exit: Node ID of the exit node (None if not set).
        node_mapping: Mapping from node ID to Node object.
        router_mapping: Mapping from outbound node ID to set of inbound node IDs.
        routing_override: Experimental override for fanout numbers, mapping
            (outbound_id, inbound_id) tuples to fixed fanout values.
        predictor_client: Client for predicting routing fanout.
    """

    def __init__(self, graph_id: str, predictor_url: str):
        """Initialize Graph.

        Args:
            graph_id: Unique identifier for this graph.
            predictor_url: URL of the predictor service for fanout prediction.
        """
        self.graph_id = graph_id

        self.entrance: Optional[str] = None
        self.exit: Optional[str] = None
        self.node_mapping: Dict[str, Node] = {}
        self.router_mapping: Dict[str, Set[str]] = {}

        # For experiment use only: set a fixed fanout number rather than
        # letting the predictor predict it.
        self.routing_override: Dict[tuple[str, str], int] = {}

        self.predictor_client = PredictorClient(predictor_url=predictor_url)

    def add_node(
        self,
        node_or_model_id,
        predictor_url: Optional[str] = None,
        scheduler_host: Optional[str] = None,
        scheduler_port: Optional[int] = None,
        entrance: bool = False,
        exit: bool = False,
    ):
        """Add a node to the graph.

        This method accepts either an existing Node object or parameters to create
        a new Node. If creating a new Node, all parameters must be provided.

        Args:
            node_or_model_id: Either a Node object or a model_id string.
            predictor_url: Predictor service URL (required if creating new Node).
            scheduler_host: Scheduler host (required if creating new Node).
            scheduler_port: Scheduler port (required if creating new Node).
            entrance: If True, sets this node as the graph entrance.
            exit: If True, sets this node as the graph exit.

        Raises:
            ValueError: If creating a new Node but missing required parameters,
                or if entrance and exit are both True.
            RuntimeError: If attempting to set a second entrance or exit node.
        """
        # Handle two cases: passing a Node object or creating a new Node.
        if isinstance(node_or_model_id, Node):
            # Case 1: Node object passed directly.
            node = node_or_model_id
        else:
            # Case 2: model_id and parameters passed - create new Node.
            if (
                predictor_url is None
                or scheduler_host is None
                or scheduler_port is None
            ):
                raise ValueError(
                    "When passing model_id, predictor_url, scheduler_host, "
                    "and scheduler_port must be provided"
                )
            node = Node(
                model_id=node_or_model_id,
                predictor_url=predictor_url,
                scheduler_host=scheduler_host,
                scheduler_port=scheduler_port,
            )

        # Create the mapping relationship.
        node_id = node.node_id
        self.node_mapping[node_id] = node

        # Validate entrance and exit constraints.
        if entrance and exit:
            raise ValueError("A node cannot be both entrance and exit")

        if entrance and self.entrance is not None:
            raise RuntimeError("A graph should only have one entrance")

        if exit and self.exit is not None:
            raise RuntimeError("A graph should only have one exit")

        if entrance:
            self.entrance = node_id

        if exit:
            self.exit = node_id

    def add_router(
        self,
        outbound_node: Node,
        inbound_node: Node,
        fix_fanout_num: Optional[int] = None,
    ):
        """Add a routing edge from outbound_node to inbound_node.

        Creates a directed edge in the graph routing table. Optionally sets
        a fixed fanout number for experiments, bypassing predictor-based fanout.

        Args:
            outbound_node: Source node of the routing edge.
            inbound_node: Destination node of the routing edge.
            fix_fanout_num: If provided, sets a fixed fanout number for this
                routing edge instead of using the predictor. For experimental
                use only.
        """
        outbound_id = outbound_node.node_id
        inbound_id = inbound_node.node_id

        # Create basic routing mapping (no fanout information).
        if outbound_id not in self.router_mapping:
            self.router_mapping[outbound_id] = {inbound_id}
        else:
            self.router_mapping[outbound_id].add(inbound_id)

        # Experiment only: bypass structure predictor, set fixed fanout number
        # for a router.
        if fix_fanout_num is not None:
            self.routing_override[(outbound_id, inbound_id)] = fix_fanout_num

    def overwrite_fanout(
        self, outbound_node: Node, inbound_node: Node, fanout_num: int
    ):
        """Overwrite the fanout number for an existing routing edge.

        This method allows modifying the fanout number after the router has been
        added. The routing edge must already exist in the graph.

        Args:
            outbound_node: Source node of the routing edge.
            inbound_node: Destination node of the routing edge.
            fanout_num: New fanout number to set for this routing edge.

        Raises:
            KeyError: If the routing edge doesn't exist in the graph.
        """
        outbound_id = outbound_node.node_id
        inbound_id = inbound_node.node_id

        # Make sure the path outbound_node -> inbound_node exists.
        if outbound_id not in self.router_mapping:
            raise KeyError(
                f"Routing from node {outbound_id} hasn't been added to the routing map"
            )
        if inbound_id not in self.router_mapping[outbound_id]:
            raise KeyError(
                f"Routing from {outbound_id} to {inbound_id} hasn't been added "
                "to the routing map"
            )

        self.routing_override[(outbound_id, inbound_id)] = fanout_num

    def exec(self):
        """Execute the graph.

        TODO: Implement graph execution logic.
        """
        pass
