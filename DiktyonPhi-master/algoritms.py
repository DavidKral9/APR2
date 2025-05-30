from diktyonphi import Graph, GraphType, Node, Hashable
import heapq
from collections import deque
from typing import List, Dict, Tuple, Set, Optional
def is_cyclic(self) -> bool:
    visited = set()
    stack = set()

    def visit(node_id):
        if node_id in stack:
            return True
        if node_id in visited:
            return False

        visited.add(node_id)
        stack.add(node_id)

        for neighbor in self.node(node_id).neighbor_ids:
            if visit(neighbor):
                return True
        stack.remove(node_id)
        return False

    for node_id in self.node_ids():
        if visit(node_id):
            return True
    return False

def strongly_connected_components(self) -> list[set[Hashable]]:
    if self.type != GraphType.DIRECTED:
        raise ValueError("SCC hledání funguje jen pro orientované grafy")

    visited = set()
    order = []

    def dfs1(nid):
        if nid in visited:
            return
        visited.add(nid)
        for neighbor in self.node(nid).neighbor_ids:
            dfs1(neighbor)
        order.append(nid)

    for node_id in self.node_ids():
        dfs1(node_id)

    # Reverzní graf
    reverse_graph = Graph(GraphType.DIRECTED)
    for node in self:
        reverse_graph.add_node(node.id)
    for node in self:
        for neighbor in node.neighbor_ids:
            reverse_graph.add_edge(neighbor, node.id)

    visited.clear()
    components = []

    def dfs2(nid, comp):
        if nid in visited:
            return
        visited.add(nid)
        comp.add(nid)
        for neighbor in reverse_graph.node(nid).neighbor_ids:
            dfs2(neighbor, comp)

    for nid in reversed(order):
        if nid not in visited:
            comp = set()
            dfs2(nid, comp)
            components.append(comp)

    return components


def topological_sort(self) -> list[Hashable]:
    if self.type != GraphType.DIRECTED:
        raise ValueError("Topological sort is only for directed graphs")

    visited = set()
    result = []

    def visit(nid):
        if nid in visited:
            return
        visited.add(nid)
        for neighbor in self.node(nid).neighbor_ids:
            visit(neighbor)
        result.append(nid)

    for node_id in self.node_ids():
        visit(node_id)

    return result[::-1]  # reverzní post-order

def kruskal(self) -> list[Tuple[Hashable, Hashable, float]]:
    if self.type != GraphType.UNDIRECTED:
        raise ValueError("Kruskal works only on undirected graphs")

    parent = {}
    def find(u):  # Find s kompresí cesty
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    def union(u, v):
        root_u, root_v = find(u), find(v)
        if root_u == root_v:
            return False
        parent[root_v] = root_u
        return True

    for node_id in self.node_ids():
        parent[node_id] = node_id

    edges = []
    for node in self:
        for neighbor_id in node.neighbor_ids:
            if node.id < neighbor_id:  # abychom neměli hrany dvakrát
                weight = node.to(neighbor_id)._attrs.get("weight", 1.0)
                edges.append((weight, node.id, neighbor_id))

    edges.sort()
    mst = []
    for weight, u, v in edges:
        if union(u, v):
            mst.append((u, v, weight))
    return mst

def dijkstra(graph: Graph, start: Hashable) -> Dict[Hashable, float]:
    """
    Vypočítá nejkratší vzdálenosti od uzlu `start` ke všem ostatním uzlům pomocí Dijkstrova algoritmu.

    :param graph: instance třídy Graph
    :param start: výchozí uzel
    :return: slovník {node_id: vzdálenost}
    """
    distances: Dict[Hashable, float] = {node_id: float('inf') for node_id in graph.node_ids()}
    distances[start] = 0

    # fronta priorit (min-heap): (vzdálenost, uzel)
    priority_queue: list[Tuple[float, Hashable]] = [(0, start)]

    while priority_queue:
        current_distance, current_node_id = heapq.heappop(priority_queue)
        current_node = graph.node(current_node_id)

        if current_distance > distances[current_node_id]:
            continue  

        for neighbor_id in current_node.neighbor_ids:
            edge = current_node.to(neighbor_id)
            weight = edge._attrs.get("weight", 1.0) 

            distance_through_current = current_distance + weight
            if distance_through_current < distances[neighbor_id]:
                distances[neighbor_id] = distance_through_current
                heapq.heappush(priority_queue, (distance_through_current, neighbor_id))
    return distances

def print_neighbors(graph: Graph) -> None:
    for node in graph:
        print(f"{node.id}: {[n.id for n in node.neighbor_nodes]}")


def print_neighbors_with_weights(graph: Graph) -> None:
    for node in graph:
        neighbors = []
        for neighbor_id in node.neighbor_ids:
            edge = node.to(neighbor_id)
            weight = edge._attrs.get("weight", 1.0)
            neighbors.append(f"{neighbor_id}({weight})")
        print(f"{node.id}: {', '.join(neighbors)}")


def adjacency_matrix(graph: Graph) -> List[List[float]]:
    nodes = list(graph.node_ids())
    index = {node_id: i for i, node_id in enumerate(nodes)}
    size = len(nodes)
    matrix = [[float("inf") for _ in range(size)] for _ in range(size)]

    for node_id in nodes:
        for neighbor_id in graph.node(node_id).neighbor_ids:
            weight = graph.node(node_id).to(neighbor_id)._attrs.get("weight", 1.0)
            matrix[index[node_id]][index[neighbor_id]] = weight

    return matrix

def dfs(self, start_id: Hashable) -> list[Hashable]:
    visited = set()
    stack = [start_id]
    result = []

    while stack:
        node_id = stack.pop()
        if node_id not in visited:
            visited.add(node_id)
            result.append(node_id)
            neighbors = list(self.node(node_id).neighbor_ids)
            stack.extend(reversed(neighbors))  # zachová pořadí
    return result


def bfs(self, start_id: Hashable) -> list[Hashable]:
    visited = set()
    queue = deque([start_id])
    result = []

    while queue:
        node_id = queue.popleft()
        if node_id not in visited:
            visited.add(node_id)
            result.append(node_id)
            queue.extend(self.node(node_id).neighbor_ids)
    return result


def path_weight(graph: Graph, path: List[Hashable]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        edge = graph.node(path[i]).to(path[i + 1])
        total += edge._attrs.get("weight", 1.0)
    return total


def all_paths(graph: Graph, start: Hashable, goal: Hashable) -> List[Tuple[List[Hashable], float]]:
    def dfs_all(current, path, visited):
        if current == goal:
            paths.append((list(path), path_weight(graph, path)))
            return
        for neighbor_id in graph.node(current).neighbor_ids:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                path.append(neighbor_id)
                dfs_all(neighbor_id, path, visited)
                path.pop()
                visited.remove(neighbor_id)

    paths = []
    dfs_all(start, [start], {start})
    return paths
def is_connected(self) -> bool:
    if self.type != GraphType.UNDIRECTED:
        raise ValueError("Connectivity check only for undirected graphs")
    nodes = list(self.node_ids())
    if not nodes:
        return True
    visited = set(self.dfs(nodes[0]))
    return len(visited) == len(self._nodes)
def connected_components(self) -> list[set[Hashable]]:
    if self.type != GraphType.UNDIRECTED:
        raise ValueError("Only for undirected graphs")
    
    visited = set()
    components = []

    for node_id in self.node_ids():
        if node_id not in visited:
            comp = set(self.dfs(node_id))
            visited.update(comp)
            components.append(comp)
    return components
def degree_centrality(self) -> dict[Hashable, float]:
    n = len(self._nodes)
    return {node.id: node.out_degree / (n - 1) for node in self}
def betweenness_centrality(self) -> dict[Hashable, float]:
    from collections import defaultdict, deque

    centrality = defaultdict(float)
    nodes = list(self.node_ids())

    for s in nodes:
        stack = []
        pred = {w: [] for w in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        dist = dict.fromkeys(nodes, -1)
        sigma[s] = 1.0
        dist[s] = 0
        queue = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in self.node(v).neighbor_ids:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = dict.fromkeys(nodes, 0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    # Normalizace
    scale = 1 / ((len(nodes) - 1) * (len(nodes) - 2))
    for k in centrality:
        centrality[k] *= scale
    return dict(centrality)
def clustering_coefficient(self, node_id: Hashable) -> float:
    neighbors = list(self.node(node_id).neighbor_ids)
    k = len(neighbors)
    if k < 2:
        return 0.0

    links = 0
    for i in range(k):
        for j in range(i + 1, k):
            u, v = neighbors[i], neighbors[j]
            if self.node(u).is_edge_to(v) or self.node(v).is_edge_to(u):
                links += 1

    return (2 * links) / (k * (k - 1))



if __name__ == "__main__":
    # Create a directed graph
    g = Graph(GraphType.DIRECTED)

    # Add nodes with attributes
    g.add_node("A", {"label": "Start", "color": "green"})
    g.add_node("B", {"label": "Middle", "color": "yellow"})
    g.add_node("C", {"label": "End", "color": "red"})
    g.add_node("D", {"label": "Optional", "color": "blue"})

    # Add edges with attributes
    g.add_edge("A", "B", {"weight": 1.0, "type": "normal"})
    g.add_edge("B", "C", {"weight": 2.5, "type": "critical"})
    g.add_edge("A", "D", {"weight": 0.8, "type": "optional"})
    g.add_edge("D", "C", {"weight": 1.7, "type": "fallback"})

    # Access and update node attribute
    print("Node A color:", g.node("A")["color"])
    g.node("A")["color"] = "darkgreen"

    # Access edge and modify its weight
    edge = g.node("A").to("B")
    print("Edge A→B weight:", edge["weight"])
    edge["weight"] = 1.1

    # Iterate through the graph
    print("\nGraph structure:")
    for node_id in g.node_ids():
        node = g.node(node_id)
        print(f"Node {node.id}: label={node['label']}, out_degree={node.out_degree}")
        for neighbor_id in node.neighbor_ids:
            edge = node.to(neighbor_id)
            print(f"  → {neighbor_id} (weight={edge['weight']}, type={edge['type']})")
    dist = dijkstra(g, "A")
    neighbors = print_neighbors(g)
    print("Nejkratší vzdálenosti od A:")
    for node, d in dist.items():
        print(f"{node}: {d}")
    print("-----------------")


