from typing import List, Set
from kg.retriever.graph_traverse import GraphTraversal, Entity, Relationship, TraversalResult
from collections import deque


class GraphUtils(GraphTraversal):
    """Utility functions for graph traversal."""

    def __init__(self, entities: List[Entity], relationships: List[Relationship]):
        super().__init__(entities, relationships)

    def bfs(self, 
                seed_entities: List[str], 
                max_depth: int = 2, 
                max_entities: int = 20) -> TraversalResult:
        """
        Perform breadth-first traversal from seed entities.
        
        Args:
            seed_entities: List of entity names to start from
            max_depth: Maximum depth to traverse
            max_entities: Maximum entities to collect
            
        Returns:
            TraversalResult with found entities and relationships
        """
        visited_entities = set()
        found_entities = []
        found_relationships = []
        traversal_path = []
        
        # Initialize BFS queue with seed entities
        queue = deque()
        for seed in seed_entities:
            if seed in self.entities:
                queue.append((seed, 0))  # (entity_name, depth)
                visited_entities.add(seed)
                found_entities.append(self.entities[seed])
                traversal_path.append(f"seed:{seed}")
        
        max_depth_reached = 0
        
        while queue and len(found_entities) < max_entities:
            current_entity, depth = queue.popleft()
            max_depth_reached = max(max_depth_reached, depth)
            
            if depth >= max_depth:
                continue
            
            # Get neighbors
            neighbors = self.find_neighbors(current_entity, max_neighbors=5)
            
            for neighbor_entity, relationship in neighbors:
                if len(found_entities) >= max_entities:
                    break
                
                # Add relationship
                if relationship not in found_relationships:
                    found_relationships.append(relationship)
                
                # Add neighbor if not visited
                if neighbor_entity.name not in visited_entities:
                    visited_entities.add(neighbor_entity.name)
                    found_entities.append(neighbor_entity)
                    queue.append((neighbor_entity.name, depth + 1))
                    traversal_path.append(f"depth{depth+1}:{neighbor_entity.name}")
        
        return TraversalResult(
            entities=found_entities,
            relationships=found_relationships,
            traversal_path=traversal_path,
            depth_reached=max_depth_reached
        )
    
    def dijkstra(self, start_entity: str, end_entity: str, max_length: int = 4) -> List[List[str]]:
        """
        Find paths between two entities.
        
        Args:
            start_entity: Starting entity name
            end_entity: Ending entity name
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of entity names)
        """
        if start_entity not in self.entities or end_entity not in self.entities:
            return []
        
        paths = []
        
        def dfs(current: str, target: str, path: List[str], visited: Set[str], max_len: int):
            if len(path) > max_len:
                return
            
            if current == target:
                paths.append(path.copy())
                return
            
            if current in self.adjacency_list:
                for neighbor, _ in self.adjacency_list[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, target, path, visited, max_len)
                        path.pop()
                        visited.remove(neighbor)
        
        visited = {start_entity}
        dfs(start_entity, end_entity, [start_entity], visited, max_length)
        
        return paths