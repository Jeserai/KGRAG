from typing import List, Set
from graph_traverse import GraphTraversal, Entity, Relationship, TraversalResult
from collections import deque
from embedding_search import SearchResult
from graph_traverse import EntityScore, GraphTraversal, TraversalResult
from retriever import HybridRetriever, RetrievalResult
from data.document_processor import DocumentChunk

class GraphUtils(GraphTraversal):
    """Utility functions for graph traversal."""

    def __init__(self, entities: List[Entity], relationships: List[Relationship]):
        super().__init__(entities, relationships)
    
    def find_neighbors(self, entity_name: str, max_neighbors: int = 10) -> List[Tuple[Entity, Relationship]]:
        """
        Find direct neighbors of an entity.
        
        Args:
            entity_name: Name of the entity
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            List of (neighbor_entity, connecting_relationship) tuples
        """
        if entity_name not in self.adjacency_list:
            return []
        
        neighbors = []
        for neighbor_name, relationship in self.adjacency_list[entity_name]:
            if neighbor_name in self.entities:
                neighbors.append((self.entities[neighbor_name], relationship))
        
        # Sort by relationship confidence if available
        neighbors.sort(key=lambda x: getattr(x[1], 'confidence', 0.5), reverse=True)
        return neighbors[:max_neighbors]

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
    
# Embedding search utility functions
def filter_entities_by_type(search_results: List[SearchResult], entity_types: List[str]) -> List[SearchResult]:
    """Filter search results by entity type."""
    return [result for result in search_results if result.entity.type in entity_types]

def deduplicate_results(search_results: List[SearchResult]) -> List[SearchResult]:
    """Remove duplicate entities from search results."""
    seen_names = set()
    unique_results = []
    
    for result in search_results:
        if result.entity.name not in seen_names:
            unique_results.append(result)
            seen_names.add(result.entity.name)
    
    return unique_results

# Graph traversal utility functions
def merge_traversal_results(results: List[TraversalResult]) -> TraversalResult:
    """Merge multiple traversal results."""
    all_entities = []
    all_relationships = []
    all_paths = []
    max_depth = 0
    
    seen_entities = set()
    seen_relationships = set()
    
    for result in results:
        # Merge entities (avoid duplicates)
        for entity in result.entities:
            if entity.name not in seen_entities:
                all_entities.append(entity)
                seen_entities.add(entity.name)
        
        # Merge relationships (avoid duplicates)
        for rel in result.relationships:
            rel_key = (rel.source_entity, rel.target_entity, rel.relationship_type)
            if rel_key not in seen_relationships:
                all_relationships.append(rel)
                seen_relationships.add(rel_key)
        
        all_paths.extend(result.traversal_path)
        max_depth = max(max_depth, result.depth_reached)
    
    return TraversalResult(
        entities=all_entities,
        relationships=all_relationships,
        traversal_path=all_paths,
        depth_reached=max_depth
    )

# Utility functions for retriever integration
def create_retriever_from_pipeline_results(entities: List[Entity], 
                                          relationships: List[Relationship],
                                          embedding_model,
                                          chunks: List[DocumentChunk] = None) -> HybridRetriever:
    """Create retriever from pipeline extraction results."""
    return HybridRetriever(entities, relationships, embedding_model, chunks)


def format_retrieval_for_context(result: RetrievalResult, max_length: int = 2000) -> str:
    """Format retrieval result for use as LLM context."""
    context_parts = []
    
    # Add entities
    if result.entities:
        context_parts.append("Relevant Entities:")
        for entity in result.entities[:10]:  # Limit entities
            entity_text = f"- {entity.name} ({entity.type}): {entity.description or 'N/A'}"
            context_parts.append(entity_text)
    
    # Add relationships  
    if result.relationships:
        context_parts.append("\nRelevant Relationships:")
        for rel in result.relationships[:5]:  # Limit relationships
            rel_text = f"- {rel.source_entity} → {rel.relationship_type} → {rel.target_entity}"
            context_parts.append(rel_text)
    
    # Add text chunks
    if result.context_chunks:
        context_parts.append("\nRelevant Context:")
        for chunk in result.context_chunks[:3]:  # Limit chunks
            chunk_text = f"- {chunk.text[:200]}..." if len(chunk.text) > 200 else f"- {chunk.text}"
            context_parts.append(chunk_text)
    
    # Combine and truncate if needed
    full_context = "\n".join(context_parts)
    if len(full_context) > max_length:
        full_context = full_context[:max_length] + "..."
    
    return full_context