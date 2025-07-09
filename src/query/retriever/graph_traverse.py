"""
Graph traversal for exploring knowledge graph from seed entities.
Simple implementation focused on finding related entities and relationships.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

from kg.extractor.entity_extractor import Entity, Relationship

logger = logging.getLogger(__name__)


@dataclass
class TraversalResult:
    """Result from graph traversal."""
    entities: List[Entity]
    relationships: List[Relationship]
    traversal_path: List[str]
    depth_reached: int


@dataclass
class EntityScore:
    """Entity with relevance score."""
    entity: Entity
    score: float
    source_path: List[str]


class GraphTraversal:
    """Simple graph traversal for finding related entities and relationships."""
    
    def __init__(self, entities: List[Entity], relationships: List[Relationship]):
        """
        Initialize graph traversal.
        
        Args:
            entities: List of all entities
            relationships: List of all relationships
        """
        self.entities = {entity.name: entity for entity in entities}
        self.relationships = relationships
        
        # Build adjacency lists for efficient traversal
        self.adjacency_list = self._build_adjacency_list()
        
        logger.info(f"Initialized GraphTraversal with {len(entities)} entities and {len(relationships)} relationships")
    
    def _build_adjacency_list(self) -> Dict[str, List[Tuple[str, Relationship]]]:
        """Build adjacency list from relationships."""
        adj_list = defaultdict(list)
        
        for relationship in self.relationships:
            # Add bidirectional connections (treating graph as undirected for traversal)
            adj_list[relationship.source_entity].append((relationship.target_entity, relationship))
            adj_list[relationship.target_entity].append((relationship.source_entity, relationship))
        
        return dict(adj_list)
    
    def find_central_entities(self, entity_names: List[str], centrality_type: str = "degree") -> List[EntityScore]:
        """
        Find central entities within a subgraph.
        
        Args:
            entity_names: List of entity names to consider
            centrality_type: Type of centrality ("degree", "betweenness_approx")
            
        Returns:
            List of EntityScore objects sorted by centrality
        """
        entity_scores = []
        
        if centrality_type == "degree":
            # Simple degree centrality
            for entity_name in entity_names:
                if entity_name in self.adjacency_list:
                    degree = len(self.adjacency_list[entity_name])
                    if entity_name in self.entities:
                        entity_scores.append(EntityScore(
                            entity=self.entities[entity_name],
                            score=degree,
                            source_path=[f"degree_centrality:{degree}"]
                        ))
        
        elif centrality_type == "betweenness_approx":
            # Approximate betweenness centrality
            entity_scores = self._approximate_betweenness(entity_names)
        
        # Sort by score
        entity_scores.sort(key=lambda x: x.score, reverse=True)
        return entity_scores
    
    def _approximate_betweenness(self, entity_names: List[str]) -> List[EntityScore]:
        """Approximate betweenness centrality calculation."""
        betweenness_scores = defaultdict(float)
        
        # Sample entity pairs and find shortest paths
        entities_to_check = entity_names[:10]  # Limit for performance
        
        for i, start in enumerate(entities_to_check):
            for end in entities_to_check[i+1:]:
                paths = self.find_paths(start, end, max_length=3)
                if paths:
                    shortest_path = min(paths, key=len)
                    # Add score to intermediate entities
                    for entity in shortest_path[1:-1]:  # Exclude start and end
                        betweenness_scores[entity] += 1.0
        
        # Convert to EntityScore objects
        entity_scores = []
        for entity_name, score in betweenness_scores.items():
            if entity_name in self.entities:
                entity_scores.append(EntityScore(
                    entity=self.entities[entity_name],
                    score=score,
                    source_path=[f"betweenness_approx:{score}"]
                ))
        
        return entity_scores
    
    def expand_entity_neighborhood(self, 
                                  entity_name: str, 
                                  expansion_strategy: str = "balanced",
                                  max_entities: int = 15) -> TraversalResult:
        """
        Expand around a single entity using different strategies.
        
        Args:
            entity_name: Entity to expand from
            expansion_strategy: "balanced", "high_confidence", "diverse_types"
            max_entities: Maximum entities to include
            
        Returns:
            TraversalResult with expanded neighborhood
        """
        if entity_name not in self.entities:
            return TraversalResult([], [], [], 0)
        
        neighbors = self.find_neighbors(entity_name, max_neighbors=20)
        
        if expansion_strategy == "high_confidence":
            # Sort by relationship confidence
            neighbors.sort(key=lambda x: getattr(x[1], 'confidence', 0.5), reverse=True)
        
        elif expansion_strategy == "diverse_types":
            # Ensure diversity in entity types
            type_counts = defaultdict(int)
            diverse_neighbors = []
            
            for neighbor_entity, relationship in neighbors:
                entity_type = neighbor_entity.type
                if type_counts[entity_type] < 3:  # Max 3 per type
                    diverse_neighbors.append((neighbor_entity, relationship))
                    type_counts[entity_type] += 1
            
            neighbors = diverse_neighbors
        
        # Limit results
        selected_neighbors = neighbors[:max_entities-1]  # -1 for seed entity
        
        entities = [self.entities[entity_name]]  # Include seed
        relationships = []
        
        for neighbor_entity, relationship in selected_neighbors:
            entities.append(neighbor_entity)
            relationships.append(relationship)
        
        return TraversalResult(
            entities=entities,
            relationships=relationships,
            traversal_path=[f"expansion_from:{entity_name}"],
            depth_reached=1
        )
    
    def find_entity_clusters(self, entity_names: List[str]) -> List[List[str]]:
        """
        Find clusters of closely connected entities.
        
        Args:
            entity_names: List of entity names to cluster
            
        Returns:
            List of clusters (each cluster is a list of entity names)
        """
        # Build subgraph adjacency
        subgraph = defaultdict(set)
        for entity in entity_names:
            if entity in self.adjacency_list:
                for neighbor, _ in self.adjacency_list[entity]:
                    if neighbor in entity_names:
                        subgraph[entity].add(neighbor)
        
        # Find connected components
        visited = set()
        clusters = []
        
        for entity in entity_names:
            if entity not in visited:
                cluster = []
                stack = [entity]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.append(current)
                        
                        # Add unvisited neighbors
                        for neighbor in subgraph[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def get_relationship_summary(self, relationships: List[Relationship]) -> Dict[str, int]:
        """Get summary statistics for relationships."""
        rel_types = defaultdict(int)
        for rel in relationships:
            rel_types[rel.relationship_type] += 1
        
        return dict(rel_types)