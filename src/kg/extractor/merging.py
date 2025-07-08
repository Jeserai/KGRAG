"""
Context-aware entity merging strategy that balances deduplication with context preservation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
import numpy as np
from kg.extractor.entity_extractor import Entity, Relationship

logger = logging.getLogger(__name__)


class EntityMerger:
    
    def __init__(self, 
                 name_threshold: float = 0.8,
                 use_embeddings: bool = False,
                 embedding_model=None):
        """
        Args:
            name_threshold: Similarity threshold for name matching
            use_embeddings: Whether to use embedding similarity  
            embedding_model: Model for computing embeddings (optional)
        """
        self.name_threshold = name_threshold
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
    
    def merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Simple merging strategy:
        1. Exact type matching first
        2. Name similarity within same type
        3. Optional: embedding similarity as tiebreaker
        """
        
        if not entities:
            return []
        
        # Step 1: Group by exact type match
        type_groups = defaultdict(list)
        for entity in entities:
            type_groups[entity.type.upper()].append(entity)
        
        merged_entities = []
        
        # Step 2: Within each type, merge by name similarity
        for entity_type, type_entities in type_groups.items():
            merged_type_entities = self._merge_within_type(type_entities)
            merged_entities.extend(merged_type_entities)
        
        print(f"Merged {len(entities)} entities into {len(merged_entities)} ({len(entities) - len(merged_entities)} merged)")
        return merged_entities
    
    def _merge_within_type(self, entities: List[Entity]) -> List[Entity]:
        """Merge entities of the same type based on name similarity."""
        
        if len(entities) <= 1:
            return entities
        
        # Build similarity matrix based on names
        n = len(entities)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Primary similarity: normalized name comparison
                name_sim = self._name_similarity(entities[i].name, entities[j].name)
                
                # Optional: embedding similarity
                if self.use_embeddings and self.embedding_model:
                    emb_sim = self._embedding_similarity(entities[i], entities[j])
                    # Weight: 70% name, 30% embedding
                    total_sim = 0.7 * name_sim + 0.3 * emb_sim
                else:
                    total_sim = name_sim
                
                similarity_matrix[i][j] = total_sim
                similarity_matrix[j][i] = total_sim
        
        # Find groups to merge using simple clustering
        merged_groups = self._cluster_entities(entities, similarity_matrix)
        
        # Merge each group
        result = []
        for group in merged_groups:
            if len(group) == 1:
                result.append(group[0])
            else:
                merged = self._merge_entity_group(group)
                result.append(merged)
        
        return result
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity - simple but effective."""
        
        # Normalize names
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return 1.0
        
        # Check if one is contained in the other (for abbreviations)
        if norm1 in norm2 or norm2 in norm1:
            return 0.9
        
        # Jaccard similarity of words
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        # Convert to lowercase
        name = name.lower().strip()
        
        # Remove common business suffixes
        suffixes = [r'\s+inc\.?$', r'\s+corp\.?$', r'\s+ltd\.?$', r'\s+llc$', r'\s+co\.?$']
        for suffix in suffixes:
            name = re.sub(suffix, '', name)
        
        # Remove punctuation except spaces and dots
        name = re.sub(r'[^\w\s\.]', '', name)
        
        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _embedding_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate embedding similarity between entities."""
        if not self.embedding_model:
            return 0.0
        
        try:
            # Create text representations
            text1 = f"{entity1.name}: {entity1.description or ''}"
            text2 = f"{entity2.name}: {entity2.description or ''}"
            
            # Get embeddings
            embeddings = self.embedding_model.encode([text1, text2])
            emb1, emb2 = embeddings[0], embeddings[1]
            
            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            print(f"Embedding similarity error: {e}")
            return 0.0
    
    def _cluster_entities(self, entities: List[Entity], similarity_matrix: np.ndarray) -> List[List[Entity]]:
        """Simple clustering based on similarity threshold."""
        n = len(entities)
        used = [False] * n
        clusters = []
        
        for i in range(n):
            if used[i]:
                continue
            
            # Start new cluster
            cluster = [entities[i]]
            used[i] = True
            
            # Add similar entities to cluster
            for j in range(i + 1, n):
                if not used[j] and similarity_matrix[i][j] >= self.name_threshold:
                    cluster.append(entities[j])
                    used[j] = True
            
            clusters.append(cluster)
        
        return clusters
    
    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """Merge a group of entities into one."""
        if len(entities) == 1:
            return entities[0]
        
        # Choose best name (longest and most complete)
        best_name = max(entities, key=lambda e: len(e.name.strip())).name
        
        # Use most common type (should be same anyway)
        entity_type = entities[0].type
        
        # Combine descriptions (keep unique ones)
        descriptions = []
        for entity in entities:
            if entity.description and entity.description.strip():
                desc = entity.description.strip()
                if desc not in descriptions:
                    descriptions.append(desc)
        
        combined_description = " | ".join(descriptions) if descriptions else ""
        
        # Combine source chunks
        all_chunks = []
        for entity in entities:
            all_chunks.extend(entity.source_chunks or [])
        unique_chunks = list(set(all_chunks))
        
        # Average confidence
        avg_confidence = sum(e.confidence for e in entities) / len(entities)
        
        return Entity(
            name=best_name,
            type=entity_type,
            description=combined_description,
            source_chunks=unique_chunks,
            confidence=avg_confidence
        )