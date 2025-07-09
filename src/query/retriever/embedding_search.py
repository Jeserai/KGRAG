"""
Embedding-based search for finding relevant entities and text units.
Simple implementation focused on semantic similarity.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from src.kg.extractor.entity_extractor import Entity, Relationship
from src.data.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from embedding search."""
    entity: Entity
    similarity_score: float
    source: str  # "entity" or "text"


class EmbeddingSearch:
    """Embedding-based search for entities and text units."""
    
    def __init__(self, embedding_model, entities: List[Entity], chunks: List[DocumentChunk] = None):
        """
        Initialize embedding search.
        
        Args:
            embedding_model: Model for generating embeddings
            entities: List of entities with embeddings
            chunks: Optional list of document chunks with embeddings
        """
        self.embedding_model = embedding_model
        self.entities = entities
        self.chunks = chunks or []
        
        # Pre-compute entity embeddings if not available
        self._ensure_entity_embeddings()
        
        logger.info(f"Initialized EmbeddingSearch with {len(entities)} entities")
    
    def _ensure_entity_embeddings(self):
        """Ensure all entities have embeddings."""
        for entity in self.entities:
            if not hasattr(entity, 'embedding') or entity.embedding is None:
                # Create text representation for embedding
                text = f"{entity.name}: {entity.description or ''}"
                try:
                    entity.embedding = self.embedding_model.encode([text])[0]
                except Exception as e:
                    logger.warning(f"Failed to embed entity {entity.name}: {e}")
                    entity.embedding = np.zeros(384)  # Default embedding size
    
    def search_entities(self, query: str, top_k: int = 10, min_similarity: float = 0.0) -> List[SearchResult]:
        """
        Search for entities most similar to query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        if not query.strip():
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for entity in self.entities:
                if hasattr(entity, 'embedding') and entity.embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, entity.embedding)
                    if similarity >= min_similarity:
                        similarities.append(SearchResult(
                            entity=entity,
                            similarity_score=similarity,
                            source="entity"
                        ))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in entity search: {e}")
            return []
    
    def search_similar_entities(self, seed_entity: Entity, top_k: int = 5) -> List[SearchResult]:
        """
        Find entities similar to a seed entity.
        
        Args:
            seed_entity: Entity to find similar entities for
            top_k: Number of similar entities to return
            
        Returns:
            List of similar entities
        """
        if not hasattr(seed_entity, 'embedding') or seed_entity.embedding is None:
            return []
        
        similarities = []
        for entity in self.entities:
            if entity.name == seed_entity.name:  # Skip self
                continue
                
            if hasattr(entity, 'embedding') and entity.embedding is not None:
                similarity = self._cosine_similarity(seed_entity.embedding, entity.embedding)
                similarities.append(SearchResult(
                    entity=entity,
                    similarity_score=similarity,
                    source="entity"
                ))
        
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]
    
    def search_text_chunks(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for relevant text chunks (if available).
        
        Args:
            query: Search query
            top_k: Number of chunks to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.chunks or not query.strip():
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            
            chunk_similarities = []
            for chunk in self.chunks:
                # Generate chunk embedding if not available
                if not hasattr(chunk, 'embedding') or chunk.embedding is None:
                    try:
                        chunk.embedding = self.embedding_model.encode([chunk.text])[0]
                    except Exception as e:
                        logger.warning(f"Failed to embed chunk {chunk.id}: {e}")
                        continue
                
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                chunk_similarities.append((chunk, similarity))
            
            # Sort and return top k
            chunk_similarities.sort(key=lambda x: x[1], reverse=True)
            return chunk_similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in text chunk search: {e}")
            return []
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Cosine similarity using sentence-transformers util (handles tensors on GPU)."""
        from sentence_transformers.util import cos_sim  # local import to avoid optional dep issues at import time
        try:
            score = cos_sim(vec1, vec2)
            # cos_sim returns a 1×1 tensor or ndarray; convert to float
            if hasattr(score, "item"):
                return float(score.item())
            return float(score)
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_entity_context(self, entities: List[Entity], max_chars: int = 2000) -> str:
        """
        Create a context string from selected entities.
        
        Args:
            entities: List of entities to include in context
            max_chars: Maximum characters in context
            
        Returns:
            Context string
        """
        context_parts = []
        current_length = 0
        
        for entity in entities:
            entity_text = f"Entity: {entity.name}\nType: {entity.type}\nDescription: {entity.description or 'N/A'}\n"
            
            if current_length + len(entity_text) > max_chars:
                break
            
            context_parts.append(entity_text)
            current_length += len(entity_text)
        
        return "\n".join(context_parts)
    
    def search_with_fallback(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search with fallback strategies.
        
        First tries entity search, if insufficient results, 
        can fallback to text-based search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        # Primary: entity embedding search
        results = self.search_entities(query, top_k=top_k, min_similarity=0.3)
        
        if len(results) < top_k // 2 and self.chunks:
            # Fallback: search through text chunks and extract related entities
            logger.info("Using text chunk fallback search")
            chunk_results = self.search_text_chunks(query, top_k=5)
            
            # Find entities mentioned in top chunks
            additional_entities = set()
            for chunk, _ in chunk_results:
                # Simple approach: find entities mentioned in chunk
                for entity in self.entities:
                    if entity.name.lower() in chunk.text.lower():
                        additional_entities.add(entity)
            
            # Add these entities to results
            for entity in additional_entities:
                if len(results) >= top_k:
                    break
                if not any(r.entity.name == entity.name for r in results):
                    results.append(SearchResult(
                        entity=entity,
                        similarity_score=0.5,  # Lower score for fallback
                        source="text"
                    ))
        
        return results[:top_k]