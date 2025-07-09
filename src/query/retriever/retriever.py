"""
Hybrid retriever that combines embedding search and graph traversal.
Main interface for GraphRAG retrieval following Microsoft's approach.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

from src.kg.extractor.entity_extractor import Entity, Relationship
from src.data.document_processor import DocumentChunk
from embedding_search import EmbeddingSearch, SearchResult as EmbedSearchResult
from graph_traverse import GraphTraversal, TraversalResult, EntityScore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    entities: List[Entity]
    relationships: List[Relationship]
    context_chunks: List[DocumentChunk]
    retrieval_method: str
    confidence_score: float
    retrieval_time: float
    metadata: Dict[str, any]


class HybridRetriever:
    """
    Hybrid retriever combining embedding search and graph traversal.
    
    Strategy:
    1. Use embedding search to find seed entities
    2. Use graph traversal to expand from seed entities
    3. Combine and rank results
    """
    
    def __init__(self, 
                 entities: List[Entity], 
                 relationships: List[Relationship],
                 embedding_model,
                 chunks: List[DocumentChunk] = None):
        """
        Initialize hybrid retriever.
        
        Args:
            entities: List of all entities
            relationships: List of all relationships  
            embedding_model: Model for generating embeddings
            chunks: Optional document chunks for text search
        """
        self.entities = entities
        self.relationships = relationships
        self.embedding_model = embedding_model
        self.chunks = chunks or []
        
        # Initialize components
        self.embedding_search = EmbeddingSearch(embedding_model, entities, chunks)
        self.graph_traversal = GraphTraversal(entities, relationships)
        
        logger.info(f"Initialized HybridRetriever with {len(entities)} entities, {len(relationships)} relationships")
    
    def retrieve(self, 
                query: str, 
                method: str = "hybrid",
                top_k: int = 10,
                max_depth: int = 2,
                include_chunks: bool = False) -> RetrievalResult:
        """
        Main retrieval method.
        
        Args:
            query: Search query
            method: "hybrid", "embedding_only", "graph_only"
            top_k: Number of entities to retrieve
            max_depth: Maximum graph traversal depth
            include_chunks: Whether to include related text chunks
            
        Returns:
            RetrievalResult
        """
        start_time = time.time()
        
        if method == "embedding_only":
            return self._embedding_only_retrieve(query, top_k, include_chunks)
        elif method == "graph_only":
            return self._graph_only_retrieve(query, top_k, max_depth, include_chunks)
        else:
            return self._hybrid_retrieve(query, top_k, max_depth, include_chunks)
    
    def _hybrid_retrieve(self, 
                        query: str, 
                        top_k: int, 
                        max_depth: int, 
                        include_chunks: bool) -> RetrievalResult:
        """
        Hybrid retrieval: embedding search + graph traversal.
        
        This is the main method following Microsoft's approach.
        """
        start_time = time.time()
        
        # Step 1: Find seed entities using embedding similarity
        embedding_results = self.embedding_search.search_entities(
            query, 
            top_k=min(top_k // 2, 5),  # Start with fewer seeds
            min_similarity=0.3
        )
        
        if not embedding_results:
            # Fallback to broader search
            embedding_results = self.embedding_search.search_with_fallback(query, top_k=3)
        
        seed_entities = [result.entity.name for result in embedding_results]
        logger.info(f"Found {len(seed_entities)} seed entities: {seed_entities}")
        
        # Step 2: Graph traversal from seed entities
        traversal_result = self.graph_traversal.breadth_first_traversal(
            seed_entities=seed_entities,
            max_depth=max_depth,
            max_entities=top_k
        )
        
        # Step 3: Combine and rank results
        final_entities = self._rank_and_combine_entities(embedding_results, traversal_result, query)
        
        # Step 4: Get related text chunks if requested
        context_chunks = []
        if include_chunks and self.chunks:
            context_chunks = self._get_related_chunks(final_entities[:5], query)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(embedding_results, traversal_result)
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            entities=final_entities[:top_k],
            relationships=traversal_result.relationships,
            context_chunks=context_chunks,
            retrieval_method="hybrid",
            confidence_score=confidence,
            retrieval_time=retrieval_time,
            metadata={
                "seed_entities": len(seed_entities),
                "traversal_depth": traversal_result.depth_reached,
                "embedding_results": len(embedding_results),
                "graph_entities": len(traversal_result.entities)
            }
        )
    
    def _embedding_only_retrieve(self, query: str, top_k: int, include_chunks: bool) -> RetrievalResult:
        """Pure embedding-based retrieval."""
        start_time = time.time()
        
        embedding_results = self.embedding_search.search_entities(query, top_k=top_k)
        entities = [result.entity for result in embedding_results]
        
        # Get relationships between found entities
        relationships = self._get_relationships_between_entities(entities)
        
        context_chunks = []
        if include_chunks:
            context_chunks = self._get_related_chunks(entities[:5], query)
        
        confidence = sum(r.similarity_score for r in embedding_results) / len(embedding_results) if embedding_results else 0.0
        
        return RetrievalResult(
            entities=entities,
            relationships=relationships,
            context_chunks=context_chunks,
            retrieval_method="embedding_only", 
            confidence_score=confidence,
            retrieval_time=time.time() - start_time,
            metadata={"embedding_results": len(embedding_results)}
        )
    
    def _graph_only_retrieve(self, query: str, top_k: int, max_depth: int, include_chunks: bool) -> RetrievalResult:
        """Pure graph-based retrieval (requires seed entities)."""
        start_time = time.time()
        
        # Find seed entities by simple text matching as fallback
        seed_entities = []
        query_lower = query.lower()
        for entity in self.entities[:20]:  # Check first 20 entities
            if any(term in entity.name.lower() for term in query_lower.split()):
                seed_entities.append(entity.name)
            if len(seed_entities) >= 3:
                break
        
        if not seed_entities:
            # Random seed as last resort
            seed_entities = [self.entities[0].name] if self.entities else []
        
        traversal_result = self.graph_traversal.breadth_first_traversal(
            seed_entities=seed_entities,
            max_depth=max_depth,
            max_entities=top_k
        )
        
        context_chunks = []
        if include_chunks:
            context_chunks = self._get_related_chunks(traversal_result.entities[:5], query)
        
        return RetrievalResult(
            entities=traversal_result.entities,
            relationships=traversal_result.relationships,
            context_chunks=context_chunks,
            retrieval_method="graph_only",
            confidence_score=0.5,  # Default confidence for graph-only
            retrieval_time=time.time() - start_time,
            metadata={
                "seed_entities": len(seed_entities),
                "traversal_depth": traversal_result.depth_reached
            }
        )
    
    def _rank_and_combine_entities(self, 
                                  embedding_results: List[EmbedSearchResult], 
                                  traversal_result: TraversalResult,
                                  query: str) -> List[Entity]:
        """
        Rank and combine entities from embedding and graph results.
        
        Strategy:
        1. Seed entities get highest priority
        2. Graph-expanded entities get medium priority  
        3. Remove duplicates
        4. Re-rank by relevance
        """
        
        entity_scores = {}
        
        # Score seed entities (from embedding search)
        for i, result in enumerate(embedding_results):
            entity_name = result.entity.name
            # Higher score for higher embedding similarity and earlier rank
            score = result.similarity_score * (1.0 - i * 0.1)
            entity_scores[entity_name] = {
                'entity': result.entity,
                'score': score + 0.5,  # Boost for being a seed
                'source': 'embedding'
            }
        
        # Score traversal entities
        for i, entity in enumerate(traversal_result.entities):
            if entity.name not in entity_scores:
                # Score decreases with position in traversal
                base_score = max(0.1, 0.8 - i * 0.05)
                entity_scores[entity.name] = {
                    'entity': entity,
                    'score': base_score,
                    'source': 'traversal'
                }
            else:
                # Boost score for entities found by both methods
                entity_scores[entity.name]['score'] += 0.3
                entity_scores[entity.name]['source'] = 'both'
        
        # Optional: Re-rank by query relevance
        self._adjust_scores_by_relevance(entity_scores, query)
        
        # Sort by score and return entities
        sorted_entities = sorted(
            entity_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        return [item['entity'] for item in sorted_entities]
    
    def _adjust_scores_by_relevance(self, entity_scores: Dict[str, Dict], query: str):
        """Adjust entity scores based on text relevance to query."""
        query_terms = set(query.lower().split())
        
        for entity_data in entity_scores.values():
            entity = entity_data['entity']
            
            # Check name relevance
            name_terms = set(entity.name.lower().split())
            name_overlap = len(query_terms & name_terms) / len(query_terms) if query_terms else 0
            
            # Check description relevance  
            desc_overlap = 0
            if entity.description:
                desc_terms = set(entity.description.lower().split())
                desc_overlap = len(query_terms & desc_terms) / len(query_terms) if query_terms else 0
            
            # Adjust score
            relevance_boost = (name_overlap * 0.3) + (desc_overlap * 0.2)
            entity_data['score'] += relevance_boost
    
    def _get_relationships_between_entities(self, entities: List[Entity]) -> List[Relationship]:
        """Find relationships between a set of entities."""
        entity_names = {entity.name for entity in entities}
        
        relevant_relationships = []
        for relationship in self.relationships:
            if (relationship.source_entity in entity_names and 
                relationship.target_entity in entity_names):
                relevant_relationships.append(relationship)
        
        return relevant_relationships
    
    def _get_related_chunks(self, entities: List[Entity], query: str, max_chunks: int = 5) -> List[DocumentChunk]:
        """Get text chunks related to entities and query."""
        if not self.chunks:
            return []
        
        # Find chunks mentioned in entity source_chunks
        related_chunks = []
        entity_chunk_ids = set()
        
        for entity in entities:
            if hasattr(entity, 'source_chunks') and entity.source_chunks:
                entity_chunk_ids.update(entity.source_chunks)
        
        # Get chunks by ID
        for chunk in self.chunks:
            if chunk.id in entity_chunk_ids:
                related_chunks.append(chunk)
        
        # If not enough, add chunks by similarity search
        if len(related_chunks) < max_chunks:
            similarity_chunks = self.embedding_search.search_text_chunks(query, top_k=max_chunks-len(related_chunks))
            for chunk, _ in similarity_chunks:
                if chunk not in related_chunks:
                    related_chunks.append(chunk)
        
        return related_chunks[:max_chunks]
    
    def _calculate_confidence(self, embedding_results: List[EmbedSearchResult], traversal_result: TraversalResult) -> float:
        """Calculate overall confidence score for retrieval."""
        if not embedding_results:
            return 0.3
        
        # Average embedding similarity
        avg_similarity = sum(r.similarity_score for r in embedding_results) / len(embedding_results)
        
        # Boost for successful graph expansion
        expansion_bonus = min(0.2, len(traversal_result.entities) * 0.02)
        
        # Relationship bonus
        relationship_bonus = min(0.1, len(traversal_result.relationships) * 0.01)
        
        return min(1.0, avg_similarity + expansion_bonus + relationship_bonus)
    
    def query_expansion(self, query: str) -> List[str]:
        """
        Expand query using related entities.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded query terms
        """
        # Find related entities
        initial_results = self.embedding_search.search_entities(query, top_k=3)
        
        expanded_terms = query.split()
        
        # Add entity names and descriptions as expansion terms
        for result in initial_results:
            entity = result.entity
            expanded_terms.extend(entity.name.split())
            if entity.description:
                expanded_terms.extend(entity.description.split()[:5])  # Limit description terms
        
        # Remove duplicates and return
        return list(set(expanded_terms))
    
    def get_retrieval_stats(self) -> Dict[str, any]:
        """Get statistics about the retrieval system."""
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships), 
            "total_chunks": len(self.chunks),
            "avg_entity_connections": len(self.relationships) / len(self.entities) if self.entities else 0,
            "entity_types": list(set(entity.type for entity in self.entities)),
            "relationship_types": list(set(rel.relationship_type for rel in self.relationships))
        }