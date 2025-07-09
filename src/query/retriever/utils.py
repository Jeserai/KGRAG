from typing import List, Set, Tuple
from collections import deque

from src.kg.extractor.entity_extractor import Entity, Relationship
from src.data.document_processor import DocumentChunk

from .graph_traverse import GraphTraversal, EntityScore, TraversalResult
from .embedding_search import SearchResult
from .retriever import HybridRetriever, RetrievalResult
    
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