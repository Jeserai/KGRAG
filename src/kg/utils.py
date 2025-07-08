from entity_extractor import EntityExtractor, Entity, Relationship
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
from embedding_search import SearchResult
from graph_traverse import EntityScore, GraphTraversal, TraversalResult
from retriever import HybridRetriever, RetrievalResult
from data.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class KGUtils(EntityExtractor):
    
    def merge_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        Merge duplicate relationships.
        
        Args:
            relationships: List of relationships to merge
            
        Returns:
            List of merged relationships
        """
        if not relationships:
            return []
        
        # Group relationships by source, target, and type
        relationship_groups = {}
        
        for relationship in relationships:
            key = (
                self._normalize_entity_name(relationship.source_entity),
                self._normalize_entity_name(relationship.target_entity),
                relationship.relationship_type
            )
            
            if key not in relationship_groups:
                relationship_groups[key] = []
            relationship_groups[key].append(relationship)
        
        # Merge relationships in each group
        merged_relationships = []
        for group in relationship_groups.values():
            merged_relationship = self._merge_relationship_group(group)
            merged_relationships.append(merged_relationship)
        
        logger.info(f"Merged {len(relationships)} relationships into {len(merged_relationships)} unique relationships")
        return merged_relationships
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        if not name:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', name.lower().strip())
        
        # Remove common prefixes/suffixes
        prefixes = ['the ', 'a ', 'an ']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        return normalized
    
    def _merge_relationship_group(self, relationships: List[Relationship]) -> Relationship:
        """Merge a group of similar relationships."""
        if len(relationships) == 1:
            return relationships[0]
        
        # Use the first relationship as template
        template = relationships[0]
        
        # Combine descriptions
        descriptions = [r.description for r in relationships if r.description]
        merged_description = "; ".join(set(descriptions))
        
        # Combine source chunks
        source_chunks = []
        for relationship in relationships:
            source_chunks.extend(relationship.source_chunks)
        
        return Relationship(
            source_entity=template.source_entity,
            target_entity=template.target_entity,
            relationship_type=template.relationship_type,
            description=merged_description,
            source_chunks=list(set(source_chunks)),
            confidence=sum(r.confidence for r in relationships) / len(relationships)
        )
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics and configuration."""
        return {
            'max_entities_per_chunk': self.max_entities_per_chunk,
            'entity_types': self.entity_types,
            'tuple_delimiter': self.tuple_delimiter,
            'record_delimiter': self.record_delimiter,
            'model_info': self.model_manager.get_model_info() if hasattr(self.model_manager, 'get_model_info') else {}
        }

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