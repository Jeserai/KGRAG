"""
Entity and relationship extraction for GraphRAG using local LLMs.
Supports Qwen and Llama models for offline entity extraction.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
from src.models.prompt import get_extraction_prompt

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str
    description: str
    source_chunks: List[str]
    confidence: float = 1.0
    properties: Optional[Dict[str, Any]] = None


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_entity: str
    target_entity: str
    relationship_type: str
    description: str
    source_chunks: List[str]
    confidence: float = 1.0
    properties: Optional[Dict[str, Any]] = None


class EntityExtractor:
    """Extract entities and relationships from text using local LLMs."""
    
    def __init__(self, model_manager, max_entities_per_chunk: int = 20):
        """
        Initialize entity extractor.
        
        Args:
            model_manager: LocalLLMManager instance
            max_entities_per_chunk: Maximum entities to extract per chunk
        """
        self.model_manager = model_manager
        self.max_entities_per_chunk = max_entities_per_chunk
        
        # Entity types to extract
        self.entity_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "EVENT", 
            "CONCEPT", "TECHNOLOGY", "PRODUCT", "DATE"
        ]
        
        # Delimiter tokens for parsing structured output
        self.tuple_delimiter = "<|>"
        self.record_delimiter = "##"
        self.completion_delimiter = "<|COMPLETE|>"
        
        logger.info("Initialized EntityExtractor")
    
    def extract_entities_and_relationships(self, text: str, chunk_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from a text chunk.
        
        Args:
            text: Text content to process
            chunk_id: Identifier for the text chunk
            
        Returns:
            Tuple of (entities, relationships)
        """
        if not text or not text.strip():
            return [], []
        
        try:
            # Augement a prompt for extraction
            prompt = get_extraction_prompt(text)
            
            # Get LLM response
            response = self.model_manager.inference(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.1,
                stop_sequences=[self.completion_delimiter]
            )
            
            # Parse the structured response
            entities, relationships = self._parse_extraction_response(response, chunk_id)
            
            logger.debug(f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk {chunk_id}")
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Error extracting from chunk {chunk_id}: {e}")
            return [], []
    
    def extract_batch(self, text_chunks: List[Tuple[str, str]]) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from multiple text chunks in parallel.
        
        Args:
            text_chunks: List of (text, chunk_id) tuples
            
        Returns:
            Tuple of (all_entities, all_relationships)
        """
        all_entities = []
        all_relationships = []
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.extract_entities_and_relationships, text, chunk_id)
                for text, chunk_id in text_chunks
            ]
            
            for future in futures:
                try:
                    entities, relationships = future.result()
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
        
        logger.info(f"Batch extracted {len(all_entities)} entities and {len(all_relationships)} relationships")
        return all_entities, all_relationships
    
    def _parse_extraction_response(self, response: str, chunk_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Parse the structured LLM response into entities and relationships."""
        entities = []
        relationships = []
        
        if not response:
            return entities, relationships
        
        # Split by record delimiter
        records = response.split(self.record_delimiter)
        
        for record in records:
            record = record.strip()
            if not record or record == self.completion_delimiter:
                continue
            
            # Check if this is a valid tuple format
            if record.startswith('(') and record.endswith(')'):
                # Remove parentheses
                content = record[1:-1]
                parts = content.split(self.tuple_delimiter)
                
                if len(parts) == 3:
                    # This is an entity: (name, type, description)
                    name, entity_type, description = [p.strip() for p in parts]
                    if name and entity_type:
                        entity = Entity(
                            name=name,
                            type=entity_type.upper(),
                            description=description,
                            source_chunks=[chunk_id]
                        )
                        entities.append(entity)
                
                elif len(parts) == 4:
                    # This is a relationship: (source, relation, target, description)
                    source, relation_type, target, description = [p.strip() for p in parts]
                    if source and target and relation_type:
                        relationship = Relationship(
                            source_entity=source,
                            target_entity=target,
                            relationship_type=relation_type.upper(),
                            description=description,
                            source_chunks=[chunk_id]
                        )
                        relationships.append(relationship)
        
        return entities, relationships
    
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