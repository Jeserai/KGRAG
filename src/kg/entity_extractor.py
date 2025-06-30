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
            # Generate the extraction prompt
            prompt = self._create_extraction_prompt(text)
            
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
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create the entity extraction prompt."""
        
        entity_types_str = ", ".join(self.entity_types)
        
        prompt = f"""You are an AI assistant that helps extract entities and relationships from text for knowledge graph construction.

## Task
Extract entities and relationships from the provided text. Focus on the most important and relevant entities that would be valuable in a knowledge graph.

## Entity Types
Extract entities of these types: {entity_types_str}

## Output Format
For each entity, provide:
(entity_name{self.tuple_delimiter}entity_type{self.tuple_delimiter}entity_description){self.record_delimiter}

For each relationship, provide:
(source_entity{self.tuple_delimiter}relationship_type{self.tuple_delimiter}target_entity{self.tuple_delimiter}relationship_description){self.record_delimiter}

## Guidelines
1. Entity names should be specific and descriptive
2. Descriptions should be concise but informative
3. Relationships should capture meaningful connections
4. Use consistent naming (e.g., "John Smith" not "John" and "Smith" separately)
5. Focus on entities that appear multiple times or are central to the text
6. Maximum {self.max_entities_per_chunk} entities per text

## Text to Process:
{text}

## Extracted Entities and Relationships:
"""
        return prompt
    
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
    
    def merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Merge duplicate entities based on name similarity.
        
        Args:
            entities: List of entities to merge
            
        Returns:
            List of merged entities
        """
        if not entities:
            return []
        
        # Group entities by normalized name
        entity_groups = {}
        
        for entity in entities:
            # Normalize entity name for grouping
            normalized_name = self._normalize_entity_name(entity.name)
            
            if normalized_name not in entity_groups:
                entity_groups[normalized_name] = []
            entity_groups[normalized_name].append(entity)
        
        # Merge entities in each group
        merged_entities = []
        for group in entity_groups.values():
            merged_entity = self._merge_entity_group(group)
            merged_entities.append(merged_entity)
        
        logger.info(f"Merged {len(entities)} entities into {len(merged_entities)} unique entities")
        return merged_entities
    
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
    
    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """Merge a group of similar entities."""
        if len(entities) == 1:
            return entities[0]
        
        # Use the most common name
        names = [e.name for e in entities]
        merged_name = max(set(names), key=names.count)
        
        # Use the most common type
        types = [e.type for e in entities]
        merged_type = max(set(types), key=types.count)
        
        # Combine descriptions
        descriptions = [e.description for e in entities if e.description]
        merged_description = "; ".join(set(descriptions))
        
        # Combine source chunks
        source_chunks = []
        for entity in entities:
            source_chunks.extend(entity.source_chunks)
        
        return Entity(
            name=merged_name,
            type=merged_type,
            description=merged_description,
            source_chunks=list(set(source_chunks)),
            confidence=sum(e.confidence for e in entities) / len(entities)
        )
    
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