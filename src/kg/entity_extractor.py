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