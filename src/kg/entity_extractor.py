"""
Entity and relationship extraction using local LLMs for Knowledge Graph construction.
Optimized prompts and processing for open source models.
"""

import logging
import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

from ..models.local_llm import LocalLLMManager
from ..models.model_config import config_manager
from ..data.document_processor import DocumentChunk
from .neo4j_connector import Entity, Relationship


logger = logging.getLogger(__name__)


@dataclass
class ExtractedTriple:
    """Represents an extracted entity-relationship-entity triple."""
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    confidence: float
    context: str


class EntityExtractor:
    """Extracts entities and relationships from text using local LLMs."""
    
    def __init__(self, llm_manager: Optional[LocalLLMManager] = None):
        """Initialize the entity extractor.
        
        Args:
            llm_manager: Local LLM manager. If None, creates a new one.
        """
        self.llm_manager = llm_manager or LocalLLMManager()
        self.kg_config = config_manager.get_kg_config()
        
        # Extraction thresholds
        self.entity_confidence_threshold = self.kg_config.get('entity_confidence_threshold', 0.7)
        self.relationship_confidence_threshold = self.kg_config.get('relationship_confidence_threshold', 0.6)
        self.max_entities_per_chunk = self.kg_config.get('max_entities_per_chunk', 20)
        self.max_relationships_per_chunk = self.kg_config.get('max_relationships_per_chunk', 30)
        
        # Common entity types for filtering
        self.common_entity_types = {
            'PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'TIME', 'MONEY', 
            'PERCENT', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
            'CONCEPT', 'TECHNOLOGY', 'METHODOLOGY', 'RESEARCH', 'THEORY'
        }
        
        logger.info("Initialized EntityExtractor with local LLM")
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a unique entity ID."""
        content = f"{entity_type}:{name}".lower()
        entity_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"entity_{entity_hash}"
    
    def _generate_relationship_id(self, subject_id: str, predicate: str, object_id: str) -> str:
        """Generate a unique relationship ID."""
        content = f"{subject_id}:{predicate}:{object_id}".lower()
        rel_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"rel_{rel_hash}"
    
    def _create_entity_extraction_prompt(self, text: str) -> str:
        """Create optimized prompt for entity extraction."""
        prompt = f"""Extract entities from the following text. Focus on meaningful concepts, people, organizations, locations, and domain-specific terms.

Text: "{text}"

Instructions:
1. Identify important entities (nouns, proper nouns, concepts)
2. Classify each entity with an appropriate type
3. Only include entities that are significant to understanding the text
4. Avoid overly common words or stop words
5. Format as JSON list with "name" and "type" fields

Entity Types to use: PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY, PRODUCT, EVENT, DATE, RESEARCH, THEORY, METHODOLOGY

Example output:
[
  {{"name": "artificial intelligence", "type": "TECHNOLOGY"}},
  {{"name": "Stanford University", "type": "ORGANIZATION"}},
  {{"name": "machine learning", "type": "CONCEPT"}}
]

Output (JSON only, no explanation):"""
        
        return prompt
    
    def _create_relationship_extraction_prompt(self, text: str, entities: List[Dict[str, str]]) -> str:
        """Create optimized prompt for relationship extraction."""
        entity_list = ", ".join([f"{e['name']} ({e['type']})" for e in entities])
        
        prompt = f"""Given the text and extracted entities, identify relationships between the entities.

Text: "{text}"

Entities: {entity_list}

Instructions:
1. Find explicit relationships mentioned in the text between the given entities
2. Use clear, descriptive relationship types
3. Only include relationships that are directly stated or clearly implied
4. Focus on meaningful semantic relationships
5. Format as JSON list with "subject", "predicate", "object" fields

Common relationship types: relates_to, part_of, used_by, created_by, located_in, works_for, develops, implements, studies, improves, based_on

Example output:
[
  {{"subject": "machine learning", "predicate": "is_part_of", "object": "artificial intelligence"}},
  {{"subject": "Stanford University", "predicate": "researches", "object": "machine learning"}}
]

Output (JSON only, no explanation):"""
        
        return prompt
    
    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON response from LLM, handling common formatting issues."""
        # Clean the response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1])
        
        # Remove any text before the first [ or {
        json_start = max(response.find('['), response.find('{'))
        if json_start > 0:
            response = response[json_start:]
        
        # Remove any text after the last ] or }
        json_end = max(response.rfind(']'), response.rfind('}'))
        if json_end > 0:
            response = response[:json_end + 1]
        
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return [parsed]
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response[:200]}...")
            return []
    
    def _validate_entity(self, entity_data: Dict[str, Any]) -> bool:
        """Validate extracted entity data."""
        required_fields = {'name', 'type'}
        if not all(field in entity_data for field in required_fields):
            return False
        
        name = entity_data['name'].strip()
        entity_type = entity_data['type'].strip().upper()
        
        # Filter out invalid entities
        if not name or len(name) < 2:
            return False
        
        # Filter out common stop words and very generic terms
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'this', 'that', 'these', 'those', 'a', 'an', 'is', 'are', 'was', 
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'
        }
        
        if name.lower() in stop_words:
            return False
        
        # Accept known entity types or reasonable alternatives
        if entity_type in self.common_entity_types:
            return True
        
        # Accept domain-specific types that seem reasonable
        if len(entity_type) > 2 and entity_type.isalpha():
            return True
        
        return False
    
    def _validate_relationship(self, rel_data: Dict[str, Any], entities: Set[str]) -> bool:
        """Validate extracted relationship data."""
        required_fields = {'subject', 'predicate', 'object'}
        if not all(field in rel_data for field in required_fields):
            return False
        
        subject = rel_data['subject'].strip()
        predicate = rel_data['predicate'].strip()
        obj = rel_data['object'].strip()
        
        # Check that subject and object are in our entity list
        if subject not in entities or obj not in entities:
            return False
        
        # Check that predicate is meaningful
        if not predicate or len(predicate) < 2:
            return False
        
        # Don't allow self-relationships
        if subject == obj:
            return False
        
        return True
    
    def extract_entities_from_text(self, text: str) -> List[Entity]:
        """Extract entities from text using local LLM.
        
        Args:
            text: Input text to extract entities from.
            
        Returns:
            List of Entity objects.
        """
        if not text.strip():
            return []
        
        prompt = self._create_entity_extraction_prompt(text)
        
        try:
            response = self.llm_manager.generate_text(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.1
            )
            
            entity_data = self._parse_json_response(response)
            entities = []
            
            for data in entity_data[:self.max_entities_per_chunk]:
                if self._validate_entity(data):
                    entity_name = data['name'].strip()
                    entity_type = data['type'].strip().upper()
                    entity_id = self._generate_entity_id(entity_name, entity_type)
                    
                    entity = Entity(
                        id=entity_id,
                        name=entity_name,
                        type=entity_type,
                        properties={
                            'source_text': text[:200] + "..." if len(text) > 200 else text,
                            'extraction_method': 'local_llm'
                        },
                        confidence=0.8  # Default confidence for LLM extraction
                    )
                    entities.append(entity)
            
            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []
    
    def extract_relationships_from_text(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities using local LLM.
        
        Args:
            text: Input text.
            entities: List of entities to find relationships between.
            
        Returns:
            List of Relationship objects.
        """
        if not text.strip() or len(entities) < 2:
            return []
        
        # Convert entities to format for prompt
        entity_data = [{'name': e.name, 'type': e.type} for e in entities]
        entity_names = {e.name for e in entities}
        
        prompt = self._create_relationship_extraction_prompt(text, entity_data)
        
        try:
            response = self.llm_manager.generate_text(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.1
            )
            
            relationship_data = self._parse_json_response(response)
            relationships = []
            
            # Create entity name to ID mapping
            entity_name_to_id = {e.name: e.id for e in entities}
            
            for data in relationship_data[:self.max_relationships_per_chunk]:
                if self._validate_relationship(data, entity_names):
                    subject_name = data['subject'].strip()
                    predicate = data['predicate'].strip()
                    object_name = data['object'].strip()
                    
                    if subject_name in entity_name_to_id and object_name in entity_name_to_id:
                        subject_id = entity_name_to_id[subject_name]
                        object_id = entity_name_to_id[object_name]
                        rel_id = self._generate_relationship_id(subject_id, predicate, object_id)
                        
                        relationship = Relationship(
                            id=rel_id,
                            source_entity_id=subject_id,
                            target_entity_id=object_id,
                            relation_type=predicate,
                            properties={
                                'source_text': text[:200] + "..." if len(text) > 200 else text,
                                'extraction_method': 'local_llm'
                            },
                            confidence=0.7  # Default confidence for LLM extraction
                        )
                        relationships.append(relationship)
            
            logger.debug(f"Extracted {len(relationships)} relationships from text")
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to extract relationships: {e}")
            return []
    
    def extract_from_chunk(self, chunk: DocumentChunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a document chunk.
        
        Args:
            chunk: Document chunk to process.
            
        Returns:
            Tuple of (entities, relationships).
        """
        logger.debug(f"Processing chunk: {chunk.id}")
        
        # Extract entities first
        entities = self.extract_entities_from_text(chunk.text)
        
        # Add chunk metadata to entities
        for entity in entities:
            entity.properties.update({
                'chunk_id': chunk.id,
                'document_source': chunk.metadata.get('source', 'unknown'),
                'chunk_index': chunk.metadata.get('chunk_index', 0)
            })
        
        # Extract relationships between entities
        relationships = []
        if len(entities) >= 2:
            relationships = self.extract_relationships_from_text(chunk.text, entities)
            
            # Add chunk metadata to relationships
            for relationship in relationships:
                relationship.properties.update({
                    'chunk_id': chunk.id,
                    'document_source': chunk.metadata.get('source', 'unknown'),
                    'chunk_index': chunk.metadata.get('chunk_index', 0)
                })
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk {chunk.id}")
        return entities, relationships
    
    def extract_from_chunks_batch(self, chunks: List[DocumentChunk]) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from multiple chunks.
        
        Args:
            chunks: List of document chunks to process.
            
        Returns:
            Tuple of (all_entities, all_relationships).
        """
        all_entities = []
        all_relationships = []
        
        for i, chunk in enumerate(chunks):
            try:
                entities, relationships = self.extract_from_chunk(chunk)
                all_entities.extend(entities)
                all_relationships.extend(relationships)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk.id}: {e}")
                continue
        
        # Deduplicate entities by name and type
        unique_entities = self._deduplicate_entities(all_entities)
        
        # Filter relationships to only include those with valid entities
        entity_ids = {e.id for e in unique_entities}
        valid_relationships = [
            r for r in all_relationships 
            if r.source_entity_id in entity_ids and r.target_entity_id in entity_ids
        ]
        
        logger.info(f"Extracted {len(unique_entities)} unique entities and {len(valid_relationships)} relationships from {len(chunks)} chunks")
        return unique_entities, valid_relationships
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities based on name and type."""
        seen = {}
        unique_entities = []
        
        for entity in entities:
            key = (entity.name.lower(), entity.type.lower())
            if key not in seen:
                seen[key] = entity
                unique_entities.append(entity)
            else:
                # Merge properties from duplicate entities
                existing = seen[key]
                existing.properties.update(entity.properties)
                # Use higher confidence
                existing.confidence = max(existing.confidence, entity.confidence)
        
        return unique_entities
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get statistics about the extraction process."""
        return {
            "entity_confidence_threshold": self.entity_confidence_threshold,
            "relationship_confidence_threshold": self.relationship_confidence_threshold,
            "max_entities_per_chunk": self.max_entities_per_chunk,
            "max_relationships_per_chunk": self.max_relationships_per_chunk,
            "model_info": self.llm_manager.get_model_info() if self.llm_manager else None
        } 