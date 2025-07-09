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
            # Augment a prompt for extraction using the LLM
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

            logger.debug(
                "Extracted %d entities and %d relationships from chunk %s using LLM", 
                len(entities), len(relationships), chunk_id
            )
            return entities, relationships

        except Exception as e:
            # If the LLM is unavailable (e.g., model not downloaded / no GPU),
            # fall back to a very simple regex-based entity extraction so that
            # the rest of the pipeline continues to function in CPU-only test
            # environments. This heuristic is *far* from perfect but provides
            # enough signal for unit tests and quick demos.
            logger.warning(
                "LLM extraction failed for chunk %s (%s). Falling back to regex-based extraction.",
                chunk_id, e
            )
            return self._regex_fallback_extraction(text, chunk_id)

    # ---------------------------------------------------------------------
    # Fallback Helpers
    # ---------------------------------------------------------------------

    def _regex_fallback_extraction(self, text: str, chunk_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Very naive noun-phrase / proper-noun extractor as a last resort.

        It identifies capitalised words (and simple multi-word phrases) as
        candidate entities and labels them as type "CONCEPT". No relationships
        are extracted in this mode.

        Args:
            text: The raw text to process.
            chunk_id: Originating chunk ID.

        Returns:
            A tuple of (entities, relationships).
        """
        # Match capitalised words or up to 3-token capitalised phrases
        candidate_pattern = r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b"
        candidates = set(re.findall(candidate_pattern, text))

        entities: List[Entity] = []
        for name in candidates:
            # Skip very common stop words / false positives
            if name.lower() in {"The", "This", "That", "It", "Its", "And", "But", "With"}:
                continue
            entities.append(
                Entity(
                    name=name,
                    type="CONCEPT",
                    description="N/A (regex fallback)",
                    source_chunks=[chunk_id],
                    confidence=0.3,
                )
            )

        # No relationships discovered by the regex fallback
        return entities, []

    # ------------------------------------------------------------------
    # Utility helpers (merging, normalisation)
    # ------------------------------------------------------------------

    @staticmethod
    def merge_relationships(relationships: List[Relationship]) -> List[Relationship]:
        """Merge duplicate relationships based on source, target and type.

        The implementation is intentionally lightweight — we simply treat
        relationships with the same (normalized) source entity, target
        entity and relationship type as identical and keep the first
        occurrence while merging metadata.

        Args:
            relationships: A list of Relationship objects (possibly with
                duplicates).

        Returns:
            A list where duplicate relationships have been collapsed.
        """
        if not relationships:
            return []

        def _norm(name: str) -> str:
            return re.sub(r"\s+", " ", name.lower().strip())

        grouped: Dict[Tuple[str, str, str], List[Relationship]] = {}
        for rel in relationships:
            key = (_norm(rel.source_entity), _norm(rel.target_entity), rel.relationship_type.upper())
            grouped.setdefault(key, []).append(rel)

        merged: List[Relationship] = []
        for rels in grouped.values():
            if len(rels) == 1:
                merged.append(rels[0])
                continue
            # Merge descriptions and source chunks; average confidence
            template = rels[0]
            description = "; ".join({r.description for r in rels if r.description})
            source_chunks = list({chunk for r in rels for chunk in r.source_chunks})
            confidence = sum(r.confidence for r in rels) / len(rels)

            merged.append(
                Relationship(
                    source_entity=template.source_entity,
                    target_entity=template.target_entity,
                    relationship_type=template.relationship_type,
                    description=description,
                    source_chunks=source_chunks,
                    confidence=confidence,
                )
            )

        logger.info("Merged %d relationships into %d unique relationships", len(relationships), len(merged))
        return merged
    
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