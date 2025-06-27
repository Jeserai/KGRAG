"""
Tests for entity extraction functionality.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from src.kg.entity_extractor import EntityExtractor, ExtractedTriple
from src.kg.neo4j_connector import Entity, Relationship
from src.data.document_processor import DocumentChunk


class TestExtractedTriple:
    """Test ExtractedTriple dataclass."""
    
    def test_extracted_triple_creation(self):
        """Test creating an ExtractedTriple instance."""
        triple = ExtractedTriple(
            subject="artificial intelligence",
            subject_type="TECHNOLOGY",
            predicate="is_part_of",
            object="computer science",
            object_type="CONCEPT",
            confidence=0.85,
            context="AI is a branch of computer science."
        )
        
        assert triple.subject == "artificial intelligence"
        assert triple.subject_type == "TECHNOLOGY"
        assert triple.predicate == "is_part_of"
        assert triple.object == "computer science"
        assert triple.object_type == "CONCEPT"
        assert triple.confidence == 0.85
        assert triple.context == "AI is a branch of computer science."


class TestEntityExtractor:
    """Test EntityExtractor class."""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager for testing."""
        mock_manager = MagicMock()
        
        # Mock entity extraction response
        entity_response = json.dumps([
            {"name": "artificial intelligence", "type": "TECHNOLOGY"},
            {"name": "Stanford University", "type": "ORGANIZATION"},
            {"name": "machine learning", "type": "CONCEPT"}
        ])
        
        # Mock relationship extraction response
        rel_response = json.dumps([
            {"subject": "machine learning", "predicate": "is_part_of", "object": "artificial intelligence"},
            {"subject": "Stanford University", "predicate": "researches", "object": "machine learning"}
        ])
        
        mock_manager.generate_text.side_effect = [entity_response, rel_response]
        return mock_manager
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing entity extraction."""
        return """
        Artificial intelligence (AI) is a branch of computer science that focuses on creating 
        intelligent machines. Stanford University has been at the forefront of AI research, 
        particularly in machine learning. Machine learning is a subset of AI that enables 
        computers to learn from data without being explicitly programmed.
        """
    
    @pytest.fixture
    def sample_chunk(self, sample_text):
        """Create a sample document chunk."""
        return DocumentChunk(
            content=sample_text,
            chunk_id="test_chunk_001",
            document_id="test_doc_001",
            metadata={"source": "test.txt"}
        )
    
    def test_entity_extractor_initialization(self, mock_llm_manager):
        """Test EntityExtractor initialization."""
        extractor = EntityExtractor(mock_llm_manager)
        
        assert extractor.llm_manager == mock_llm_manager
        assert extractor.entity_confidence_threshold == 0.7
        assert extractor.relationship_confidence_threshold == 0.6
        assert extractor.max_entities_per_chunk == 20
        assert extractor.max_relationships_per_chunk == 30
    
    def test_entity_extractor_default_initialization(self):
        """Test EntityExtractor initialization with default LLM manager."""
        with patch('src.kg.entity_extractor.LocalLLMManager') as mock_llm_class:
            mock_llm_manager = MagicMock()
            mock_llm_class.return_value = mock_llm_manager
            
            extractor = EntityExtractor()
            
            assert extractor.llm_manager == mock_llm_manager
            mock_llm_class.assert_called_once()
    
    def test_generate_entity_id(self, mock_llm_manager):
        """Test entity ID generation."""
        extractor = EntityExtractor(mock_llm_manager)
        
        entity_id = extractor._generate_entity_id("artificial intelligence", "TECHNOLOGY")
        
        assert entity_id.startswith("entity_")
        assert len(entity_id) == 15  # "entity_" + 8 hex chars
    
    def test_generate_relationship_id(self, mock_llm_manager):
        """Test relationship ID generation."""
        extractor = EntityExtractor(mock_llm_manager)
        
        rel_id = extractor._generate_relationship_id("entity_123", "is_part_of", "entity_456")
        
        assert rel_id.startswith("rel_")
        assert len(rel_id) == 11  # "rel_" + 8 hex chars
    
    def test_create_entity_extraction_prompt(self, mock_llm_manager):
        """Test entity extraction prompt creation."""
        extractor = EntityExtractor(mock_llm_manager)
        
        text = "AI is a technology that enables machine learning."
        prompt = extractor._create_entity_extraction_prompt(text)
        
        assert "Extract entities" in prompt
        assert text in prompt
        assert "JSON" in prompt
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
    
    def test_create_relationship_extraction_prompt(self, mock_llm_manager):
        """Test relationship extraction prompt creation."""
        extractor = EntityExtractor(mock_llm_manager)
        
        text = "AI enables machine learning."
        entities = [
            {"name": "AI", "type": "TECHNOLOGY"},
            {"name": "machine learning", "type": "CONCEPT"}
        ]
        
        prompt = extractor._create_relationship_extraction_prompt(text, entities)
        
        assert "relationships" in prompt
        assert text in prompt
        assert "AI (TECHNOLOGY)" in prompt
        assert "machine learning (CONCEPT)" in prompt
        assert "JSON" in prompt
    
    def test_parse_json_response_valid(self, mock_llm_manager):
        """Test parsing valid JSON response."""
        extractor = EntityExtractor(mock_llm_manager)
        
        response = '[{"name": "AI", "type": "TECHNOLOGY"}]'
        result = extractor._parse_json_response(response)
        
        assert len(result) == 1
        assert result[0]["name"] == "AI"
        assert result[0]["type"] == "TECHNOLOGY"
    
    def test_parse_json_response_with_markdown(self, mock_llm_manager):
        """Test parsing JSON response with markdown formatting."""
        extractor = EntityExtractor(mock_llm_manager)
        
        response = '```json\n[{"name": "AI", "type": "TECHNOLOGY"}]\n```'
        result = extractor._parse_json_response(response)
        
        assert len(result) == 1
        assert result[0]["name"] == "AI"
    
    def test_parse_json_response_with_text(self, mock_llm_manager):
        """Test parsing JSON response with surrounding text."""
        extractor = EntityExtractor(mock_llm_manager)
        
        response = 'Here are the entities: [{"name": "AI", "type": "TECHNOLOGY"}] and that\'s it.'
        result = extractor._parse_json_response(response)
        
        assert len(result) == 1
        assert result[0]["name"] == "AI"
    
    def test_parse_json_response_invalid(self, mock_llm_manager):
        """Test parsing invalid JSON response."""
        extractor = EntityExtractor(mock_llm_manager)
        
        response = 'Invalid JSON content'
        result = extractor._parse_json_response(response)
        
        assert result == []
    
    def test_validate_entity_valid(self, mock_llm_manager):
        """Test validation of valid entity."""
        extractor = EntityExtractor(mock_llm_manager)
        
        entity_data = {"name": "artificial intelligence", "type": "TECHNOLOGY"}
        is_valid = extractor._validate_entity(entity_data)
        
        assert is_valid is True
    
    def test_validate_entity_invalid_name(self, mock_llm_manager):
        """Test validation of entity with invalid name."""
        extractor = EntityExtractor(mock_llm_manager)
        
        # Too short name
        entity_data = {"name": "a", "type": "TECHNOLOGY"}
        is_valid = extractor._validate_entity(entity_data)
        assert is_valid is False
        
        # Empty name
        entity_data = {"name": "", "type": "TECHNOLOGY"}
        is_valid = extractor._validate_entity(entity_data)
        assert is_valid is False
    
    def test_validate_entity_stop_word(self, mock_llm_manager):
        """Test validation of entity that is a stop word."""
        extractor = EntityExtractor(mock_llm_manager)
        
        entity_data = {"name": "the", "type": "CONCEPT"}
        is_valid = extractor._validate_entity(entity_data)
        
        assert is_valid is False
    
    def test_validate_entity_missing_fields(self, mock_llm_manager):
        """Test validation of entity with missing fields."""
        extractor = EntityExtractor(mock_llm_manager)
        
        # Missing type
        entity_data = {"name": "AI"}
        is_valid = extractor._validate_entity(entity_data)
        assert is_valid is False
        
        # Missing name
        entity_data = {"type": "TECHNOLOGY"}
        is_valid = extractor._validate_entity(entity_data)
        assert is_valid is False
    
    def test_validate_relationship_valid(self, mock_llm_manager):
        """Test validation of valid relationship."""
        extractor = EntityExtractor(mock_llm_manager)
        
        entities = {"AI", "machine learning"}
        rel_data = {"subject": "AI", "predicate": "enables", "object": "machine learning"}
        is_valid = extractor._validate_relationship(rel_data, entities)
        
        assert is_valid is True
    
    def test_validate_relationship_invalid_entities(self, mock_llm_manager):
        """Test validation of relationship with invalid entities."""
        extractor = EntityExtractor(mock_llm_manager)
        
        entities = {"AI", "machine learning"}
        rel_data = {"subject": "AI", "predicate": "enables", "object": "unknown_entity"}
        is_valid = extractor._validate_relationship(rel_data, entities)
        
        assert is_valid is False
    
    def test_validate_relationship_missing_fields(self, mock_llm_manager):
        """Test validation of relationship with missing fields."""
        extractor = EntityExtractor(mock_llm_manager)
        
        entities = {"AI", "machine learning"}
        
        # Missing predicate
        rel_data = {"subject": "AI", "object": "machine learning"}
        is_valid = extractor._validate_relationship(rel_data, entities)
        assert is_valid is False
        
        # Missing subject
        rel_data = {"predicate": "enables", "object": "machine learning"}
        is_valid = extractor._validate_relationship(rel_data, entities)
        assert is_valid is False
    
    def test_extract_entities_from_text(self, mock_llm_manager, sample_text):
        """Test entity extraction from text."""
        extractor = EntityExtractor(mock_llm_manager)
        
        entities = extractor.extract_entities_from_text(sample_text)
        
        assert len(entities) > 0
        assert all(isinstance(entity, Entity) for entity in entities)
        
        # Check that entities have required attributes
        for entity in entities:
            assert hasattr(entity, 'entity_id')
            assert hasattr(entity, 'name')
            assert hasattr(entity, 'entity_type')
            assert hasattr(entity, 'properties')
    
    def test_extract_relationships_from_text(self, mock_llm_manager, sample_text):
        """Test relationship extraction from text."""
        extractor = EntityExtractor(mock_llm_manager)
        
        # Create some entities first
        entities = [
            Entity("entity_1", "artificial intelligence", "TECHNOLOGY", {}),
            Entity("entity_2", "machine learning", "CONCEPT", {}),
            Entity("entity_3", "Stanford University", "ORGANIZATION", {})
        ]
        
        relationships = extractor.extract_relationships_from_text(sample_text, entities)
        
        assert len(relationships) > 0
        assert all(isinstance(rel, Relationship) for rel in relationships)
        
        # Check that relationships have required attributes
        for rel in relationships:
            assert hasattr(rel, 'relationship_id')
            assert hasattr(rel, 'source_entity_id')
            assert hasattr(rel, 'target_entity_id')
            assert hasattr(rel, 'relationship_type')
            assert hasattr(rel, 'properties')
    
    def test_extract_from_chunk(self, mock_llm_manager, sample_chunk):
        """Test extraction from document chunk."""
        extractor = EntityExtractor(mock_llm_manager)
        
        entities, relationships = extractor.extract_from_chunk(sample_chunk)
        
        assert len(entities) > 0
        assert len(relationships) > 0
        assert all(isinstance(entity, Entity) for entity in entities)
        assert all(isinstance(rel, Relationship) for rel in relationships)
    
    def test_extract_from_chunks_batch(self, mock_llm_manager):
        """Test batch extraction from multiple chunks."""
        extractor = EntityExtractor(mock_llm_manager)
        
        chunks = [
            DocumentChunk("AI is a technology.", "chunk_1", "doc_1"),
            DocumentChunk("Machine learning is part of AI.", "chunk_2", "doc_1")
        ]
        
        entities, relationships = extractor.extract_from_chunks_batch(chunks)
        
        assert len(entities) > 0
        assert len(relationships) > 0
    
    def test_deduplicate_entities(self, mock_llm_manager):
        """Test entity deduplication."""
        extractor = EntityExtractor(mock_llm_manager)
        
        entities = [
            Entity("entity_1", "AI", "TECHNOLOGY", {}),
            Entity("entity_2", "AI", "TECHNOLOGY", {}),  # Duplicate
            Entity("entity_3", "machine learning", "CONCEPT", {}),
            Entity("entity_4", "Machine Learning", "CONCEPT", {})  # Case variant
        ]
        
        deduplicated = extractor._deduplicate_entities(entities)
        
        # Should have fewer entities after deduplication
        assert len(deduplicated) < len(entities)
        
        # Check that unique entities are preserved
        unique_names = {entity.name.lower() for entity in deduplicated}
        assert len(unique_names) == len(deduplicated)
    
    def test_get_extraction_statistics(self, mock_llm_manager):
        """Test extraction statistics."""
        extractor = EntityExtractor(mock_llm_manager)
        
        # Simulate some extractions
        extractor._extraction_stats = {
            'entities_extracted': 10,
            'relationships_extracted': 15,
            'chunks_processed': 5
        }
        
        stats = extractor.get_extraction_statistics()
        
        assert stats['entities_extracted'] == 10
        assert stats['relationships_extracted'] == 15
        assert stats['chunks_processed'] == 5
        assert 'avg_entities_per_chunk' in stats
        assert 'avg_relationships_per_chunk' in stats
    
    @patch('src.kg.entity_extractor.config_manager')
    def test_custom_configuration(self, mock_config_manager, mock_llm_manager):
        """Test EntityExtractor with custom configuration."""
        # Mock configuration
        mock_config_manager.get_kg_config.return_value = {
            'entity_confidence_threshold': 0.8,
            'relationship_confidence_threshold': 0.7,
            'max_entities_per_chunk': 15,
            'max_relationships_per_chunk': 25
        }
        
        extractor = EntityExtractor(mock_llm_manager)
        
        assert extractor.entity_confidence_threshold == 0.8
        assert extractor.relationship_confidence_threshold == 0.7
        assert extractor.max_entities_per_chunk == 15
        assert extractor.max_relationships_per_chunk == 25 