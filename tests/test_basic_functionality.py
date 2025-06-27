"""
Basic functionality tests for KGRAG system.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.model_config import ModelConfig, ModelConfigManager
from src.data.document_processor import DocumentProcessor, DocumentChunk


class TestBasicModelConfig:
    """Basic tests for model configuration."""
    
    def test_model_config_creation(self):
        """Test creating a ModelConfig instance."""
        config = ModelConfig(
            name="test-model",
            type="llm",
            device="cpu",
            max_length=512
        )
        
        assert config.name == "test-model"
        assert config.type == "llm"
        assert config.device == "cpu"
        assert config.max_length == 512
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig(
            name="test-model",
            type="embedding",
            device="cpu"
        )
        
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.normalize_embeddings is True


class TestBasicDocumentProcessor:
    """Basic tests for document processing."""
    
    def test_document_processor_initialization(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(
            chunk_size=512,
            chunk_overlap=50
        )
        
        assert processor.chunk_size == 512
        assert processor.chunk_o