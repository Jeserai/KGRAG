"""
Tests for model configuration management.
"""

import pytest
import tempfile
import yaml
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.model_config import ModelConfig, ModelConfigManager


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test creating a ModelConfig instance."""
        config = ModelConfig(
            name="test-model",
            type="llm",
            device="cuda",  # Use cuda for GPU servers
            max_length=512,
            temperature=0.7
        )
        
        assert config.name == "test-model"
        assert config.type == "llm"
        assert config.device == "cuda"
        assert config.max_length == 512
        assert config.temperature == 0.7
        assert config.batch_size == 1  # default value
    
    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = ModelConfig(
            name="test-model",
            type="embedding",
            device="cuda"  # Use cuda for GPU servers
        )
        
        assert config.max_length is None
        assert config.temperature is None
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.normalize_embeddings is True


class TestModelConfigManager:
    """Test ModelConfigManager class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {
            "models": {
                "llama_3_2_3b": {
                    "name": "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct",
                    "type": "llm",
                    "device": "cuda",
                    "max_length": 2048,
                    "temperature": 0.7,
                    "load_in_4bit": True
                },
                "qwen_embedding": {
                    "name": "/data/models/huggingface/qwen/Qwen1.5-7B",
                    "type": "embedding",
                    "device": "cuda",
                    "max_seq_length": 512,
                    "normalize_embeddings": True
                }
            },
            "default": {
                "llm_model": "llama_3_2_3b",
                "embedding_model": "qwen_embedding"
            },
            "performance": {
                "max_memory_gb": 32,
                "enable_model_caching": True
            },
            "kg": {
                "entity_confidence_threshold": 0.7,
                "relationship_confidence_threshold": 0.6
            }
        }
    
    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_config_manager_initialization(self, temp_config_file):
        """Test ModelConfigManager initialization."""
        manager = ModelConfigManager(temp_config_file)
        
        assert manager.config_path == temp_config_file
        assert "models" in manager._config
        assert "default" in manager._config
    
    def test_get_model_config(self, temp_config_file):
        """Test getting a specific model configuration."""
        manager = ModelConfigManager(temp_config_file)
        
        config = manager.get_model_config("llama_3_2_3b")
        
        assert config.name == "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct"
        assert config.type == "llm"
        # Device should be cuda if available, otherwise cpu
        if torch.cuda.is_available():
            assert config.device == "cuda"
        else:
            assert config.device == "cpu"
        assert config.max_length == 2048
        assert config.temperature == 0.7
        # Quantization should be disabled for CPU
        if torch.cuda.is_available():
            assert config.load_in_4bit is True
        else:
            assert config.load_in_4bit is False
    
    def test_get_model_config_not_found(self, temp_config_file):
        """Test getting a non-existent model configuration."""
        manager = ModelConfigManager(temp_config_file)
        
        with pytest.raises(ValueError, match="Model configuration not found"):
            manager.get_model_config("non_existent_model")
    
    def test_get_default_configs(self, temp_config_file):
        """Test getting default configurations."""
        manager = ModelConfigManager(temp_config_file)
        
        llm_config = manager.get_default_llm_config()
        embedding_config = manager.get_default_embedding_config()
        
        assert llm_config.name == "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct"
        assert embedding_config.name == "/data/models/huggingface/qwen/Qwen1.5-7B"
    
    def test_list_available_models(self, temp_config_file):
        """Test listing available models."""
        manager = ModelConfigManager(temp_config_file)
        
        all_models = manager.list_available_models()
        llm_models = manager.list_available_models("llm")
        embedding_models = manager.list_available_models("embedding")
        
        assert len(all_models) == 2
        assert len(llm_models) == 1
        assert len(embedding_models) == 1
        
        assert "llama_3_2_3b" in all_models
        assert "qwen_embedding" in all_models
    
    def test_get_performance_config(self, temp_config_file):
        """Test getting performance configuration."""
        manager = ModelConfigManager(temp_config_file)
        
        perf_config = manager.get_performance_config()
        
        assert perf_config["max_memory_gb"] == 32
        assert perf_config["enable_model_caching"] is True
    
    def test_get_kg_config(self, temp_config_file):
        """Test getting knowledge graph configuration."""
        manager = ModelConfigManager(temp_config_file)
        
        kg_config = manager.get_kg_config()
        
        assert kg_config["entity_confidence_threshold"] == 0.7
        assert kg_config["relationship_confidence_threshold"] == 0.6
    
    def test_update_model_config(self, temp_config_file):
        """Test updating model configuration."""
        manager = ModelConfigManager(temp_config_file)
        
        updates = {"temperature": 0.8, "max_length": 1024}
        manager.update_model_config("llama_3_2_3b", updates)
        
        config = manager.get_model_config("llama_3_2_3b")
        assert config.temperature == 0.8
        assert config.max_length == 1024
    
    @patch('torch.cuda.is_available')
    def test_auto_device_detection_cuda_unavailable(self, mock_cuda_available, temp_config_file):
        """Test automatic device detection when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        
        manager = ModelConfigManager(temp_config_file)
        config = manager.get_model_config("llama_3_2_3b")
        
        assert config.device == "cpu"
        assert config.load_in_4bit is False
        assert config.load_in_8bit is False
    
    @patch('torch.cuda.is_available')
    def test_auto_device_detection_cuda_available(self, mock_cuda_available, temp_config_file):
        """Test automatic device detection when CUDA is available."""
        mock_cuda_available.return_value = True
        
        manager = ModelConfigManager(temp_config_file)
        config = manager.get_model_config("llama_3_2_3b")
        
        assert config.device == "cuda"
        assert config.load_in_4bit is True
    
    def test_device_auto_detection_real_environment(self, temp_config_file):
        """Test device detection in real environment (GPU-aware)."""
        manager = ModelConfigManager(temp_config_file)
        config = manager.get_model_config("llama_3_2_3b")
        
        # Should work regardless of whether CUDA is available
        assert config.device in ["cuda", "cpu"]
        
        # Quantization should be disabled for CPU
        if config.device == "cpu":
            assert config.load_in_4bit is False
            assert config.load_in_8bit is False
        else:
            # On GPU, quantization settings should be preserved
            assert config.load_in_4bit is True
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        with pytest.raises(FileNotFoundError):
            ModelConfigManager(Path("/non/existent/path.yaml"))
    
    def test_invalid_yaml_config(self):
        """Test handling of invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid YAML configuration"):
                ModelConfigManager(Path(temp_path))
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_missing_required_sections(self):
        """Test handling of missing required configuration sections."""
        invalid_config = {
            "models": {
                "test": {"name": "test", "type": "llm", "device": "cuda"}
            }
            # Missing "default" section
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required configuration section"):
                ModelConfigManager(Path(temp_path))
        finally:
            Path(temp_path).unlink(missing_ok=True) 