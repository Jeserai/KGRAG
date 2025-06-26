"""
Model configuration management for local Knowledge Graph RAG system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    type: str  # 'llm' or 'embedding'
    device: str
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    batch_size: int = 1
    quantization: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    max_seq_length: Optional[int] = None
    normalize_embeddings: bool = True


class ModelConfigManager:
    """Manager for loading and accessing model configurations."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "model_configs.yaml"
        
        self.config_path = config_path
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = ['models', 'default']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get_model_config(self, model_id: str) -> ModelConfig:
        """Get configuration for a specific model.
        
        Args:
            model_id: Identifier for the model in the config.
            
        Returns:
            ModelConfig object with the model's configuration.
        """
        if model_id not in self._config['models']:
            raise ValueError(f"Model configuration not found: {model_id}")
        
        model_data = self._config['models'][model_id].copy()
        
        # Auto-detect device if cuda not available
        if model_data.get('device') == 'cuda' and not torch.cuda.is_available():
            model_data['device'] = 'cpu'
            # Disable quantization for CPU
            model_data['load_in_4bit'] = False
            model_data['load_in_8bit'] = False
            model_data['quantization'] = None
        
        return ModelConfig(**model_data)
    
    def get_default_llm_config(self) -> ModelConfig:
        """Get the default LLM configuration."""
        default_llm = self._config['default']['llm_model']
        return self.get_model_config(default_llm)
    
    def get_default_embedding_config(self) -> ModelConfig:
        """Get the default embedding model configuration."""
        default_embedding = self._config['default']['embedding_model']
        return self.get_model_config(default_embedding)
    
    def list_available_models(self, model_type: Optional[str] = None) -> Dict[str, str]:
        """List all available models or models of a specific type.
        
        Args:
            model_type: Filter by model type ('llm' or 'embedding').
            
        Returns:
            Dictionary mapping model IDs to model names.
        """
        models = {}
        for model_id, config in self._config['models'].items():
            if model_type is None or config.get('type') == model_type:
                models[model_id] = config['name']
        return models
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return self._config.get('performance', {})
    
    def get_kg_config(self) -> Dict[str, Any]:
        """Get knowledge graph related configuration."""
        return self._config.get('kg', {})
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval related configuration."""
        return self._config.get('retrieval', {})
    
    def update_model_config(self, model_id: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a specific model.
        
        Args:
            model_id: Identifier for the model.
            updates: Dictionary of configuration updates.
        """
        if model_id not in self._config['models']:
            raise ValueError(f"Model configuration not found: {model_id}")
        
        self._config['models'][model_id].update(updates)
    
    def save_config(self) -> None:
        """Save the current configuration back to the YAML file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)


# Global configuration manager instance
config_manager = ModelConfigManager() 