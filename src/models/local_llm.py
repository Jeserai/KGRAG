"""
Local LLM management with efficient inference and memory optimization.
"""

import gc
import logging
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import psutil

from .model_config import ModelConfig, config_manager


logger = logging.getLogger(__name__)


class LocalLLMManager:
    """Manager for local LLM inference with memory optimization."""
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize the LLM manager.
        
        Args:
            model_config: Configuration for the model. If None, uses default.
        """
        self.config = model_config or config_manager.get_default_llm_config()
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._generation_config = None
        
        # Memory management
        self.performance_config = config_manager.get_performance_config()
        self.max_memory_gb = self.performance_config.get('max_memory_gb', 16)
        self.enable_caching = self.performance_config.get('enable_model_caching', True)
        
        logger.info(f"Initialized LLM manager for model: {self.config.name}")
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for memory optimization."""
        if self.config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        return None
    
    def _check_memory_availability(self) -> bool:
        """Check if sufficient memory is available for model loading."""
        if self.config.device == 'cuda' and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            gpu_free = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            available_gpu = gpu_memory - gpu_free
            
            logger.info(f"GPU memory - Total: {gpu_memory:.1f}GB, Available: {available_gpu:.1f}GB")
            return available_gpu >= 4.0  # Minimum 4GB for 7B models with quantization
        else:
            ram_available = psutil.virtual_memory().available / (1024**3)  # GB
            logger.info(f"RAM available: {ram_available:.1f}GB")
            return ram_available >= 8.0  # Minimum 8GB RAM for CPU inference
    
    def load_model(self, force_reload: bool = False) -> None:
        """Load the model and tokenizer with memory optimization.
        
        Args:
            force_reload: Force reload even if model is already loaded.
        """
        if self._model_loaded and not force_reload:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading model: {self.config.name}")
        
        # Check memory availability
        if not self._check_memory_availability():
            logger.warning("Insufficient memory detected, switching to CPU with optimizations")
            self.config.device = 'cpu'
            self.config.load_in_4bit = False
            self.config.load_in_8bit = False
        
        start_time = time.time()
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="left"
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare model loading arguments
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": torch.float16 if self.config.device == 'cuda' else torch.float32,
                "device_map": "auto" if self.config.device == 'cuda' else None,
            }
            
            # Add quantization config
            quant_config = self._get_quantization_config()
            if quant_config:
                model_kwargs["quantization_config"] = quant_config
                logger.info(f"Using quantization: {self.config.quantization}")
            
            # Load model
            logger.info("Loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if model_kwargs.get("device_map") is None:
                self.model = self.model.to(self.config.device)
            
            # Set generation config
            self._generation_config = GenerationConfig(
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True if self.config.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            self._model_loaded = True
            load_time = time.time() - start_time
            
            # Log memory usage
            if self.config.device == 'cuda' and torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"Model loaded in {load_time:.1f}s, GPU memory used: {memory_used:.1f}GB")
            else:
                logger.info(f"Model loaded in {load_time:.1f}s")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._cleanup_model()
            raise
    
    def _cleanup_model(self) -> None:
        """Clean up model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._model_loaded = False
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cleaned up from memory")
    
    def generate_text(self, 
                     prompt: str, 
                     max_new_tokens: Optional[int] = None,
                     temperature: Optional[float] = None,
                     top_p: Optional[float] = None) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt for generation.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            
        Returns:
            Generated text.
        """
        if not self._model_loaded:
            self.load_model()
        
        # Prepare generation config
        gen_config = self._generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature
            gen_config.do_sample = temperature > 0
        if top_p is not None:
            gen_config.top_p = top_p
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.max_length - (max_new_tokens or 512)
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_batch(self, 
                      prompts: List[str], 
                      max_new_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      top_p: Optional[float] = None) -> List[str]:
        """Generate text for a batch of prompts.
        
        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            
        Returns:
            List of generated texts.
        """
        if not prompts:
            return []
        
        if not self._model_loaded:
            self.load_model()
        
        # Process in batches to manage memory
        batch_size = self.config.batch_size
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length - (max_new_tokens or 512)
                )
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                
                # Generate
                gen_config = self._generation_config
                if max_new_tokens is not None:
                    gen_config.max_new_tokens = max_new_tokens
                if temperature is not None:
                    gen_config.temperature = temperature
                    gen_config.do_sample = temperature > 0
                if top_p is not None:
                    gen_config.top_p = top_p
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                # Decode responses
                for j, output in enumerate(outputs):
                    generated_text = self.tokenizer.decode(
                        output[inputs['input_ids'][j].shape[0]:],
                        skip_special_tokens=True
                    )
                    batch_results.append(generated_text.strip())
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Batch generation failed for batch {i//batch_size}: {e}")
                # Add empty results for failed batch
                results.extend([""] * len(batch_prompts))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "name": self.config.name,
            "device": self.config.device,
            "loaded": self._model_loaded,
            "quantization": self.config.quantization,
        }
        
        if self._model_loaded and self.model is not None:
            info["num_parameters"] = self.model.num_parameters()
            
            if self.config.device == 'cuda' and torch.cuda.is_available():
                info["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024**2)
        
        return info
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        logger.info("Unloading model...")
        self._cleanup_model()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._model_loaded:
            self._cleanup_model() 