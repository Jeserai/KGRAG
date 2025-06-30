"""
Local LLM model manager for GraphRAG implementation.
Supports loading and inference with Qwen and Llama models offline.
"""

import logging
import gc
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TextGenerationPipeline,
        pipeline,
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence-transformers not available. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages local LLM and embedding models for GraphRAG."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model manager.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_pipeline = None
        self.embedding_model = None
        
        # Model configuration
        self.llm_name = config.get('llm', {}).get('name', 'Qwen/Qwen2.5-7B-Instruct')
        self.embedding_name = config.get('embedding', {}).get('name', 'sentence-transformers/msmarco-distilbert-base-tas-b')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.load_in_8bit = config.get('load_in_8bit', True)
        self.max_memory = config.get('max_memory', None)
        
        # Check dependencies
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Please install: pip install transformers")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("Sentence-transformers not available. Please install: pip install sentence-transformers")
        
        logger.info(f"Initialized ModelManager with LLM: {self.llm_name}, Embedding: {self.embedding_name}")
    
    def load_llm(self) -> bool:
        """
        Load the LLM model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Cannot load LLM: transformers not available")
            return False
        
        try:
            logger.info(f"Loading LLM model: {self.llm_name}")
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if self.load_in_8bit and self.device == 'cuda':
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
            
            # Load tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            # Add pad token if not present
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
                'device_map': 'auto' if self.device == 'cuda' else None,
                'low_cpu_mem_usage': True,
            }
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            
            if self.max_memory:
                model_kwargs['max_memory'] = self.max_memory
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                **model_kwargs
            )
            
            # Create pipeline
            self.llm_pipeline = pipeline(
                'text-generation',
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                device_map='auto' if self.device == 'cuda' else None,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            )
            
            logger.info(f"Successfully loaded LLM model: {self.llm_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            return False
    
    def load_embedding_model(self) -> bool:
        """
        Load the embedding model.
        
        Returns:
            True if successful, False otherwise
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("Cannot load embedding model: sentence-transformers not available")
            return False
        
        try:
            logger.info(f"Loading embedding model: {self.embedding_name}")
            
            model_kwargs = {
                'device': self.device
            }
            # Qwen models require trusting remote code
            if 'qwen' in self.embedding_name.lower():
                model_kwargs['trust_remote_code'] = True

            self.embedding_model = SentenceTransformer(
                self.embedding_name,
                **model_kwargs
            )
            
            logger.info(f"Successfully loaded embedding model: {self.embedding_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return False
    
    def ensure_models_loaded(self):
        """Ensure both LLM and embedding models are loaded."""
        if self.llm_model is None:
            if not self.load_llm():
                raise RuntimeError("Failed to load LLM model")
        
        if self.embedding_model is None:
            if not self.load_embedding_model():
                raise RuntimeError("Failed to load embedding model")
    
    def inference(self, 
                prompt: str, 
                max_tokens: int = 512,
                temperature: float = 0.1,
                top_p: float = 0.9,
                stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Stop sequences to end generation
            
        Returns:
            Generated text
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Cannot generate: transformers not available")
            return ""
        
        self.ensure_models_loaded()
        
        try:
            # Prepare generation parameters
            generation_kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': temperature > 0,
                'pad_token_id': self.llm_tokenizer.eos_token_id,
                'eos_token_id': self.llm_tokenizer.eos_token_id,
                'return_full_text': False
            }
            
            # Add stop sequences if provided
            if stop_sequences:
                # Convert stop sequences to token IDs
                stop_token_ids = []
                for seq in stop_sequences:
                    tokens = self.llm_tokenizer.encode(seq, add_special_tokens=False)
                    if tokens:
                        stop_token_ids.extend(tokens)
                
                if stop_token_ids:
                    generation_kwargs['eos_token_id'] = stop_token_ids
            
            # Generate response
            response = self.llm_pipeline(prompt, **generation_kwargs)
            
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0]['generated_text']
            else:
                generated_text = str(response)
            
            # Clean up the response
            generated_text = generated_text.strip()
            
            # Remove stop sequences from the end
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if generated_text.endswith(stop_seq):
                        generated_text = generated_text[:-len(stop_seq)].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("Cannot generate embedding: sentence-transformers not available")
            return []
        
        self.ensure_models_loaded()
        
        try:
            # Clean the text
            text = text.strip()
            if not text:
                return []
            
            # Generate embedding
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            
            return list(embedding)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("Cannot generate embeddings: sentence-transformers not available")
            return []
        
        self.ensure_models_loaded()
        
        try:
            # Clean texts
            clean_texts = [text.strip() for text in texts if text.strip()]
            
            if not clean_texts:
                return []
            
            # Generate embeddings in batches
            embeddings = self.embedding_model.encode(
                clean_texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=len(clean_texts) > 100
            )
            
            # Convert to list format
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            
            return [list(emb) for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            embeddings = self.generate_embeddings_batch([text1, text2])
            
            if len(embeddings) != 2:
                return 0.0
            
            # Calculate cosine similarity
            emb1, emb2 = embeddings
            
            # Convert to tensors for calculation
            tensor1 = torch.tensor(emb1)
            tensor2 = torch.tensor(emb2)
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                tensor1.unsqueeze(0), 
                tensor2.unsqueeze(0)
            ).item()
            
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'llm_model': self.llm_name,
            'embedding_model': self.embedding_name,
            'device': self.device,
            'llm_loaded': self.llm_model is not None,
            'embedding_loaded': self.embedding_model is not None,
            'quantization': self.load_in_8bit,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
        }
        
        # Add memory usage if on CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated()
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved()
        
        return info
    
    def cleanup(self):
        """Clean up models and free memory."""
        logger.info("Cleaning up models...")
        
        # Clear models
        if self.llm_model is not None:
            del self.llm_model
            self.llm_model = None
        
        if self.llm_tokenizer is not None:
            del self.llm_tokenizer
            self.llm_tokenizer = None
        
        if self.llm_pipeline is not None:
            del self.llm_pipeline
            self.llm_pipeline = None
        
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cleanup completed")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
