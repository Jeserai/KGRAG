from model import ModelManager
import logging
import gc
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelUtils(ModelManager):
    def __init__(self, llm_model: str, embedding_model: str, device: str = 'cuda', load_in_8bit: bool = False):
        super().__init__(llm_model, embedding_model, device, load_in_8bit)

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        
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
            'quantization': self.load_in_8bit
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