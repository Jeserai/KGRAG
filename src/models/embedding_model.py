"""
Local embedding model management for Knowledge Graph RAG.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss

from .model_config import ModelConfig, config_manager


logger = logging.getLogger(__name__)


class LocalEmbeddingManager:
    """Manager for local embedding models with efficient batch processing."""
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize the embedding manager.
        
        Args:
            model_config: Configuration for the embedding model. If None, uses default.
        """
        self.config = model_config or config_manager.get_default_embedding_config()
        self.model = None
        self._model_loaded = False
        
        # Performance settings
        self.performance_config = config_manager.get_performance_config()
        self.enable_caching = self.performance_config.get('enable_model_caching', True)
        
        # Embedding cache for frequently used texts
        self._embedding_cache = {} if self.enable_caching else None
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Initialized embedding manager for model: {self.config.name}")
    
    def load_model(self, force_reload: bool = False) -> None:
        """Load the embedding model.
        
        Args:
            force_reload: Force reload even if model is already loaded.
        """
        if self._model_loaded and not force_reload:
            logger.info("Embedding model already loaded")
            return
        
        logger.info(f"Loading embedding model: {self.config.name}")
        start_time = time.time()
        
        try:
            # Load sentence transformer model
            self.model = SentenceTransformer(
                self.config.name,
                device=self.config.device
            )
            
            # Configure model settings
            if hasattr(self.model, 'max_seq_length'):
                if self.config.max_seq_length:
                    self.model.max_seq_length = self.config.max_seq_length
            
            self._model_loaded = True
            load_time = time.time() - start_time
            
            logger.info(f"Embedding model loaded in {load_time:.1f}s")
            logger.info(f"Model max sequence length: {getattr(self.model, 'max_seq_length', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._cleanup_model()
            raise
    
    def _cleanup_model(self) -> None:
        """Clean up model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self._model_loaded = False
        
        # Clear cache
        if self._embedding_cache:
            self._embedding_cache.clear()
        
        logger.info("Embedding model cleaned up from memory")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"{hash(text)}_{self.config.name}"
    
    def encode_single(self, text: str, normalize: Optional[bool] = None) -> np.ndarray:
        """Encode a single text into embeddings.
        
        Args:
            text: Text to encode.
            normalize: Whether to normalize embeddings. Uses config default if None.
            
        Returns:
            Embedding vector as numpy array.
        """
        if not self._model_loaded:
            self.load_model()
        
        # Check cache first
        if self._embedding_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                self._cache_hits += 1
                return self._embedding_cache[cache_key].copy()
        
        try:
            # Encode text
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize or self.config.normalize_embeddings,
                show_progress_bar=False
            )
            
            # Cache result
            if self._embedding_cache:
                cache_key = self._get_cache_key(text)
                self._embedding_cache[cache_key] = embedding.copy()
                self._cache_misses += 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def encode_batch(self, 
                    texts: List[str], 
                    normalize: Optional[bool] = None,
                    batch_size: Optional[int] = None) -> np.ndarray:
        """Encode a batch of texts into embeddings.
        
        Args:
            texts: List of texts to encode.
            normalize: Whether to normalize embeddings. Uses config default if None.
            batch_size: Batch size for processing. Uses config default if None.
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        
        if not self._model_loaded:
            self.load_model()
        
        batch_size = batch_size or self.config.batch_size
        normalize = normalize or self.config.normalize_embeddings
        
        # Check for cached embeddings
        embeddings = []
        texts_to_encode = []
        cache_indices = {}
        
        if self._embedding_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    embeddings.append(self._embedding_cache[cache_key].copy())
                    self._cache_hits += 1
                else:
                    cache_indices[len(texts_to_encode)] = i
                    texts_to_encode.append(text)
                    embeddings.append(None)  # Placeholder
        else:
            texts_to_encode = texts
            embeddings = [None] * len(texts)
        
        # Encode uncached texts
        if texts_to_encode:
            try:
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=len(texts_to_encode) > 100
                )
                
                # Insert new embeddings and cache them
                for j, embedding in enumerate(new_embeddings):
                    original_index = cache_indices.get(j, j)
                    embeddings[original_index] = embedding
                    
                    if self._embedding_cache:
                        cache_key = self._get_cache_key(texts_to_encode[j])
                        self._embedding_cache[cache_key] = embedding.copy()
                        self._cache_misses += 1
                
            except Exception as e:
                logger.error(f"Failed to encode batch: {e}")
                raise
        
        return np.array(embeddings)
    
    def compute_similarity(self, 
                          embeddings1: np.ndarray, 
                          embeddings2: np.ndarray,
                          metric: str = "cosine") -> np.ndarray:
        """Compute similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings.
            embeddings2: Second set of embeddings.
            metric: Similarity metric ('cosine', 'euclidean', 'dot').
            
        Returns:
            Similarity matrix.
        """
        if metric == "cosine":
            # Ensure embeddings are normalized for cosine similarity
            embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            return np.dot(embeddings1_norm, embeddings2_norm.T)
        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "euclidean":
            # Compute negative euclidean distance (higher = more similar)
            distances = np.sqrt(np.sum((embeddings1[:, np.newaxis] - embeddings2) ** 2, axis=2))
            return -distances
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def find_most_similar(self, 
                         query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5,
                         metric: str = "cosine") -> List[tuple]:
        """Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding vector.
            candidate_embeddings: Candidate embeddings to search.
            top_k: Number of top results to return.
            metric: Similarity metric to use.
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity.
        """
        if len(candidate_embeddings) == 0:
            return []
        
        # Compute similarities
        similarities = self.compute_similarity(
            query_embedding.reshape(1, -1),
            candidate_embeddings,
            metric=metric
        )[0]
        
        # Get top-k indices and scores
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def create_faiss_index(self, 
                          embeddings: np.ndarray,
                          index_type: str = "flat") -> faiss.Index:
        """Create a FAISS index for efficient similarity search.
        
        Args:
            embeddings: Embeddings to index.
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw').
            
        Returns:
            FAISS index.
        """
        embedding_dim = embeddings.shape[1]
        
        if index_type == "flat":
            index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = min(100, max(1, embeddings.shape[0] // 39))  # Rule of thumb
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        
        # Train index if needed
        if hasattr(index, 'train'):
            index.train(normalized_embeddings)
        
        # Add embeddings to index
        index.add(normalized_embeddings)
        
        logger.info(f"Created FAISS {index_type} index with {embeddings.shape[0]} embeddings")
        return index
    
    def search_faiss_index(self, 
                          index: faiss.Index,
                          query_embedding: np.ndarray,
                          top_k: int = 5) -> List[tuple]:
        """Search a FAISS index for similar embeddings.
        
        Args:
            index: FAISS index to search.
            query_embedding: Query embedding vector.
            top_k: Number of top results to return.
            
        Returns:
            List of (index, similarity_score) tuples.
        """
        # Normalize query embedding
        query_norm = query_embedding.copy().reshape(1, -1)
        faiss.normalize_L2(query_norm)
        
        # Search index
        scores, indices = index.search(query_norm, top_k)
        
        # Return results
        results = [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]
        return results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding model and cache."""
        stats = {
            "model_name": self.config.name,
            "model_loaded": self._model_loaded,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "normalize_embeddings": self.config.normalize_embeddings,
        }
        
        if self._model_loaded and self.model:
            stats["embedding_dimension"] = self.model.get_sentence_embedding_dimension()
            stats["max_seq_length"] = getattr(self.model, 'max_seq_length', None)
        
        if self._embedding_cache is not None:
            stats["cache_size"] = len(self._embedding_cache)
            stats["cache_hits"] = self._cache_hits
            stats["cache_misses"] = self._cache_misses
            if self._cache_hits + self._cache_misses > 0:
                stats["cache_hit_rate"] = self._cache_hits / (self._cache_hits + self._cache_misses)
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._embedding_cache:
            self._embedding_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("Embedding cache cleared")
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        logger.info("Unloading embedding model...")
        self._cleanup_model()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._model_loaded:
            self._cleanup_model() 