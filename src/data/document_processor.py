"""
Document processing pipeline for Knowledge Graph RAG.
Handles document loading, preprocessing, and chunking optimized for local models.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import hashlib

from ..models.model_config import config_manager


logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    id: str
    text: str
    metadata: Dict[str, Any]
    start_char: int
    end_char: int
    token_count: Optional[int] = None


@dataclass
class Document:
    """Represents a processed document."""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    chunks: List[DocumentChunk]
    source: Optional[str] = None


class DocumentProcessor:
    """Processes documents for Knowledge Graph RAG pipeline."""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1024):
        """Initialize the document processor.
        
        Args:
            chunk_size: Target size for text chunks in tokens.
            chunk_overlap: Number of tokens to overlap between chunks.
            min_chunk_size: Minimum chunk size in tokens.
            max_chunk_size: Maximum chunk size in tokens.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Get model configuration for context optimization
        self.model_config = config_manager.get_default_llm_config()
        
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}")
    
    def _generate_doc_id(self, title: str, content: str) -> str:
        """Generate a unique document ID based on content."""
        content_hash = hashlib.md5((title + content).encode()).hexdigest()[:8]
        return f"doc_{content_hash}"
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{doc_id}_chunk_{chunk_index:04d}"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text.
        
        Uses a rough approximation: 1 token ≈ 4 characters for English text.
        """
        return len(text) // 4
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunking."""
        # Simple sentence splitting - can be enhanced with NLTK/spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunks(self, text: str, doc_id: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create chunks from text with optimal sizing for local models."""
        chunks = []
        sentences = self._split_by_sentences(text)
        
        if not sentences:
            return chunks
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            potential_tokens = self._estimate_token_count(potential_chunk)
            
            if potential_tokens > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_tokens = self._estimate_token_count(current_chunk)
                if chunk_tokens >= self.min_chunk_size:
                    chunk_end = current_start + len(current_chunk)
                    chunk = DocumentChunk(
                        id=self._generate_chunk_id(doc_id, chunk_index),
                        text=current_chunk.strip(),
                        metadata={
                            **metadata,
                            "chunk_index": chunk_index,
                            "sentence_start": max(0, i - len(current_chunk.split('. '))),
                            "sentence_end": i
                        },
                        start_char=current_start,
                        end_char=chunk_end,
                        token_count=chunk_tokens
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and chunks:
                    # Add some sentences from the end of previous chunk for overlap
                    overlap_sentences = current_chunk.split('. ')[-2:]  # Last 2 sentences
                    overlap_text = '. '.join(overlap_sentences)
                    if overlap_text and self._estimate_token_count(overlap_text) <= self.chunk_overlap:
                        current_chunk = overlap_text + ". " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
                
                current_start = chunk_end - len(overlap_text) if self.chunk_overlap > 0 and 'overlap_text' in locals() else chunk_end
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk:
            chunk_tokens = self._estimate_token_count(current_chunk)
            if chunk_tokens >= self.min_chunk_size:
                chunk = DocumentChunk(
                    id=self._generate_chunk_id(doc_id, chunk_index),
                    text=current_chunk.strip(),
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "sentence_start": len(sentences) - len(current_chunk.split('. ')),
                        "sentence_end": len(sentences)
                    },
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    token_count=chunk_tokens
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks
    
    def process_text(self, 
                    text: str, 
                    title: str = "",
                    metadata: Optional[Dict[str, Any]] = None,
                    source: Optional[str] = None) -> Document:
        """Process a text document into chunks.
        
        Args:
            text: Raw text content.
            title: Document title.
            metadata: Additional metadata to include.
            source: Source identifier for the document.
            
        Returns:
            Processed Document object with chunks.
        """
        if not text.strip():
            raise ValueError("Text content cannot be empty")
        
        # Clean text
        cleaned_text = self._clean_text(text)
        if not title:
            title = cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text
        
        # Generate document ID
        doc_id = self._generate_doc_id(title, cleaned_text)
        
        # Prepare metadata
        doc_metadata = {
            "title": title,
            "content_length": len(cleaned_text),
            "estimated_tokens": self._estimate_token_count(cleaned_text),
            "source": source,
            **(metadata or {})
        }
        
        # Create chunks
        chunks = self._create_chunks(cleaned_text, doc_id, doc_metadata)
        
        return Document(
            id=doc_id,
            title=title,
            content=cleaned_text,
            metadata=doc_metadata,
            chunks=chunks,
            source=source
        )
    
    def process_file(self, file_path: Union[str, Path]) -> Document:
        """Process a text file into a document.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            Processed Document object.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use filename as title
            title = file_path.stem
            
            # Add file metadata
            metadata = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix,
            }
            
            return self.process_text(
                text=content,
                title=title,
                metadata=metadata,
                source=str(file_path)
            )
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise
    
    def process_multiple_files(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """Process multiple text files into documents.
        
        Args:
            file_paths: List of file paths to process.
            
        Returns:
            List of processed Document objects.
        """
        documents = []
        for file_path in file_paths:
            try:
                doc = self.process_file(file_path)
                documents.append(doc)
                logger.info(f"Processed file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue
        
        return documents
    
    def get_chunk_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about document chunks.
        
        Args:
            documents: List of processed documents.
            
        Returns:
            Dictionary containing chunk statistics.
        """
        all_chunks = []
        for doc in documents:
            all_chunks.extend(doc.chunks)
        
        if not all_chunks:
            return {"total_chunks": 0}
        
        chunk_sizes = [chunk.token_count for chunk in all_chunks if chunk.token_count]
        text_lengths = [len(chunk.text) for chunk in all_chunks]
        
        stats = {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "avg_chunks_per_doc": len(all_chunks) / len(documents) if documents else 0,
            "avg_chunk_tokens": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            "min_chunk_tokens": min(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_tokens": max(chunk_sizes) if chunk_sizes else 0,
            "avg_chunk_chars": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "total_tokens": sum(chunk_sizes) if chunk_sizes else 0,
        }
        
        return stats
    
    def optimize_for_model(self, model_name: str) -> None:
        """Optimize chunk sizes for a specific model.
        
        Args:
            model_name: Name of the model to optimize for.
        """
        model_config = config_manager.get_model_config(model_name)
        
        # Adjust chunk size based on model's context length
        if model_config.max_length:
            # Use 25% of context length for chunks to leave room for prompts
            optimal_chunk_size = min(self.max_chunk_size, model_config.max_length // 4)
            self.chunk_size = optimal_chunk_size
            
            # Adjust overlap proportionally
            self.chunk_overlap = min(self.chunk_overlap, optimal_chunk_size // 10)
            
            logger.info(f"Optimized chunking for {model_name}: "
                       f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def merge_small_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Merge chunks that are too small to improve efficiency.
        
        Args:
            chunks: List of document chunks.
            
        Returns:
            List of merged chunks.
        """
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if chunk.token_count and chunk.token_count < self.min_chunk_size:
                if current_chunk is None:
                    current_chunk = chunk
                else:
                    # Merge with current chunk
                    merged_text = current_chunk.text + " " + chunk.text
                    merged_tokens = self._estimate_token_count(merged_text)
                    
                    if merged_tokens <= self.max_chunk_size:
                        current_chunk = DocumentChunk(
                            id=current_chunk.id + "_merged",
                            text=merged_text,
                            metadata={**current_chunk.metadata, "merged": True},
                            start_char=current_chunk.start_char,
                            end_char=chunk.end_char,
                            token_count=merged_tokens
                        )
                    else:
                        # Current chunk is big enough, add it and start new one
                        merged_chunks.append(current_chunk)
                        current_chunk = chunk
            else:
                # Add any pending small chunk
                if current_chunk is not None:
                    merged_chunks.append(current_chunk)
                    current_chunk = None
                
                # Add the current chunk (it's big enough)
                merged_chunks.append(chunk)
        
        # Add final pending chunk
        if current_chunk is not None:
            merged_chunks.append(current_chunk)
        
        return merged_chunks 