"""
Document processor for GraphRAG implementation.
Handles document loading, cleaning, and chunking operations.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    id: str
    text: str
    source_doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: int


@dataclass 
class Document:
    """Represents a source document."""
    id: str
    title: str
    content: str
    source_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> Document:
        """Load a document from file path."""
        pass


class PDFDocumentLoader(DocumentLoader):
    """Load PDF documents."""
    
    def load(self, file_path: Union[str, Path]) -> Document:
        file_path = Path(file_path)
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        
        content = "\n\n".join([page.page_content for page in pages])
        metadata = pages[0].metadata if pages else {}
        
        return Document(
            id=file_path.stem,
            title=file_path.stem,
            content=content,
            source_path=str(file_path),
            metadata=metadata
        )


class TextDocumentLoader(DocumentLoader):
    """Load plain text documents."""
    
    def load(self, file_path: Union[str, Path]) -> Document:
        file_path = Path(file_path)
        loader = TextLoader(str(file_path))
        doc = loader.load()[0]
        
        return Document(
            id=file_path.stem,
            title=file_path.stem,
            content=doc.page_content,
            source_path=str(file_path),
            metadata=doc.metadata
        )


class WordDocumentLoader(DocumentLoader):
    """Load Word documents."""
    
    def load(self, file_path: Union[str, Path]) -> Document:
        file_path = Path(file_path)
        loader = UnstructuredWordDocumentLoader(str(file_path))
        doc = loader.load()[0]
        
        return Document(
            id=file_path.stem,
            title=file_path.stem,
            content=doc.page_content,
            source_path=str(file_path),
            metadata=doc.metadata
        )


class DocumentProcessor:
    """Main document processing class."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 encoding_model: str = "cl100k_base"):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks in tokens
            encoding_model: Tokenizer model for counting tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_model)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {encoding_model}: {e}")
            self.tokenizer = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Document loaders
        self.loaders = {
            '.pdf': PDFDocumentLoader(),
            '.txt': TextDocumentLoader(),
            '.docx': WordDocumentLoader(),
            '.doc': WordDocumentLoader()
        }
        
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def load_document(self, file_path: Union[str, Path]) -> Document:
        """
        Load a single document from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.loaders:
            raise ValueError(f"Unsupported file type: {extension}")
        
        loader = self.loaders[extension]
        document = loader.load(file_path)
        
        # Clean the content
        document.content = self._clean_text(document.content)
        
        logger.info(f"Loaded document: {document.title} ({len(document.content)} chars)")
        return document
    
    def load_documents(self, directory: Union[str, Path]) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of Document objects
        """
        directory = Path(directory)
        documents = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.loaders:
                try:
                    document = self.load_document(file_path)
                    documents.append(document)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a document into chunks."""
        if not document.content.strip():
            return []

        text = document.content
        chunks = []

        # Simple chunking by character count
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)

                if break_point > start:
                    end = break_point + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = DocumentChunk(
                    id=f"{document.id}_chunk_{chunk_index}",
                    text=chunk_text,
                    source_doc_id=document.id,
                    chunk_index=chunk_index,
                    metadata={'source_title': document.title}
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end

        logger.info(f" Created {len(chunks)} chunks from document '{document.title}'")
        return chunks
    
    def process_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Process multiple documents into chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        return text.strip()
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if self.tokenizer is None:
            # Fallback to rough character-based estimation
            return len(text) // 4
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text) // 4
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'supported_formats': list(self.loaders.keys()),
            'tokenizer_available': self.tokenizer is not None
        }