"""
Tests for document processing functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.document_processor import DocumentProcessor, DocumentChunk


class TestDocumentChunk:
    """Test DocumentChunk dataclass."""
    
    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk instance."""
        chunk = DocumentChunk(
            content="This is a test document chunk.",
            chunk_id="chunk_001",
            document_id="doc_001",
            start_char=0,
            end_char=30,
            metadata={"source": "test.txt", "page": 1}
        )
        
        assert chunk.content == "This is a test document chunk."
        assert chunk.chunk_id == "chunk_001"
        assert chunk.document_id == "doc_001"
        assert chunk.start_char == 0
        assert chunk.end_char == 30
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.metadata["page"] == 1
    
    def test_document_chunk_defaults(self):
        """Test DocumentChunk with default values."""
        chunk = DocumentChunk(
            content="Test content",
            chunk_id="chunk_001",
            document_id="doc_001"
        )
        
        assert chunk.start_char == 0
        assert chunk.end_char == 0
        assert chunk.metadata == {}


class TestDocumentProcessor:
    """Test DocumentProcessor class."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        This is a sample document for testing the document processor.
        It contains multiple sentences that should be processed.
        The processor should split this into appropriate chunks.
        Each chunk should maintain the semantic meaning of the content.
        This is important for knowledge graph construction.
        """
    
    @pytest.fixture
    def temp_document(self, sample_text):
        """Create a temporary document file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text)
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_document_processor_initialization(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(
            chunk_size=512,
            chunk_overlap=50,
            min_chunk_size=100
        )
        
        assert processor.chunk_size == 512
        assert processor.chunk_overlap == 50
        assert processor.min_chunk_size == 100
    
    def test_document_processor_defaults(self):
        """Test DocumentProcessor with default values."""
        processor = DocumentProcessor()
        
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
        assert processor.min_chunk_size == 200
    
    def test_process_text(self, sample_text):
        """Test processing text into chunks."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        chunks = processor.process_text(sample_text, "test_doc")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == "test_doc" for chunk in chunks)
        
        # Check that chunks have reasonable sizes
        for chunk in chunks:
            assert len(chunk.content) <= 100
            assert len(chunk.content) >= 20  # min_chunk_size
    
    def test_process_text_small_content(self):
        """Test processing very small text content."""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        small_text = "This is a very short text."
        chunks = processor.process_text(small_text, "small_doc")
        
        assert len(chunks) == 1
        assert chunks[0].content == small_text.strip()
    
    def test_process_text_empty_content(self):
        """Test processing empty text content."""
        processor = DocumentProcessor()
        
        chunks = processor.process_text("", "empty_doc")
        
        assert len(chunks) == 0
    
    def test_process_text_whitespace_only(self):
        """Test processing whitespace-only content."""
        processor = DocumentProcessor()
        
        chunks = processor.process_text("   \n\t   ", "whitespace_doc")
        
        assert len(chunks) == 0
    
    def test_process_file(self, temp_document):
        """Test processing a file."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        chunks = processor.process_file(temp_document)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == temp_document.name for chunk in chunks)
    
    def test_process_file_not_found(self):
        """Test processing a non-existent file."""
        processor = DocumentProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.process_file(Path("/non/existent/file.txt"))
    
    def test_process_directory(self, temp_document):
        """Test processing a directory of files."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        # Create a temporary directory with the document
        temp_dir = temp_document.parent
        chunks = processor.process_directory(temp_dir, file_patterns=["*.txt"])
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    
    def test_process_directory_no_files(self, tempfile):
        """Test processing a directory with no matching files."""
        processor = DocumentProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = processor.process_directory(Path(temp_dir), file_patterns=["*.txt"])
            assert len(chunks) == 0
    
    def test_chunk_overlap_validation(self):
        """Test that chunk overlap is properly handled."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=50)
        
        text = "This is a test document. " * 10  # Create longer text
        chunks = processor.process_text(text, "overlap_test")
        
        # Check that consecutive chunks have some overlap
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].content
            next_chunk = chunks[i + 1].content
            
            # Find overlap by checking if end of current chunk appears in next chunk
            overlap_found = False
            for j in range(len(current_chunk) - 20, len(current_chunk)):
                if current_chunk[j:] in next_chunk:
                    overlap_found = True
                    break
            
            assert overlap_found, f"No overlap found between chunks {i} and {i+1}"
    
    def test_chunk_id_generation(self, sample_text):
        """Test that chunk IDs are properly generated."""
        processor = DocumentProcessor()
        
        chunks = processor.process_text(sample_text, "test_doc")
        
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Check that all IDs are unique
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # Check that IDs follow expected pattern
        for chunk_id in chunk_ids:
            assert chunk_id.startswith("test_doc_")
    
    def test_metadata_preservation(self, sample_text):
        """Test that metadata is properly preserved in chunks."""
        processor = DocumentProcessor()
        
        metadata = {"source": "test.txt", "author": "Test Author", "date": "2024-01-01"}
        chunks = processor.process_text(sample_text, "test_doc", metadata=metadata)
        
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["date"] == "2024-01-01"
    
    def test_sentence_boundary_respect(self, sample_text):
        """Test that chunks respect sentence boundaries when possible."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        
        chunks = processor.process_text(sample_text, "sentence_test")
        
        # Check that most chunks end with sentence-ending punctuation
        sentence_endings = ['.', '!', '?']
        chunks_ending_with_sentence = 0
        
        for chunk in chunks:
            if chunk.content.strip() and chunk.content.strip()[-1] in sentence_endings:
                chunks_ending_with_sentence += 1
        
        # At least 50% of chunks should end with sentence endings
        assert chunks_ending_with_sentence >= len(chunks) * 0.5
    
    def test_content_continuity(self, sample_text):
        """Test that chunk content maintains continuity."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        chunks = processor.process_text(sample_text, "continuity_test")
        
        # Reconstruct the original text from chunks
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                reconstructed += chunk.content
            else:
                # Find overlap and add only new content
                prev_chunk = chunks[i-1].content
                current_chunk = chunk.content
                
                # Find the overlap point
                overlap_found = False
                for j in range(len(prev_chunk) - 20, len(prev_chunk)):
                    if prev_chunk[j:] in current_chunk:
                        reconstructed += current_chunk[len(prev_chunk[j:]):]
                        overlap_found = True
                        break
                
                if not overlap_found:
                    reconstructed += " " + current_chunk
        
        # The reconstructed text should contain most of the original content
        original_clean = sample_text.replace('\n', ' ').replace('  ', ' ').strip()
        reconstructed_clean = reconstructed.replace('  ', ' ').strip()
        
        # Check that at least 80% of original words are in reconstructed text
        original_words = set(original_clean.lower().split())
        reconstructed_words = set(reconstructed_clean.lower().split())
        
        overlap_ratio = len(original_words.intersection(reconstructed_words)) / len(original_words)
        assert overlap_ratio >= 0.8 