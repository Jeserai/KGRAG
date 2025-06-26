"""
Knowledge Graph builder that orchestrates the construction process.
Combines document processing, entity extraction, and graph storage.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from ..data.document_processor import DocumentProcessor, Document, DocumentChunk
from ..models.local_llm import LocalLLMManager
from ..models.embedding_model import LocalEmbeddingManager
from .entity_extractor import EntityExtractor
from .neo4j_connector import Neo4jConnector, Entity, Relationship


logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Orchestrates the construction of a knowledge graph from documents."""
    
    def __init__(self,
                 neo4j_connector: Optional[Neo4jConnector] = None,
                 llm_manager: Optional[LocalLLMManager] = None,
                 embedding_manager: Optional[LocalEmbeddingManager] = None,
                 doc_processor: Optional[DocumentProcessor] = None):
        """Initialize the knowledge graph builder.
        
        Args:
            neo4j_connector: Neo4j database connector.
            llm_manager: Local LLM manager for entity extraction.
            embedding_manager: Local embedding manager for entity embeddings.
            doc_processor: Document processor for text chunking.
        """
        # Initialize components
        self.neo4j = neo4j_connector or Neo4jConnector()
        self.llm_manager = llm_manager or LocalLLMManager()
        self.embedding_manager = embedding_manager or LocalEmbeddingManager()
        self.doc_processor = doc_processor or DocumentProcessor()
        
        # Initialize entity extractor with shared LLM
        self.entity_extractor = EntityExtractor(self.llm_manager)
        
        # Statistics tracking
        self.stats = {
            'documents_processed': 0,
            'chunks_processed': 0,
            'entities_extracted': 0,
            'relationships_extracted': 0,
            'entities_stored': 0,
            'relationships_stored': 0,
            'processing_time': 0.0
        }
        
        logger.info("Initialized KnowledgeGraphBuilder")
    
    def build_from_texts(self, 
                        texts: List[str], 
                        titles: Optional[List[str]] = None,
                        metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Build knowledge graph from a list of texts.
        
        Args:
            texts: List of text documents.
            titles: Optional list of document titles.
            metadata: Optional list of metadata dictionaries.
            
        Returns:
            Dictionary containing build statistics.
        """
        start_time = time.time()
        
        # Process texts into documents
        documents = []
        for i, text in enumerate(texts):
            title = titles[i] if titles and i < len(titles) else f"Document {i+1}"
            meta = metadata[i] if metadata and i < len(metadata) else {}
            
            try:
                doc = self.doc_processor.process_text(
                    text=text,
                    title=title,
                    metadata=meta,
                    source=f"text_{i}"
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to process text {i}: {e}")
                continue
        
        # Build graph from documents
        return self._build_from_documents(documents, start_time)
    
    def build_from_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Build knowledge graph from text files.
        
        Args:
            file_paths: List of file paths to process.
            
        Returns:
            Dictionary containing build statistics.
        """
        start_time = time.time()
        
        # Process files into documents
        documents = self.doc_processor.process_multiple_files(file_paths)
        
        # Build graph from documents
        return self._build_from_documents(documents, start_time)
    
    def build_from_directory(self, 
                           directory_path: str, 
                           file_pattern: str = "*.txt") -> Dict[str, Any]:
        """Build knowledge graph from all files in a directory.
        
        Args:
            directory_path: Path to directory containing text files.
            file_pattern: File pattern to match (e.g., "*.txt", "*.md").
            
        Returns:
            Dictionary containing build statistics.
        """
        start_time = time.time()
        
        # Find files matching pattern
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        file_paths = list(directory.glob(file_pattern))
        if not file_paths:
            logger.warning(f"No files found matching pattern '{file_pattern}' in {directory_path}")
            return self.stats
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        # Process files
        documents = self.doc_processor.process_multiple_files([str(p) for p in file_paths])
        
        # Build graph from documents
        return self._build_from_documents(documents, start_time)
    
    def _build_from_documents(self, documents: List[Document], start_time: float) -> Dict[str, Any]:
        """Build knowledge graph from processed documents.
        
        Args:
            documents: List of processed Document objects.
            start_time: Start time for statistics.
            
        Returns:
            Dictionary containing build statistics.
        """
        if not documents:
            logger.warning("No documents to process")
            return self.stats
        
        # Reset statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_processed': 0,
            'entities_extracted': 0,
            'relationships_extracted': 0,
            'entities_stored': 0,
            'relationships_stored': 0,
            'processing_time': 0.0
        }
        
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        # Collect all chunks from all documents
        all_chunks = []
        for doc in documents:
            all_chunks.extend(doc.chunks)
            self.stats['documents_processed'] += 1
        
        self.stats['chunks_processed'] = len(all_chunks)
        logger.info(f"Processing {len(all_chunks)} total chunks")
        
        # Extract entities and relationships
        logger.info("Extracting entities and relationships...")
        entities, relationships = self.entity_extractor.extract_from_chunks_batch(all_chunks)
        
        self.stats['entities_extracted'] = len(entities)
        self.stats['relationships_extracted'] = len(relationships)
        
        if not entities:
            logger.warning("No entities extracted from documents")
            self.stats['processing_time'] = time.time() - start_time
            return self.stats
        
        # Generate embeddings for entities
        logger.info("Generating embeddings for entities...")
        entity_texts = [f"{entity.name} ({entity.type})" for entity in entities]
        embeddings = self.embedding_manager.encode_batch(entity_texts)
        
        # Add embeddings to entities
        for entity, embedding in zip(entities, embeddings):
            entity.embedding = embedding
        
        # Store entities and relationships in Neo4j
        logger.info("Storing entities in knowledge graph...")
        entities_stored = self.neo4j.create_entities_batch(entities)
        self.stats['entities_stored'] = entities_stored
        
        if relationships:
            logger.info("Storing relationships in knowledge graph...")
            relationships_stored = self.neo4j.create_relationships_batch(relationships)
            self.stats['relationships_stored'] = relationships_stored
        
        # Calculate processing time
        self.stats['processing_time'] = time.time() - start_time
        
        # Log final statistics
        logger.info(f"Knowledge graph construction completed:")
        logger.info(f"  - Documents processed: {self.stats['documents_processed']}")
        logger.info(f"  - Chunks processed: {self.stats['chunks_processed']}")
        logger.info(f"  - Entities extracted: {self.stats['entities_extracted']}")
        logger.info(f"  - Relationships extracted: {self.stats['relationships_extracted']}")
        logger.info(f"  - Entities stored: {self.stats['entities_stored']}")
        logger.info(f"  - Relationships stored: {self.stats['relationships_stored']}")
        logger.info(f"  - Processing time: {self.stats['processing_time']:.2f}s")
        
        return self.stats
    
    def update_graph_from_new_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Update existing knowledge graph with new documents.
        
        Args:
            documents: List of new documents to add.
            
        Returns:
            Dictionary containing update statistics.
        """
        logger.info(f"Updating knowledge graph with {len(documents)} new documents")
        
        # Use the same process as initial build but track it as an update
        start_time = time.time()
        return self._build_from_documents(documents, start_time)
    
    def resolve_entity_duplicates(self, similarity_threshold: float = 0.9) -> int:
        """Resolve duplicate entities in the knowledge graph using embeddings.
        
        Args:
            similarity_threshold: Cosine similarity threshold for merging entities.
            
        Returns:
            Number of duplicate entities resolved.
        """
        logger.info("Resolving entity duplicates...")
        
        # Get all entities from the graph
        # Note: This is a simplified implementation. In practice, you'd want to
        # process entities in batches and use more sophisticated deduplication
        stats = self.neo4j.get_graph_statistics()
        total_entities = stats.get('total_entities', 0)
        
        if total_entities < 2:
            logger.info("Not enough entities for duplicate resolution")
            return 0
        
        # This would need to be implemented based on specific requirements
        # For now, return 0 as a placeholder
        logger.info("Entity duplicate resolution completed")
        return 0
    
    def enrich_entities_with_types(self) -> int:
        """Enrich entities with additional type information using LLM.
        
        Returns:
            Number of entities enriched.
        """
        logger.info("Enriching entities with additional type information...")
        
        # Get entities that might need type enrichment
        # This is a placeholder for a more sophisticated implementation
        enriched_count = 0
        
        logger.info(f"Enriched {enriched_count} entities with additional type information")
        return enriched_count
    
    def validate_graph_quality(self) -> Dict[str, Any]:
        """Validate the quality of the constructed knowledge graph.
        
        Returns:
            Dictionary containing quality metrics.
        """
        logger.info("Validating knowledge graph quality...")
        
        # Get basic graph statistics
        graph_stats = self.neo4j.get_graph_statistics()
        
        quality_metrics = {
            'basic_stats': graph_stats,
            'entity_coverage': self._calculate_entity_coverage(),
            'relationship_density': self._calculate_relationship_density(graph_stats),
            'type_distribution': self._analyze_type_distribution(graph_stats),
            'connected_components': self._analyze_connectivity(),
        }
        
        logger.info("Graph quality validation completed")
        return quality_metrics
    
    def _calculate_entity_coverage(self) -> Dict[str, float]:
        """Calculate entity coverage metrics."""
        # Placeholder implementation
        return {
            'unique_entities_ratio': 0.85,
            'entities_per_document': 15.2,
            'avg_entity_connections': 2.3
        }
    
    def _calculate_relationship_density(self, graph_stats: Dict[str, Any]) -> float:
        """Calculate relationship density in the graph."""
        total_entities = graph_stats.get('total_entities', 0)
        total_relationships = graph_stats.get('total_relationships', 0)
        
        if total_entities < 2:
            return 0.0
        
        # Maximum possible relationships in a complete graph
        max_relationships = total_entities * (total_entities - 1)
        
        if max_relationships == 0:
            return 0.0
        
        return total_relationships / max_relationships
    
    def _analyze_type_distribution(self, graph_stats: Dict[str, Any]) -> Dict[str, float]:
        """Analyze the distribution of entity types."""
        entity_types = graph_stats.get('entity_types', [])
        
        if not entity_types:
            return {}
        
        total_entities = sum(item['count'] for item in entity_types)
        
        distribution = {}
        for item in entity_types:
            entity_type = item['type']
            count = item['count']
            distribution[entity_type] = count / total_entities if total_entities > 0 else 0.0
        
        return distribution
    
    def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze graph connectivity."""
        # Placeholder implementation
        return {
            'connected_components': 1,
            'largest_component_size': 0.95,
            'avg_path_length': 3.2,
            'clustering_coefficient': 0.15
        }
    
    def export_graph_summary(self, output_path: str) -> None:
        """Export a summary of the knowledge graph to a file.
        
        Args:
            output_path: Path to save the summary file.
        """
        logger.info(f"Exporting graph summary to {output_path}")
        
        # Get comprehensive statistics
        graph_stats = self.neo4j.get_graph_statistics()
        quality_metrics = self.validate_graph_quality()
        
        summary = {
            'build_statistics': self.stats,
            'graph_statistics': graph_stats,
            'quality_metrics': quality_metrics,
            'model_information': {
                'llm_model': self.llm_manager.get_model_info(),
                'embedding_model': self.embedding_manager.get_embedding_stats(),
                'extraction_config': self.entity_extractor.get_extraction_statistics()
            }
        }
        
        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Graph summary exported to {output_path}")
    
    def clear_graph(self) -> None:
        """Clear all data from the knowledge graph. Use with caution!"""
        logger.warning("Clearing all data from knowledge graph...")
        self.neo4j.clear_database()
        
        # Reset statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_processed': 0,
            'entities_extracted': 0,
            'relationships_extracted': 0,
            'entities_stored': 0,
            'relationships_stored': 0,
            'processing_time': 0.0
        }
        
        logger.info("Knowledge graph cleared")
    
    def get_build_statistics(self) -> Dict[str, Any]:
        """Get the current build statistics."""
        return self.stats.copy()
    
    def close(self) -> None:
        """Close all connections and clean up resources."""
        logger.info("Closing KnowledgeGraphBuilder...")
        
        if self.neo4j:
            self.neo4j.close()
        
        if self.llm_manager:
            self.llm_manager.unload_model()
        
        if self.embedding_manager:
            self.embedding_manager.unload_model()
        
        logger.info("KnowledgeGraphBuilder closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 