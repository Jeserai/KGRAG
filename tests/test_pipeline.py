import yaml
import logging
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import sys
import os
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.document_processor import DocumentProcessor, DocumentChunk, Document
from src.kg.entity_extractor import EntityExtractor, Entity, Relationship
from src.models.model import ModelManager
from src.models.prompt import get_extraction_prompt
from test_data import get_test_documents
from src.kg.storage import GraphStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graphrag.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Main GraphRAG pipeline orchestrator."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the GraphRAG pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = self._load_config(config_path)
        self.model_manager = ModelManager(self.config.get('models', {}))
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.get('processing', {}).get('chunk_size', 1000),
            chunk_overlap=self.config.get('processing', {}).get('chunk_overlap', 200)
        )
        self.entity_extractor = EntityExtractor(
            model_manager=self.model_manager,
            max_entities_per_chunk=self.config.get('extraction', {}).get('max_entities_per_chunk', 20)
        )
        self.storage = GraphStorage()
        self.stats = {
            'start_time': None,
            'end_time': None,
            'documents_processed': 0,
            'chunks_created': 0,
            'entities_extracted': 0,
            'relationships_extracted': 0,
            'entities_stored': 0,
            'relationships_stored': 0
        }
        
        logger.info("GraphRAG pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return

    def load_models(self) -> bool:
        """Load all required models."""
        return self.model_manager.load_model()

    def log_statistics(self):
        """Log pipeline statistics."""
        stats = self.stats
        logger.info(" === Pipeline Statistics ===")
        logger.info(f"Total time: {stats['total_time']:.2f} seconds")
        logger.info(f"Documents processed: {stats['documents_processed']}")
        logger.info(f"Chunks created: {stats['chunks_created']}")
        logger.info(f"Entities extracted: {stats['entities_extracted']}")
        logger.info(f"Relationships extracted: {stats['relationships_extracted']}")

        if stats['chunks_created'] > 0:
            logger.info(f"Avg entities per chunk: {stats['entities_extracted'] / stats['chunks_created']:.1f}")
            logger.info(f"Avg relationships per chunk: {stats['relationships_extracted'] / stats['chunks_created']:.1f}")

    def display_results(self, results: Dict[str, Any]):
        """Display results in a nice format."""
        entities = results['entities']
        relationships = results['relationships']

        print("\n" + "="*80)
        print("GRAPHRAG EXTRACTION RESULTS")
        print("="*80)

        print(f"\n ENTITIES ({len(entities)}):")
        print("-" * 50)
        for i, entity in enumerate(entities[:20], 1):  # Show first 20
            print(f"{i:2d}. {entity.name} ({entity.type})")
            print(f"    {entity.description}")
            if len(entity.source_chunks) > 1:
                print(f"    Found in {len(entity.source_chunks)} chunks")

        if len(entities) > 20:
            print(f"    ... and {len(entities) - 20} more entities")

        print(f"\n RELATIONSHIPS ({len(relationships)}):")
        print("-" * 50)
        for i, rel in enumerate(relationships[:15], 1):  # Show first 15
            print(f"{i:2d}. {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity}")
            print(f"    {rel.description}")

        if len(relationships) > 15:
            print(f"    ... and {len(relationships) - 15} more relationships")

        print(f"\n STATISTICS:")
        print("-" * 50)
        stats = results['statistics']
        print(f"Processing time: {stats['total_time']:.2f} seconds")
        print(f"Documents: {stats['documents_processed']}")
        print(f"Chunks: {stats['chunks_created']}")
        print(f"Total entities: {len(entities)}")
        print(f"Total relationships: {len(relationships)}")

        if self.device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {memory_used:.1f} / {memory_total:.1f} GB")

        print("\n" + "="*80)

    def save_results(self, results: Dict[str, Any], filename: str = None):
        self.storage.save(results['entities'], results['relationships'], filename)
        return f"Saved as {filename}"
    
    def load_results(self, filename: str = None):
        return self.storage.load(filename)
    
    def run_pipeline(self, documents: List[Document] = None) -> Dict[str, Any]:
        """Run the complete GraphRAG pipeline."""
        start_time = time.time()

        logger.info(" Starting GraphRAG pipeline")

        documents = get_test_documents()

        self.stats['documents_processed'] = len(documents)

        # Step 1: Process documents into chunks
        logger.info(" Processing documents into chunks:")
        all_chunks = []
        for doc in documents:
            chunks = self.doc_processor.chunk_document(doc)
            all_chunks.extend(chunks)

        self.stats['chunks_created'] = len(all_chunks)
        logger.info(f" Created {len(all_chunks)} total chunks")

        # Step 2: Extract entities and relationships
        logger.info("Extracting entities and relationships:")
        all_entities = []
        all_relationships = []

        for i, chunk in enumerate(all_chunks):
            logger.info(f"Processing chunk {i+1}/{len(all_chunks)}")
            entities, relationships = self.entity_extractor.extract_from_chunk(chunk)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        self.stats['entities_extracted'] = len(all_entities)
        self.stats['relationships_extracted'] = len(all_relationships)

        # Step 3: Merge duplicates
        logger.info("Merging duplicates:")
        merged_entities = self.entity_extractor.merge_entities(all_entities)
        merged_relationships = self.entity_extractor.merge_relationships(all_relationships)

        # Finalize
        end_time = time.time()
        self.stats['total_time'] = end_time - start_time

        logger.info("Pipeline completed successfully!")
        self.log_statistics()

        return {
            'entities': merged_entities,
            'relationships': merged_relationships,
            'statistics': self.stats
        }

    def cleanup(self):
        """Clean up resources."""
        self.model_manager.cleanup()