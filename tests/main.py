"""
Main orchestration script for GraphRAG implementation.
Provides testing, configuration management, and pipeline execution.
"""

import yaml
import logging
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.document_processor import DocumentProcessor, DocumentChunk, Document
from src.kg.entity_extractor import EntityExtractor, Entity, Relationship
from src.kg.neo4j import GraphStorage
from src.models.model import ModelManager
from src.models.prompt import get_extraction_prompt
from test_data import get_test_documents

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
        
        self.graph_storage = None
        try:
            self.graph_storage = GraphStorage(
                uri=self.config.get('neo4j', {}).get('uri', 'bolt://localhost:7687'),
                username=self.config.get('neo4j', {}).get('username', 'neo4j'),
                password=self.config.get('neo4j', {}).get('password', 'password'),
                database=self.config.get('neo4j', {}).get('database', 'neo4j')
            )
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            logger.info("Pipeline will run without graph storage")
        
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
            return self._get_default_config()
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'models': {
                'llm': {
                    'name': 'Qwen/Qwen2.5-7B-Instruct',
                    'device': 'cuda',
                    'load_in_8bit': True
                },
                'embedding': {
                    'name': 'sentence-transformers/msmarco-distilbert-base-tas-b',
                    'device': 'cuda'
                }
            },
            'processing': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'batch_size': 4
            },
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': 'password',
                'database': 'neo4j'
            },
            'extraction': {
                'max_entities_per_chunk': 20,
                'entity_types': ['PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 'CONCEPT']
            }
        }
    
    def run_pipeline(self, input_path: str = None, clear_graph: bool = False) -> Dict[str, Any]:
        """
        Run the complete GraphRAG pipeline.
        
        Args:
            input_path: Path to documents (file or directory), if None uses test data
            clear_graph: Whether to clear existing graph data
            
        Returns:
            Pipeline statistics
        """
        self.stats['start_time'] = time.time()
        logger.info(f"Starting GraphRAG pipeline")
        
        try:
            # Setup
            if self.graph_storage and clear_graph:
                logger.info("Clearing existing graph data")
                self.graph_storage.clear_graph()
            
            if self.graph_storage:
                self.graph_storage.create_schema()
            
            # Step 1: Load and process documents
            logger.info("Step 1: Loading and processing documents")
            
            if input_path:
                input_path = Path(input_path)
                if input_path.is_file():
                    documents = [self.document_processor.load_document(input_path)]
                elif input_path.is_dir():
                    documents = self.document_processor.load_documents(input_path)
                else:
                    raise ValueError(f"Invalid input path: {input_path}")
            else:
                # Use test data from the dedicated test_data module
                documents = get_test_documents()
            
            self.stats['documents_processed'] = len(documents)
            
            # Step 2: Create chunks
            logger.info("Step 2: Creating document chunks")
            all_chunks = self.document_processor.process_documents(documents)
            self.stats['chunks_created'] = len(all_chunks)
            
            if not all_chunks:
                logger.warning("No chunks created, exiting pipeline")
                return self.stats
            
            # Step 3: Extract entities and relationships
            logger.info("Step 3: Extracting entities and relationships")
            
            # Prepare chunks for batch processing
            chunk_data = [(chunk.text, chunk.id) for chunk in all_chunks]
            
            # Extract in batches
            batch_size = self.config.get('processing', {}).get('batch_size', 4)
            all_entities = []
            all_relationships = []
            
            for i in range(0, len(chunk_data), batch_size):
                batch = chunk_data[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(chunk_data) + batch_size - 1)//batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                entities, relationships = self.entity_extractor.extract_batch(batch)
                all_entities.extend(entities)
                all_relationships.extend(relationships)
                
                logger.info(f"Batch {batch_num}: extracted {len(entities)} entities, {len(relationships)} relationships")
            
            self.stats['entities_extracted'] = len(all_entities)
            self.stats['relationships_extracted'] = len(all_relationships)
            
            # Step 4: Merge duplicates
            logger.info("Step 4: Merging duplicate entities and relationships")
            merged_entities = self.entity_extractor.merge_entities(all_entities)
            merged_relationships = self.entity_extractor.merge_relationships(all_relationships)
            
            logger.info(f"After merging: {len(merged_entities)} unique entities, {len(merged_relationships)} unique relationships")
            
            # Step 5: Store in graph (if available)
            if self.graph_storage:
                logger.info("Step 5: Storing in knowledge graph")
                entities_stored = self.graph_storage.store_entities(merged_entities)
                relationships_stored = self.graph_storage.store_relationships(merged_relationships)
                
                self.stats['entities_stored'] = entities_stored
                self.stats['relationships_stored'] = relationships_stored
            else:
                logger.info("Step 5: Skipping graph storage (Neo4j not available)")
                self.stats['entities_stored'] = 0
                self.stats['relationships_stored'] = 0
            
            # Finalize
            self.stats['end_time'] = time.time()
            self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
            
            logger.info("GraphRAG pipeline completed successfully")
            self._log_statistics()
            
            self._save_results(merged_entities, merged_relationships)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats['error'] = str(e)
            return self.stats
    
    def _save_results(self, entities: List[Entity], relationships: List[Relationship]):
        """Save extracted entities and relationships to a JSON file in the 'results' directory."""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

        output_data = {
            'entities': [e.__dict__ for e in entities],
            'relationships': [r.__dict__ for r in relationships],
            'statistics': self.stats
        }

        output_file = output_dir / f"graph_results_{int(time.time())}.json"

        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=4)
            logger.info(f"Saved results to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def test_model_loading(self):
        """Test LLM and embedding model loading and basic inference."""
        logger.info("Testing model loading...")
        
        # Test LLM generation
        logger.info("Testing LLM generation...")
        # Use the centralized prompt generator
        test_content = "OpenAI is an AI research company based in San Francisco, founded by Sam Altman."
        test_prompt = get_extraction_prompt(test_content)
        
        response = self.model_manager.inference(
            test_prompt,
            max_tokens=150, # Increased for a more realistic extraction
            stop_sequences=["<|COMPLETE|>"]
        )
        if response and response.strip():
            logger.info(f"✓ LLM generated response: {response[:100]}...")
        else:
            logger.warning("⚠ LLM generation returned empty response")
        
        # Test embedding model loading
        embedding_success = self.model_manager.load_embedding_model()
        if embedding_success:
            logger.info("✓ Embedding model loaded successfully")
            
            # Test embedding generation
            test_embedding = self.model_manager.generate_embedding("This is a test sentence.")
            if isinstance(test_embedding, list) and len(test_embedding) > 0:
                logger.info(f"✓ Embedding generated with dimension: {len(test_embedding)}")
            else:
                logger.warning("⚠ Embedding generation failed or returned empty.")
                embedding_success = False
        else:
            logger.error("✗ Failed to load embedding model")

        # Get model info
        info = self.model_manager.get_model_info()
        logger.info(f"Model info: {info}")
        
        return response.strip() and embedding_success
    
    def query_graph(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query the graph for entities."""
        if not self.graph_storage:
            logger.error("Graph storage not available")
            return []
        
        return self.graph_storage.search_entities(query, max_results)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self.graph_storage:
            logger.error("Graph storage not available")
            return {}
        
        return self.graph_storage.get_graph_statistics()
    
    def validate_graph(self) -> Dict[str, Any]:
        """Validate graph integrity."""
        if not self.graph_storage:
            logger.error("Graph storage not available")
            return {}
        
        return self.graph_storage.validate_graph_integrity()
    
    def export_graph(self, output_path: str):
        """Export graph data."""
        if not self.graph_storage:
            logger.error("Graph storage not available")
            return
        
        graph_data = self.graph_storage.export_graph_data()
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
        
        logger.info(f"Graph data exported to {output_path}")
    
    def _log_statistics(self):
        """Log pipeline statistics."""
        stats = self.stats
        logger.info("=== Pipeline Statistics ===")
        logger.info(f"Total time: {stats.get('total_time', 0):.2f} seconds")
        logger.info(f"Documents processed: {stats['documents_processed']}")
        logger.info(f"Chunks created: {stats['chunks_created']}")
        logger.info(f"Entities extracted: {stats['entities_extracted']}")
        logger.info(f"Relationships extracted: {stats['relationships_extracted']}")
        logger.info(f"Entities stored: {stats['entities_stored']}")
        logger.info(f"Relationships stored: {stats['relationships_stored']}")
        
        if stats['chunks_created'] > 0:
            logger.info(f"Average entities per chunk: {stats['entities_extracted'] / stats['chunks_created']:.1f}")
            logger.info(f"Average relationships per chunk: {stats['relationships_extracted'] / stats['chunks_created']:.1f}")
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up pipeline resources")
        
        if hasattr(self, 'model_manager'):
            self.model_manager.cleanup()
        
        if hasattr(self, 'graph_storage') and self.graph_storage:
            self.graph_storage.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GraphRAG Pipeline with Qwen 7B")
    parser.add_argument('--input', '-i', type=str, help='Input file or directory')
    parser.add_argument('--config', '-c', type=str, default='configs/config.yaml', help='Config file path')
    parser.add_argument('--clear', action='store_true', help='Clear existing graph data')
    parser.add_argument('--query', '-q', type=str, help='Query the knowledge graph')
    parser.add_argument('--stats', action='store_true', help='Show graph statistics')
    parser.add_argument('--validate', action='store_true', help='Validate graph integrity')
    parser.add_argument('--export', '-e', type=str, help='Export graph data to file')
    parser.add_argument('--test-models', action='store_true', help='Test model loading and generation')
    parser.add_argument('--run-pipeline', action='store_true', help='Run the full pipeline with test data')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GraphRAGPipeline(args.config)
    
    try:
        if args.test_models:
            # Test model loading
            success = pipeline.test_model_loading()
            if success:
                print("\n✓ All models loaded successfully!")
                print("Ready to run the full pipeline.")
            else:
                print("\n✗ Model loading failed. Check your configuration and dependencies.")
            return
        
        if args.run_pipeline or args.input:
            # Run the main pipeline
            stats = pipeline.run_pipeline(args.input, clear_graph=args.clear)
            print("\nPipeline completed!")
            print(f"Processed {stats['documents_processed']} documents in {stats.get('total_time', 0):.2f} seconds")
            print(f"Extracted {stats['entities_extracted']} entities and {stats['relationships_extracted']} relationships")
        
        if args.query:
            # Query the graph
            results = pipeline.query_graph(args.query)
            print(f"\nQuery results for '{args.query}':")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.get('name', 'Unknown')} ({result.get('type', 'Unknown')})")
                print(f"   Description: {result.get('description', 'No description')[:100]}...")
        
        if args.stats:
            # Show graph statistics
            stats = pipeline.get_graph_statistics()
            print("\nGraph Statistics:")
            print(f"Total entities: {stats.get('total_entities', 0)}")
            print(f"Total relationships: {stats.get('total_relationships', 0)}")
            print(f"Average degree: {stats.get('average_degree', 0):.2f}")
            print(f"Graph density: {stats.get('density', 0):.4f}")
        
        if args.validate:
            # Validate graph
            validation = pipeline.validate_graph()
            print("\nGraph Validation:")
            print(f"Validation passed: {validation.get('validation_passed', False)}")
            if not validation.get('validation_passed', False):
                print("Issues found:")
                for key, value in validation.items():
                    if key != 'validation_passed':
                        print(f"  {key}: {value}")
        
        if args.export:
            # Export graph data
            pipeline.export_graph(args.export)
            print(f"Graph data exported to {args.export}")
        
        # If no specific action, show help
        if not any([args.input, args.query, args.stats, args.validate, args.export, args.test_models, args.run_pipeline]):
            print("GraphRAG Pipeline with Qwen 7B")
            print("\nQuick start:")
            print("1. Test model loading:     python main.py --test-models")
            print("2. Run with test data:     python main.py --run-pipeline")
            print("3. Process your files:     python main.py --input /path/to/documents")
            print("4. Query the graph:        python main.py --query 'artificial intelligence'")
            print("\nUse --help for all options.")
    
    finally:
        # Cleanup
        pipeline.cleanup()


if __name__ == "__main__":
    main()