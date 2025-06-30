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

# -- Path Setup --
# Add the project root to the Python path to allow imports from 'src'
# This is necessary for running this script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.document_processor import DocumentProcessor, DocumentChunk, Document
from src.kg.entity_extractor import EntityExtractor, Entity, Relationship
from src.kg.neo4j import GraphStorage
from src.models.model import ModelManager


# Configure logging
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
        
        # Initialize components
        self.model_manager = ModelManager(self.config.get('models', {}))
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.get('processing', {}).get('chunk_size', 1000),
            chunk_overlap=self.config.get('processing', {}).get('chunk_overlap', 200)
        )
        self.entity_extractor = EntityExtractor(
            model_manager=self.model_manager,
            max_entities_per_chunk=self.config.get('extraction', {}).get('max_entities_per_chunk', 20)
        )
        
        # Initialize Neo4j (with graceful fallback)
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
        
        # Statistics
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
                    'name': 'sentence-transformers/all-MiniLM-L6-v2',
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
                # Use test data
                documents = self._create_test_documents()
            
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
            
            # Save results to file
            self._save_results(merged_entities, merged_relationships)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats['error'] = str(e)
            return self.stats
    
    def _create_test_documents(self) -> List[Document]:
        """Create test documents for demonstration."""
        test_docs = [
            Document(
                id="ai_overview",
                title="AI and Technology Overview",
                content="""
                Artificial Intelligence (AI) has revolutionized the technology landscape. Leading companies like 
                OpenAI, Google, and Microsoft are at the forefront of AI development. OpenAI, founded by Sam Altman 
                and others, has created groundbreaking models like GPT-3 and GPT-4. These large language models 
                demonstrate remarkable capabilities in natural language understanding and generation.
                
                Google's DeepMind has made significant breakthroughs in areas like AlphaGo, which defeated world 
                champion Go players, and AlphaFold, which solved protein folding predictions. The company's 
                headquarters in London serves as a hub for cutting-edge AI research.
                
                Machine Learning, a subset of AI, enables computers to learn patterns from data without explicit 
                programming. Deep Learning, using neural networks with multiple layers, has been particularly 
                successful in computer vision and natural language processing tasks.
                """
            ),
            Document(
                id="tech_companies",
                title="Technology Companies and Innovation",
                content="""
                The technology sector is dominated by several major players. Apple, based in Cupertino, California, 
                is known for innovative consumer products like the iPhone and Mac computers. Tim Cook serves as the 
                CEO, continuing the legacy established by Steve Jobs.
                
                Amazon, led by Andy Jassy (who succeeded Jeff Bezos), has expanded from e-commerce to cloud computing 
                with Amazon Web Services (AWS). The company's Seattle headquarters oversees operations that span 
                logistics, retail, and enterprise technology solutions.
                
                Tesla, under Elon Musk's leadership, has pioneered electric vehicle technology and autonomous driving 
                systems. The company's Gigafactories in Nevada, Texas, and other locations represent the future of 
                sustainable manufacturing.
                """
            ),
            Document(
                id="research_institutions",
                title="Research Institutions and Academia",
                content="""
                Stanford University, located in Silicon Valley, has been instrumental in training many technology 
                leaders and fostering innovation. The Stanford AI Lab has contributed significantly to machine 
                learning research and computer vision advances.
                
                MIT (Massachusetts Institute of Technology) in Cambridge, Massachusetts, is renowned for its 
                Computer Science and Artificial Intelligence Laboratory (CSAIL). Researchers there work on 
                robotics, human-computer interaction, and distributed systems.
                
                Carnegie Mellon University in Pittsburgh has a strong tradition in AI research, particularly in 
                areas like computer vision, natural language processing, and robotics. The university's partnerships 
                with industry have led to numerous technological breakthroughs.
                """
            )
        ]
        
        logger.info(f"Created {len(test_docs)} test documents")
        return test_docs
    
    def _save_results(self, entities: List[Entity], relationships: List[Relationship]):
        """Save extraction results to JSON file."""
        results = {
            'entities': [
                {
                    'name': e.name,
                    'type': e.type,
                    'description': e.description,
                    'source_chunks': e.source_chunks,
                    'confidence': e.confidence
                }
                for e in entities
            ],
            'relationships': [
                {
                    'source_entity': r.source_entity,
                    'target_entity': r.target_entity,
                    'relationship_type': r.relationship_type,
                    'description': r.description,
                    'source_chunks': r.source_chunks,
                    'confidence': r.confidence
                }
                for r in relationships
            ],
            'statistics': self.stats
        }
        
        output_file = f"graphrag_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def test_model_loading(self):
        """Test model loading and basic functionality."""
        logger.info("Testing model loading...")
        
        # Test LLM loading
        llm_success = self.model_manager.load_llm()
        if llm_success:
            logger.info("✓ LLM loaded successfully")
            
            # Test generation
            test_prompt = """You are an AI assistant that helps extract entities and relationships from text for knowledge graph construction.

## Task
Extract entities and relationships from the provided text. Focus on the most important and relevant entities that would be valuable in a knowledge graph.

## Entity Types
Extract entities of these types: PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, TECHNOLOGY, PRODUCT, DATE

## Output Format
For each entity, provide:
(entity_name<|>entity_type<|>entity_description)##

For each relationship, provide:
(source_entity<|>relationship_type<|>target_entity<|>relationship_description)##

## Guidelines
1. Entity names should be specific and descriptive
2. Descriptions should be concise but informative
3. Relationships should capture meaningful connections
4. Use consistent naming (e.g., "John Smith" not "John" and "Smith" separately)
5. Focus on entities that appear multiple times or are central to the text
6. Maximum 20 entities per text

## Text to Process:
OpenAI is an AI research company founded by Sam Altman. They developed GPT-3, a large language model that uses machine learning.

## Extracted Entities and Relationships:
"""
            
            response = self.model_manager.generate(
                test_prompt, 
                max_tokens=300, 
                temperature=0.1,
                stop_sequences=["<|COMPLETE|>"]
            )
            
            if response.strip():
                logger.info("✓ LLM generation successful")
                logger.info(f"Sample response: {response[:200]}...")
            else:
                logger.warning("⚠ LLM generation returned empty response")
        else:
            logger.error("✗ Failed to load LLM")
        
        # Test embedding model loading
        embedding_success = self.model_manager.load_embedding_model()
        if embedding_success:
            logger.info("✓ Embedding model loaded successfully")
            
            # Test embedding generation
            test_embedding = self.model_manager.generate_embedding("artificial intelligence research")
            if test_embedding:
                logger.info(f"✓ Embedding generation successful (dimension: {len(test_embedding)})")
            else:
                logger.warning("⚠ Embedding generation failed")
        else:
            logger.error("✗ Failed to load embedding model")
        
        # Show model info
        info = self.model_manager.get_model_info()
        logger.info(f"Model info: {info}")
        
        return llm_success and embedding_success
    
    def query_graph(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge graph."""
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