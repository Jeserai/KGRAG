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
from src.models.model import ModelManager
from src.models.prompt import get_extraction_prompt
from test_data import get_test_documents
from test_pipeline import GraphRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graphrag.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

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
        # Load models
        print("\n Loading models...")
        if not pipeline.load_models():
            print("Failed to load models")
            return

        print("Models loaded successfully!")

        # Run pipeline
        print("\n Running the pipeline...")
        results = pipeline.run_pipeline()

        # Display results
        pipeline.display_results(results)

        # Save results
        filename = pipeline.save_results(results)
        print(f"\n Complete results saved to: {filename}")

        return results

    except Exception as e:
        logger.error(f" Pipeline error: {e}")
        print(f" Error: {e}")

    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main()