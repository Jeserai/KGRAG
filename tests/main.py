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
from pipeline import run_pipeline, load_config, to_serializable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graphrag.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GraphRAG pipeline")
    parser.add_argument("--input", "-i", type=str, help="Path to file OR directory of docs")
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    start = time.time()
    results = run_pipeline(cfg, Path(args.input) if args.input else None)
    elapsed = time.time() - start

    # Quick summary to stdout
    print("\n=== SUMMARY ===")
    print(f"Documents: {results['docs']}")
    print(f"Chunks: {results['chunks']}")
    print(f"Entities: {len(results['entities'])}")
    print(f"Relationships: {len(results['relationships'])}")
    print(f"Total time: {elapsed:.1f}s")

if __name__ == "__main__":
    main() 