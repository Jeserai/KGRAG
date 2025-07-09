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

from tests.pipeline import run_pipeline, load_config, to_serializable

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
    parser = argparse.ArgumentParser(description="Run the GraphRAG pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a single document or directory of documents to process.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = time.time()
    stats = run_pipeline(cfg, Path(args.input) if args.input else None)
    print("\n=== SUMMARY ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    end = time.time()
    print(f"Time taken: {end - start} seconds")

if __name__ == "__main__":
    main() 