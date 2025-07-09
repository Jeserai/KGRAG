from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.document_processor import DocumentProcessor, Document
from src.models.model import ModelManager
from tests.test_data import get_test_documents
from kg.extractor.entity_extractor import EntityExtractor
from kg.extractor.merging import EntityMerger
from src.kg.storage import GraphStorage
from src.query.retriever.retriever import HybridRetriever
from src.query.augment import build_augmented_prompt
from tests.test_queries import get_test_queries

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")


def load_config(config_path: Path) -> Dict[str, Any]:
    import yaml  

    if not config_path.exists():
        logger.warning("Config file not found – using defaults")
        return {}

    with open(config_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    return cfg or {}


def to_serializable(obj):
    """Convert dataclass or custom objects to dicts for JSON serialization."""
    if hasattr(obj, "__dict__"):
        return {k: to_serializable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    return obj


def run_pipeline(cfg: Dict[str, Any], input_path: Path | None = None) -> Dict[str, Any]:
    """Run the GraphRAG pipeline and return a results dict."""

    model_mgr = ModelManager(cfg.get("models", {}))
    doc_proc = DocumentProcessor(
        chunk_size=cfg.get("processing", {}).get("chunk_size", 800),
        chunk_overlap=cfg.get("processing", {}).get("chunk_overlap", 150),
    )
    extractor = EntityExtractor(
        model_manager=model_mgr,
        max_entities_per_chunk=cfg.get("extraction", {}).get("max_entities_per_chunk", 15),
    )

    logger.info("Loading LLM & embedding models (eager)…")
    model_mgr.ensure_models_loaded()

    merger = EntityMerger(name_threshold=0.8)
    storage = GraphStorage(base_path="results")

    # Load documents
    if input_path is None:
        documents: List[Document] = get_test_documents()
        logger.info("Loaded %d built-in test documents", len(documents))
    else:
        input_path = input_path.expanduser()
        if input_path.is_file():
            documents = [doc_proc.load_document(input_path)]
        elif input_path.is_dir():
            documents = doc_proc.load_documents(input_path)
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
        logger.info("Loaded %d documents from %s", len(documents), input_path)

    # Chunk documents
    chunks = doc_proc.process_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    # Extract entities & relationships in batches
    chunk_data = [(c.text, c.id) for c in chunks]
    entities, relationships = extractor.extract_batch(chunk_data)
    logger.info("Extracted %d entities and %d relationships", len(entities), len(relationships))

    # Merge duplicates
    entities = merger.merge_entities(entities)
    relationships = extractor.merge_relationships(relationships)
    logger.info("After merge: %d unique entities, %d unique relationships", len(entities), len(relationships))

    storage.save(entities, relationships, name="graphs/graph")

    # ---------------- Query step -----------------
    queries_dir = Path("results/queries")
    queries_dir.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever(
        entities=entities,
        relationships=relationships,
        embedding_model=model_mgr.embedding_model,
        chunks=chunks,
    )

    for q in get_test_queries():
        ret = retriever.retrieve(q, include_chunks=True)
        prompt = build_augmented_prompt(q, ret)
        out_file = queries_dir / f"query_{q[:20].replace(' ','_')}.json"
        with open(out_file, "w") as f:
            json.dump({"query": q, "prompt": prompt}, f, indent=2)
        logger.info("Saved augmented prompt for query '%s'", q)

    logger.info("Graph & query prompts saved under results/")

    return {"docs": len(documents), "chunks": len(chunks), "entities": len(entities), "relationships": len(relationships)}