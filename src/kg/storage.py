"""
Graph storage system: JSON for lightweight, Parquet for heavyweight.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict

from kg.extractor.entity_extractor import Entity, Relationship

class GraphStorage:
    def __init__(self, base_path: str = "data", threshold: int = 1000):
        """
        Input:
            base_path: Directory to store files
            threshold: Use JSON below this count, Parquet above
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.threshold = threshold
    
    def save(self, entities: List[Entity], relationships: List[Relationship], name: str = "graph"):
        """Save entities and relationships, auto-choosing format."""
        total_count = len(entities) + len(relationships)
        
        if total_count < self.threshold:
            self._save_json(entities, relationships, name)
        else:
            self._save_parquet(entities, relationships, name)
    
    def load(self, name: str = "graph"):
        """Load entities and relationships, auto-detecting format."""
        json_file = self.base_path / f"{name}.json"
        parquet_entities = self.base_path / f"{name}_entities.parquet"
        
        if json_file.exists():
            return self._load_json(name)
        elif parquet_entities.exists():
            return self._load_parquet(name)
        else:
            return [], []
    
    def _save_json(self, entities: List[Entity], relationships: List[Relationship], name: str):
        """Save to single JSON file."""
        data = {
            'entities': [asdict(e) for e in entities],
            'relationships': [asdict(r) for r in relationships]
        }
        
        filepath = self.base_path / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f" Saved to JSON: {filepath} ({len(entities)} entities, {len(relationships)} relationships)")
    
    def _save_parquet(self, entities: List[Entity], relationships: List[Relationship], name: str):
        """Save to separate Parquet files."""
        # Entities
        if entities:
            entities_df = pd.DataFrame([asdict(e) for e in entities])
            entities_file = self.base_path / f"{name}_entities.parquet"
            entities_df.to_parquet(entities_file)
        
        # Relationships  
        if relationships:
            relationships_df = pd.DataFrame([asdict(r) for r in relationships])
            relationships_file = self.base_path / f"{name}_relationships.parquet"
            relationships_df.to_parquet(relationships_file)
        
        print(f" Saved to Parquet: {name}_*.parquet ({len(entities)} entities, {len(relationships)} relationships)")
    
    def _load_json(self, name: str):
        """Load from JSON file."""
        filepath = self.base_path / f"{name}.json"
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        entities = [Entity(**e) for e in data['entities']]
        relationships = [Relationship(**r) for r in data['relationships']]
        
        print(f" Loaded from JSON: {filepath}")
        return entities, relationships
    
    def _load_parquet(self, name: str):
        """Load from Parquet files."""
        entities, relationships = [], []
        
        # Load entities
        entities_file = self.base_path / f"{name}_entities.parquet"
        if entities_file.exists():
            entities_df = pd.read_parquet(entities_file)
            entities = [Entity(**row.to_dict()) for _, row in entities_df.iterrows()]
        
        # Load relationships
        relationships_file = self.base_path / f"{name}_relationships.parquet"
        if relationships_file.exists():
            relationships_df = pd.read_parquet(relationships_file)
            relationships = [Relationship(**row.to_dict()) for _, row in relationships_df.iterrows()]
        
        print(f" Loaded from Parquet: {name}_*.parquet")
        return entities, relationships