"""
Neo4j database connector for Knowledge Graph RAG.
Handles all database operations for entities, relationships, and embeddings.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
import time

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float = 1.0


class Neo4jConnector:
    """Connector for Neo4j database operations."""
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 username: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j"):
        """Initialize Neo4j connection.
        
        Args:
            uri: Neo4j database URI.
            username: Database username.
            password: Database password.
            database: Database name.
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        
        self._connect()
        self._create_constraints()
        
        logger.info(f"Connected to Neo4j database at {uri}")
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _create_constraints(self) -> None:
        """Create necessary constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS FOR (r:Relationship) REQUIRE r.id IS UNIQUE",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX relationship_type_index IF NOT EXISTS FOR (r:Relationship) ON (r.type)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Failed to create constraint/index: {e}")
    
    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def clear_database(self) -> None:
        """Clear all data from the database. Use with caution!"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
    
    def create_entity(self, entity: Entity) -> bool:
        """Create an entity in the database.
        
        Args:
            entity: Entity object to create.
            
        Returns:
            True if successful, False otherwise.
        """
        query = """
        MERGE (e:Entity {id: $entity_id})
        SET e.name = $name,
            e.type = $type,
            e.confidence = $confidence,
            e.properties = $properties,
            e.created_at = datetime(),
            e.updated_at = datetime()
        RETURN e.id as id
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, 
                    entity_id=entity.id,
                    name=entity.name,
                    type=entity.type,
                    confidence=entity.confidence,
                    properties=entity.properties
                )
                
                if result.single():
                    logger.debug(f"Created entity: {entity.id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to create entity {entity.id}: {e}")
            return False
    
    def create_entities_batch(self, entities: List[Entity]) -> int:
        """Create multiple entities in batch.
        
        Args:
            entities: List of Entity objects to create.
            
        Returns:
            Number of entities successfully created.
        """
        if not entities:
            return 0
        
        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {id: entity.id})
        SET e.name = entity.name,
            e.type = entity.type,
            e.confidence = entity.confidence,
            e.properties = entity.properties,
            e.created_at = datetime(),
            e.updated_at = datetime()
        RETURN COUNT(e) as count
        """
        
        entity_data = [
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "confidence": entity.confidence,
                "properties": entity.properties
            }
            for entity in entities
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entities=entity_data)
                count = result.single()["count"]
                logger.info(f"Created {count} entities in batch")
                return count
                
        except Exception as e:
            logger.error(f"Failed to create entity batch: {e}")
            return 0
    
    def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship between entities.
        
        Args:
            relationship: Relationship object to create.
            
        Returns:
            True if successful, False otherwise.
        """
        query = """
        MATCH (source:Entity {id: $source_id})
        MATCH (target:Entity {id: $target_id})
        MERGE (source)-[r:RELATED {id: $rel_id}]->(target)
        SET r.type = $rel_type,
            r.confidence = $confidence,
            r.properties = $properties,
            r.created_at = datetime(),
            r.updated_at = datetime()
        RETURN r.id as id
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query,
                    source_id=relationship.source_entity_id,
                    target_id=relationship.target_entity_id,
                    rel_id=relationship.id,
                    rel_type=relationship.relation_type,
                    confidence=relationship.confidence,
                    properties=relationship.properties
                )
                
                if result.single():
                    logger.debug(f"Created relationship: {relationship.id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to create relationship {relationship.id}: {e}")
            return False
    
    def create_relationships_batch(self, relationships: List[Relationship]) -> int:
        """Create multiple relationships in batch.
        
        Args:
            relationships: List of Relationship objects to create.
            
        Returns:
            Number of relationships successfully created.
        """
        if not relationships:
            return 0
        
        query = """
        UNWIND $relationships AS rel
        MATCH (source:Entity {id: rel.source_id})
        MATCH (target:Entity {id: rel.target_id})
        MERGE (source)-[r:RELATED {id: rel.id}]->(target)
        SET r.type = rel.type,
            r.confidence = rel.confidence,
            r.properties = rel.properties,
            r.created_at = datetime(),
            r.updated_at = datetime()
        RETURN COUNT(r) as count
        """
        
        rel_data = [
            {
                "id": rel.id,
                "source_id": rel.source_entity_id,
                "target_id": rel.target_entity_id,
                "type": rel.relation_type,
                "confidence": rel.confidence,
                "properties": rel.properties
            }
            for rel in relationships
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, relationships=rel_data)
                count = result.single()["count"]
                logger.info(f"Created {count} relationships in batch")
                return count
                
        except Exception as e:
            logger.error(f"Failed to create relationship batch: {e}")
            return 0
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID.
        
        Args:
            entity_id: ID of the entity to retrieve.
            
        Returns:
            Entity object if found, None otherwise.
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e.id as id, e.name as name, e.type as type, 
               e.confidence as confidence, e.properties as properties
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)
                record = result.single()
                
                if record:
                    return Entity(
                        id=record["id"],
                        name=record["name"],
                        type=record["type"],
                        confidence=record["confidence"],
                        properties=record["properties"] or {}
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get entity {entity_id}: {e}")
            return None
    
    def find_entities_by_name(self, name: str, fuzzy: bool = True) -> List[Entity]:
        """Find entities by name.
        
        Args:
            name: Entity name to search for.
            fuzzy: If True, performs fuzzy matching.
            
        Returns:
            List of matching Entity objects.
        """
        if fuzzy:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e.id as id, e.name as name, e.type as type,
                   e.confidence as confidence, e.properties as properties
            ORDER BY e.confidence DESC
            LIMIT 50
            """
        else:
            query = """
            MATCH (e:Entity {name: $name})
            RETURN e.id as id, e.name as name, e.type as type,
                   e.confidence as confidence, e.properties as properties
            """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, name=name)
                entities = []
                
                for record in result:
                    entities.append(Entity(
                        id=record["id"],
                        name=record["name"],
                        type=record["type"],
                        confidence=record["confidence"],
                        properties=record["properties"] or {}
                    ))
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to find entities by name '{name}': {e}")
            return []
    
    def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Find entities by type.
        
        Args:
            entity_type: Type of entities to find.
            
        Returns:
            List of Entity objects of the specified type.
        """
        query = """
        MATCH (e:Entity {type: $entity_type})
        RETURN e.id as id, e.name as name, e.type as type,
               e.confidence as confidence, e.properties as properties
        ORDER BY e.confidence DESC
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_type=entity_type)
                entities = []
                
                for record in result:
                    entities.append(Entity(
                        id=record["id"],
                        name=record["name"],
                        type=record["type"],
                        confidence=record["confidence"],
                        properties=record["properties"] or {}
                    ))
                
                return entities
                
        except Exception as e:
            logger.error(f"Failed to find entities by type '{entity_type}': {e}")
            return []
    
    def get_entity_relationships(self, entity_id: str, max_hops: int = 1) -> List[Tuple[Entity, Relationship, Entity]]:
        """Get relationships for an entity with multi-hop traversal.
        
        Args:
            entity_id: ID of the central entity.
            max_hops: Maximum number of hops to traverse.
            
        Returns:
            List of (source_entity, relationship, target_entity) tuples.
        """
        query = f"""
        MATCH path = (source:Entity {{id: $entity_id}})-[r:RELATED*1..{max_hops}]-(target:Entity)
        UNWIND relationships(path) as rel
        MATCH (start)-[rel]-(end)
        RETURN DISTINCT 
               start.id as source_id, start.name as source_name, start.type as source_type,
               start.confidence as source_confidence, start.properties as source_properties,
               rel.id as rel_id, rel.type as rel_type, rel.confidence as rel_confidence,
               rel.properties as rel_properties,
               end.id as target_id, end.name as target_name, end.type as target_type,
               end.confidence as target_confidence, end.properties as target_properties
        ORDER BY rel.confidence DESC
        LIMIT 100
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)
                relationships = []
                
                for record in result:
                    source_entity = Entity(
                        id=record["source_id"],
                        name=record["source_name"],
                        type=record["source_type"],
                        confidence=record["source_confidence"],
                        properties=record["source_properties"] or {}
                    )
                    
                    target_entity = Entity(
                        id=record["target_id"],
                        name=record["target_name"],
                        type=record["target_type"],
                        confidence=record["target_confidence"],
                        properties=record["target_properties"] or {}
                    )
                    
                    relationship = Relationship(
                        id=record["rel_id"],
                        source_entity_id=record["source_id"],
                        target_entity_id=record["target_id"],
                        relation_type=record["rel_type"],
                        confidence=record["rel_confidence"],
                        properties=record["rel_properties"] or {}
                    )
                    
                    relationships.append((source_entity, relationship, target_entity))
                
                return relationships
                
        except Exception as e:
            logger.error(f"Failed to get relationships for entity {entity_id}: {e}")
            return []
    
    def find_path_between_entities(self, source_id: str, target_id: str, max_hops: int = 3) -> List[List[Tuple[Entity, Relationship]]]:
        """Find paths between two entities.
        
        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            max_hops: Maximum path length.
            
        Returns:
            List of paths, each path is a list of (entity, relationship) tuples.
        """
        query = f"""
        MATCH path = (source:Entity {{id: $source_id}})-[r:RELATED*1..{max_hops}]-(target:Entity {{id: $target_id}})
        RETURN nodes(path) as entities, relationships(path) as relationships
        ORDER BY length(path)
        LIMIT 10
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, source_id=source_id, target_id=target_id)
                paths = []
                
                for record in result:
                    entities = record["entities"]
                    relationships = record["relationships"]
                    
                    path = []
                    for i, entity_node in enumerate(entities[:-1]):  # Exclude last entity
                        entity = Entity(
                            id=entity_node["id"],
                            name=entity_node["name"],
                            type=entity_node["type"],
                            confidence=entity_node["confidence"],
                            properties=entity_node.get("properties", {})
                        )
                        
                        if i < len(relationships):
                            rel_node = relationships[i]
                            relationship = Relationship(
                                id=rel_node["id"],
                                source_entity_id=rel_node.start_node["id"],
                                target_entity_id=rel_node.end_node["id"],
                                relation_type=rel_node["type"],
                                confidence=rel_node["confidence"],
                                properties=rel_node.get("properties", {})
                            )
                            path.append((entity, relationship))
                    
                    paths.append(path)
                
                return paths
                
        except Exception as e:
            logger.error(f"Failed to find path between {source_id} and {target_id}: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.
        
        Returns:
            Dictionary containing graph statistics.
        """
        queries = {
            "total_entities": "MATCH (e:Entity) RETURN COUNT(e) as count",
            "total_relationships": "MATCH ()-[r:RELATED]->() RETURN COUNT(r) as count",
            "entity_types": "MATCH (e:Entity) RETURN e.type as type, COUNT(e) as count ORDER BY count DESC",
            "relationship_types": "MATCH ()-[r:RELATED]->() RETURN r.type as type, COUNT(r) as count ORDER BY count DESC",
            "avg_degree": """MATCH (e:Entity) 
                           WITH e, size((e)-[]-()) as degree 
                           RETURN AVG(degree) as avg_degree""",
        }
        
        stats = {}
        
        try:
            with self.driver.session(database=self.database) as session:
                # Get basic counts
                for stat_name, query in queries.items():
                    if stat_name in ["entity_types", "relationship_types"]:
                        result = session.run(query)
                        stats[stat_name] = [{"type": record["type"], "count": record["count"]} 
                                          for record in result]
                    else:
                        result = session.run(query)
                        record = result.single()
                        stats[stat_name] = record["count"] if "count" in record else record["avg_degree"]
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            stats = {"error": str(e)}
        
        return stats
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships.
        
        Args:
            entity_id: ID of the entity to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        DETACH DELETE e
        RETURN COUNT(e) as deleted
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)
                deleted = result.single()["deleted"]
                return deleted > 0
                
        except Exception as e:
            logger.error(f"Failed to delete entity {entity_id}: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 