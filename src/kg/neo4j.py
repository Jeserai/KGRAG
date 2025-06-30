"""
Neo4j graph storage for GraphRAG implementation.
Handles entity and relationship storage, retrieval, and graph operations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


class GraphStorage:
    """Neo4j-based graph storage for knowledge graph."""
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 username: str = "neo4j", 
                 password: str = "password",
                 database: str = "neo4j"):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password  
            database: Database name
        """
        self.uri = uri
        self.username = username
        self.database = database
        self.driver: Optional[Driver] = None
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test connection
            with self.driver.session(database=database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {uri}")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    @contextmanager
    def get_session(self):
        """Context manager for Neo4j sessions."""
        if not self.driver:
            raise RuntimeError("Database driver not initialized")
        
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()
    
    def create_schema(self):
        """Create necessary indexes and constraints."""
        with self.get_session() as session:
            # Create constraints for entity uniqueness
            constraints = [
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r]-() ON (r.type)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created constraint/index: {constraint}")
                except Exception as e:
                    logger.warning(f"Failed to create constraint/index: {e}")
        
        logger.info("Schema creation completed")
    
    def clear_graph(self):
        """Clear all nodes and relationships from the graph. Use with caution!"""
        with self.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")
    
    def store_entity(self, entity) -> bool:
        """
        Store or update an entity in the graph.
        
        Args:
            entity: Entity object to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                query = """
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.description = $description,
                    e.confidence = $confidence,
                    e.source_chunks = $source_chunks,
                    e.updated_at = datetime()
                ON CREATE SET e.created_at = datetime()
                RETURN e.name
                """
                
                result = session.run(query, {
                    'name': entity.name,
                    'type': entity.type,
                    'description': entity.description,
                    'confidence': entity.confidence,
                    'source_chunks': entity.source_chunks
                })
                
                return bool(result.single())
        except Exception as e:
            logger.error(f"Error storing entity {entity.name}: {e}")
            return False
    
    def store_entities(self, entities) -> int:
        """
        Store multiple entities in batch.
        
        Args:
            entities: List of entities to store
            
        Returns:
            Number of entities successfully stored
        """
        if not entities:
            return 0
        
        success_count = 0
        
        try:
            with self.get_session() as session:
                # Prepare batch data
                entity_data = []
                for entity in entities:
                    entity_data.append({
                        'name': entity.name,
                        'type': entity.type,
                        'description': entity.description,
                        'confidence': entity.confidence,
                        'source_chunks': entity.source_chunks
                    })
                
                # Batch insert query
                query = """
                UNWIND $entities AS entity
                MERGE (e:Entity {name: entity.name})
                SET e.type = entity.type,
                    e.description = entity.description,
                    e.confidence = entity.confidence,
                    e.source_chunks = entity.source_chunks,
                    e.updated_at = datetime()
                ON CREATE SET e.created_at = datetime()
                """
                
                session.run(query, {'entities': entity_data})
                success_count = len(entities)
                
        except Exception as e:
            logger.error(f"Error in batch entity storage: {e}")
            # Fallback to individual storage
            for entity in entities:
                if self.store_entity(entity):
                    success_count += 1
        
        logger.info(f"Stored {success_count}/{len(entities)} entities")
        return success_count
    
    def store_relationship(self, relationship) -> bool:
        """
        Store a relationship between entities.
        
        Args:
            relationship: Relationship object to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_session() as session:
                query = """
                MATCH (source:Entity {name: $source_name})
                MATCH (target:Entity {name: $target_name})
                MERGE (source)-[r:RELATES {type: $rel_type}]->(target)
                SET r.description = $description,
                    r.confidence = $confidence,
                    r.source_chunks = $source_chunks,
                    r.updated_at = datetime()
                ON CREATE SET r.created_at = datetime()
                RETURN r
                """
                
                result = session.run(query, {
                    'source_name': relationship.source_entity,
                    'target_name': relationship.target_entity,
                    'rel_type': relationship.relationship_type,
                    'description': relationship.description,
                    'confidence': relationship.confidence,
                    'source_chunks': relationship.source_chunks
                })
                
                return bool(result.single())
        except Exception as e:
            logger.error(f"Error storing relationship {relationship.source_entity} -> {relationship.target_entity}: {e}")
            return False
    
    def store_relationships(self, relationships) -> int:
        """
        Store multiple relationships in batch.
        
        Args:
            relationships: List of relationships to store
            
        Returns:
            Number of relationships successfully stored
        """
        if not relationships:
            return 0
        
        success_count = 0
        
        try:
            with self.get_session() as session:
                # Prepare batch data
                rel_data = []
                for rel in relationships:
                    rel_data.append({
                        'source_name': rel.source_entity,
                        'target_name': rel.target_entity,
                        'rel_type': rel.relationship_type,
                        'description': rel.description,
                        'confidence': rel.confidence,
                        'source_chunks': rel.source_chunks
                    })
                
                # Batch insert query
                query = """
                UNWIND $relationships AS rel
                MATCH (source:Entity {name: rel.source_name})
                MATCH (target:Entity {name: rel.target_name})
                MERGE (source)-[r:RELATES {type: rel.rel_type}]->(target)
                SET r.description = rel.description,
                    r.confidence = rel.confidence,
                    r.source_chunks = rel.source_chunks,
                    r.updated_at = datetime()
                ON CREATE SET r.created_at = datetime()
                """
                
                session.run(query, {'relationships': rel_data})
                success_count = len(relationships)
                
        except Exception as e:
            logger.error(f"Error in batch relationship storage: {e}")
            # Fallback to individual storage
            for rel in relationships:
                if self.store_relationship(rel):
                    success_count += 1
        
        logger.info(f"Stored {success_count}/{len(relationships)} relationships")
        return success_count
    
    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an entity by name."""
        try:
            with self.get_session() as session:
                query = "MATCH (e:Entity {name: $name}) RETURN e"
                result = session.run(query, {'name': name})
                record = result.single()
                
                if record:
                    return dict(record['e'])
                return None
        except Exception as e:
            logger.error(f"Error retrieving entity {name}: {e}")
            return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific type."""
        try:
            with self.get_session() as session:
                query = "MATCH (e:Entity {type: $type}) RETURN e"
                result = session.run(query, {'type': entity_type})
                
                return [dict(record['e']) for record in result]
        except Exception as e:
            logger.error(f"Error retrieving entities of type {entity_type}: {e}")
            return []
    
    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get entities related to a given entity within specified depth."""
        try:
            with self.get_session() as session:
                query = f"""
                MATCH (start:Entity {{name: $name}})
                MATCH (start)-[*1..{max_depth}]-(related:Entity)
                WHERE related <> start
                RETURN DISTINCT related
                """
                result = session.run(query, {'name': entity_name})
                
                return [dict(record['related']) for record in result]
        except Exception as e:
            logger.error(f"Error retrieving related entities for {entity_name}: {e}")
            return []
    
    def search_entities(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by name or description."""
        try:
            with self.get_session() as session:
                query = """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $term OR e.description CONTAINS $term
                RETURN e
                LIMIT $limit
                """
                result = session.run(query, {
                    'term': search_term,
                    'limit': limit
                })
                
                return [dict(record['e']) for record in result]
        except Exception as e:
            logger.error(f"Error searching entities with term {search_term}: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the graph."""
        try:
            with self.get_session() as session:
                # Count entities
                entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()['count']
                
                # Count relationships
                rel_count = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as count").single()['count']
                
                # Entity types distribution
                type_dist = session.run("""
                    MATCH (e:Entity)
                    RETURN e.type as type, count(e) as count
                    ORDER BY count DESC
                """).data()
                
                # Average degree
                avg_degree = session.run("""
                    MATCH (e:Entity)
                    OPTIONAL MATCH (e)-[r:RELATES]-()
                    RETURN avg(count(r)) as avg_degree
                """).single()['avg_degree'] or 0
                
                return {
                    'total_entities': entity_count,
                    'total_relationships': rel_count,
                    'entity_types': type_dist,
                    'average_degree': float(avg_degree),
                    'density': rel_count / max(entity_count * (entity_count - 1), 1) if entity_count > 1 else 0
                }
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def validate_graph_integrity(self) -> Dict[str, Any]:
        """Validate graph integrity and find potential issues."""
        issues = {}
        
        try:
            with self.get_session() as session:
                # Find orphaned relationships (relationships without valid entities)
                orphaned_rels = session.run("""
                    MATCH ()-[r:RELATES]->()
                    WHERE NOT EXISTS((r)-[:HAS_SOURCE]->(:Entity)) 
                       OR NOT EXISTS((r)-[:HAS_TARGET]->(:Entity))
                    RETURN count(r) as count
                """).single()['count']
                
                # Find entities without relationships
                isolated_entities = session.run("""
                    MATCH (e:Entity)
                    WHERE NOT (e)-[:RELATES]-()
                    RETURN count(e) as count
                """).single()['count']
                
                # Find duplicate entities (same name, different cases)
                duplicates = session.run("""
                    MATCH (e1:Entity), (e2:Entity)
                    WHERE e1 <> e2 AND toLower(e1.name) = toLower(e2.name)
                    RETURN count(DISTINCT e1) as count
                """).single()['count']
                
                issues = {
                    'orphaned_relationships': orphaned_rels,
                    'isolated_entities': isolated_entities,
                    'potential_duplicates': duplicates,
                    'validation_passed': orphaned_rels == 0 and duplicates == 0
                }
                
        except Exception as e:
            logger.error(f"Error validating graph integrity: {e}")
            issues['validation_error'] = str(e)
        
        return issues
    
    def export_graph_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export all graph data for backup or analysis."""
        try:
            with self.get_session() as session:
                # Export entities
                entities = session.run("MATCH (e:Entity) RETURN e").data()
                entities = [dict(record['e']) for record in entities]
                
                # Export relationships
                relationships = session.run("""
                    MATCH (s:Entity)-[r:RELATES]->(t:Entity)
                    RETURN s.name as source, r, t.name as target
                """).data()
                
                rel_data = []
                for record in relationships:
                    rel_dict = dict(record['r'])
                    rel_dict['source_entity'] = record['source']
                    rel_dict['target_entity'] = record['target']
                    rel_data.append(rel_dict)
                
                return {
                    'entities': entities,
                    'relationships': rel_data
                }
        except Exception as e:
            logger.error(f"Error exporting graph data: {e}")
            return {'entities': [], 'relationships': []}
    
    def get_entity_neighborhood(self, entity_name: str, radius: int = 1) -> Dict[str, Any]:
        """Get the neighborhood of an entity (entities and relationships within radius)."""
        try:
            with self.get_session() as session:
                query = f"""
                MATCH (center:Entity {{name: $name}})
                MATCH path = (center)-[*1..{radius}]-(neighbor:Entity)
                WITH center, neighbor, path
                MATCH (start)-[rel]->(end)
                WHERE start IN nodes(path) AND end IN nodes(path)
                RETURN DISTINCT
                    center,
                    collect(DISTINCT neighbor) as neighbors,
                    collect(DISTINCT {{source: start.name, target: end.name, type: rel.type, description: rel.description}}) as relationships
                """
                
                result = session.run(query, {'name': entity_name})
                record = result.single()
                
                if record:
                    return {
                        'center_entity': dict(record['center']),
                        'neighbors': [dict(n) for n in record['neighbors']],
                        'relationships': record['relationships']
                    }
                return {}
        except Exception as e:
            logger.error(f"Error getting neighborhood for {entity_name}: {e}")
            return {}