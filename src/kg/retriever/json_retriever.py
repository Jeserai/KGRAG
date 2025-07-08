class JSONKnowledgeGraph:
    def __init__(self, json_file_path):
        self.data = self.load_json(json_file_path)
        self.entities = self.data.get('entities', [])
        self.relationships = self.data.get('relationships', [])
        self.entity_index = self.build_entity_index()
        
    def build_entity_index(self):
        """Create fast lookup index for entities"""
        index = {}
        for entity in self.entities:
            # Index by name and type
            name = entity.get('name', '').lower()
            entity_type = entity.get('type', '')
            index[name] = entity
            # Add type-based indexing
            if entity_type not in index:
                index[entity_type] = []
            index[entity_type].append(entity)
        return index
    
    def find_entities(self, query_text):
        """Find entities mentioned in query"""
        query_lower = query_text.lower()
        found_entities = []
        for name, entity in self.entity_index.items():
            if isinstance(entity, dict) and name in query_lower:
                found_entities.append(entity)
        return found_entities
    
    def get_related_entities(self, entity_name, max_hops=2):
        """Get entities connected through relationships"""
        related = set()
        current_entities = {entity_name}
        
        for hop in range(max_hops):
            next_entities = set()
            for rel in self.relationships:
                source = rel.get('source', '')
                target = rel.get('target', '')
                
                if source in current_entities:
                    next_entities.add(target)
                    related.add(target)
                elif target in current_entities:
                    next_entities.add(source)
                    related.add(source)
            
            current_entities = next_entities
            if not next_entities:
                break
        
        return list(related)

class JSONRetriever:
    def __init__(self, knowledge_graph, documents):
        self.kg = knowledge_graph
        self.documents = documents  # Your processed documents
        
    def retrieve_context(self, question, max_entities=10, max_hops=2):
        """Retrieve relevant context for a question"""
        # 1. Find entities mentioned in question
        query_entities = self.kg.find_entities(question)
        
        # 2. Expand through relationships
        expanded_entities = set()
        for entity in query_entities[:5]:  # Limit to avoid explosion
            related = self.kg.get_related_entities(
                entity.get('name', ''), max_hops=max_hops
            )
            expanded_entities.update(related)
        
        # 3. Retrieve context about these entities
        context_chunks = []
        for entity_name in list(expanded_entities)[:max_entities]:
            entity_context = self.find_entity_context(entity_name)
            context_chunks.extend(entity_context)
        
        return context_chunks
    
    def find_entity_context(self, entity_name):
        """Find document chunks mentioning this entity"""
        context = []
        for doc in self.documents:
            if entity_name.lower() in doc.get('text', '').lower():
                context.append({
                    'text': doc.get('text', ''),
                    'source': doc.get('source', ''),
                    'entity': entity_name
                })
        return context