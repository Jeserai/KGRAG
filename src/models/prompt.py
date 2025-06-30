from typing import List, Optional

# Delimiters for parsing structured output from the LLM
TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# Default entity types to extract
DEFAULT_ENTITY_TYPES: List[str] = [
    "PERSON", "ORGANIZATION", "LOCATION", "EVENT",
    "CONCEPT", "TECHNOLOGY", "PRODUCT", "DATE"
]

def get_extraction_prompt(
    text: str, 
    entity_types: Optional[List[str]] = None,
    max_entities: int = 15
) -> str:
    """
    Generates a prompt for entity and relationship extraction.
    
    Args:
        text: The input text to process.
        entity_types: A list of entity types to extract.
        max_entities: The maximum number of entities to extract.
        
    Returns:
        A formatted prompt string.
    """
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    
    entity_types_str = ", ".join(entity_types)
    
    return f"""You are an AI assistant that helps extract entities and relationships from text for knowledge graph construction.

## Task
Extract entities and relationships from the provided text. Focus on the most important and relevant entities that would be valuable in a knowledge graph.

## Entity Types
Extract entities of these types: {entity_types_str}

## Output Format
For each entity, provide:
(entity_name{TUPLE_DELIMITER}entity_type{TUPLE_DELIMITER}entity_description){RECORD_DELIMITER}

For each relationship, provide:
(source_entity{TUPLE_DELIMITER}relationship_type{TUPLE_DELIMITER}target_entity{TUPLE_DELIMITER}relationship_description){RECORD_DELIMITER}

When you are finished, output the completion token: {COMPLETION_DELIMITER}

## Guidelines
1. Entity names should be specific and descriptive.
2. Descriptions should be concise but informative.
3. Relationships should capture meaningful connections.
4. Use consistent naming (e.g., "John Smith" not "John" and "Smith" separately).
5. Focus on entities that are central to the text.
6. Adhere strictly to the output format.
7. Do not exceed a maximum of {max_entities} entities per text.

## Text to Process:
{text}

## Extracted Entities and Relationships:
"""