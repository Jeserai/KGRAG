"""
Takes a user query and the output of the HybridRetriever (entities / relationships
/ context chunks) and constructs an augmented prompt suitable for passing to the
LLM.
"""
from __future__ import annotations

from typing import List
from textwrap import shorten

from src.query.retriever.retriever import RetrievalResult, Entity, Relationship, DocumentChunk  # noqa: E402


def _entities_to_text(entities: List[Entity], max_entities: int = 10) -> str:
    lines = []
    for ent in entities[:max_entities]:
        desc = ent.description or "N/A"
        lines.append(f"- {ent.name} ({ent.type}): {shorten(desc, width=120, placeholder='…')}")
    return "\n".join(lines)


def _relationships_to_text(rels: List[Relationship], max_rels: int = 15) -> str:
    lines = []
    for r in rels[:max_rels]:
        lines.append(f"- {r.source_entity} --[{r.relationship_type}]--> {r.target_entity}: {shorten(r.description or '', 100, '…')}")
    return "\n".join(lines)


def _chunks_to_text(chunks: List[DocumentChunk], max_chunks: int = 3, max_chars: int = 400) -> str:
    lines = []
    for c in chunks[:max_chunks]:
        snippet = shorten(c.text.replace("\n", " "), width=max_chars, placeholder="…")
        lines.append(f"[Chunk {c.id}] {snippet}")
    return "\n".join(lines)


def build_augmented_prompt(user_query: str, retrieval: RetrievalResult) -> str:
    """Return a single textual prompt that embeds KG context for the LLM."""
    parts: List[str] = []

    parts.append("## User Query\n" + user_query.strip())

    if retrieval.entities:
        parts.append("\n## Relevant Entities\n" + _entities_to_text(retrieval.entities))

    if retrieval.relationships:
        parts.append("\n## Key Relationships\n" + _relationships_to_text(retrieval.relationships))

    if retrieval.context_chunks:
        parts.append("\n## Supporting Passages\n" + _chunks_to_text(retrieval.context_chunks))

    parts.append("\n## Instruction\nUse the entities, relationships and passages above to answer the user query comprehensively.")

    return "\n".join(parts) 