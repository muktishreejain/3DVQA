"""Extract target object, relation, and attribute from VQA questions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

RELATIONS = {
    "behind",
    "in front of",
    "in_front_of",
    "left of",
    "left_of",
    "right of",
    "right_of",
    "near",
    "on",
    "under",
    "above",
    "next to",
    "next_to",
}

ATTRIBUTES = {
    "color",
    "size",
    "shape",
    "material",
    "type",
    "height",
    "width",
    "length",
}


def parse_query(question: str) -> Dict[str, Optional[str]]:
    """
    Parse question into structured query fields.

    Example:
        "What is the color of the object behind the red cylinder?"
        -> target=red cylinder, relation=behind, attribute=color
    """
    q = question.strip().lower()
    relation = _extract_relation(q)
    attribute = _extract_attribute(q)
    target = _extract_target(q, relation)
    return {
        "target": target,
        "relation": relation,
        "attribute": attribute,
        "raw_question": question.strip(),
    }


def _extract_relation(q: str) -> Optional[str]:
    for rel in sorted(RELATIONS, key=len, reverse=True):
        pattern = rel.replace("_", " ")
        if pattern in q:
            return pattern.replace(" ", "_")
    return None


def _extract_attribute(q: str) -> Optional[str]:
    for attr in ATTRIBUTES:
        if attr in q:
            return attr
    m = re.search(r"what (?:is|are) the (\w+)", q)
    if m and m.group(1) not in ("object", "objects"):
        return m.group(1)
    return None


def _extract_target(q: str, relation: Optional[str]) -> Optional[str]:
    if relation:
        rel_text = relation.replace("_", " ")
        parts = re.split(rf"\b{re.escape(rel_text)}\b", q, maxsplit=1)
        if len(parts) > 1:
            tail = parts[-1].strip()
            tail = re.sub(r"^(the|a|an)\s+", "", tail)
            tail = re.sub(r"\?.*$", "", tail).strip()
            if tail:
                return tail
    m = re.search(r"(?:the|a|an)\s+([\w\s-]+?)(?:\?|$)", q)
    if m:
        phrase = m.group(1).strip()
        for stop in ("object", "thing", "item"):
            phrase = re.sub(rf"\b{stop}\b", "", phrase).strip()
        if phrase:
            return phrase
    return None


def detection_phrases(query: Dict[str, Optional[str]]) -> List[str]:
    """Build Grounding DINO text prompts from parsed query."""
    phrases: List[str] = []
    if query.get("target"):
        phrases.append(query["target"])
    if query.get("relation"):
        phrases.append(query["relation"].replace("_", " "))
    phrases.append("object")
    seen = set()
    out = []
    for p in phrases:
        key = p.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out
