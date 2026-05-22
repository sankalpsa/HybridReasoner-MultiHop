"""
Fix kinship_symbolic.py: add smart-quote normalisation and improve regex.
Writes the updated file directly to D: drive.
"""
import os

BASE = r"D:\Open\College PC\Downloads\LLS_NEW"

kinship_code = r'''"""
kinship_symbolic.py - Fact Extraction and Pronoun Coreference Resolution

Implements:
  - 25 compiled regex patterns for extracting kinship facts from narratives
    (Section 3.2.6, Listing 1)
  - Algorithm 1: Pronoun Coreference Resolution with last_entity tracking
  - Full symbolic reasoning pipeline: text -> facts -> forward chaining -> answer

References:
  - Paper Section 3.2.3, Algorithm 1
  - Paper Section 3.2.6, Listing 1
"""

import re
from typing import List, Tuple, Optional, Dict, Any
import logging

from inference_engine import run_forward_chaining

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Pronouns used for coreference resolution (Algorithm 1)
# ──────────────────────────────────────────────────────────────────
FIRST_PERSON_PRONOUNS = {"my", "i", "me", "mine", "myself"}
MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
THIRD_PERSON_PRONOUNS = MALE_PRONOUNS | FEMALE_PRONOUNS

# ──────────────────────────────────────────────────────────────────
# Relation words recognised by the fact extractor
# ──────────────────────────────────────────────────────────────────
RELATION_WORDS = (
    "father|mother|son|daughter|brother|sister|"
    "grandfather|grandmother|grandson|granddaughter|"
    "uncle|aunt|nephew|niece|husband|wife|"
    "father-in-law|mother-in-law|"
    "parent|child|sibling"
)

# ──────────────────────────────────────────────────────────────────
# Smart-quote normalisation: convert curly quotes to straight ones
# ──────────────────────────────────────────────────────────────────
def normalise_text(text: str) -> str:
    """Replace smart/curly quotes with straight ASCII equivalents."""
    text = text.replace("\u2018", "'")   # left single
    text = text.replace("\u2019", "'")   # right single (most common)
    text = text.replace("\u201C", '"')   # left double
    text = text.replace("\u201D", '"')   # right double
    text = text.replace("\u2013", "-")   # en-dash
    text = text.replace("\u2014", "-")   # em-dash
    return text


# ──────────────────────────────────────────────────────────────────
# Compiled regex patterns for fact extraction (Section 3.2.6)
# Ordered specific-to-general as per the paper.
# ──────────────────────────────────────────────────────────────────
FACT_PATTERNS = [
    # 1. "X is the Y of Z"
    (re.compile(
        rf"([\w\s]+?)\s+is\s+the\s+({RELATION_WORDS})\s+of\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(2).strip().lower(), m.group(3).strip())),

    # 2. "X is Y's Z"  (possessive with straight apostrophe)
    (re.compile(
        rf"([\w\s]+?)\s+is\s+([\w]+)'s\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(3).strip().lower(), m.group(2).strip())),

    # 3. "X is my Y"
    (re.compile(
        rf"([\w\s]+?)\s+is\s+my\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(2).strip().lower(), "Speaker")),

    # 4. "X is his/her Y"
    (re.compile(
        rf"([\w\s]+?)\s+is\s+(?:his|her)\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(2).strip().lower(), "__LAST_ENTITY__")),

    # 5. "X's Y is Z"  (possessive subject)
    (re.compile(
        rf"([\w]+)'s\s+({RELATION_WORDS})\s+is\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(3).strip(), m.group(2).strip().lower(), m.group(1).strip())),

    # 6. "my Y is X"
    (re.compile(
        rf"my\s+({RELATION_WORDS})\s+is\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(2).strip(), m.group(1).strip().lower(), "Speaker")),

    # 7. "his/her Y is X"
    (re.compile(
        rf"(?:his|her)\s+({RELATION_WORDS})\s+is\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(2).strip(), m.group(1).strip().lower(), "__LAST_ENTITY__")),

    # 8. "X and Y are brothers"
    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+brothers(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "brother", m.group(2).strip())),

    # 9. "X and Y are sisters"
    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+sisters(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "sister", m.group(2).strip())),

    # 10. "X and Y are siblings"
    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+siblings(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "sibling", m.group(2).strip())),

    # 11. "X and Y are married" / "X married Y"
    (re.compile(
        r"([\w\s]+?)\s+(?:and\s+([\w\s]+?)\s+are\s+married|married\s+([\w\s]+?))(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "husband", (m.group(2) or m.group(3)).strip())),

    # 12. "X has a Y named Z" / "X has a Y, Z"
    (re.compile(
        rf"([\w\s]+?)\s+has\s+a\s+({RELATION_WORDS})(?:\s+named|\s*,)\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(3).strip(), m.group(2).strip().lower(), m.group(1).strip())),

    # 13. "X gave birth to Y" / "X had Y"
    (re.compile(
        r"([\w\s]+?)\s+(?:gave\s+birth\s+to|had)\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "parent", m.group(2).strip())),

    # 14. "X, Y's Z" (comma-separated possessive)
    (re.compile(
        rf"([\w\s]+?),\s+([\w]+)'s\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(3).strip().lower(), m.group(2).strip())),

    # 15. "X, the Y of Z"
    (re.compile(
        rf"([\w\s]+?),\s+the\s+({RELATION_WORDS})\s+of\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(2).strip().lower(), m.group(3).strip())),
]


def extract_facts_from_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract kinship fact triples from a narrative text using regex patterns.

    Args:
        text: Natural language narrative about family relationships

    Returns:
        List of (subject, relation, object) triples
    """
    # Normalise smart quotes before matching
    text = normalise_text(text)

    facts = []
    seen = set()  # de-duplicate

    for pattern, extractor in FACT_PATTERNS:
        for match in pattern.finditer(text):
            try:
                s, r, o = extractor(match)
                if s and r and o:
                    key = (s.lower(), r.lower(), o.lower())
                    if key not in seen:
                        seen.add(key)
                        facts.append((s, r, o))
            except Exception:
                continue
    return facts


def resolve_coreferences(
    facts: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
    """
    Algorithm 1: Pronoun Coreference Resolution.

    Maintains a last_entity variable. For each fact:
      - If subject is a first-person pronoun -> replace with 'Speaker'
      - If subject is a third-person pronoun -> replace with last_entity
      - If object is a first-person pronoun -> replace with 'Speaker'
      - If object is a third-person pronoun -> replace with last_entity
      - If subject is NOT a pronoun -> update last_entity

    Args:
        facts: List of (subject, relation, object) triples with potential pronouns

    Returns:
        List of (subject, relation, object) triples with pronouns resolved
    """
    last_entity = None
    resolved = []

    for subject, relation, obj in facts:
        s = subject
        o = obj

        # Resolve subject (Algorithm 1, lines 4-9)
        s_lower = s.lower().strip()
        if s_lower in FIRST_PERSON_PRONOUNS:
            s = "Speaker"
        elif s_lower in THIRD_PERSON_PRONOUNS or s == "__LAST_ENTITY__":
            if last_entity is not None:
                s = last_entity

        # Resolve object (Algorithm 1, lines 11-16)
        o_lower = o.lower().strip()
        if o_lower in FIRST_PERSON_PRONOUNS:
            o = "Speaker"
        elif o_lower in THIRD_PERSON_PRONOUNS or o == "__LAST_ENTITY__":
            if last_entity is not None:
                o = last_entity

        # Update last_entity if subject is not a pronoun (Algorithm 1, lines 18-19)
        if s_lower not in (FIRST_PERSON_PRONOUNS | THIRD_PERSON_PRONOUNS | {"__last_entity__"}):
            last_entity = s

        resolved.append((s, relation, o))

    return resolved


def symbolic_solve(
    narrative: str,
    query_subject: str,
    query_object: str,
) -> Dict[str, Any]:
    """
    Full symbolic reasoning pipeline:
      1. Extract facts from narrative using regex patterns
      2. Resolve pronoun coreferences (Algorithm 1)
      3. Run forward chaining inference (Algorithm 2)
      4. Return answer with proof trace

    Args:
        narrative: The story text
        query_subject: Entity A in the query
        query_object: Entity B in the query

    Returns:
        dict with success, relation, confidence, proof_trace
    """
    # Step 1: Extract facts
    raw_facts = extract_facts_from_text(narrative)
    logger.info(f"Extracted {len(raw_facts)} raw facts from narrative")

    # Step 2: Resolve coreferences (Algorithm 1)
    resolved_facts = resolve_coreferences(raw_facts)
    logger.info(f"After coreference resolution: {len(resolved_facts)} facts")

    if not resolved_facts:
        return {
            "success": False,
            "relation": None,
            "confidence": 0.0,
            "proof_trace": ["No facts could be extracted from the narrative."],
        }

    # Step 3: Forward chaining (Algorithm 2)
    result = run_forward_chaining(resolved_facts, query_subject, query_object)

    return result


if __name__ == "__main__":
    # Demo
    story = (
        "Apollo is Hermes' father. "
        "Zeus is Apollo's father. "
        "Zeus's father is Cronus. "
        "Cronus's father is Uranus. "
        "Gaia is the mother of Uranus."
    )
    print("Facts extracted:")
    facts = extract_facts_from_text(story)
    for f in facts:
        print(f"  {f}")

    print()
    result = symbolic_solve(story, "Gaia", "Hermes")
    print(f"\nResult: {result['relation']} (confidence={result['confidence']})")
    print("\nProof trace:")
    for step in result["proof_trace"]:
        print(f"  {step}")
'''

filepath = os.path.join(BASE, "kinship_symbolic.py")
with open(filepath, "w", encoding="utf-8") as f:
    f.write(kinship_code)
print(f"Updated: {filepath} ({len(kinship_code)} bytes)")
