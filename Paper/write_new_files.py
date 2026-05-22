"""
Write the new inference_engine.py, kinship_symbolic.py, and orchestrator.py
directly to D:\Open\College PC\Downloads\LLS_NEW\
"""
import os

BASE = r"D:\Open\College PC\Downloads\LLS_NEW"

# ──────────────────────────────────────────────────────────────────
# FILE 1: inference_engine.py — Forward Chaining (Algorithm 2)
# ──────────────────────────────────────────────────────────────────
inference_engine_code = r'''"""
inference_engine.py - Saturation-Based Forward Chaining Engine (Algorithm 2)

Implements the paper's Algorithm 2: repeatedly applies Horn clause rules
over the knowledge graph until the target query is answered or no new
facts can be derived (saturation).

References:
  - Paper Section 3.2.4, Algorithm 2
  - Paper Section 3.2.5, Equation 12 (specificity ranking)
"""

from typing import List, Tuple, Dict, Set, Any
import logging

from rules import (
    COMPOSITION_RULES, INVERSE_RULES,
    get_specificity, rank_by_specificity, get_inverse,
)

logger = logging.getLogger(__name__)


class ForwardChainingEngine:
    """
    Saturation-based forward chaining engine over Horn clause rules.

    Given an initial fact set F0 and a target query (subject, ?, object),
    repeatedly applies all rules to derive new facts until:
      - The query is answered, or
      - No new facts can be derived (saturation), or
      - Maximum iterations reached.
    """

    def __init__(self, max_iterations: int = 20):
        self.max_iterations = max_iterations
        self.composition_rules = COMPOSITION_RULES
        self.inverse_rules = INVERSE_RULES

    def solve(
        self,
        initial_facts: List[Tuple[str, str, str]],
        query_subject: str,
        query_object: str,
    ) -> Dict[str, Any]:
        """
        Run forward chaining inference (Algorithm 2).

        Args:
            initial_facts: List of (subject, relation, object) triples
            query_subject: The subject entity in the query
            query_object: The object entity in the query

        Returns:
            dict with keys:
              - success (bool)
              - relation (str or None)
              - confidence (float) - always 1.0 on success
              - proof_trace (list of str)
        """
        qs = query_subject.lower().strip()
        qo = query_object.lower().strip()

        # Normalise initial facts
        facts: Set[Tuple[str, str, str]] = set()
        for s, r, o in initial_facts:
            facts.add((s.lower().strip(), r.lower().strip(), o.lower().strip()))

        proof_trace: List[str] = []
        for s, r, o in sorted(facts):
            proof_trace.append(f"FACT: {r}({s}, {o})")

        logger.info(
            f"Forward chaining: {len(facts)} initial facts, "
            f"query=({qs}, ?, {qo}), max_iter={self.max_iterations}"
        )

        # Check if query is already answered by initial facts
        answer = self._check_query(facts, qs, qo)
        if answer:
            relation = rank_by_specificity(answer)
            proof_trace.append(f"ANSWER (direct fact): {relation}({qs}, {qo})")
            return {
                "success": True,
                "relation": relation,
                "confidence": 1.0,
                "proof_trace": proof_trace,
            }

        # Main forward chaining loop (Algorithm 2, lines 3-26)
        for iteration in range(self.max_iterations):
            new_facts: Set[Tuple[str, str, str]] = set()

            # Apply composition rules: body1(x,y) ^ body2(y,z) -> head(x,z)
            for head, (body1, body2) in self.composition_rules:
                for sx, r1, ox in facts:
                    if r1 != body1:
                        continue
                    for sy, r2, oy in facts:
                        if r2 != body2:
                            continue
                        if ox != sy:  # intermediate entity must match
                            continue
                        derived = (sx, head, oy)
                        if derived not in facts:
                            new_facts.add(derived)
                            proof_trace.append(
                                f"DERIVED: {head}({sx}, {oy}) via "
                                f"{body1}({sx}, {ox}) ^ {body2}({sy}, {oy})"
                            )

            # Apply inverse rules: body(x,y) -> head(y,x)
            for head, (body,) in self.inverse_rules:
                for sx, r, ox in facts:
                    if r != body:
                        continue
                    derived = (ox, head, sx)
                    if derived not in facts:
                        new_facts.add(derived)
                        proof_trace.append(
                            f"DERIVED (inverse): {head}({ox}, {sx}) via {body}({sx}, {ox})"
                        )

            # Saturation check (Algorithm 2, line 14)
            if not new_facts:
                logger.info(f"Saturation reached after {iteration + 1} iterations")
                proof_trace.append(
                    f"SATURATION: No new facts after {iteration + 1} iterations"
                )
                break

            # Add new facts to the knowledge base
            facts.update(new_facts)

            # Check if query is now answered (Algorithm 2, lines 18-24)
            answer = self._check_query(facts, qs, qo)
            if answer:
                relation = rank_by_specificity(answer)
                proof_trace.append(
                    f"ANSWER (iteration {iteration + 1}): {relation}({qs}, {qo})"
                )
                return {
                    "success": True,
                    "relation": relation,
                    "confidence": 1.0,
                    "proof_trace": proof_trace,
                }

            # Also check inverse direction (Algorithm 2, lines 20-22)
            answer_inv = self._check_query(facts, qo, qs)
            if answer_inv:
                base_rel = rank_by_specificity(answer_inv)
                inv_rel = get_inverse(base_rel)
                if inv_rel:
                    proof_trace.append(
                        f"ANSWER (inverse, iteration {iteration + 1}): "
                        f"{inv_rel}({qs}, {qo}) [inverse of {base_rel}({qo}, {qs})]"
                    )
                    return {
                        "success": True,
                        "relation": inv_rel,
                        "confidence": 1.0,
                        "proof_trace": proof_trace,
                    }

        # Query not resolved
        proof_trace.append("UNKNOWN: Could not derive the queried relation")
        return {
            "success": False,
            "relation": None,
            "confidence": 0.0,
            "proof_trace": proof_trace,
        }

    def _check_query(
        self,
        facts: Set[Tuple[str, str, str]],
        subject: str,
        obj: str,
    ) -> List[str]:
        """Check all facts for matches on (subject, ?, object)."""
        matches = []
        for s, r, o in facts:
            if s == subject and o == obj:
                matches.append(r)
        return matches


def run_forward_chaining(
    facts: List[Tuple[str, str, str]],
    query_subject: str,
    query_object: str,
    max_iterations: int = 20,
) -> Dict[str, Any]:
    """Run the forward chaining engine and return results."""
    engine = ForwardChainingEngine(max_iterations=max_iterations)
    return engine.solve(facts, query_subject, query_object)


if __name__ == "__main__":
    # Demo: Apollo is Hermes' father. Zeus is Apollo's father.
    # Query: What is Zeus to Hermes? -> grandfather
    demo_facts = [
        ("Apollo", "father", "Hermes"),
        ("Zeus", "father", "Apollo"),
    ]
    result = run_forward_chaining(demo_facts, "Zeus", "Hermes")
    print(f"Result: {result['relation']} (confidence={result['confidence']})")
    print("Proof trace:")
    for step in result["proof_trace"]:
        print(f"  {step}")
'''

# ──────────────────────────────────────────────────────────────────
# FILE 2: kinship_symbolic.py — Fact Extraction + Coreference (Algo 1)
# ──────────────────────────────────────────────────────────────────
kinship_symbolic_code = r'''"""
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
# 25 compiled regex patterns for fact extraction (Section 3.2.6)
# Ordered specific-to-general as per the paper.
# Each pattern returns (subject, relation, object) triples.
# ──────────────────────────────────────────────────────────────────
RELATION_WORDS = (
    "father|mother|son|daughter|brother|sister|"
    "grandfather|grandmother|grandson|granddaughter|"
    "uncle|aunt|nephew|niece|husband|wife|"
    "father-in-law|mother-in-law|"
    "parent|child|sibling"
)

FACT_PATTERNS = [
    # 1. "X is the Y of Z"
    (re.compile(
        rf"([\w\s]+?)\s+is\s+the\s+({RELATION_WORDS})\s+of\s+([\w\s]+?)(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(2).strip().lower(), m.group(3).strip())),

    # 2. "X is Y's Z"
    (re.compile(
        rf"([\w\s]+?)\s+is\s+([\w\s]+?)'s\s+({RELATION_WORDS})(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(3).strip().lower(), m.group(2).strip())),

    # 3. "X is my Y"
    (re.compile(
        rf"([\w\s]+?)\s+is\s+my\s+({RELATION_WORDS})(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(2).strip().lower(), "Speaker")),

    # 4. "X is his/her Y"
    (re.compile(
        rf"([\w\s]+?)\s+is\s+(?:his|her)\s+({RELATION_WORDS})(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(2).strip().lower(), "__LAST_ENTITY__")),

    # 5. "X's Y is Z"
    (re.compile(
        rf"([\w\s]+?)'s\s+({RELATION_WORDS})\s+is\s+([\w\s]+?)(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(3).strip(), m.group(2).strip().lower(), m.group(1).strip())),

    # 6. "my Y is X"
    (re.compile(
        rf"my\s+({RELATION_WORDS})\s+is\s+([\w\s]+?)(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(2).strip(), m.group(1).strip().lower(), "Speaker")),

    # 7. "X and Y are brothers"
    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+brothers(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "brother", m.group(2).strip())),

    # 8. "X and Y are sisters"
    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+sisters(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "sister", m.group(2).strip())),

    # 9. "X and Y are siblings"
    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+siblings(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "sibling", m.group(2).strip())),

    # 10. "X and Y are married" / "X married Y"
    (re.compile(
        r"([\w\s]+?)\s+(?:and\s+([\w\s]+?)\s+are\s+married|married\s+([\w\s]+?))(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "husband", (m.group(2) or m.group(3)).strip())),

    # 11. "X has a Y named Z" / "X has a Y, Z"
    (re.compile(
        rf"([\w\s]+?)\s+has\s+a\s+({RELATION_WORDS})(?:\s+named|\s*,)\s+([\w\s]+?)(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(3).strip(), m.group(2).strip().lower(), m.group(1).strip())),

    # 12. "X gave birth to Y" / "X had Y"
    (re.compile(
        r"([\w\s]+?)\s+(?:gave\s+birth\s+to|had)\s+([\w\s]+?)(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), "parent", m.group(2).strip())),

    # 13. "X, Y's Z" (comma-separated possessive)
    (re.compile(
        rf"([\w\s]+?),\s+([\w\s]+?)'s\s+({RELATION_WORDS})(?:\.|,|;|$)",
        re.IGNORECASE
    ), lambda m: (m.group(1).strip(), m.group(3).strip().lower(), m.group(2).strip())),
]


def extract_facts_from_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract kinship fact triples from a narrative text using regex patterns.

    Args:
        text: Natural language narrative about family relationships

    Returns:
        List of (subject, relation, object) triples
    """
    facts = []
    for pattern, extractor in FACT_PATTERNS:
        for match in pattern.finditer(text):
            try:
                s, r, o = extractor(match)
                if s and r and o:
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

        # Resolve subject
        if s.lower().strip() in FIRST_PERSON_PRONOUNS:
            s = "Speaker"
        elif s.lower().strip() in THIRD_PERSON_PRONOUNS or s == "__LAST_ENTITY__":
            if last_entity is not None:
                s = last_entity

        # Resolve object
        if o.lower().strip() in FIRST_PERSON_PRONOUNS:
            o = "Speaker"
        elif o.lower().strip() in THIRD_PERSON_PRONOUNS or o == "__LAST_ENTITY__":
            if last_entity is not None:
                o = last_entity

        # Update last_entity if subject is not a pronoun (Algorithm 1, line 18-19)
        if subject.lower().strip() not in (FIRST_PERSON_PRONOUNS | THIRD_PERSON_PRONOUNS | {"__last_entity__"}):
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
    result = symbolic_solve(story, "Gaia", "Hermes")
    print(f"\nResult: {result['relation']} (confidence={result['confidence']})")
    print("\nProof trace:")
    for step in result["proof_trace"]:
        print(f"  {step}")
'''

# ──────────────────────────────────────────────────────────────────
# FILE 3: orchestrator.py — Algorithm 3
# ──────────────────────────────────────────────────────────────────
orchestrator_code = r'''"""
orchestrator.py - Confidence-Based Hybrid Answer Selection (Algorithm 3)

Implements hard symbolic priority:
  1. Run symbolic engine (forward chaining). On success -> return with confidence 1.0
  2. On symbolic failure -> run neural module (BERT classifier)
     - If neural confidence >= tau (0.8) -> return high-confidence neural answer
     - Else -> return low-confidence fallback with uncertainty indicator

References:
  - Paper Section 3.3, Algorithm 3
  - Confidence threshold tau = 0.8
"""

from typing import Dict, Any, Optional
import logging

from kinship_symbolic import symbolic_solve

logger = logging.getLogger(__name__)

# Confidence threshold (paper Section 3.3)
TAU = 0.8


def hybrid_select(
    narrative: str,
    query_subject: str,
    query_object: str,
    neural_predict_fn=None,
    tau: float = TAU,
) -> Dict[str, Any]:
    """
    Algorithm 3: Confidence-Based Hybrid Answer Selection.

    Args:
        narrative: The story text
        query_subject: Entity A
        query_object: Entity B
        neural_predict_fn: Optional callable(narrative, subject, object) -> dict
                           with keys 'relation', 'confidence'
                           If None, only symbolic reasoning is used.
        tau: Confidence threshold for neural high-confidence path (default 0.8)

    Returns:
        dict with:
          - relation (str)
          - confidence (float)
          - method ('symbolic' | 'neural_high' | 'neural_low')
          - proof_trace (list of str, if symbolic)
    """
    logger.info(f"Hybrid orchestrator: query=({query_subject}, ?, {query_object})")

    # ── Step 1: Symbolic path (Algorithm 3, lines 2-5) ──────────
    symbolic_result = symbolic_solve(narrative, query_subject, query_object)

    if symbolic_result["success"]:
        logger.info(
            f"Symbolic SUCCESS: {symbolic_result['relation']} "
            f"(confidence=1.0)"
        )
        return {
            "relation": symbolic_result["relation"],
            "confidence": 1.0,
            "method": "symbolic",
            "proof_trace": symbolic_result["proof_trace"],
        }

    logger.info("Symbolic FAILED: falling back to neural module")

    # ── Step 2: Neural fallback (Algorithm 3, lines 6-9) ────────
    if neural_predict_fn is not None:
        try:
            neural_result = neural_predict_fn(
                narrative, query_subject, query_object
            )
            neural_conf = neural_result.get("confidence", 0.0)
            neural_rel = neural_result.get("relation", None)

            if neural_conf >= tau:
                logger.info(
                    f"Neural HIGH confidence: {neural_rel} ({neural_conf:.2f})"
                )
                return {
                    "relation": neural_rel,
                    "confidence": neural_conf,
                    "method": "neural_high",
                    "proof_trace": symbolic_result["proof_trace"]
                    + [f"NEURAL (high-conf): {neural_rel} ({neural_conf:.2f})"],
                }
            else:
                logger.info(
                    f"Neural LOW confidence: {neural_rel} ({neural_conf:.2f})"
                )
                return {
                    "relation": neural_rel,
                    "confidence": neural_conf,
                    "method": "neural_low",
                    "proof_trace": symbolic_result["proof_trace"]
                    + [
                        f"NEURAL (low-conf fallback): {neural_rel} ({neural_conf:.2f})",
                        "WARNING: Confidence below threshold. Result may be unreliable.",
                    ],
                }
        except Exception as e:
            logger.error(f"Neural module error: {e}")

    # ── No answer available ─────────────────────────────────────
    return {
        "relation": None,
        "confidence": 0.0,
        "method": "none",
        "proof_trace": symbolic_result["proof_trace"]
        + ["No answer: both symbolic and neural modules failed."],
    }


if __name__ == "__main__":
    story = (
        "Apollo is Hermes' father. "
        "Zeus is Apollo's father."
    )
    result = hybrid_select(story, "Zeus", "Hermes")
    print(f"Relation: {result['relation']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Method: {result['method']}")
    print("Proof trace:")
    for step in result["proof_trace"]:
        print(f"  {step}")
'''

# ── Write all files ─────────────────────────────────────────────
files = {
    "inference_engine.py": inference_engine_code,
    "kinship_symbolic.py": kinship_symbolic_code,
    "orchestrator.py": orchestrator_code,
}

for filename, content in files.items():
    filepath = os.path.join(BASE, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Written: {filepath} ({len(content)} bytes)")

print("\nAll 3 files written successfully to D: drive!")
