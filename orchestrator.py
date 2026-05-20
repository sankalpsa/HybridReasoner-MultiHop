"""
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

    # Early check for identity / self-relationship
    qs = query_subject.lower().strip()
    qo = query_object.lower().strip()
    if qs == qo and qs and qs != "?":
        logger.info(f"Identity query detected: {query_subject} is the same person as {query_object}")
        return {
            "relation": "self",
            "confidence": 1.0,
            "method": "symbolic",
            "proof_trace": [f"IDENTITY: {query_subject} is the same person as {query_object}"],
        }

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
