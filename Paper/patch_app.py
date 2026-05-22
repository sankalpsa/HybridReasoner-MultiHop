"""Patch app.py to defer BERT loading so the server starts immediately."""
import os
BASE = r"D:\Open\College PC\Downloads\LLS_NEW"

app_code = r'''"""
app.py - Hybrid Reasoner Flask Web Server

Main entry point. Implements the full pipeline from the paper:
  1. Receives narrative + query via /reason endpoint
  2. Routes through orchestrator (Algorithm 3):
     - Symbolic first (forward chaining, confidence 1.0)
     - Neural fallback (BERT classifier, threshold tau=0.8)
  3. Returns relation, confidence, method, and proof trace
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Lazy neural module loading ──────────────────────────────────
_neural_fn = None
_neural_attempted = False

def get_neural_fn():
    """Lazily load the BERT classifier on first neural fallback."""
    global _neural_fn, _neural_attempted
    if _neural_attempted:
        return _neural_fn
    _neural_attempted = True
    try:
        from bert_classifier import get_classifier
        clf = get_classifier()
        _neural_fn = clf.neural_predict_fn
        logger.info("BERT neural module loaded")
    except Exception as e:
        logger.warning(f"BERT not available ({e}). Symbolic-only mode.")
        _neural_fn = None
    return _neural_fn


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reason", methods=["POST"])
def reason():
    """Hybrid reasoning endpoint (Algorithm 3)."""
    try:
        data = request.json or {}
        context = data.get("context", "").strip()
        question = data.get("question", "").strip()
        entity_a = data.get("entity_a", "").strip()
        entity_b = data.get("entity_b", "").strip()

        if not question and not context:
            return jsonify({"error": "Provide a context or question"}), 400

        # Extract entities from question if not provided
        if not entity_a or not entity_b:
            match = re.search(r"between\s+(\w+)\s+and\s+(\w+)", question, re.I)
            if match:
                entity_a, entity_b = match.group(1), match.group(2)
            else:
                words = re.findall(r"\b[A-Z][a-z]+\b", question + " " + context)
                if len(words) >= 2:
                    entity_a, entity_b = words[0], words[1]
                else:
                    return jsonify({"error": "Could not extract entities. Provide entity_a and entity_b."}), 400

        narrative = context if context else question
        logger.info(f"Query: {entity_a} -> {entity_b}")

        from orchestrator import hybrid_select
        result = hybrid_select(
            narrative=narrative,
            query_subject=entity_a,
            query_object=entity_b,
            neural_predict_fn=get_neural_fn(),
        )

        return jsonify({
            "question": question,
            "entities": {"entity_a": entity_a, "entity_b": entity_b},
            "answer": {
                "relation": result["relation"],
                "confidence": result["confidence"],
                "method": result["method"],
                "proof_trace": result.get("proof_trace", []),
            },
        })
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "mode": "hybrid" if _neural_fn else "symbolic-only",
    })


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("HYBRID REASONER - Starting...")
    logger.info("BERT loading deferred to first neural fallback")
    logger.info("=" * 50)
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
'''

with open(os.path.join(BASE, "app.py"), "w", encoding="utf-8") as f:
    f.write(app_code)
print(f"Written app.py ({len(app_code)} bytes)")
