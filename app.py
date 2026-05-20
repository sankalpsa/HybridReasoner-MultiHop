"""
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

@app.after_request
def add_header(r):
    """Disable caching for development."""
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

# ── Lazy neural module loading ──────────────────────────────────
_neural_fn = None
_neural_attempted = False

def get_neural_fn():
    """Lazily load neural module. Tries Kaggle BERT classifier first, MLP second."""
    global _neural_fn, _neural_attempted
    if _neural_attempted:
        return _neural_fn
    _neural_attempted = True

    # 1) Try the Kaggle BERT model first (Highest accuracy, matches paper Sec 3.1)
    try:
        from kaggle_inference import get_kaggle_fn
        fn = get_kaggle_fn()
        if fn is not None:
            _neural_fn = fn
            logger.info("✅ Kaggle BERT classifier loaded as primary neural fallback")
            return _neural_fn
    except Exception as e:
        logger.warning(f"Kaggle BERT classifier not available ({e}), trying MLP...")

    # 2) Fallback to MLP if Kaggle model fails
    try:
        from trained_neural import get_trained_neural
        module = get_trained_neural()
        if module.loaded:
            _neural_fn = module.neural_predict_fn
            logger.info("✅ Trained CLUTRR MLP loaded as backup neural fallback")
            return _neural_fn
    except Exception as e:
        logger.warning(f"Trained MLP not available ({e}). Symbolic-only mode.")
        _neural_fn = None

    return _neural_fn

# ── Universal Query Entity Resolver ────────────────────────────────
FIRST_PERSON_PRONOUNS = {"my", "i", "me", "mine", "myself"}

STOP_WORDS_LOWER = {
    "what", "how", "who", "whom", "whose", "which", "whatever", "is", "are", "do", "does", "can", "could", "the", "a", "an",
    "tell", "related", "relationship", "relation", "family", "was", "were", "been", "being",
    "have", "has", "had", "named", "called", "this", "that", "these", "those",
    "in", "on", "at", "for", "with", "about", "by", "from", "up", "out", "into", "of", "to", "between", "and", "or",
    "they", "their", "them", "theirs", "themselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "i", "me", "my", "mine", "myself", "we", "our", "us", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "so", "then", "there", "here", "just", "very", "too", "also", 
    "nice", "meet", "hi", "hello", "hey", "please", "would", "should", "will", "shall", "be", "am", "go", "went", "gone",
    "see", "saw", "seen", "say", "said", "thanks", "thank", "kindly", "determine", "find", "show", "give", "display", 
    "list", "get", "ask", "query", "question", "whats", "what's", "hows", "how's", "whos", "who's", "relation", 
    "relationship", "related", "between", "among"
}

RELATION_WORDS_LOWER = {
    "father", "mother", "son", "daughter", "brother", "sister",
    "grandfather", "grandmother", "grandson", "granddaughter",
    "uncle", "aunt", "nephew", "niece", "husband", "wife",
    "father-in-law", "mother-in-law", "son-in-law", "daughter-in-law",
    "parent", "child", "sibling", "cousin",
    "fathers", "mothers", "sons", "daughters", "brothers", "sisters",
    "grandfathers", "grandmothers", "grandsons", "granddaughters",
    "uncles", "aunts", "nephews", "nieces", "husbands", "wives",
    "parents", "children", "siblings", "cousins",
    "father-in-laws", "mother-in-laws", "son-in-laws", "daughter-in-laws",
    "brother-in-law", "sister-in-law", "brother-in-laws", "sister-in-laws",
    "dad", "dads", "daddy", "daddies", "papa", "papas", "pop", "pops",
    "mom", "moms", "mommy", "mommies", "mama", "mamas", "mum", "mums",
    "hubby", "hubbies", "wifey", "wifeys", "bro", "bros",
    "sis", "sises", "grandpa", "grandpas", "granddad", "granddads", 
    "gramps", "grandma", "grandmas", "granny", "grannies", "kid", "kids", "spouse", "spouses",
    "aunty", "aunties"
}

def resolve_query_entities(question: str, context: str) -> tuple:
    """
    Extract and resolve the query subject and object from a natural language question
    using the resolved entities and speaker identity from the context.
    """
    try:
        from kinship_symbolic import extract_facts_from_text, resolve_coreferences
    except ImportError:
        return "?", "?"

    # 1. Extract facts and resolved entities from context
    raw_facts = extract_facts_from_text(context)
    resolved_facts = resolve_coreferences(raw_facts)
    
    known_entities = set()
    speaker_name = None
    
    for s, r, o in resolved_facts:
        if s.lower() not in ("speaker", "__last_entity__") and s.lower() not in FIRST_PERSON_PRONOUNS:
            known_entities.add(s)
        if o.lower() not in ("speaker", "__last_entity__") and o.lower() not in FIRST_PERSON_PRONOUNS:
            known_entities.add(o)
            
    # Check if a speaker identity was resolved
    speaker_name = getattr(resolved_facts, 'speaker_name', None)
    if not speaker_name:
        for s, r, o in raw_facts:
            if r == "identity":
                if s == "Speaker":
                    speaker_name = o
                elif o == "Speaker":
                    speaker_name = s


    # Normalize question
    question_clean = re.sub(r"[^\w\s'-]", " ", question)
    words = question_clean.split()
    
    extracted_entities = [] # list of (value, category)
    
    i = 0
    while i < len(words):
        w = words[i]
        w_clean = re.sub(r"'s$", "", w, flags=re.IGNORECASE).strip()
        w_lower = w_clean.lower()
        
        # 1. Multi-word known entities first
        matched_len = 0
        matched_entity = None
        for ke in sorted(known_entities, key=len, reverse=True):
            ke_words = ke.lower().split()
            if words[i:i+len(ke_words)] and [x.lower() for x in words[i:i+len(ke_words)]] == ke_words:
                matched_len = len(ke_words)
                matched_entity = ke
                break
                
        if matched_entity:
            extracted_entities.append((matched_entity, 'known'))
            i += matched_len
            continue
            
        # 2. Check single-word known entities
        matched_single = None
        for ke in known_entities:
            if ke.lower() == w_lower:
                matched_single = ke
                break
        if matched_single:
            extracted_entities.append((matched_single, 'known'))
            i += 1
            continue
            
        # 3. Check first-person pronoun
        if w_lower in FIRST_PERSON_PRONOUNS:
            resolved_val = speaker_name if speaker_name else "Speaker"
            extracted_entities.append((resolved_val, 'pronoun'))
            i += 1
            continue
            
        # 4. Check stop words or relationship words
        if w_lower in STOP_WORDS_LOWER or w_lower in RELATION_WORDS_LOWER:
            i += 1
            continue
            
        # 5. Treat any other word (capitalized or not) of len > 1 as entity candidate
        if len(w_clean) > 1:
            extracted_entities.append((w_clean.title(), 'other'))
            i += 1
            continue
            
        i += 1
        
    known_only = [val for val, cat in extracted_entities if cat == 'known']
    pronoun_only = [val for val, cat in extracted_entities if cat == 'pronoun']
    
    # Apply prioritized matching:
    # 1. If we have at least 2 known entities, use only known entities (prioritized)
    if len(known_only) >= 2:
        final_candidates = known_only
    # 2. If known + pronouns give at least 2 candidates, use those (e.g. Sadhana and me)
    elif len(known_only) + len(pronoun_only) >= 2:
        final_candidates = [val for val, cat in extracted_entities if cat in ('known', 'pronoun')]
    # 3. Fallback to all candidates
    else:
        final_candidates = [val for val, cat in extracted_entities]
        
    if len(final_candidates) >= 2:
        return final_candidates[0], final_candidates[1]
    elif len(final_candidates) == 1:
        ent = final_candidates[0]
        # If the question contains reflexive pronouns, query is reflexive (self)
        reflexive_pronouns = {"himself", "herself", "myself", "self", "itself", "yourself"}
        words_lower = {w.lower() for w in words}
        if words_lower & reflexive_pronouns:
            return ent, ent
        # Fallback to other entities from context if available
        other_entities = [ke for ke in known_entities if ke.lower() != ent.lower()]
        if other_entities:
            return ent, other_entities[0]
        else:
            return ent, ent
            
    return "?", "?"


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

        # Resolve/extract entities
        first_person_pronouns = {"my", "i", "me", "mine", "myself"}
        needs_resolution = (
            not entity_a or not entity_b or 
            entity_a.lower() in first_person_pronouns or 
            entity_b.lower() in first_person_pronouns or
            (entity_a and not entity_a[0].isupper()) or
            (entity_b and not entity_b[0].isupper())
        )
        
        if needs_resolution:
            resolved_a, resolved_b = resolve_query_entities(question or context, context)
            if resolved_a != "?" and resolved_b != "?":
                entity_a, entity_b = resolved_a, resolved_b
            else:
                # 1. Try explicit 'between A and B'
                match = re.search(r"between\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)", question, re.IGNORECASE)
                if match:
                    entity_a, entity_b = match.group(1).title(), match.group(2).title()
                else:
                    # 2. Extract capitalized words ignoring stop words
                    potential_names = [w for w in re.findall(r"\b[A-Z][a-z]+\b", question) if w.lower() not in STOP_WORDS_LOWER]
                    if len(potential_names) >= 2:
                        entity_a, entity_b = potential_names[0], potential_names[-1]
                    else:
                        # 3. Fallback to any noun-like words
                        any_words = [w for w in re.findall(r"\b[a-zA-Z]+\b", question) if w.lower() not in STOP_WORDS_LOWER and w.lower() not in RELATION_WORDS_LOWER and len(w) > 2]
                        if len(any_words) >= 2:
                            entity_a, entity_b = any_words[0].title(), any_words[-1].title()
                        else:
                            # Graceful failure instead of 400
                            return jsonify({
                                "answer": {"relation": "unknown", "confidence": 0.0, "method": "neural_low", "proof_trace": []},
                                "entities": {"entity_a": "?", "entity_b": "?"},
                                "graph": {"nodes": [], "edges": []}
                            })

        if entity_a:
            entity_a = entity_a.strip().title()
        if entity_b:
            entity_b = entity_b.strip().title()
        narrative = context if context else question
        logger.info(f"Query: {entity_a} -> {entity_b}")

        from orchestrator import hybrid_select
        result = hybrid_select(
            narrative=narrative,
            query_subject=entity_a,
            query_object=entity_b,
            neural_predict_fn=get_neural_fn(),
        )
        
        # Parse proof trace into graph nodes/edges
        nodes_set = set()
        edges = []
        for step in result.get("proof_trace", []):
            match = re.search(r"(FACT|DERIVED|ANSWER)[^:]*:\s*([a-zA-Z0-9_-]+)\(([^,]+),\s*([^)]+)\)", step)
            if match and match.group(1) in ["FACT", "ANSWER"]:
                type_flag = match.group(1)
                relation = match.group(2)
                sub = match.group(3).strip()
                obj = match.group(4).strip()
                nodes_set.add(sub)
                nodes_set.add(obj)
                edges.append({
                    "from": sub,
                    "to": obj,
                    "label": relation,
                    "type": type_flag
                })

        return jsonify({
            "question": question,
            "entities": {"entity_a": entity_a, "entity_b": entity_b},
            "answer": {
                "relation": result["relation"],
                "confidence": result["confidence"],
                "method": result["method"],
                "proof_trace": result.get("proof_trace", []),
            },
            "graph": {
                "nodes": [{"id": n, "label": n.capitalize()} for n in nodes_set],
                "edges": edges
            }
        })
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "GRAPH_READY",
        "mode": "hybrid" if _neural_fn else "symbolic-only",
    })

@app.route("/extract_graph", methods=["POST"])
def extract_graph():
    try:
        from kinship_symbolic import extract_facts_from_text, resolve_coreferences
        data = request.json or {}
        context = data.get("context", "").strip()
        if not context:
            return jsonify({"nodes": [], "edges": []})
            
        raw_facts = extract_facts_from_text(context)
        resolved_facts = resolve_coreferences(raw_facts)
        
        nodes_set = set()
        edges = []
        for s, r, o in resolved_facts:
            s_label = s.title()
            o_label = o.title()
            nodes_set.add(s_label)
            nodes_set.add(o_label)
            edges.append({
                "from": s_label,
                "to": o_label,
                "label": r,
                "type": "FACT"
            })
            
        return jsonify({
            "nodes": [{"id": n, "label": n} for n in nodes_set],
            "edges": edges
        })
    except Exception as e:
        logger.error(f"Error in extract_graph: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("HYBRID REASONER - Starting...")
    logger.info("BERT loading deferred to first neural fallback")
    logger.info("=" * 50)
    app.run(debug=True, host="127.0.0.1", port=5001, use_reloader=False)
