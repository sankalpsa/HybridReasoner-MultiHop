"""Write new app.py (Flask server) and updated index.html to D: drive."""
import os, shutil
BASE = r"D:\Open\College PC\Downloads\LLS_NEW"

# First, rename old app.py -> train_joint.py (preserve it)
old_app = os.path.join(BASE, "app.py")
renamed = os.path.join(BASE, "train_joint.py")
if os.path.exists(old_app) and not os.path.exists(renamed):
    shutil.copy2(old_app, renamed)
    print(f"Backed up old app.py -> train_joint.py")

# ── New app.py ──────────────────────────────────────────────────
app_code = r'''"""
app.py - Hybrid Reasoner Flask Web Server

Main entry point for the Neural-Symbolic Hybrid Reasoner.
Implements the full pipeline from the paper:
  1. Receives narrative + query via /reason endpoint
  2. Routes through orchestrator (Algorithm 3):
     - Symbolic first (forward chaining, confidence 1.0)
     - Neural fallback (BERT classifier, threshold tau=0.8)
  3. Returns relation, confidence, method, and proof trace
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Global state ────────────────────────────────────────────────
neural_predict_fn = None


def init_neural_module():
    """Try to load the BERT classifier for neural fallback."""
    global neural_predict_fn
    try:
        from bert_classifier import get_classifier
        clf = get_classifier()
        neural_predict_fn = clf.neural_predict_fn
        logger.info("BERT neural module loaded successfully")
    except Exception as e:
        logger.warning(f"BERT module not available: {e}")
        logger.warning("Running in symbolic-only mode")
        neural_predict_fn = None


# ── Routes ──────────────────────────────────────────────────────

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
            import re
            match = re.search(
                r"between\s+(\w+)\s+and\s+(\w+)", question, re.IGNORECASE
            )
            if match:
                entity_a, entity_b = match.group(1), match.group(2)
            else:
                words = re.findall(r"\b[A-Z][a-z]+\b", question)
                if len(words) >= 2:
                    entity_a, entity_b = words[0], words[1]
                else:
                    return jsonify({"error": "Could not extract entities. Provide entity_a and entity_b."}), 400

        narrative = context if context else question

        logger.info(f"Query: {entity_a} -> {entity_b}")
        logger.info(f"Narrative: {narrative[:100]}...")

        # Run hybrid orchestrator (Algorithm 3)
        from orchestrator import hybrid_select
        result = hybrid_select(
            narrative=narrative,
            query_subject=entity_a,
            query_object=entity_b,
            neural_predict_fn=neural_predict_fn,
        )

        response = {
            "question": question,
            "entities": {"entity_a": entity_a, "entity_b": entity_b},
            "answer": {
                "relation": result["relation"],
                "confidence": result["confidence"],
                "method": result["method"],
                "proof_trace": result.get("proof_trace", []),
            },
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "neural_loaded": neural_predict_fn is not None,
        "mode": "hybrid" if neural_predict_fn else "symbolic-only",
    })


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("HYBRID REASONER - Neural-Symbolic Web Server")
    logger.info("=" * 60)

    init_neural_module()

    logger.info("=" * 60)
    logger.info("Server ready: http://localhost:5000")
    logger.info("=" * 60)

    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
'''

with open(os.path.join(BASE, "app.py"), "w", encoding="utf-8") as f:
    f.write(app_code)
print(f"Written app.py ({len(app_code)} bytes)")

# ── Updated index.html ─────────────────────────────────────────
html_code = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid Reasoner - Neural-Symbolic Inference</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh; padding: 20px;
            display: flex; justify-content: center; align-items: center;
        }
        .container {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            box-shadow: 0 25px 80px rgba(0,0,0,0.5);
            max-width: 780px; width: 100%; padding: 40px;
            color: #e0e0e0;
        }
        .header { display: flex; align-items: center; margin-bottom: 30px;
                   border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 20px; }
        .header h1 { font-size: 26px; color: #fff; margin-left: 12px; }
        .header .icon { font-size: 32px; }
        .subtitle { color: #aaa; font-size: 13px; margin-top: 4px; }
        .status { display: inline-block; width: 10px; height: 10px;
                  background: #4caf50; border-radius: 50%; margin-right: 6px;
                  animation: pulse 2s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
        .status-text { color: #4caf50; font-size: 13px; font-weight: 600; }
        .form-group { margin-bottom: 18px; }
        .form-group label { display: block; font-weight: 600; color: #ccc;
                            margin-bottom: 8px; font-size: 13px; }
        textarea, input[type="text"] {
            width: 100%; padding: 12px 14px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px; color: #fff;
            font-family: inherit; font-size: 14px; resize: vertical;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        textarea:focus, input:focus {
            outline: none; border-color: #7c4dff;
            box-shadow: 0 0 0 3px rgba(124,77,255,0.2);
        }
        textarea { min-height: 90px; }
        .entity-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .button-group { display: flex; gap: 10px; margin-top: 22px; }
        button { flex: 1; padding: 12px 20px; border: none; border-radius: 10px;
                 font-size: 15px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-primary {
            background: linear-gradient(135deg, #7c4dff, #448aff);
            color: #fff;
        }
        .btn-primary:hover { transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(124,77,255,0.4); }
        .btn-secondary { background: rgba(255,255,255,0.08); color: #ccc; }
        .btn-secondary:hover { background: rgba(255,255,255,0.15); }
        .result-box { margin-top: 22px; padding: 20px; border-radius: 14px; display: none; }
        .result-box.show { display: block; }
        .result-box.success { background: rgba(76,175,80,0.1); border-left: 4px solid #4caf50; }
        .result-box.error { background: rgba(244,67,54,0.1); border-left: 4px solid #f44336; }
        .result-box.loading { background: rgba(33,150,243,0.1); border-left: 4px solid #2196f3; }
        .answer-main { font-size: 28px; font-weight: 700; color: #fff;
                       margin: 12px 0; text-transform: capitalize; }
        .confidence { display: inline-block; padding: 4px 14px; border-radius: 20px;
                      font-size: 13px; font-weight: 600; }
        .conf-high { background: rgba(76,175,80,0.2); color: #66bb6a; }
        .conf-med { background: rgba(255,152,0,0.2); color: #ffa726; }
        .conf-low { background: rgba(244,67,54,0.2); color: #ef5350; }
        .method-badge { display: inline-block; padding: 3px 10px; border-radius: 6px;
                        font-size: 11px; font-weight: 600; margin-left: 8px; text-transform: uppercase; }
        .method-symbolic { background: rgba(124,77,255,0.2); color: #b388ff; }
        .method-neural { background: rgba(0,188,212,0.2); color: #4dd0e1; }
        .proof-trace { margin-top: 16px; background: rgba(0,0,0,0.3); border-radius: 10px;
                       padding: 14px; max-height: 250px; overflow-y: auto; }
        .proof-trace summary { cursor: pointer; font-weight: 600; font-size: 13px; color: #aaa; }
        .proof-trace pre { font-size: 12px; color: #8a8; margin-top: 8px;
                           white-space: pre-wrap; line-height: 1.6; }
        .loading-spinner { display: inline-block; width: 16px; height: 16px;
            border: 2px solid rgba(124,77,255,0.3); border-top-color: #7c4dff;
            border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .footer { text-align: center; margin-top: 28px; color: #555; font-size: 11px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="icon">&#9889;</div>
            <div>
                <h1>Hybrid Reasoner</h1>
                <div class="subtitle">Neural-Symbolic Multi-hop Inference Engine</div>
                <div style="margin-top: 6px;">
                    <span class="status"></span>
                    <span class="status-text">System Ready</span>
                </div>
            </div>
        </div>

        <form id="reasonForm">
            <div class="form-group">
                <label>1. Context / Story</label>
                <textarea id="context" placeholder="Enter the narrative story..."></textarea>
            </div>
            <div class="form-group">
                <label>2. Question</label>
                <input type="text" id="question" placeholder="What is the relation between X and Y?" required>
            </div>
            <div class="entity-row">
                <div class="form-group">
                    <label>Entity A (optional)</label>
                    <input type="text" id="entity_a" placeholder="e.g. Zeus">
                </div>
                <div class="form-group">
                    <label>Entity B (optional)</label>
                    <input type="text" id="entity_b" placeholder="e.g. Hermes">
                </div>
            </div>
            <div class="button-group">
                <button type="submit" class="btn-primary">Execute Reasoning</button>
                <button type="reset" class="btn-secondary">Clear</button>
            </div>
        </form>

        <div id="resultBox" class="result-box">
            <div id="resultContent"></div>
        </div>

        <div class="footer">Hybrid Reasoner &mdash; BERT + FOL Forward Chaining</div>
    </div>

    <script>
        const form = document.getElementById('reasonForm');
        const resultBox = document.getElementById('resultBox');
        const resultContent = document.getElementById('resultContent');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = document.getElementById('question').value.trim();
            const context = document.getElementById('context').value.trim();
            const entity_a = document.getElementById('entity_a').value.trim();
            const entity_b = document.getElementById('entity_b').value.trim();

            if (!question && !context) { showError('Provide a question or context'); return; }
            showLoading();

            try {
                const response = await fetch('/reason', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ question, context, entity_a, entity_b })
                });
                const data = await response.json();
                if (!response.ok) { showError(data.error || 'Error'); return; }
                showSuccess(data);
            } catch (err) {
                showError('Connection failed: ' + err.message);
            }
        });

        function showLoading() {
            resultBox.className = 'result-box show loading';
            resultContent.innerHTML = '<div class="loading-spinner"></div><strong>Processing...</strong><p style="color:#aaa;margin-top:6px">Running symbolic + neural inference...</p>';
        }

        function showSuccess(data) {
            resultBox.className = 'result-box show success';
            const a = data.answer || {};
            const rel = a.relation || 'unknown';
            const conf = a.confidence || 0;
            const method = a.method || 'unknown';
            const trace = a.proof_trace || [];

            let confClass = conf >= 0.8 ? 'conf-high' : conf >= 0.5 ? 'conf-med' : 'conf-low';
            let methodClass = method.startsWith('symbolic') ? 'method-symbolic' : 'method-neural';

            let html = `<h3 style="color:#aaa;font-size:14px">Answer</h3>
                <div class="answer-main">${esc(rel)}</div>
                <span class="confidence ${confClass}">${(conf*100).toFixed(1)}% confidence</span>
                <span class="method-badge ${methodClass}">${esc(method)}</span>`;

            if (trace.length > 0) {
                html += `<div class="proof-trace"><details open>
                    <summary>Proof Trace (${trace.length} steps)</summary>
                    <pre>${trace.map(s => esc(s)).join('\n')}</pre>
                </details></div>`;
            }
            resultContent.innerHTML = html;
        }

        function showError(msg) {
            resultBox.className = 'result-box show error';
            resultContent.innerHTML = `<strong style="color:#ef5350">Error:</strong> <span style="color:#ccc">${esc(msg)}</span>`;
        }

        function esc(t) {
            const d = document.createElement('div');
            d.textContent = t;
            return d.innerHTML;
        }
    </script>
</body>
</html>
'''

tpl_dir = os.path.join(BASE, "templates")
os.makedirs(tpl_dir, exist_ok=True)
with open(os.path.join(tpl_dir, "index.html"), "w", encoding="utf-8") as f:
    f.write(html_code)
print(f"Written templates/index.html ({len(html_code)} bytes)")

print("\nAll files written to D: drive!")
