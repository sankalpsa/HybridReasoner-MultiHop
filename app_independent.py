from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import re
import logging
import os
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

CSV_FILE = 'clutrr_train.csv'
SELECTED_DOMAIN = 'family'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    logger.info("🟢 GPU detected - using TinyLlama (1.1B)")
else:
    MODEL_NAME = "distilgpt2"
    logger.info("🔴 CPU detected - using DistilGPT2 (82M) - FASTEST")

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using model: {MODEL_NAME}")

csv_data = None
all_stories = []
tokenizer = None
model = None
text_generator = None

def load_model():
    """Load the language model with proper optimization."""
    global tokenizer, model, text_generator
    
    try:
        logger.info(f"\n{'='*70}")
        logger.info("LOADING LOCAL LANGUAGE MODEL")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Device: {DEVICE}")
        
        logger.info("\n1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("   ✓ Tokenizer loaded")
        
        logger.info("2. Loading model...")
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if DEVICE == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
        
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        
        model.eval()
        logger.info("   ✓ Model loaded")
        
        logger.info("3. Creating generation pipeline...")
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if DEVICE == "cuda" else -1,
            max_new_tokens=200
        )
        logger.info("   ✓ Pipeline created")
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ MODEL READY")
        logger.info(f"{'='*70}\n")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error("Install: pip install torch transformers")
        return False

def local_reasoning(question: str, entity_a: str, entity_b: str, stories: List) -> Dict:
    """
    Perform fast reasoning using local transformer model.
    Optimized for speed on both CPU and GPU.
    """
    
    stories_context = ""
    if stories:
        stories_context = "Information:\n"
        for i, story in enumerate(stories[:3]):
            stories_context += f"{i+1}. {story['story']}\n"
    
    prompt = f"""{stories_context}

Q: {question}

Answer this relation question with:
Relation: [one word]
Confidence: [0-100]
Reasoning: [one sentence]

Answer:"""
    
    try:
        logger.info(f"\n{'='*70}")
        logger.info("LOCAL REASONING ENGINE")
        logger.info(f"{'='*70}")
        logger.info(f"Question: {question}")
        logger.info(f"Entities: {entity_a} → {entity_b}")
        logger.info(f"Using: {MODEL_NAME} on {DEVICE}")
        logger.info("Generating response...")
        
        with torch.no_grad():
            response = text_generator(
                prompt,
                max_length=300,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = response[0]['generated_text']
        
        if "Answer:" in generated_text:
            llm_response = generated_text.split("Answer:")[-1]
        else:
            llm_response = generated_text
        
        logger.info(f"Generated: {llm_response[:150]}...\n")
        
        result = {
            "raw_response": llm_response,
            "relation": None,
            "confidence": 0,
            "reasoning": ""
        }
        
        rel_match = re.search(r"Relation\s*:\s*([^\n]+?)\s*(?:\n|$)", llm_response, re.IGNORECASE)
        if rel_match:
            relation_text = rel_match.group(1).strip()
            relation_text = re.sub(r'[^a-zA-Z\s\-_]', '', relation_text).strip()
            result["relation"] = relation_text.lower() if relation_text else "unknown"
        
        conf_match = re.search(r"Confidence\s*:\s*(\d+)", llm_response, re.IGNORECASE)
        if conf_match:
            result["confidence"] = min(100, max(0, int(conf_match.group(1))))
        else:
            result["confidence"] = 65
        
        reason_match = re.search(r"Reasoning\s*:\s*(.+?)(?=\n|$)", llm_response, re.IGNORECASE | re.DOTALL)
        if reason_match:
            result["reasoning"] = reason_match.group(1).strip()[:150]
        else:
            result["reasoning"] = "Inference based on relationship chain analysis"
        
        logger.info(f"✓ Relation: {result['relation']}")
        logger.info(f"✓ Confidence: {result['confidence']}%\n")
        
        return result
    
    except Exception as e:
        logger.error(f"✗ Reasoning error: {e}", exc_info=True)
        return {
            "raw_response": str(e),
            "relation": None,
            "confidence": 0,
            "reasoning": f"ERROR: {str(e)}"
        }

def load_csv_data():
    """Load CSV knowledge base."""
    global csv_data, all_stories
    try:
        csv_data = pd.read_csv(CSV_FILE)
        all_stories = csv_data.to_dict('records')
        logger.info(f"✓ Loaded {len(all_stories)} stories from {CSV_FILE}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load CSV: {e}")
        return False

def get_relevant_stories(entity_a: str, entity_b: str) -> list:
    """Retrieve relevant stories from knowledge base."""
    relevant = []
    entity_a_lower = entity_a.lower()
    entity_b_lower = entity_b.lower()
    
    for story in all_stories:
        story_text = str(story.get('story', '')).lower()
        query = str(story.get('query', '')).lower()
        target = str(story.get('target', ''))
        
        if entity_a_lower in story_text and entity_b_lower in story_text:
            if entity_a_lower in query and entity_b_lower in query:
                clean_story = story.get('clean_story', story.get('story', ''))
                relevant.append({
                    'story': clean_story,
                    'relation': target if target != 'nan' else 'unknown',
                    'query': query
                })
    
    return relevant

def extract_entities(question: str) -> Tuple[str, str]:
    """Extract two entities from question."""
    common_words = [
        'what', 'is', 'the', 'relation', 'relationship', 'between',
        'and', 'a', 'an', 'of', 'in', 'for', 'to', 'with', 'from'
    ]
    
    match = re.search(r'between\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)', question, re.IGNORECASE)
    if match:
        return (match.group(1), match.group(2))
    
    match = re.search(r'([a-zA-Z]+)\s+and\s+([a-zA-Z]+)', question, re.IGNORECASE)
    if match:
        return (match.group(1), match.group(2))
    
    words = re.findall(r'\b[A-Z][a-z]+\b', question)
    if len(words) >= 2:
        return (words[0], words[1])
    
    all_words = question.split('/\s+/')
    cleaned_words = [w.strip('?,!.;:').strip() for w in all_words 
                     if w.lower() not in common_words and len(w.strip('?,!.;:')) > 1]
    
    if len(cleaned_words) >= 2:
        return (cleaned_words[0], cleaned_words[1])
    
    return None

@app.route('/')
def index():
    """Serve HTML interface."""
    return render_template('index.html')

@app.route('/reason', methods=['POST'])
def reason():
    """Main reasoning endpoint (completely local, optimized)."""
    try:
        data = request.json or {}
        question = data.get('question', '').strip()
        context = data.get('context', '').strip()
        entity_a = data.get('entity_a', '').strip()
        entity_b = data.get('entity_b', '').strip()
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Received: {question}")
        logger.info(f"{'='*70}")
        
        if not entity_a or not entity_b:
            extracted = extract_entities(question)
            if not extracted:
                return jsonify({
                    "error": "Could not extract two entities",
                    "hint": "Format: 'What is the relation between Name1 and Name2?'"
                }), 400
            entity_a, entity_b = extracted
        
        logger.info(f"Entities: {entity_a}, {entity_b}")
        
        stories = get_relevant_stories(entity_a, entity_b)
        logger.info(f"Found {len(stories)} relevant stories")
        
        if context:
            stories = [{
                'story': context,
                'relation': '?',
                'query': f"({entity_a}, {entity_b})"
            }]
        
        result = local_reasoning(question, entity_a, entity_b, stories)
        
        response = {
            "question": question,
            "provided_context": context if context else None,
            "entities": {
                "entity_a": entity_a,
                "entity_b": entity_b
            },
            "stories_used": stories,
            "answer": {
                "relation": result.get("relation"),
                "confidence": result.get("confidence"),
                "reasoning": result.get("reasoning"),
                "model": MODEL_NAME,
                "device": DEVICE,
                "raw_response": result.get("raw_response")
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    model_ok = text_generator is not None
    csv_ok = csv_data is not None
    
    return jsonify({
        "status": "healthy" if (model_ok and csv_ok) else "degraded",
        "model_loaded": model_ok,
        "csv_loaded": csv_ok,
        "stories_count": len(all_stories) if csv_ok else 0,
        "model": MODEL_NAME,
        "device": DEVICE
    })

if __name__ == '__main__':
    logger.info("\n" + "="*70)
    logger.info("NEURAL-SYMBOLIC REASONER (Optimized)")
    logger.info("="*70)
    
    model_ready = load_model()
    
    csv_ready = load_csv_data()
    
    logger.info("\n" + "="*70)
    if model_ready and csv_ready:
        logger.info("✓ ALL SYSTEMS READY")
    else:
        logger.warning("⚠ SOME SYSTEMS NOT READY")
    logger.info("="*70)
    logger.info(f"Server: http://localhost:5000")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info("="*70 + "\n")
    
    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,
        threaded=True
    )
