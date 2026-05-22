from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import json
import re
import logging
import pandas as pd
import os
from typing import Dict, List, Tuple, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
CSV_FILE = 'clutrr_train.csv'
SELECTED_DOMAIN = 'family'

class HornClause:
    """
    Represents a Horn Clause in the form:
    head :- body1, body2, body3
    Example: aunt(X, Z) :- sister(X, Y), father(Y, Z)
    """
    def __init__(self, head: str, body: List[str]):
        self.head = head
        self.body = body

    def __repr__(self):
        body_str = ", ".join(self.body)
        return f"{self.head} :- {body_str}"

class KnowledgeGraph:
    """
    Semantic Knowledge Graph with Horn Clauses
    Supports symbolic inference using backward chaining
    """
    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[HornClause] = []
        self.predicate_index: Dict[str, List[str]] = {}

    def add_fact(self, fact: str):
        """Add a ground fact to the knowledge graph."""
        self.facts.add(fact)
        
        predicate = fact.split('(')[0]
        if predicate not in self.predicate_index:
            self.predicate_index[predicate] = []
        self.predicate_index[predicate].append(fact)

    def add_rule(self, rule: HornClause):
        """Add a Horn clause rule."""
        self.rules.append(rule)

    def has_fact(self, fact: str) -> bool:
        """Check if a fact exists."""
        return fact in self.facts

    def get_facts_by_predicate(self, predicate: str) -> List[str]:
        """Get all facts matching a predicate."""
        return self.predicate_index.get(predicate, [])

    def __repr__(self):
        facts_str = f"\nFacts ({len(self.facts)}):\n"
        facts_str += "\n".join(sorted(self.facts)[:10])
        if len(self.facts) > 10:
            facts_str += f"\n... and {len(self.facts) - 10} more"
        
        rules_str = f"\n\nRules ({len(self.rules)}):\n"
        rules_str += "\n".join(str(r) for r in self.rules[:10])
        if len(self.rules) > 10:
            rules_str += f"\n... and {len(self.rules) - 10} more"
        
        return facts_str + rules_str

class SymbolicInferenceEngine:
    """
    Backward chaining inference engine
    Uses Horn clauses and knowledge graph for symbolic reasoning
    """
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.proof_cache = {}
        self.recursion_depth = 0
        self.max_depth = 10

    def prove(self, goal: str, bindings: Dict = None, depth: int = 0) -> Tuple[bool, List[str]]:
        """
        Prove a goal using backward chaining.
        
        Returns:
            (success: bool, proof_trace: List[str])
        """
        if depth > self.max_depth:
            return False, []
        
        if bindings is None:
            bindings = {}
        
        logger.debug(f"{'  ' * depth}Trying to prove: {goal}")
        
        if self.kg.has_fact(goal):
            logger.debug(f"{'  ' * depth}✓ Found direct fact: {goal}")
            return True, [goal]
        
        for rule in self.kg.rules:
            if self._unify(goal, rule.head):
                logger.debug(f"{'  ' * depth}Applying rule: {rule}")
                
                proofs = []
                success = True
                
                for body_pred in rule.body:
                    proved, trace = self.prove(body_pred, bindings, depth + 1)
                    if proved:
                        proofs.extend(trace)
                    else:
                        success = False
                        break
                
                if success:
                    proof_str = f"DERIVED: {goal} by {rule}"
                    return True, proofs + [proof_str]
        
        logger.debug(f"{'  ' * depth}✗ Could not prove: {goal}")
        return False, []

    def _unify(self, goal: str, pattern: str) -> bool:
        """
        Unification: check if goal matches pattern.
        Simplified version - just checks predicate names.
        """
        goal_pred = goal.split('(')[0]
        pattern_pred = pattern.split('(')[0]
        return goal_pred == pattern_pred

def extract_knowledge_from_csv(csv_file: str) -> KnowledgeGraph:
    """
    Extract facts and rules from CSV using Natural Language Processing.
    This builds the Knowledge Graph with Horn Clauses.
    """
    kg = KnowledgeGraph()
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Extracting knowledge from {csv_file}...")
        
        for idx, row in df.iterrows():
            story = str(row.get('clean_story', row.get('story', '')))
            
            facts = extract_facts_from_text(story)
            for fact in facts:
                kg.add_fact(fact)
            
            edge_types = str(row.get('edge_types', ''))
            target = str(row.get('target', ''))
            
            if edge_types and target != 'nan':
                rule = create_rule_from_edges(edge_types, target)
                if rule:
                    kg.add_rule(rule)
        
        logger.info(f"Knowledge Graph Summary:")
        logger.info(f"  Facts: {len(kg.facts)}")
        logger.info(f"  Rules: {len(kg.rules)}")
        
        return kg
    
    except Exception as e:
        logger.error(f"Error extracting knowledge: {e}")
        return kg

def extract_facts_from_text(text: str) -> Set[str]:
    """
    Extract predicates from natural language text.
    Example: "Alice is Bob's mother" → {mother(alice, bob), person(alice), person(bob)}
    """
    facts = set()
    text = text.lower()
    
    patterns = [
        (r"(\w+)\s+(?:is\s+)?(\w+)'s\s+(\w+)", lambda m: f"{m.group(3)}({m.group(2)}, {m.group(1)})"),
        (r"(\w+)\s+is\s+the\s+(\w+)\s+of\s+(\w+)", lambda m: f"{m.group(2)}({m.group(1)}, {m.group(3)})"),
        (r"(\w+)\s+(?:works for|works for)\s+(\w+)", lambda m: f"works_for({m.group(1)}, {m.group(2)})"),
    ]
    
    for pattern, extractor in patterns:
        for match in re.finditer(pattern, text):
            try:
                fact = extractor(match)
                facts.add(fact)
            except:
                pass
    
    return facts

def create_rule_from_edges(edge_types: str, target: str) -> HornClause:
    """
    Create a Horn clause from edge types.
    Example: edge_types="parent-sibling", target="uncle"
    Result: uncle(X, Z) :- parent(X, Y), sibling(Y, Z)
    """
    if not edge_types or target == 'nan':
        return None
    
    edges = [e.strip().lower() for e in re.split(r'[-,]', edge_types)]
    edges = [e for e in edges if e]
    
    if len(edges) < 1:
        return None
    
    return HornClause(target, edges)

def symbolic_reasoning(kg: KnowledgeGraph, entity_a: str, entity_b: str) -> Dict:
    """
    Perform pure symbolic reasoning using Horn clauses.
    """
    engine = SymbolicInferenceEngine(kg)
    results = {}
    
    for rule in kg.rules:
        goal = f"{rule.head}({entity_a.lower()}, {entity_b.lower()})"
        proved, trace = engine.prove(goal)
        
        if proved:
            results[rule.head] = {
                'confidence': 95,
                'trace': trace,
                'method': 'symbolic'
            }
    
    return results

def neural_reasoning(question: str, entity_a: str, entity_b: str, stories: List) -> Dict:
    """
    Perform neural reasoning using Ollama LLM.
    """
    stories_context = ""
    if stories:
        stories_context = "### Stories:\n"
        for i, story in enumerate(stories, 1):
            stories_context += f"{i}. {story['story']}\n"
    
    prompt = f"""Determine the relationship between {entity_a} and {entity_b}.

{stories_context}

Question: {question}

Answer Format:
Relation: [relationship]
Confidence: [0-100]
Reasoning: [explanation]"""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 4096}
            },
            timeout=120
        )
        
        response.raise_for_status()
        llm_response = response.json().get('response', '')
        
        result = {}
        
        rel_match = re.search(r"Relation\s*:\s*([^\n]+)", llm_response, re.IGNORECASE)
        conf_match = re.search(r"Confidence\s*:\s*(\d+)", llm_response, re.IGNORECASE)
        reason_match = re.search(r"Reasoning\s*:\s*(.+?)(?=\n\n|$)", llm_response, re.IGNORECASE | re.DOTALL)
        
        if rel_match:
            relation = rel_match.group(1).strip().lower()
            confidence = int(conf_match.group(1)) if conf_match else 50
            reasoning = reason_match.group(1).strip() if reason_match else ""
            
            result[relation] = {
                'confidence': confidence,
                'reasoning': reasoning,
                'method': 'neural'
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Neural reasoning error: {e}")
        return {}

def hybrid_reasoning(kg: KnowledgeGraph, question: str, entity_a: str, entity_b: str, stories: List) -> Dict:
    """
    Combine symbolic and neural reasoning.
    Returns the most confident answer.
    """
    logger.info(f"\n{'='*70}")
    logger.info("HYBRID REASONING (Symbolic + Neural)")
    logger.info(f"{'='*70}\n")
    
    logger.info("1. SYMBOLIC REASONING (Horn Clauses + Backward Chaining)")
    logger.info("-" * 70)
    symbolic_results = symbolic_reasoning(kg, entity_a, entity_b)
    for relation, data in symbolic_results.items():
        logger.info(f"   ✓ {relation}: {data['confidence']}% (Formal Proof)")
    
    logger.info("\n2. NEURAL REASONING (Ollama LLM)")
    logger.info("-" * 70)
    neural_results = neural_reasoning(question, entity_a, entity_b, stories)
    for relation, data in neural_results.items():
        logger.info(f"   → {relation}: {data['confidence']}% (LLM Reasoning)")
    
    combined_results = {**neural_results, **symbolic_results}
    
    logger.info("\n3. FINAL RESULT")
    logger.info("-" * 70)
    
    if combined_results:
        best_relation = max(combined_results.items(), 
                          key=lambda x: x[1]['confidence'])
        logger.info(f"   ANSWER: {best_relation[0].upper()}")
        logger.info(f"   Confidence: {best_relation[1]['confidence']}%")
        logger.info(f"   Method: {best_relation[1]['method'].upper()}")
        logger.info(f"{'='*70}\n")
        
        return {
            'relation': best_relation[0],
            'confidence': best_relation[1]['confidence'],
            'method': best_relation[1]['method'],
            'symbolic_results': symbolic_results,
            'neural_results': neural_results
        }
    
    logger.info("   NO RESULTS")
    logger.info(f"{'='*70}\n")
    return {'relation': None, 'confidence': 0, 'method': 'none'}

csv_data = None
all_stories = []
knowledge_graph = None

def load_csv_data():
    """Load CSV data."""
    global csv_data, all_stories
    try:
        csv_data = pd.read_csv(CSV_FILE)
        all_stories = csv_data.to_dict('records')
        logger.info(f"✓ Loaded {len(all_stories)} stories")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load CSV: {e}")
        return False

def build_knowledge_graph():
    """Build knowledge graph with horn clauses."""
    global knowledge_graph
    logger.info("\n" + "="*70)
    logger.info("BUILDING KNOWLEDGE GRAPH WITH HORN CLAUSES")
    logger.info("="*70 + "\n")
    
    knowledge_graph = extract_knowledge_from_csv(CSV_FILE)
    logger.info(knowledge_graph)
    
    return knowledge_graph is not None

def check_ollama():
    """Check Ollama connection."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get('models', [])
        logger.info(f"✓ Ollama running with {len(models)} models")
        return True
    except:
        logger.error("✗ Ollama NOT running")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reason', methods=['POST'])
def reason():
    """Hybrid reasoning endpoint."""
    try:
        data = request.json or {}
        question = data.get('question', '').strip()
        context = data.get('context', '').strip()
        entity_a = data.get('entity_a', '').strip()
        entity_b = data.get('entity_b', '').strip()
        
        if not question:
            return jsonify({"error": "Question required"}), 400
        
        logger.info(f"\nIncoming: {question}")
        
        if not entity_a or not entity_b:
            all_words = question.split()
            common_words = ['what', 'is', 'the', 'relation', 'between', 'and', 'a']
            words = [w for w in all_words if w.lower() not in common_words]
            if len(words) < 2:
                return jsonify({"error": "Need two entities"}), 400
            entity_a, entity_b = words[0], words[1]
        
        stories = []
        entity_a_lower = entity_a.lower()
        entity_b_lower = entity_b.lower()
        for story in all_stories:
            story_text = str(story.get('story', '')).lower()
            if entity_a_lower in story_text and entity_b_lower in story_text:
                stories.append({
                    'story': story.get('clean_story', story.get('story', '')),
                    'relation': story.get('target', '')
                })
        
        if context:
            stories = [{'story': context, 'relation': '?'}]
        
        result = hybrid_reasoning(knowledge_graph, question, entity_a, entity_b, stories)
        
        response = {
            "question": question,
            "entities": {"entity_a": entity_a, "entity_b": entity_b},
            "stories": stories,
            "answer": {
                "relation": result['relation'],
                "confidence": result['confidence'],
                "method": result['method'],
                "symbolic_results": result.get('symbolic_results', {}),
                "neural_results": result.get('neural_results', {})
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/kb', methods=['GET'])
def get_knowledge_graph():
    """Return knowledge graph info."""
    if knowledge_graph:
        return jsonify({
            "facts_count": len(knowledge_graph.facts),
            "rules_count": len(knowledge_graph.rules),
            "sample_facts": list(knowledge_graph.facts)[:10],
            "sample_rules": [str(r) for r in knowledge_graph.rules[:5]]
        })
    return jsonify({"error": "KB not loaded"}), 500

if __name__ == '__main__':
    logger.info("\n" + "="*70)
    logger.info("NEURAL-SYMBOLIC REASONER (With Horn Clauses & Knowledge Graph)")
    logger.info("="*70)
    
    check_ollama()
    load_csv_data()
    build_knowledge_graph()
    
    logger.info("\n" + "="*70)
    logger.info("SERVER READY")
    logger.info("="*70)
    logger.info(f"http://localhost:5000")
    logger.info(f"Knowledge Graph API: http://localhost:5000/kb")
    logger.info("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
