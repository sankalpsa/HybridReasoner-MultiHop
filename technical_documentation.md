# Neuro-Symbolic Hybrid Reasoner for Kinship Logic
**Technical Project Documentation**

## 1. Project Overview
This project implements a **Neuro-Symbolic Hybrid Reasoning Engine** designed to parse unstructured, conversational natural language and deduce complex kinship relationships. 

Standard large language models often hallucinate or fail at multi-hop logical deduction (e.g., deducing a 4-hop cousin relationship). To solve this, the project combines the interpretability and strict correctness of **Symbolic Logic (Forward Chaining)** with the robustness of a **Neural Fallback**.

## 2. System Architecture

The hybrid orchestrator pipeline follows three distinct phases:

### Phase A: Conversational Information Extraction (NLP Module)
The system receives unstructured text (e.g., `"Hey guys, let me tell you about my family! Tejas has a sister named Ananya. She is married to Bob. Bob's dad is Charles."`). 
- **Noise Filtering:** Advanced string normalization strips conversational colloquialisms (`"this guy"`, `"she recently"`, `"meet my"`) and parses complex possessives.
- **Fact Extraction:** A suite of cascading regular expressions extracts binary logical predicates (e.g., `sister(Ananya, Tejas)`, `wife(Ananya, Bob)`).
- **Coreference Resolution (Algorithm 1):** The engine dynamically tracks conversational state, resolving gendered pronouns (`he`, `she`) and speaker references (`my`, `I`, `me`) to the correct antecedent entities without losing the chain of context.

### Phase B: Symbolic Inference Engine
The extracted facts form a Knowledge Graph. The symbolic engine executes a Forward-Chaining logic algorithm to saturate the graph.
- **Compositional Rules:** The engine continuously applies logical compositions (e.g., `sister(A, B) ∧ husband(B, C) → sister-in-law(A, C)`).
- **Dynamic Inverse Relations:** Whenever a fact is derived, the engine computes its inverse (e.g., if `A` is the `father` of `B`, then `B` is the `child` of `A`). Crucially, the engine performs **dynamic gender resolution**—it evaluates the gender of `B` to determine if the inverse should specifically be `son` or `daughter`, rather than falling back to static assumptions.
- **Proof Tracing:** Every derived conclusion is accompanied by a mathematically sound "proof trace" guaranteeing exactly how the answer was reached.

### Phase C: Neural Fallback
In real-world data, relation chains are sometimes broken by implicit knowledge gaps that cannot be solved by strict boolean logic.
- If the Symbolic Engine cannot bridge the path between the Query Subject and Query Object, the system routes the sub-graph to a **Neural Embedding Model**.
- The Neural Model generates a probabilistic prediction to bridge the gap, meaning the system never fails catastrophically on dirty data.

## 3. Continuous Adversarial Testing
To guarantee 100% logical robustness, the system features an autonomous **Adversarial Testing Protocol**. 
- A background generator continuously constructs mutated, high-complexity test queries. 
- These generated tests include massive multi-hop relationships (e.g., deep cousin branches), pronoun soup, missing gender identifiers, and disconnected negative controls. 
- This ensures the logical ruleset remains mathematically impenetrable against regressions over time.

## 4. Evaluation and Strengths
- **Explainability:** Unlike pure neural networks, the symbolic component provides a deterministic, transparent proof of exactly *why* a relationship exists.
- **Precision:** Zero hallucination on explicit logical paths.
- **Resilience:** Graceful degradation into neural approximation when symbolic logic is insufficient.
