# 🚀 Future Work & Enhancements

This document outlines the roadmap and future research directions for the **Neuro-Symbolic Hybrid Kinship Reasoner**. While the current architecture achieves 100% logical consistency on adversarial benchmarks, there are several exciting avenues for expanding the model's capabilities.

## 1. Multilingual & Cross-Cultural Reasoning
Currently, the relation extraction pipeline is heavily optimized for English semantics. Future work will extend the Natural Language Processing (NLP) module to support multilingual contexts. 
* Many languages (like Hindi, Mandarin, and Arabic) have highly granular kinship terminology (e.g., distinguishing between a maternal uncle and a paternal uncle). 
* The symbolic engine can be extended with localized rule sets to inherently map these granular terms into a universal semantic graph.

## 2. Temporal & Dynamic Logic Integration
The current knowledge graph treats relationships as static facts. A major advancement would be integrating **Temporal Logic**:
* Tracking life events such as marriages, divorces, births, and deaths across a timeline.
* Resolving queries like *"Who was David's wife in 2010?"* or handling state-changes where affinal (in-law) relationships change due to external events.

## 3. Large Language Model (LLM) Integration
While the current architecture uses a specialized RoBERTa classifier for fast and efficient relation extraction, future iterations could integrate parameter-efficient fine-tuned LLMs (like Llama 3 or Mistral).
* LLMs excel at zero-shot coreference resolution in highly noisy, multi-speaker conversational dialogue.
* The LLM would act as the "perception" layer, translating messy human transcripts into clean logical triples (JSON/RDF), which are then handed off to our deterministic symbolic engine for hallucination-free reasoning.

## 4. Probabilistic & Fuzzy Logic
The symbolic engine currently operates on absolute truths (Confidence = 1.0) or failures (Confidence = 0.0). We plan to implement probabilistic Soft Rules (e.g., Markov Logic Networks or Probabilistic Soft Logic).
* Handling uncertain inputs like *"I think he is my cousin"* or *"She might be my aunt."*
* Calculating confidence intervals across multi-hop inference paths where the source facts possess varying degrees of reliability.

## 5. Domain Expansion Beyond Kinship
The hybrid architecture (combining neural extraction with graph-based symbolic deduction) is theoretically domain-agnostic. 
* The system can be scaled beyond the CLUTRR and ConceptNet kinship subsets to tackle general-purpose reasoning.
* Potential target domains include **Medical Diagnostics** (symptom-disease-treatment hops), **Legal Reasoning** (contractual obligation pathways), and **Financial Forensics** (tracking multi-hop ownership structures).

## 6. Advanced Coreference Resolution
Improving the continuous adversarial tester to handle massive, novel-length transcripts where pronouns (he/she/they) refer to entities mentioned chapters ago. Implementing an attention-based memory buffer for the symbolic engine would allow it to track entity states across vast context windows.
