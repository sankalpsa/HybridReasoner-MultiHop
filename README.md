# Hybrid Neural-Symbolic Reasoner for Multi-hop Kinship Logic

<div align="center">

[![Accuracy](https://img.shields.io/badge/Accuracy-94.7%25-10B981?style=for-the-badge&logo=checkmarx&logoColor=white)](#key-results)
[![Logical Consistency](https://img.shields.io/badge/Consistency-100%25-F59E0B?style=for-the-badge&logo=academia&logoColor=white)](#key-results)
[![Status](https://img.shields.io/badge/Status-In%20Preparation-3B82F6?style=for-the-badge&logo=googlescholar&logoColor=white)](#overview)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](#prerequisites)
[![Frameworks](https://img.shields.io/badge/PyTorch%20%7C%20Flask-D33C3C?style=for-the-badge&logo=pytorch&logoColor=white)](#prerequisites)
[![Adversarial Tests](https://img.shields.io/badge/Adversarial%20Tests-Passing%20(80%2F80)-emerald?style=for-the-badge&logo=githubactions&logoColor=white)](#testing)

</div>

---

### 📖 Publication Details
> **Paper:** *Multi-hop Reasoning in Neural, Symbolic, and Hybrid Models*  
> **Authors:** Sankalp B C, Sk Abdur Razaaq, Vathsala H, Dinesh Naik, Ch. Janaki, Sathyanarayana K B, S D Sudarsan  
> **Affiliations:** National Institute of Technology Surathkal (NITK) & Centre for Development of Advanced Computing (C-DAC)  
> **Status:** In Preparation for *Applied Intelligence*

---

## ⚡ Overview

**Hybrid Reasoner** is a state-of-the-art modular neural-symbolic framework designed for high-precision, multi-hop relational (kinship) reasoning over natural language stories. By merging the linguistic perception of **Deep Learning** with the absolute mathematical consistency of **First-Order Logic (FOL) forward chaining**, it solves a fundamental limitation of modern LLMs: *performance degradation over long-chain deduction*.

### 🏆 Core Breakthroughs
* 🎯 **Depth-Invariant Precision:** Maintains near-constant accuracy from 2-hop to 10-hop inference chains (whereas pure neural model accuracies degrade from 95.2% to 62.1% past 7 hops).
* 🔒 **100% Logical Consistency:** Employs saturation-based forward chaining to enforce zero-hallucination logical proofs.
* 🛡️ **Extremely Robust:** Features dynamic coreference tracking, pronoun resolution, and localized name-matching to handle messy conversational preambles and pronoun soups seamlessly.

---

## 📐 System Architecture

The workflow routes unstructured text inputs through a modular, confidence-based hybrid selector (Algorithm 3):

![Hybrid Reasoner Architecture](Paper/architecture.svg?v=2)

### 🧩 The Modules
1. **Conversational NLP Extraction (Algorithm 1):** Cascading regular expressions strip conversational noise, parse complex possessives, and dynamically resolve gendered pronouns (`he`, `she`) and first-person speaker references (`I`, `me`, `my`) to their correct entity antecedents.
2. **Symbolic Logic Saturation (Algorithm 2):** Performs forward chaining over a Knowledge Graph using **47 custom Horn clause rules** (e.g. `sister(A,B) ∧ husband(B,C) → sister-in-law(A,C)`) with dynamic inverse gender resolution (deriving `son-in-law` vs `daughter-in-law` based on gender checking).
3. **Neural Fallback Module:** Fine-tuned relational BERT encoder evaluates complex, implicit relationship paths when strict symbolic rules are broken by information gaps.

---

## 📈 Evaluation & Benchmarks (CLUTRR Test Set)

The framework has been thoroughly evaluated on the compositional generalization testbed **CLUTRR** (Cognitive Logic Unit for Truth Value Relational Reasoning):

| Evaluation Metric | Neural-Only Baseline (BERT) | Pure Symbolic Engine | Hybrid Reasoner (Ours) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 88.3% | 79.1% | **94.7%** |
| **Macro F1** | 0.861 | 0.770 | **0.923** |
| **Micro F1** | 0.883 | 0.791 | **0.947** |
| **Logical Consistency** | 73.2% | **100.0%** | **100.0%** |

---

## 📂 Repository Structure

The curated repository contains only the clean, relevant, and submission-ready files:

```text
HybridReasoner-MultiHop/
├── app.py                      # Flask web server (Main entry point)
├── kinship_symbolic.py         # Symbolic core, regex NLP, and pronoun tracker
├── orchestrator.py             # Confidence-based hybrid selector (Algorithm 3)
├── inference_engine.py         # Universal reasoning wrapper
├── rules.py                    # 47 mathematical Horn clause rules
├── requirements.txt            # Project dependencies
├── LAUNCH_HYBRID_REASONER.bat  # Portable, one-click Windows launcher
├── README.md                   # Visual repository guide
├── technical_documentation.md  # Detailed technical project specifications
├── templates/
│   └── index.html              # Dynamic web UI with visual graph drawing
├── tests/
│   ├── hard_test_suite.py               # 10 adversarial extreme edge case scenarios
│   └── continuous_adversarial_tester.py  # 80 generated randomized mutation tests
└── Paper/
    ├── Applied_intelligence_journal.pdf # Draft manuscript / Preprint
    └── architecture.svg                 # High-fidelity workflow graphic
```

---

## 🚀 Quick Start

### 📋 Prerequisites
* Python 3.10+
* (Optional) NVIDIA GPU with CUDA for neural acceleration

### ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/sankalpsa/HybridReasoner-MultiHop.git
cd HybridReasoner-MultiHop

# 2. Set up virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install core dependencies
pip install -r requirements.txt
```

### 🖥️ Running the Interactive Web App

Run the Flask server:
```bash
python app.py
```
Open [http://localhost:5001](http://localhost:5001) in your browser. 

* **Windows One-Click Launcher:** Double-click `LAUNCH_HYBRID_REASONER.bat` to automatically kill stale port-binding processes, activate the virtual environment, start the Flask server on Port 5001, and launch the web app instantly!

---

## 🧪 Testing and Verification

Execute the continuous adversarial testing suite:
```bash
python tests/hard_test_suite.py
python tests/continuous_adversarial_tester.py
```

These suites verify multi-hop paths, reflexive same-name inputs, conversational noise, pronoun pollution, and affinal (in-law) relationships across all **18 kinship classes** to guarantee zero regressions.

---

## 🎓 Citation

```bibtex
@article{sankalp2026multihop,
  title   = {Multi-hop Reasoning in Neural, Symbolic, and Hybrid Models},
  author  = {Sankalp, B C and Razaaq, Sk Abdur and Vathsala, H and 
             Naik, Dinesh and Janaki, Ch and Sathyanarayana, K B and 
             Sudarsan, S D},
  journal = {Applied Intelligence (In Preparation)},
  year    = {2026}
}
```

---

## 📄 License
This project is developed as part of academic research at **NITK Surathkal** and **C-DAC Bangalore**. All rights reserved.
