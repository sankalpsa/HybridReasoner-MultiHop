# Hybrid Reasoner — Multi-hop Kinship Reasoning

> **Paper:** *Multi-hop Reasoning in Neural, Symbolic, and Hybrid Models*  
> **Authors:** Sankalp B C, Sk Abdur Razaaq, Vathsala H, Dinesh Naik, Ch. Janaki, Sathyanarayana K B, S D Sudarsan  
> **Affiliations:** National Institute of Technology Surathkal (NITK) & Centre for Development of Advanced Computing (C-DAC)  
> **Journal:** Applied Intelligence

---

## Overview

**Hybrid Reasoner** is a modular neural-symbolic architecture for multi-hop relational (kinship) reasoning. It combines:

| Module | Role | Method |
|---|---|---|
| **Neural Module** | Language perception | Fine-tuned BERT encoder for relational classification |
| **Symbolic Module** | Logical inference | First-Order Logic (FOL) forward chaining over 47 Horn clause rules |
| **Orchestrator** | Decision routing | Confidence-based selection (Algorithm 3) with symbolic priority |

The system is evaluated on the [CLUTRR benchmark](https://github.com/facebookresearch/clutrr) — a compositional generalization testbed with 18 kinship relations across 2–10 hop inference chains.

### Key Results (CLUTRR Test Set)

| Metric | Neural-Only | Hybrid Reasoner |
|---|---|---|
| Accuracy | 88.3% | **94.7%** |
| Macro F1 | 0.861 | **0.923** |
| Micro F1 | 0.883 | **0.947** |
| Logical Consistency | 73.2% | **100.0%** |

The Hybrid Reasoner achieves **depth-invariant performance** — maintaining near-constant accuracy from 2-hop to 10-hop chains, unlike pure neural models whose accuracy degrades from 95.2% to 62.1% beyond 7 hops.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Natural Language Input                 │
│            (Context narrative + Relational query)        │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌───────────────┐       ┌─────────────────┐
  │   Symbolic     │       │   Neural         │
  │   Reasoner     │       │   Reasoner       │
  │                │       │                  │
  │  • Regex NLP   │       │  • BERT encoder  │
  │  • Coreference │       │  • Softmax over  │
  │    resolution  │       │    18 classes    │
  │  • 47 Horn     │       │  • Confidence    │
  │    clause rules│       │    score         │
  │  • Forward     │       │                  │
  │    chaining    │       │                  │
  └───────┬───────┘       └────────┬─────────┘
          │                        │
          └────────┬───────────────┘
                   ▼
        ┌─────────────────┐
        │   Orchestrator   │
        │   (Algorithm 3)  │
        │                  │
        │  Symbolic result │
        │  → conf = 1.0   │
        │  Neural ≥ 0.8   │
        │  → high conf    │
        │  Neural < 0.8   │
        │  → low conf     │
        └────────┬────────┘
                 ▼
           Final Answer
         + Proof Trace
```

---

## Repository Structure

```
HybridReasoner-MultiHop/
├── app.py                     # Flask web interface
├── kinship_symbolic.py        # Symbolic FOL reasoning engine
├── orchestrator.py            # Confidence-based hybrid orchestrator
├── inference_engine.py        # Inference pipeline wrapper
├── rules.py                   # 47 Horn clause inference rules
├── requirements.txt           # Python dependencies
├── LAUNCH_HYBRID_REASONER.bat # One-click Windows launcher
├── technical_documentation.md # Detailed technical documentation
├── templates/
│   └── index.html             # Web UI template
├── tests/
│   ├── hard_test_suite.py              # Adversarial test battery (80 cases)
│   └── continuous_adversarial_tester.py # Automated regression tester
└── Paper/
    ├── Applied_intelligence_journal.pdf # Published paper
    └── *.png                            # Extracted figures
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for the neural module

### Installation

```bash
# Clone the repository
git clone https://github.com/sankalpsa/HybridReasoner-MultiHop.git
cd HybridReasoner-MultiHop

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Interface

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser. Enter a kinship narrative in the context field and a relational query, then click **Execute Logical Reasoning**.

### Windows One-Click Launch

Double-click `LAUNCH_HYBRID_REASONER.bat` to start the Flask server automatically.

---

## How It Works

1. **Input Processing** — The narrative undergoes regex-based fact extraction and pronoun coreference resolution (Algorithm 1) to produce structured triples.

2. **Symbolic Inference** — The FOL engine applies 47 Horn clause rules via saturation-based forward chaining (Algorithm 2) over the extracted knowledge graph.

3. **Neural Inference** — In parallel, a fine-tuned BERT model generates a softmax probability distribution over 18 kinship classes.

4. **Hybrid Selection** — The orchestrator (Algorithm 3) applies symbolic priority: if the symbolic engine succeeds, its answer is returned with confidence 1.0. Otherwise, the neural prediction is used, flagged with its confidence score.

---

## Testing

Run the adversarial test suite:

```bash
python tests/hard_test_suite.py
python tests/continuous_adversarial_tester.py
```

The test suite covers multi-hop chains, conversational noise, pronoun resolution, inverse relations, and edge cases across all 18 kinship types.

---

## Citation

If you use this work, please cite:

```bibtex
@article{sankalp2026multihop,
  title   = {Multi-hop Reasoning in Neural, Symbolic, and Hybrid Models},
  author  = {Sankalp, B C and Razaaq, Sk Abdur and Vathsala, H and 
             Naik, Dinesh and Janaki, Ch and Sathyanarayana, K B and 
             Sudarsan, S D},
  journal = {Applied Intelligence},
  year    = {2026}
}
```

---

## License

This project is developed as part of academic research at NITK Surathkal and C-DAC Bangalore.
