"""Test the forward chaining engine, symbolic pipeline, and orchestrator."""
import sys
sys.path.insert(0, r"D:\Open\College PC\Downloads\LLS_NEW")

from inference_engine import run_forward_chaining
from kinship_symbolic import symbolic_solve, extract_facts_from_text

print("=" * 60)
print("TEST 1: Forward Chaining Engine (direct facts)")
print("=" * 60)
demo_facts = [
    ("Apollo", "father", "Hermes"),
    ("Zeus", "father", "Apollo"),
]
result = run_forward_chaining(demo_facts, "Zeus", "Hermes")
print(f"  Relation: {result['relation']}")
print(f"  Confidence: {result['confidence']}")
print(f"  Success: {result['success']}")
print("  Proof trace:")
for step in result["proof_trace"]:
    print(f"    {step}")

print()
print("=" * 60)
print("TEST 2: Symbolic Pipeline (narrative -> facts -> answer)")
print("=" * 60)
# Using ASCII apostrophes explicitly
story2 = "Apollo is Hermes's father. Zeus is Apollo's father."
print(f"  Story: {story2}")
facts2 = extract_facts_from_text(story2)
print(f"  Extracted facts: {facts2}")
result2 = symbolic_solve(story2, "Zeus", "Hermes")
print(f"  Relation: {result2['relation']}")
print(f"  Confidence: {result2['confidence']}")
print(f"  Success: {result2['success']}")
print("  Proof trace:")
for step in result2["proof_trace"]:
    print(f"    {step}")

print()
print("=" * 60)
print("TEST 3: Deeper chain (5 hops) - paper example")
print("=" * 60)
story3 = (
    "Apollo is the father of Hermes. "
    "Zeus is the father of Apollo. "
    "Cronus is the father of Zeus. "
    "Uranus is the father of Cronus. "
    "Gaia is the mother of Uranus."
)
print(f"  Story: {story3}")
facts3 = extract_facts_from_text(story3)
print(f"  Extracted facts ({len(facts3)}):")
for f in facts3:
    print(f"    {f}")
result3 = symbolic_solve(story3, "Gaia", "Hermes")
print(f"  Relation: {result3['relation']}")
print(f"  Confidence: {result3['confidence']}")
print(f"  Success: {result3['success']}")
print("  Proof trace (last 5):")
for step in result3["proof_trace"][-5:]:
    print(f"    {step}")

print()
print("=" * 60)
print("TEST 4: Orchestrator (hybrid)")
print("=" * 60)
from orchestrator import hybrid_select
result4 = hybrid_select(story2, "Zeus", "Hermes")
print(f"  Relation: {result4['relation']}")
print(f"  Confidence: {result4['confidence']}")
print(f"  Method: {result4['method']}")
print("  Proof trace:")
for step in result4["proof_trace"]:
    print(f"    {step}")

print()
print("=" * 60)
print("TEST 5: Coreference resolution")
print("=" * 60)
story5 = "John is my father. His brother is Bob."
print(f"  Story: {story5}")
facts5 = extract_facts_from_text(story5)
print(f"  Raw facts: {facts5}")
from kinship_symbolic import resolve_coreferences
resolved5 = resolve_coreferences(facts5)
print(f"  Resolved: {resolved5}")
