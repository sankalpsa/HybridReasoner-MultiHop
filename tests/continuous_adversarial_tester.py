import random
import time
import json
import os
import sys

# Add parent directory to sys.path to allow importing from main directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator import hybrid_select

# Simple set of names to build relations
NAMES_M = ["sankalp", "achal", "bob", "dave", "siddharth", "rahul", "akhil", "mohit", "robert", "zeus", "apollo", "tejas"]
NAMES_F = ["sadhana", "alice", "carol", "jessica", "emily", "sarah", "jane", "ananya", "maa", "clara"]

# Simple generator for a random 3-hop relation that we know the answer to
# Format: (Story string, Query Subject, Query Object, Expected Answer)
def generate_random_test():
    patterns = [
        # pattern 1: A is father of B. B is father of C. -> A is grandfather of C
        lambda: (
            f"{random.choice(NAMES_M)} is the father of {random.choice(NAMES_M)} . The latter is the father of {(c:=random.choice(NAMES_M))} .",
            "grandfather", "grandson"
        ),
    ]
    # To keep it completely bulletproof and truly adversarial without requiring a full logical solver to generate answers,
    # we can run existing hard tests, or we can just randomly swap names in the 10 known hard patterns.
    pass

# Instead of complex dynamic generation (which needs its own prolog engine to find the ground truth),
# we will randomly mutate the names in the hard_test_suite.py patterns.

def get_mutated_hard_tests():
    m_pool = random.sample(NAMES_M, len(NAMES_M))
    f_pool = random.sample(NAMES_F, len(NAMES_F))
    
    m1, m2, m3, m4, m5 = m_pool[:5]
    f1, f2, f3, f4, f5 = f_pool[:5]
    
    return [
        # 1. Extreme Pronoun Soup + Conversational Noise (In-law via sibling)
        {
            "context": f"Hey guys, let me tell you about my family! {m1} has a sister named {f1}. She is incredibly smart. She is married to this guy {m2}. {m2}'s father is {m3}, who is a great man.",
            "q_subj": m3, "q_obj": f1,
            "expected": "father-in-law"
        },
        # 2. Tricky Reflexive Case-Insensitive (Testing the self identity bug again)
        {
            "context": f"{f2} is the sister of {m1}. {m1} is the brother of {m2}. I don't know who {m4} is.",
            "q_subj": f"  {m2.upper()} ", "q_obj": f"{m2} ",
            "expected": "self"
        },
        # 3. Deep Cousin Chain (4-hop)
        {
            "context": f"{m1} is the father of {m2}. {m2} has a brother {m3}. {m3} is the father of {f1}. {m1} also has a daughter {f2} who is the mother of {f3}. Wow what a big tree.",
            "q_subj": f1, "q_obj": f3,
            "expected": "cousin"
        },
        # 4. Unspecified Gender Parent -> Child (Testing the generic 'child' fallback)
        {
            "context": f"{m4} has a mother named {f4}. {f4} is married to {m5}.",
            "q_subj": m5, "q_obj": m4,
            "expected": "father"
        },
        # 5. Reverse Query (Child to Parent in conversational tone)
        {
            "context": f"so {f5} has a brother {m1} , {m1} has father {m2}.",
            "q_subj": f5, "q_obj": m2,
            "expected": "daughter"
        },
        # 6. Negative Control (Unrelated)
        {
            "context": f"{m1} is the son of {f1}. But over in the other town, {m2} is the son of {f2}.",
            "q_subj": m1, "q_obj": m2,
            "expected": "unknown"
        },
        # 7. Uncle through marriage
        {
            "context": f"{f1} is the wife of {m1}. {m1} is the brother of {m2}. {m2} is the father of {f2}.",
            "q_subj": m1, "q_obj": f2,
            "expected": "uncle"
        },
        # 8. Speaker query (I / my / me)
        {
            "context": f"I am so proud of my son {m1}. His sister is {f1}. She recently had a baby boy named {m2}.",
            "q_subj": m2, "q_obj": "Speaker",
            "expected": "grandson"
        }
    ]

def dummy_neural(ctx, qs, qo):
    return {"relation": "UNKNOWN", "confidence": 0.0}

def run_tests():
    stats_file = "adversarial_stats.json"
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {"total_runs": 0, "total_passed": 0, "total_failed": 0, "failures": []}

    print("Running generated adversarial batch...")
    
    # Generate 50 tests by mutating names repeatedly
    batch_tests = []
    for _ in range(10):
        batch_tests.extend(get_mutated_hard_tests())
        
    batch_passed = 0
    batch_failed = 0
    
    for i, test in enumerate(batch_tests):
        stats["total_runs"] += 1
        
        ctx = test["context"]
        qs = test["q_subj"]
        qo = test["q_obj"]
        expected = test["expected"]
        
        # We must suppress print statements from hybrid_select for speed
        res = hybrid_select(ctx, qs, qo, dummy_neural)
        result = res.get('relation', 'None') or 'None'
        
        if result.lower() == expected.lower():
            batch_passed += 1
            stats["total_passed"] += 1
        else:
            batch_failed += 1
            stats["total_failed"] += 1
            stats["failures"].append({
                "context": ctx,
                "qs": qs,
                "qo": qo,
                "expected": expected,
                "got": result,
                "proof": res.get('proof_trace', [])
            })
            
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"Batch complete. Passed: {batch_passed}, Failed: {batch_failed}")
    print(f"Total historical runs: {stats['total_runs']}, Total historical fails: {stats['total_failed']}")
    
if __name__ == '__main__':
    run_tests()
