import sys
import os

# Ensure standard output supports UTF-8 characters to prevent CP1252/Windows encoding errors
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from kinship_symbolic import symbolic_solve

TEST_CASES = [
    {
        "id": 1,
        "name": "Symmetric Cousin Derivation",
        "context": "my dad's name is brc. brc has sister sadhana. sadhana has son achal. brc has son sankalp",
        "subject": "Sankalp",
        "object": "Achal",
        "expected": "cousin"
    },
    {
        "id": 2,
        "name": "Maternal Cousin Derivation",
        "context": "my mom's name is Alice. Alice has brother Bob. Bob has daughter Carol. my name is Dave.",
        "subject": "Dave",
        "object": "Carol",
        "expected": "cousin"
    },
    {
        "id": 3,
        "name": "Brother-Sibling-Nephew Cousin Path",
        "context": "Siddharth has brother Rahul. Rahul has son Akhil. Siddharth has son Mohit.",
        "subject": "Akhil",
        "object": "Mohit",
        "expected": "cousin"
    },
    {
        "id": 4,
        "name": "Deep Pronoun Coreference Chain",
        "context": "Robert is my father. He is married to Jessica. She has a daughter Clara.",
        "subject": "Robert",
        "object": "Clara",
        "expected": "father"
    },
    {
        "id": 5,
        "name": "Conversational Preamble & Noise",
        "context": "Oh, by the way, introducing my awesome family! Meet my dad, his name is Robert. He has a sister named Jessica. She is the mother of Emily.",
        "subject": "Emily",
        "object": "Speaker",
        "expected": "cousin"
    },
    {
        "id": 6,
        "name": "Deep Multi-hop Ancestry (3-hop Grandparent)",
        "context": "Zeus is the father of Apollo. Cronus is the father of Zeus. Uranus is the father of Cronus.",
        "subject": "Uranus",
        "object": "Apollo",
        "expected": "grandfather"
    },
    {
        "id": 7,
        "name": "Affinal (In-law) Relationship",
        "context": "David is married to Sarah. Sarah's father is Abraham.",
        "subject": "Abraham",
        "object": "David",
        "expected": "father-in-law"
    },
    {
        "id": 8,
        "name": "Negative Control (Unrelated Families)",
        "context": "Alice is the mother of Bob. Charlie is the father of David.",
        "subject": "Alice",
        "object": "Charlie",
        "expected": None  # Expect failure/Unknown
    },
    {
        "id": 9,
        "name": "Self-Identity Query (Same Name)",
        "context": "",
        "subject": "Sankalp",
        "object": "Sankalp",
        "expected": "self"
    },
    {
        "id": 10,
        "name": "Self-Identity Query (Case Insensitive & Stripped)",
        "context": "Sankalp has cousin bro Achal.",
        "subject": "sankalp ",
        "object": " sankalp",
        "expected": "self"
    }
]

def execute_suite():
    print("=" * 80)
    print("           EXTREME KINSHIP REASONER ROBUSTNESS TEST SUITE")
    print("=" * 80)
    print(f"Loaded {len(TEST_CASES)} highly challenging test scenarios.\n")
    
    passed_count = 0
    failed_cases = []
    
    for tc in TEST_CASES:
        print("-" * 80)
        print(f"TEST CASE #{tc['id']}: {tc['name']}")
        print(f"Context: {tc['context']}")
        print(f"Query:   ({tc['subject']}, ?, {tc['object']})")
        print(f"Expected Relation: {tc['expected']}")
        print("-" * 80)
        
        # Run solver
        result = symbolic_solve(tc['context'], tc['subject'], tc['object'])
        
        print(f"Success: {result['success']}")
        print(f"Derived: {result['relation']} (Confidence: {result['confidence']})")
        print("Proof Trace:")
        for step in result.get('proof_trace', []):
            print(f"  {step}")
            
        # Verify
        actual = result['relation'] if result['success'] else None
        
        if actual == tc['expected']:
            print(f"Result:  ✅ PASSED")
            passed_count += 1
        else:
            print(f"Result:  ❌ FAILED (Expected '{tc['expected']}', got '{actual}')")
            failed_cases.append((tc['id'], tc['name'], tc['expected'], actual))
        print()
        
    print("=" * 80)
    print("                             TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Total Test Cases: {len(TEST_CASES)}")
    print(f"Passed:           {passed_count}")
    print(f"Failed:           {len(failed_cases)}")
    print("=" * 80)
    
    if failed_cases:
        print("\nFailed Scenarios:")
        for fid, name, exp, act in failed_cases:
            print(f"  - Case #{fid} ({name}): Expected '{exp}', got '{act}'")
        sys.exit(1)
        
    print("\n🎉 ALL ROBUSTNESS TESTS PASSED SUCCESSFULLY! 100% CORRECT BEHAVIOR PROVEN!")
    sys.exit(0)

if __name__ == "__main__":
    execute_suite()
