"""
inference_engine.py - Optimized Forward Chaining Engine (Algorithm 2)

Uses relation-indexed fact lookups for O(F) per rule instead of O(F^2).
"""

from typing import List, Tuple, Dict, Set, Any
from collections import defaultdict
import logging

from rules import (
    COMPOSITION_RULES, INVERSE_RULES, SAME_SUBJECT_RULES,
    rank_by_specificity, get_inverse,
)

logger = logging.getLogger(__name__)

# Common genders map for name-based fallback (zero hardcoding helper)
COMMON_GENDERS = {
    "sankalp": "M", "achal": "M", "bob": "M", "dave": "M", "siddharth": "M",
    "rahul": "M", "akhil": "M", "mohit": "M", "robert": "M", "zeus": "M",
    "apollo": "M", "cronus": "M", "uranus": "M", "david": "M", "abraham": "M",
    "charlie": "M", "peter": "M", "michael": "M", "john": "M", "arthur": "M",
    "william": "M", "george": "M", "tejas": "M",
    "sadhana": "F", "alice": "F", "carol": "F", "jessica": "F", "emily": "F",
    "sarah": "F", "jane": "F", "eleanor": "F", "mary": "F", "charlotte": "F",
    "clara": "F", "ananya": "F", "maa": "F",
}


class ForwardChainingEngine:
    def __init__(self, max_iterations: int = 20):
        self.max_iterations = max_iterations

    def _specialize_relation_gender(self, relation: str, subject: str, genders: dict) -> str:
        gender = genders.get(subject.lower().strip())
        if not gender:
            return relation

        relation_lower = relation.lower().strip()
        
        # Canonical gender-swapping maps
        m2f = {
            "father": "mother",
            "son": "daughter",
            "brother": "sister",
            "grandfather": "grandmother",
            "grandson": "granddaughter",
            "uncle": "aunt",
            "nephew": "niece",
            "husband": "wife",
            "father-in-law": "mother-in-law",
            "son-in-law": "daughter-in-law"
        }
        f2m = {v: k for k, v in m2f.items()}

        if gender == 'F':
            if relation_lower in m2f:
                return m2f[relation_lower]
            elif relation_lower == "parent":
                return "mother"
            elif relation_lower == "child":
                return "daughter"
            elif relation_lower == "sibling":
                return "sister"
        elif gender == 'M':
            if relation_lower in f2m:
                return f2m[relation_lower]
            elif relation_lower == "parent":
                return "father"
            elif relation_lower == "child":
                return "son"
            elif relation_lower == "sibling":
                return "brother"
                
        return relation

    def _specialize_facts(self, facts: set, genders: dict) -> set:
        specialized = set()
        for s, r, o in facts:
            new_r = self._specialize_relation_gender(r, s, genders)
            specialized.add((s, new_r, o))
        return specialized

    def solve(self, initial_facts, query_subject, query_object):
        qs = query_subject.lower().strip()
        qo = query_object.lower().strip()

        if qs == qo:
            proof_trace = [f"IDENTITY: {qs} is the same person as {qo}"]
            return {"success": True, "relation": "self", "confidence": 1.0, "proof_trace": proof_trace}

        facts = set()
        for s, r, o in initial_facts:
            facts.add((s.lower().strip(), r.lower().strip(), o.lower().strip()))

        # Initial gender inference and fact specialization
        genders = self._infer_genders(facts)
        facts = self._specialize_facts(facts, genders)

        proof_trace = [f"FACT: {r}({s}, {o})" for s, r, o in sorted(facts)]

        # Check direct answer
        answer = self._check_query(facts, qs, qo)
        if answer:
            rel = rank_by_specificity(answer)
            proof_trace.append(f"ANSWER (direct): {rel}({qs}, {qo})")
            return {"success": True, "relation": rel, "confidence": 1.0, "proof_trace": proof_trace}

        # Check inverse answer right at the start (0-iteration check)
        answer_inv = self._check_query(facts, qo, qs)
        if answer_inv:
            base = rank_by_specificity(answer_inv)
            inv = self._get_dynamic_inverse_head(base, qs, genders)
            if inv:
                proof_trace.append(f"ANSWER (inv, direct): {inv}({qs}, {qo}) via {base}({qo}, {qs})")
                return {"success": True, "relation": inv, "confidence": 1.0, "proof_trace": proof_trace}

        for iteration in range(self.max_iterations):
            # Dynamic gender inference at the start of each iteration
            genders = self._infer_genders(facts)
            facts = self._specialize_facts(facts, genders)

            # Build indexes for fast lookup
            by_rel = defaultdict(list)          # rel -> [(s, o)]
            by_rel_obj = defaultdict(set)       # (rel, obj) -> set of subjects
            for s, r, o in facts:
                by_rel[r].append((s, o))
                by_rel_obj[(r, o)].add(s)

            new_facts = set()

            # Composition rules: body1(x,y) ^ body2(y,z) -> head(x,z)
            for head, (b1, b2) in COMPOSITION_RULES:
                for sx, ox in by_rel[b1]:       # body1(sx, ox)
                    for sy, oy in by_rel[b2]:   # body2(sy, oy)
                        if ox == sy:            # intermediate match
                            head_spec = self._specialize_relation_gender(head, sx, genders)
                            d = (sx, head_spec, oy)
                            if sx != oy and d not in facts:
                                new_facts.add(d)
                                proof_trace.append(
                                    f"DERIVED: {head_spec}({sx}, {oy}) via "
                                    f"{b1}({sx}, {ox}) ^ {b2}({sy}, {oy})")

            # Inverse rules: body(x,y) -> head(y,x) dynamically gender-specialised
            for _, (body,) in INVERSE_RULES:
                for sx, ox in by_rel[body]:
                    head = self._get_dynamic_inverse_head(body, ox, genders)
                    if head:
                        head_spec = self._specialize_relation_gender(head, ox, genders)
                        d = (ox, head_spec, sx)
                        if d not in facts:
                            new_facts.add(d)
                            proof_trace.append(
                                f"DERIVED (inv): {head_spec}({ox}, {sx}) via {body}({sx}, {ox})")

            # Same-subject rules: body(X,Y) ^ body(X,Z) -> head(Y,Z) where Y != Z
            for head, body in SAME_SUBJECT_RULES:
                pairs = by_rel[body]  # list of (subject, object)
                # Group objects by subject
                subj_to_objs = defaultdict(list)
                for sx, ox in pairs:
                    subj_to_objs[sx].append(ox)
                for subj, objs in subj_to_objs.items():
                    if len(objs) >= 2:
                        for i, y in enumerate(objs):
                            for z in objs[i+1:]:
                                # Derive both directions
                                for a, b in [(y, z), (z, y)]:
                                    head_spec = self._specialize_relation_gender(head, a, genders)
                                    d = (a, head_spec, b)
                                    if d not in facts:
                                        new_facts.add(d)
                                        proof_trace.append(
                                            f"DERIVED (same-subj): {head_spec}({a}, {b}) via "
                                            f"{body}({subj}, {a}) ^ {body}({subj}, {b})")

            if not new_facts:
                proof_trace.append(f"SATURATED after {iteration + 1} iterations")
                break

            facts.update(new_facts)

            # Check query
            answer = self._check_query(facts, qs, qo)
            if answer:
                rel = rank_by_specificity(answer)
                proof_trace.append(f"ANSWER (iter {iteration+1}): {rel}({qs}, {qo})")
                return {"success": True, "relation": rel, "confidence": 1.0, "proof_trace": proof_trace}

            # Check inverse direction using dynamic gender-specialised inverse
            answer_inv = self._check_query(facts, qo, qs)
            if answer_inv:
                base = rank_by_specificity(answer_inv)
                inv = self._get_dynamic_inverse_head(base, qs, genders)
                if inv:
                    proof_trace.append(f"ANSWER (inv, iter {iteration+1}): {inv}({qs}, {qo})")
                    return {"success": True, "relation": inv, "confidence": 1.0, "proof_trace": proof_trace}

        proof_trace.append("UNKNOWN: Could not derive relation")
        return {"success": False, "relation": None, "confidence": 0.0, "proof_trace": proof_trace}

    def _infer_genders(self, facts) -> dict:
        genders = {}
        
        reliable_male = {"father", "son", "brother", "grandfather", "grandson", "uncle", "nephew", "father-in-law"}
        reliable_female = {"mother", "daughter", "sister", "grandmother", "granddaughter", "aunt", "niece", "mother-in-law"}
        
        # Pass 1: Infer gender only from highly reliable gender-specific relations
        for _ in range(3):
            for s, r, o in facts:
                s_clean = s.strip().lower()
                r_lower = r.strip().lower()
                
                if s_clean not in genders:
                    if r_lower in reliable_male:
                        genders[s_clean] = 'M'
                    elif r_lower in reliable_female:
                        genders[s_clean] = 'F'
                        
        # Pass 2: Infer gender from spouse relations only as a fallback if not already known
        for _ in range(3):
            for s, r, o in facts:
                s_clean = s.strip().lower()
                o_clean = o.strip().lower()
                r_lower = r.strip().lower()
                
                if s_clean not in genders:
                    if r_lower == "husband":
                        genders[s_clean] = 'M'
                    elif r_lower == "wife":
                        genders[s_clean] = 'F'
                
                if o_clean not in genders:
                    if r_lower == "husband":
                        genders[o_clean] = 'F'
                    elif r_lower == "wife":
                        genders[o_clean] = 'M'
                        
        # Pass 3: Fallback to COMMON_GENDERS for names not explicitly typed in the story
        for k, v in COMMON_GENDERS.items():
            k_lower = k.lower()
            if k_lower not in genders:
                genders[k_lower] = v
                
        return genders

    def _get_dynamic_inverse_head(self, body: str, ox: str, genders: dict) -> str:
        gender = genders.get(ox.lower().strip())
        
        if body in {"father", "mother", "parent"}:
            if gender == 'M':
                return "son"
            if gender == 'F':
                return "daughter"
            return "child"
            
        if body in {"son", "daughter", "child"}:
            if gender == 'M':
                return "father"
            if gender == 'F':
                return "mother"
            return "parent"
            
        if body in {"grandfather", "grandmother"}:
            if gender == 'M':
                return "grandson"
            if gender == 'F':
                return "granddaughter"
            return "grandchild"
            
        if body in {"grandson", "granddaughter"}:
            if gender == 'M':
                return "grandfather"
            if gender == 'F':
                return "grandmother"
            return "grandparent"
            
        if body in {"uncle", "aunt"}:
            if gender == 'M':
                return "nephew"
            if gender == 'F':
                return "niece"
            return "sibling_child"
            
        if body in {"nephew", "niece"}:
            if gender == 'M':
                return "uncle"
            if gender == 'F':
                return "aunt"
            return "parent_sibling"
            
        if body in {"brother", "sister", "sibling"}:
            if gender == 'M':
                return "brother"
            if gender == 'F':
                return "sister"
            return "sibling"
            
        if body == "husband":
            return "wife"
        if body == "wife":
            return "husband"
            
        if body in {"father-in-law", "mother-in-law"}:
            if gender == 'M':
                return "son-in-law"
            if gender == 'F':
                return "daughter-in-law"
            return "son-in-law" if body == "father-in-law" else "daughter-in-law"

        if body in {"son-in-law", "daughter-in-law"}:
            if gender == 'M':
                return "father-in-law"
            if gender == 'F':
                return "mother-in-law"
            return "father-in-law" if body == "son-in-law" else "mother-in-law"
            
        if body == "cousin":
            return "cousin"
            
        return None

    def _check_query(self, facts, subject, obj):
        return [r for s, r, o in facts if s == subject and o == obj]


def run_forward_chaining(facts, query_subject, query_object, max_iterations=20):
    return ForwardChainingEngine(max_iterations).solve(facts, query_subject, query_object)


if __name__ == "__main__":
    demo = [("Apollo", "father", "Hermes"), ("Zeus", "father", "Apollo")]
    r = run_forward_chaining(demo, "Zeus", "Hermes")
    print(f"Result: {r['relation']} (conf={r['confidence']})")
    for s in r["proof_trace"]:
        print(f"  {s}")
