"""
rules.py — Static library of 47 Horn clause rules for kinship reasoning.

Each rule is a tuple: (head_relation, (body_rel_1, body_rel_2))
Meaning: body_rel_1(x, y) ∧ body_rel_2(y, z) ⟹ head_relation(x, z)

For inverse rules (single-body): (head_relation, (body_rel,))
Meaning: body_rel(x, y) ⟹ head_relation(y, x)

For gender-specialisation rules: (head_relation, (body_rel, gender_predicate))
Meaning: body_rel(x, y) ∧ gender_predicate(x) ⟹ head_relation(x, y)

Categories (per Section 3.2.2 of the paper):
  1. Composition rules — chain two different relations to derive a third
  2. Transitivity rules — chain two instances of the same relation
  3. Inverse rules — derive the converse direction
  4. Gender-specific rules — speciate generic relations by gender
  5. In-law rules — derive affinal (by-marriage) relations

References:
  - Paper Section 3.2.2, Equation 10-11
  - 18 CLUTRR kinship predicates (Table 2)
"""

# ──────────────────────────────────────────────────────────────────────
# The 18 kinship relation labels used in CLUTRR (Table 2)
# ──────────────────────────────────────────────────────────────────────
KINSHIP_RELATIONS = [
    "father", "mother", "son", "daughter",
    "brother", "sister",
    "grandfather", "grandmother", "grandson", "granddaughter",
    "uncle", "aunt", "nephew", "niece",
    "husband", "wife",
    "father-in-law", "mother-in-law",
    "cousin",
]

# ──────────────────────────────────────────────────────────────────────
# Specificity ranking (Equation 12)
# Higher score = more specific; used when multiple valid answers exist
# ──────────────────────────────────────────────────────────────────────
SPECIFICITY = {
    "father-in-law": 15, "mother-in-law": 15, "son-in-law": 15, "daughter-in-law": 15,
    "father": 10, "mother": 10, "son": 10, "daughter": 10,
    "brother": 8, "sister": 8,
    "husband": 8, "wife": 8,
    "uncle": 5, "aunt": 5, "nephew": 5, "niece": 5, "cousin": 5,
    "grandfather": 3, "grandmother": 3, "grandson": 3, "granddaughter": 3,
    "parent": 1, "child": 1, "sibling": 1,
}

# ──────────────────────────────────────────────────────────────────────
# Inverse relation map — used by Algorithm 2 (line 21-22)
# If we derive r(s, o) and the query asks for (o, s), return inverse(r)
# ──────────────────────────────────────────────────────────────────────
INVERSE_MAP = {
    "father": "son",        "mother": "daughter",
    "son": "father",        "daughter": "mother",
    "brother": "brother",   "sister": "sister",
    "grandfather": "grandson",    "grandmother": "granddaughter",
    "grandson": "grandfather",    "granddaughter": "grandmother",
    "uncle": "nephew",      "aunt": "niece",
    "nephew": "uncle",      "niece": "aunt",
    "husband": "wife",      "wife": "husband",
    "father-in-law": "son-in-law",  "mother-in-law": "daughter-in-law",
    "parent": "child",      "child": "parent",
    "sibling": "sibling",
    "cousin": "cousin",
}

# ──────────────────────────────────────────────────────────────────────
# Gender knowledge — used by gender-specific rules
# ──────────────────────────────────────────────────────────────────────
MALE_RELATIONS = {"father", "son", "brother", "grandfather", "grandson",
                  "uncle", "nephew", "husband", "father-in-law"}
FEMALE_RELATIONS = {"mother", "daughter", "sister", "grandmother",
                    "granddaughter", "aunt", "niece", "wife", "mother-in-law"}

# ──────────────────────────────────────────────────────────────────────
# 47 Horn clause rules
# Format: (consequent, (antecedent_1, antecedent_2))
#   body_1(x, y) ∧ body_2(y, z) ⟹ head(x, z)
#
# For inverse/unary rules the tuple has a single element and a flag.
# We store everything uniformly as composition rules for the forward
# chaining engine.
# ──────────────────────────────────────────────────────────────────────

COMPOSITION_RULES = [
    # ── Category 1: Multi-generational composition ──────────────────
    # father(x,y) ∧ father(y,z) → grandfather(x,z)
    ("grandfather", ("father", "father")),
    # mother(x,y) ∧ father(y,z) → grandmother(x,z)   [x is female → grandmother]
    ("grandmother", ("mother", "father")),
    # father(x,y) ∧ mother(y,z) → grandfather(x,z)   [x is male → grandfather]
    ("grandfather", ("father", "mother")),
    # mother(x,y) ∧ mother(y,z) → grandmother(x,z)
    ("grandmother", ("mother", "mother")),

    # son(x,y) ∧ son(y,z) → grandson(x,z)
    ("grandson", ("son", "son")),
    # son(x,y) ∧ daughter(y,z) → granddaughter(x,z)
    ("granddaughter", ("son", "daughter")),
    # daughter(x,y) ∧ son(y,z) → grandson(x,z)
    ("grandson", ("daughter", "son")),
    # daughter(x,y) ∧ daughter(y,z) → granddaughter(x,z)
    ("granddaughter", ("daughter", "daughter")),

    # ── Category 2: Uncle / Aunt via sibling + parent ───────────────
    # brother(x,y) ∧ father(y,z) → uncle(x,z)  [x is brother of father y → uncle of z]
    ("uncle", ("brother", "father")),
    # brother(x,y) ∧ mother(y,z) → uncle(x,z)  [x is brother of mother y → uncle of z]
    ("uncle", ("brother", "mother")),
    # sister(x,y) ∧ father(y,z) → aunt(x,z)  [x is sister of father y → aunt of z]
    ("aunt", ("sister", "father")),
    # sister(x,y) ∧ mother(y,z) → aunt(x,z)  [x is sister of mother y → aunt of z]
    ("aunt", ("sister", "mother")),

    # ── Category 3: Nephew / Niece via child + sibling ──────────────
    # son(x,y) ∧ brother(y,z) → nephew(x,z)  [x is son of y, y's brother is z → x is nephew of z]
    ("nephew", ("son", "brother")),
    # son(x,y) ∧ sister(y,z) → nephew(x,z)  [x is son of y, y's sister is z → x is nephew of z]
    ("nephew", ("son", "sister")),
    # daughter(x,y) ∧ brother(y,z) → niece(x,z)  [x is daughter of y, y's brother is z → x is niece of z]
    ("niece", ("daughter", "brother")),
    # daughter(x,y) ∧ sister(y,z) → niece(x,z)  [x is daughter of y, y's sister is z → x is niece of z]
    ("niece", ("daughter", "sister")),

    # ── Category 4: Parent ↔ child via father/mother/son/daughter ───
    # father(x,y) → parent(x,y)   [gender generalisation]
    # Expressed as composition: we handle these via GENERALISATION_RULES below

    # ── Category 5: Sibling via shared parent ───────────────────────
    # son(x,y) ∧ father(y,z) → brother(z,x)  ... not standard
    # Instead: father(x,y) ∧ father(x,z) → sibling(y,z) needs same first arg
    # We handle sibling derivation through parent-child chains:
    # brother(x,y) ∧ brother(y,z) → brother(x,z)  [transitivity]
    ("brother", ("brother", "brother")),
    # sister(x,y) ∧ sister(y,z) → sister(x,z)
    ("sister", ("sister", "sister")),
    # brother(x,y) ∧ sister(y,z) → sister(x,z)  [x's brother's sister]
    ("sister", ("brother", "sister")),
    # sister(x,y) ∧ brother(y,z) → brother(x,z)
    ("brother", ("sister", "brother")),

    # ── Category 6: In-law rules ────────────────────────────────────
    # father(x,y) ∧ husband(y,z) → father-in-law(x,z) [x is father of y, y is husband of z]
    ("father-in-law", ("father", "husband")),
    ("father-in-law", ("father", "wife")),
    ("mother-in-law", ("mother", "husband")),
    ("mother-in-law", ("mother", "wife")),
    
    # son/daughter in law (inverse of above)
    ("son-in-law", ("husband", "daughter")),
    ("son-in-law", ("husband", "son")),
    ("daughter-in-law", ("wife", "son")),
    ("daughter-in-law", ("wife", "daughter")),

    # spouse + sibling → in-law sibling
    # husband(x,y) ∧ brother(y,z) → brother-in-law(z,x)
    # We don't have brother-in-law in the 18 relations, so we skip or
    # map to the closest. The paper says 18 relations; we stay within those.

    # ── Category 7: Parent inheritance through siblings ────────────
    # If x is the father of y, and y is the brother/sister of z,
    # then x is also the father of z (shared parent).
    ("father", ("father", "brother")),
    ("father", ("father", "sister")),
    # If x is the mother of y, and y is the brother/sister of z,
    # then x is also the mother of z (shared parent).
    ("mother", ("mother", "brother")),
    ("mother", ("mother", "sister")),

    # ── Category 8: Grandparent-over-parent composition ──────────────
    # grandmother(x,y) ∧ father(y,z) → grandmother(x,z)
    # If x is grandmother of y, and y is the father of z, then x is
    # grandmother of z too (chaining grandparent down through parent)
    ("grandmother", ("grandmother", "father")),
    ("grandmother", ("grandmother", "mother")),
    ("grandfather", ("grandfather", "father")),
    ("grandfather", ("grandfather", "mother")),

    # ── Category 9: Spouse chains ───────────────────────────────────
    # husband(x,y) ∧ son(y,z) → son(x,z)   [husband's son = my son]
    ("son", ("husband", "son")),
    # husband(x,y) ∧ daughter(y,z) → daughter(x,z)
    ("daughter", ("husband", "daughter")),
    # wife(x,y) ∧ son(y,z) → son(x,z)
    ("son", ("wife", "son")),
    # wife(x,y) ∧ daughter(y,z) → daughter(x,z)
    ("daughter", ("wife", "daughter")),

    # ── Category 10: Grandparent inheritance through siblings ──────
    # If x is grandfather of y, and y is brother/sister of z,
    # then x is also grandfather of z (shared grandparent).
    ("grandfather", ("grandfather", "brother")),
    ("grandfather", ("grandfather", "sister")),
    ("grandmother", ("grandmother", "brother")),
    ("grandmother", ("grandmother", "sister")),

    # ── Category 11: Son/daughter of sibling = nephew/niece ─────────
    # son(x,y) ∧ brother(y,z) → nephew... direction matters
    # These are covered by nephew/niece rules above from the other direction

    # ── Category 12: Cousin composition ──────────────────────────────
    ("cousin", ("son", "aunt")),
    ("cousin", ("daughter", "aunt")),
    ("cousin", ("son", "uncle")),
    ("cousin", ("daughter", "uncle")),
    ("cousin", ("nephew", "father")),
    ("cousin", ("nephew", "mother")),
    ("cousin", ("niece", "father")),
    ("cousin", ("niece", "mother")),
    ("cousin", ("brother", "cousin")),
    ("cousin", ("sister", "cousin")),
    ("cousin", ("sibling", "cousin")),
    ("cousin", ("cousin", "brother")),
    ("cousin", ("cousin", "sister")),
    ("cousin", ("cousin", "sibling")),
    ("cousin", ("son", "cousin")),
    ("cousin", ("daughter", "cousin")),
    ("cousin", ("child", "cousin")),
    ("cousin", ("cousin", "son")),
    ("cousin", ("cousin", "daughter")),
    ("cousin", ("cousin", "child")),

    # ── Category 13: Spouse parent composition ───────────────────────
    # husband(x,y) ∧ mother(y,z) → father(x,z)
    ("father", ("husband", "mother")),
    # husband(x,y) ∧ father(y,z) → father(x,z)
    ("father", ("husband", "father")),
    # wife(x,y) ∧ mother(y,z) → mother(x,z)
    ("mother", ("wife", "mother")),
    # wife(x,y) ∧ father(y,z) → mother(x,z)
    ("mother", ("wife", "father")),

    # ── Category 14: Multi-hop Uncle/Aunt Grandparent compositions ─
    ("grandfather", ("father", "uncle")),
    ("grandfather", ("father", "aunt")),
    ("grandmother", ("mother", "uncle")),
    ("grandmother", ("mother", "aunt")),

    # ── Category 15: Sibling child rules ─────────────────────────────
    ("son", ("brother", "son")),
    ("son", ("sister", "son")),
    ("son", ("sibling", "son")),
    ("daughter", ("brother", "daughter")),
    ("daughter", ("sister", "daughter")),
    ("daughter", ("sibling", "daughter")),
    ("son", ("brother", "daughter")),
    ("daughter", ("sister", "son")),

    # ── Category 16: Sibling grandchild rules ────────────────────────
    ("grandson", ("brother", "grandson")),
    ("grandson", ("sister", "grandson")),
    ("grandson", ("sibling", "grandson")),
    ("granddaughter", ("brother", "granddaughter")),
    ("granddaughter", ("sister", "granddaughter")),
    ("granddaughter", ("sibling", "granddaughter")),
    ("grandson", ("brother", "granddaughter")),
    ("granddaughter", ("sister", "grandson")),

    # ── Category 17: Nephew/Niece grandchild rules ───────────────────
    ("grandson", ("nephew", "son")),
    ("grandson", ("nephew", "daughter")),
    ("granddaughter", ("niece", "son")),
    ("granddaughter", ("niece", "daughter")),

    # ── Category 18: Sibling uncle/aunt rules ────────────────────────
    ("uncle", ("uncle", "brother")),
    ("uncle", ("uncle", "sister")),
    ("uncle", ("uncle", "sibling")),
    ("aunt", ("aunt", "brother")),
    ("aunt", ("aunt", "sister")),
    ("aunt", ("aunt", "sibling")),

    # ── Category 19: Sibling nephew/niece rules ──────────────────────
    ("nephew", ("brother", "nephew")),
    ("nephew", ("sister", "nephew")),
    ("nephew", ("sibling", "nephew")),
    ("niece", ("brother", "niece")),
    ("niece", ("sister", "niece")),
    ("niece", ("sibling", "niece")),
    ("nephew", ("brother", "niece")),
    ("niece", ("sister", "nephew")),
]

# ──────────────────────────────────────────────────────────────────────
# Inverse rules — single-body implications
# body(x, y) ⟹ head(y, x)
# Stored separately; the forward chaining engine applies these too.
# ──────────────────────────────────────────────────────────────────────
INVERSE_RULES = [
    ("son", ("father",)),       # father(x,y) → son(y,x)
    ("daughter", ("mother",)),  # mother(x,y) → daughter(y,x)
    ("father", ("son",)),       # son(x,y) → father(y,x)
    ("mother", ("daughter",)),  # daughter(x,y) → mother(y,x)
    ("grandson", ("grandfather",)),
    ("granddaughter", ("grandmother",)),
    ("grandfather", ("grandson",)),
    ("grandmother", ("granddaughter",)),
    ("nephew", ("uncle",)),
    ("niece", ("aunt",)),
    ("uncle", ("nephew",)),
    ("aunt", ("niece",)),
    ("wife", ("husband",)),
    ("husband", ("wife",)),
    ("brother", ("brother",)),
    ("sister", ("sister",)),
    ("sibling", ("sibling",)),
    ("cousin", ("cousin",)),
    ("son-in-law", ("father-in-law",)),
    ("daughter-in-law", ("mother-in-law",)),
    ("father-in-law", ("son-in-law",)),
    ("mother-in-law", ("daughter-in-law",)),
]

# ──────────────────────────────────────────────────────────────────────
# Same-subject rules — derive a relation between two objects that share
# the same subject in different facts.
# Pattern: body(X, Y) ∧ body(X, Z) ⟹ head(Y, Z)  where Y ≠ Z
# This handles sibling derivation from shared parents.
# ──────────────────────────────────────────────────────────────────────
SAME_SUBJECT_RULES = [
    # father(X, Y) ∧ father(X, Z) → brother(Y, Z)
    ("brother", "father"),
    # mother(X, Y) ∧ mother(X, Z) → sister(Y, Z)
    ("sister", "mother"),
]

def get_all_rules():
    """Return all rules in a unified format for the forward chaining engine."""
    return COMPOSITION_RULES, INVERSE_RULES

def get_specificity(relation: str) -> int:
    """Return the specificity score for a relation (Equation 12)."""
    return SPECIFICITY.get(relation, 0)

def rank_by_specificity(relations: list) -> str:
    """Given a list of candidate relations, return the most specific one."""
    if not relations:
        return None
    return max(relations, key=lambda r: SPECIFICITY.get(r, 0))

def get_inverse(relation: str) -> str:
    """Return the inverse of a relation, or None if unknown."""
    return INVERSE_MAP.get(relation)
