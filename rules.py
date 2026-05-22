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

KINSHIP_RELATIONS = [
    "father", "mother", "son", "daughter",
    "brother", "sister",
    "grandfather", "grandmother", "grandson", "granddaughter",
    "uncle", "aunt", "nephew", "niece",
    "husband", "wife",
    "father-in-law", "mother-in-law",
    "cousin",
]

SPECIFICITY = {
    "father-in-law": 15, "mother-in-law": 15, "son-in-law": 15, "daughter-in-law": 15,
    "father": 10, "mother": 10, "son": 10, "daughter": 10,
    "brother": 8, "sister": 8,
    "husband": 8, "wife": 8,
    "uncle": 5, "aunt": 5, "nephew": 5, "niece": 5, "cousin": 5,
    "grandfather": 3, "grandmother": 3, "grandson": 3, "granddaughter": 3,
    "parent": 1, "child": 1, "sibling": 1,
}

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

MALE_RELATIONS = {"father", "son", "brother", "grandfather", "grandson",
                  "uncle", "nephew", "husband", "father-in-law"}
FEMALE_RELATIONS = {"mother", "daughter", "sister", "grandmother",
                    "granddaughter", "aunt", "niece", "wife", "mother-in-law"}

COMPOSITION_RULES = [
    ("grandfather", ("father", "father")),
    ("grandmother", ("mother", "father")),
    ("grandfather", ("father", "mother")),
    ("grandmother", ("mother", "mother")),

    ("grandson", ("son", "son")),
    ("granddaughter", ("son", "daughter")),
    ("grandson", ("daughter", "son")),
    ("granddaughter", ("daughter", "daughter")),

    ("uncle", ("brother", "father")),
    ("uncle", ("brother", "mother")),
    ("aunt", ("sister", "father")),
    ("aunt", ("sister", "mother")),

    ("nephew", ("son", "brother")),
    ("nephew", ("son", "sister")),
    ("niece", ("daughter", "brother")),
    ("niece", ("daughter", "sister")),

    ("brother", ("brother", "brother")),
    ("sister", ("sister", "sister")),
    ("sister", ("brother", "sister")),
    ("brother", ("sister", "brother")),

    ("father-in-law", ("father", "husband")),
    ("father-in-law", ("father", "wife")),
    ("mother-in-law", ("mother", "husband")),
    ("mother-in-law", ("mother", "wife")),
    
    ("son-in-law", ("husband", "daughter")),
    ("son-in-law", ("husband", "son")),
    ("daughter-in-law", ("wife", "son")),
    ("daughter-in-law", ("wife", "daughter")),

    ("father", ("father", "brother")),
    ("father", ("father", "sister")),
    ("mother", ("mother", "brother")),
    ("mother", ("mother", "sister")),

    ("grandmother", ("grandmother", "father")),
    ("grandmother", ("grandmother", "mother")),
    ("grandfather", ("grandfather", "father")),
    ("grandfather", ("grandfather", "mother")),

    ("son", ("husband", "son")),
    ("daughter", ("husband", "daughter")),
    ("son", ("wife", "son")),
    ("daughter", ("wife", "daughter")),

    ("grandfather", ("grandfather", "brother")),
    ("grandfather", ("grandfather", "sister")),
    ("grandmother", ("grandmother", "brother")),
    ("grandmother", ("grandmother", "sister")),

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

    ("father", ("husband", "mother")),
    ("father", ("husband", "father")),
    ("mother", ("wife", "mother")),
    ("mother", ("wife", "father")),

    ("grandfather", ("father", "uncle")),
    ("grandfather", ("father", "aunt")),
    ("grandmother", ("mother", "uncle")),
    ("grandmother", ("mother", "aunt")),

    ("son", ("brother", "son")),
    ("son", ("sister", "son")),
    ("son", ("sibling", "son")),
    ("daughter", ("brother", "daughter")),
    ("daughter", ("sister", "daughter")),
    ("daughter", ("sibling", "daughter")),
    ("son", ("brother", "daughter")),
    ("daughter", ("sister", "son")),

    ("grandson", ("brother", "grandson")),
    ("grandson", ("sister", "grandson")),
    ("grandson", ("sibling", "grandson")),
    ("granddaughter", ("brother", "granddaughter")),
    ("granddaughter", ("sister", "granddaughter")),
    ("granddaughter", ("sibling", "granddaughter")),
    ("grandson", ("brother", "granddaughter")),
    ("granddaughter", ("sister", "grandson")),

    ("grandson", ("nephew", "son")),
    ("grandson", ("nephew", "daughter")),
    ("granddaughter", ("niece", "son")),
    ("granddaughter", ("niece", "daughter")),

    ("uncle", ("uncle", "brother")),
    ("uncle", ("uncle", "sister")),
    ("uncle", ("uncle", "sibling")),
    ("aunt", ("aunt", "brother")),
    ("aunt", ("aunt", "sister")),
    ("aunt", ("aunt", "sibling")),

    ("nephew", ("brother", "nephew")),
    ("nephew", ("sister", "nephew")),
    ("nephew", ("sibling", "nephew")),
    ("niece", ("brother", "niece")),
    ("niece", ("sister", "niece")),
    ("niece", ("sibling", "niece")),
    ("nephew", ("brother", "niece")),
    ("niece", ("sister", "nephew")),
]

INVERSE_RULES = [
    ("son", ("father",)),
    ("daughter", ("mother",)),
    ("father", ("son",)),
    ("mother", ("daughter",)),
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

SAME_SUBJECT_RULES = [
    ("brother", "father"),
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
