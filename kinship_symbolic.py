"""
kinship_symbolic.py - Fact Extraction and Pronoun Coreference Resolution

Implements:
  - 25 compiled regex patterns for extracting kinship facts from narratives
    (Section 3.2.6, Listing 1)
  - Algorithm 1: Pronoun Coreference Resolution with last_entity tracking
  - Full symbolic reasoning pipeline: text -> facts -> forward chaining -> answer

References:
  - Paper Section 3.2.3, Algorithm 1
  - Paper Section 3.2.6, Listing 1
"""

import re
from typing import List, Tuple, Optional, Dict, Any
import logging

from inference_engine import run_forward_chaining

logger = logging.getLogger(__name__)

class ResolvedFactsList(list):
    def __init__(self, iterable=(), speaker_name=None):
        super().__init__(iterable)
        self.speaker_name = speaker_name

COMMON_GENDERS = {
    "sankalp": "M", "achal": "M", "bob": "M", "dave": "M", "siddharth": "M",
    "rahul": "M", "akhil": "M", "mohit": "M", "robert": "M", "zeus": "M",
    "apollo": "M", "cronus": "M", "uranus": "M", "david": "M", "abraham": "M",
    "charlie": "M", "peter": "M", "michael": "M", "john": "M", "arthur": "M",
    "william": "M", "george": "M",
    "sadhana": "F", "alice": "F", "carol": "F", "jessica": "F", "emily": "F",
    "sarah": "F", "jane": "F", "eleanor": "F", "mary": "F", "charlotte": "F",
    "clara": "F",
}

FIRST_PERSON_PRONOUNS = {"my", "i", "me", "mine", "myself"}
MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
THIRD_PERSON_PRONOUNS = MALE_PRONOUNS | FEMALE_PRONOUNS

RELATION_WORDS = (
    "father|mother|son|daughter|brother|sister|"
    "grandfather|grandmother|grandson|granddaughter|"
    "uncle|aunt|nephew|niece|husband|wife|"
    "father-in-law|mother-in-law|son-in-law|daughter-in-law|"
    "parent|child|sibling|cousin"
)

def clean_name(name: str) -> str:
    """Clean up extracted name by removing leading/trailing stop words, possessives, and spaces."""
    name = name.strip()
    name = re.sub(r"'s$", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"'$", "", name, flags=re.IGNORECASE).strip()
    
    leading_strip = [
        "also", "and", "but", "then", "so", "meet", "introducing", "this is", "that is",
        "to", "with", "for", "about", "of", "the", "a", "an", "called", "named",
        "this guy", "guy", "baby boy named", "baby boy", "baby girl named", "baby girl",
        "she recently", "he recently", "recently", "just", "our", "my"
    ]
    for w in leading_strip:
        pattern = rf"^{w}\s+"
        text_len = len(name)
        name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
        while len(name) < text_len:
            text_len = len(name)
            name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
        
    trailing_strip = [
        "is", "are", "was", "were", "has", "had", "have", "married", "introduced", "called", "named", "recently"
    ]
    for w in trailing_strip:
        pattern = rf"\s+{w}$"
        name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
        
    return name.strip()

def normalize_synonyms(text: str) -> str:
    """Normalize colloquial and plural relation terms to their canonical CLUTRR equivalents."""
    replacements = {
        r"\bfather[- ]in[- ]laws?\b": "father-in-law",
        r"\bmother[- ]in[- ]laws?\b": "mother-in-law",
        r"\bson[- ]in[- ]laws?\b": "son-in-law",
        r"\bdaughter[- ]in[- ]laws?\b": "daughter-in-law",
        
        r"\bgrand[- ]fathers?\b": "grandfather",
        r"\bgrand[- ]mothers?\b": "grandmother",
        r"\bgrand[- ]sons?\b": "grandson",
        r"\bgrand[- ]daughters?\b": "granddaughter",

        r"\bcousin[- ](?:bros?|brothers?|sis(?:ses)?|sises?|sisters?)\b": "cousin",
        r"\bdads?\b": "father",
        r"\bdadas?\b": "father",
        r"\bdaddies\b": "father",
        r"\bpapas?\b": "father",
        r"\bpops?\b": "father",
        r"\bmoms?\b": "mother",
        r"\bmommies\b": "mother",
        r"\bmommy\b": "mother",
        r"\bmamas?\b": "mother",
        r"\bmammas?\b": "mother",
        r"\bmums?\b": "mother",
        r"\bhubbies\b": "husband",
        r"\bhubby\b": "husband",
        r"\bwifey\b": "wife",
        r"\bwives\b": "wife",
        r"\bbros?\b": "brother",
        r"\bsis(?:ses)?\b": "sister",
        r"\bsises?\b": "sister",
        r"\bgrandpas?\b": "grandfather",
        r"\bgranddads?\b": "grandfather",
        r"\bgramps\b": "grandfather",
        r"\bgrandmas?\b": "grandmother",
        r"\bgrannies\b": "grandmother",
        r"\bgranny\b": "grandmother",
        r"\bkids?\b": "child",
        r"\bchildren\b": "child",
        r"\bspouses?\b": "husband",
        r"\baunties?\b": "aunt",
        r"\baunty\b": "aunt",
        r"\bfathers\b": "father",
        r"\bmothers\b": "mother",
        r"\bsons\b": "son",
        r"\bdaughters\b": "daughter",
        r"\bbrothers\b": "brother",
        r"\bsisters\b": "sister",
        r"\bgrandfathers\b": "grandfather",
        r"\bgrandmothers\b": "grandmother",
        r"\bgrandsons\b": "grandson",
        r"\bgrandsubjs\b": "grandson",
        r"\bgranddaughters\b": "granddaughter",
        r"\buncles\b": "uncle",
        r"\baunts\b": "aunt",
        r"\bnephews\b": "nephew",
        r"\bnieces\b": "niece",
        r"\bhusbands\b": "husband",
        r"\bparents\b": "parent",
        r"\bsiblings\b": "sibling",
        r"\bcousins\b": "cousin"
    }
    for pattern, canonical in replacements.items():
        text = re.sub(pattern, canonical, text, flags=re.IGNORECASE)
    return text

def normalise_text(text: str) -> str:
    """Replace smart/curly quotes with straight ASCII equivalents and normalize synonyms."""
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201C", '"')
    text = text.replace("\u201D", '"')
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = normalize_synonyms(text)
    return text

FACT_PATTERNS = [
    (re.compile(
        r"(?:my\s+name\s+is|i\s+am|i'm)\s+([\w]+)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: ("Speaker", "identity", clean_name(m.group(1)))),
    (re.compile(
        r"([\w]+)\s+is\s+my\s+name(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: ("Speaker", "identity", clean_name(m.group(1)))),

    (re.compile(
        rf"(?:meet|this\s+is)\s+my\s+({RELATION_WORDS})(?:\s+named|\s*,\s+his\s+name\s+is|\s*,\s+her\s+name\s+is|\s*,)\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(2)), m.group(1).strip().lower(), "Speaker")),
    (re.compile(
        rf"(?:meet|this\s+is)\s+(?:his|her)\s+({RELATION_WORDS})(?:\s+named|\s*,\s+his\s+name\s+is|\s*,\s+her\s+name\s+is|\s*,)\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(2)), m.group(1).strip().lower(), "__LAST_ENTITY__")),
    (re.compile(
        rf"(?:my|his|her)\s+({RELATION_WORDS})'s\s+name\s+is\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(2)), m.group(1).strip().lower(), "Speaker" if "my" in m.group(0).lower() else "__LAST_ENTITY__")),

    (re.compile(
        rf"([\w\s]+?)\s+is\s+the\s+({RELATION_WORDS})\s+of\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(2).strip().lower(), clean_name(m.group(3)))),

    (re.compile(
        rf"([\w\s]+?)\s+is\s+([\w]+)'s\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(3).strip().lower(), clean_name(m.group(2)))),

    (re.compile(
        rf"([\w\s]+?)\s+is\s+my\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(2).strip().lower(), "Speaker")),

    (re.compile(
        rf"([\w\s]+?)\s+is\s+(?:his|her)\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(2).strip().lower(), "__LAST_ENTITY__")),

    (re.compile(
        rf"([\w]+)'s\s+({RELATION_WORDS})\s+is\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(3)), m.group(2).strip().lower(), clean_name(m.group(1)))),

    (re.compile(
        rf"my\s+({RELATION_WORDS})\s+is\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(2)), m.group(1).strip().lower(), "Speaker")),

    (re.compile(
        rf"(?:his|her)\s+({RELATION_WORDS})\s+is\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(2)), m.group(1).strip().lower(), "__LAST_ENTITY__")),

    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+brothers(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), "brother", clean_name(m.group(2)))),

    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+sisters(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), "sister", clean_name(m.group(2)))),

    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+siblings(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), "sibling", clean_name(m.group(2)))),

    (re.compile(
        r"([\w\s]+?)\s+(?:and\s+([\w\s]+?)\s+are\s+married|married\s+([\w\s]+?))(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), "husband", clean_name(m.group(2) or m.group(3)))),

    (re.compile(
        rf"([\w\s]+?)\s+has\s+(?:a\s+)?({RELATION_WORDS})(?:\s+named|\s*,)?\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(3)), m.group(2).strip().lower(), clean_name(m.group(1)))),

    (re.compile(
        r"([\w\s]+?)\s+(?:gave\s+birth\s+to|had)\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), "parent", clean_name(m.group(2)))),

    (re.compile(
        rf"([\w\s]+?),\s+([\w]+)'s\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(3).strip().lower(), clean_name(m.group(2)))),

    (re.compile(
        rf"([\w\s]+?),\s+the\s+({RELATION_WORDS})\s+of\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(2).strip().lower(), clean_name(m.group(3)))),

    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+parent\s+and\s+child(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), "parent", clean_name(m.group(2)))),

    (re.compile(
        rf"([\w\s]+?),\s+({RELATION_WORDS})\s+of\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(2).strip().lower(), clean_name(m.group(3)))),

    (re.compile(
        rf"([\w]+)'s\s+({RELATION_WORDS}),\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(3)), m.group(2).strip().lower(), clean_name(m.group(1)))),

    (re.compile(
        rf"([\w\s]+?)\s+is\s+a\s+({RELATION_WORDS})\s+to\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(2).strip().lower(), clean_name(m.group(3)))),

    (re.compile(
        rf"([\w\s]+?)\s+is\s+the\s+({RELATION_WORDS})\s+to\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(2).strip().lower(), clean_name(m.group(3)))),

    (re.compile(
        rf"([\w\s]+?)\s+was\s+the\s+({RELATION_WORDS})\s+of\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(2).strip().lower(), clean_name(m.group(3)))),

    (re.compile(
        rf"([\w\s]+?)\s+was\s+([\w]+)'s\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), m.group(3).strip().lower(), clean_name(m.group(2)))),

    (re.compile(
        rf"([\w\s]+?)\s+calls\s+([\w\s]+?)\s+(?:his|her)\s+({RELATION_WORDS})(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(2)), m.group(3).strip().lower(), clean_name(m.group(1)))),

    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+are\s+husband\s+and\s+wife(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), "husband", clean_name(m.group(2)))),

    (re.compile(
        r"([\w\s]+?)\s+and\s+([\w\s]+?)\s+had\s+([\w\s]+?)(?:\.|,|;|!|\?|$)",
        re.IGNORECASE
    ), lambda m: (clean_name(m.group(1)), "parent", clean_name(m.group(3)))),
]

PLURAL_MAP = {
    "fathers": "father", "mothers": "mother", "sons": "son", "daughters": "daughter",
    "brothers": "brother", "sisters": "sister", "grandfathers": "grandfather",
    "grandmothers": "grandmother", "grandsons": "grandson", "granddaughters": "granddaughter",
    "uncles": "uncle", "aunts": "aunt", "nephews": "nephew", "nieces": "niece",
    "husbands": "husband", "wives": "wife", "parents": "parent", "children": "child",
    "siblings": "sibling", "cousins": "cousin"
}

SEMANTIC_STOP_WORDS = {
    "is", "the", "of", "a", "to", "and", "are", "have", "has", "had", "named", "called", 
    "who", "they", "his", "her", "their", "our", "him", "she", "he", "it", "its", "them",
    "name", "names", "relationship", "relation", "family", "relative", "relatives", 
    "between", "with", "this", "that", "these", "those", "was", "were", "been", "being",
    "in", "on", "at", "for", "with", "about", "by", "from", "up", "out", "into"
}

def normalize_rel_word(word: str) -> str:
    word_lower = word.lower().strip()
    rel_set = set(RELATION_WORDS.split("|"))
    if word_lower in rel_set:
        return word_lower
    if word_lower in PLURAL_MAP:
        return PLURAL_MAP[word_lower]
    return None

def semantic_extract_sentence(sentence: str, known_entities: set = None) -> list:
    """Extract facts from a single sentence using semantic proximity and grammatical heuristics."""
    sentence_clean = re.sub(r"[^\w\s'-]", " ", sentence)
    words = [w for w in sentence_clean.split() if w]
    if not words:
        return []

    relations = []
    for i, w in enumerate(words):
        norm_rel = normalize_rel_word(w)
        if norm_rel:
            relations.append((norm_rel, i))

    if not relations:
        return []

    entities = []
    stop_words = {
        "what", "how", "who", "whom", "is", "are", "do", "does", "can", "could", "the", "a", "an",
        "tell", "me", "between", "and", "related", "relationship", "relation", "family", "was",
        "were", "been", "being", "have", "has", "had", "named", "called", "whose", "this", "that",
        "these", "those", "in", "on", "at", "for", "with", "about", "by", "from", "up", "out",
        "into", "of", "to", "they", "their", "them", "our", "us", "we", "you", "your", "yours",
        "it", "its", "so", "then", "there", "here", "just", "very", "too", "also", "nice", "meet",
        "hi", "hello", "hey", "please", "would", "should", "could", "will", "shall", "be", "is", "are",
        "am", "was", "were", "go", "went", "gone", "see", "saw", "seen", "say", "said",
        "younger", "older", "elder", "young", "old", "twin", "my", "his", "her", "their", "our",
        "introduced", "introduces", "introducing", "met", "meets", "meeting", "married", "marries",
        "marrying", "loves", "likes", "lives", "works", "goes", "went", "has", "have", "had", "is", "was",
        "were", "be", "been", "being", "awesome", "member", "person", "people", "friend", "friends",
        "colleague", "colleagues", "classmate", "classmates", "roommate", "roommates"
    }

    proper_names = []
    for i, w in enumerate(words):
        w_clean = w.strip("',.!?()[]{}").strip()
        w_lower = w_clean.lower()
        if not w_clean or len(w_clean) < 2:
            continue
        name = clean_name(w_clean)
        name_lower = name.lower()
        if name_lower in stop_words or normalize_rel_word(name_lower):
            continue
        if w_clean[0].isupper() or (known_entities and name.title() in known_entities):
            proper_names.append((name.title(), i))

    for i, w in enumerate(words):
        w_clean = w.strip("',.!?()[]{}").strip()
        w_lower = w_clean.lower()
        if not w_clean or len(w_clean) < 2:
            continue
        
        name = clean_name(w_clean)
        name_lower = name.lower()
        
        if name_lower in FIRST_PERSON_PRONOUNS:
            entities.append(("Speaker", i))
            continue
            
        if name_lower in THIRD_PERSON_PRONOUNS:
            preceding = [p for p in proper_names if p[1] < i]
            if preceding:
                entities.append((preceding[-1][0], i))
            else:
                entities.append((w_clean, i))
            continue
            
        if normalize_rel_word(name_lower):
            continue
            
        if known_entities and name.title() in known_entities:
            entities.append((name.title(), i))
            continue
            
        if w_clean[0].isupper() and name_lower not in stop_words:
            entities.append((name.title(), i))
            continue
            
        if name_lower not in stop_words and len(name) > 2:
            entities.append((name.title(), i))

    if not entities:
        return []

    facts = []
    seen = set()

    def determine_direction(ent1, idx1, ent2, idx2, rel, r_idx):
        """Grammatical heuristics to resolve relation(subj, obj) from context words."""
        def is_possessive(idx):
            if idx < 0 or idx >= len(words):
                return False
            w = words[idx].lower()
            return w.endswith("'s") or w.endswith("'") or w in {"my", "his", "her", "their", "our"} or (idx < len(words) - 1 and words[idx+1] == "'s")

        words_l = [w.lower() for w in words]

        r_word = words[r_idx].lower()
        if r_word.endswith("'s") or r_word.endswith("'") or (r_idx > 0 and words[r_idx-1].lower() in {"my", "his", "her", "their", "our"}):
            if idx1 < r_idx and idx2 > r_idx:
                return ent2, ent1
            if idx2 < r_idx and idx1 > r_idx:
                return ent1, ent2

        if is_possessive(idx1):
            return ent2, ent1
        if is_possessive(idx2):
            return ent1, ent2

        of_before_ent2 = False
        for j in range(max(0, idx2 - 3), idx2):
            if words_l[j] == "of":
                of_before_ent2 = True
                break
        
        of_before_ent1 = False
        for j in range(max(0, idx1 - 3), idx1):
            if words_l[j] == "of":
                of_before_ent1 = True
                break

        if of_before_ent2:
            return ent1, ent2
        if of_before_ent1:
            return ent2, ent1

        to_before_ent2 = False
        for j in range(max(0, idx2 - 3), idx2):
            if words_l[j] == "to":
                to_before_ent2 = True
                break
                
        to_before_ent1 = False
        for j in range(max(0, idx1 - 3), idx1):
            if words_l[j] == "to":
                to_before_ent1 = True
                break

        if to_before_ent2:
            return ent1, ent2
        if to_before_ent1:
            return ent2, ent1

        has_verb_near_ent1 = False
        for j in range(max(0, idx1), min(len(words), r_idx)):
            if words_l[j] in {"has", "had", "have", "owns", "calls"}:
                has_verb_near_ent1 = True
                break
                
        has_verb_near_ent2 = False
        for j in range(max(0, idx2), min(len(words), r_idx)):
            if words_l[j] in {"has", "had", "have", "owns", "calls"}:
                has_verb_near_ent2 = True
                break

        if has_verb_near_ent1:
            return ent2, ent1
        if has_verb_near_ent2:
            return ent1, ent2

        if abs(idx1 - r_idx) < abs(idx2 - r_idx):
            return ent1, ent2
        else:
            return ent2, ent1

    for rel, r_idx in relations:
        if len(entities) >= 2:
            sorted_ents = sorted(entities, key=lambda e: abs(e[1] - r_idx))
            ent1, idx1 = sorted_ents[0]
            ent2, idx2 = sorted_ents[1]

            if ent1 == ent2:
                continue

            subj, obj = determine_direction(ent1, idx1, ent2, idx2, rel, r_idx)

            key = (subj, rel, obj)
            if key not in seen and subj != obj:
                seen.add(key)
                facts.append(key)

        elif len(entities) == 1:
            ent, idx = entities[0]
            has_first = any(w.lower() in FIRST_PERSON_PRONOUNS for w in words)
            if has_first and ent != "Speaker":
                key = (ent, rel, "Speaker")
                if key not in seen:
                    seen.add(key)
                    facts.append(key)
            elif ent != "Speaker":
                key = (ent, rel, "Speaker")
                if key not in seen:
                    seen.add(key)
                    facts.append(key)

    return facts

def extract_facts_from_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract kinship fact triples from a narrative text using both strict regex patterns
    and a robust semantic fallback chunker for absolute conversational coverage.
    """
    text = normalise_text(text)

    facts_with_pos = []
    seen = set()

    known_entities = set()
    for word in re.findall(r"\b[A-Z][a-z]+\b", text):
        known_entities.add(word)

    for pattern, extractor in FACT_PATTERNS:
        for match in pattern.finditer(text):
            try:
                s, r, o = extractor(match)
                if s and r and o:
                    key = (s.lower(), r.lower(), o.lower())
                    if key not in seen:
                        seen.add(key)
                        facts_with_pos.append((match.start(), (s, r, o)))
            except Exception:
                continue

    sentences = re.split(r"[.!?\n]+", text)
    current_search_pos = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_pos = text.find(sent, current_search_pos)
        if sent_pos != -1:
            current_search_pos = sent_pos + len(sent)
        else:
            sent_pos = 0
            
        semantic_facts = semantic_extract_sentence(sent, known_entities)
        for s, r, o in semantic_facts:
            key = (s.lower(), r.lower(), o.lower())
            if key not in seen:
                seen.add(key)
                facts_with_pos.append((sent_pos, (s, r, o)))

    facts_with_pos.sort(key=lambda x: x[0])

    return [fact for pos, fact in facts_with_pos]

def resolve_coreferences(
    facts: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
    """
    Algorithm 1: Pronoun Coreference Resolution with dynamic gender and identity resolution.
    """
    genders = {}
    for k, v in COMMON_GENDERS.items():
        genders[k.title()] = v
        genders[k.lower()] = v

    reliable_male = {"father", "son", "brother", "grandfather", "grandson", "uncle", "nephew", "father-in-law"}
    reliable_female = {"mother", "daughter", "sister", "grandmother", "granddaughter", "aunt", "niece", "mother-in-law"}
    male_relations = reliable_male | {"husband"}
    female_relations = reliable_female | {"wife"}

    for _ in range(3):
        for s, r, o in facts:
            s_clean = s.strip()
            r_lower = r.lower()
            
            if not (s_clean.lower() in FIRST_PERSON_PRONOUNS or s_clean.lower() in THIRD_PERSON_PRONOUNS or s == "__LAST_ENTITY__"):
                if s_clean.lower() not in COMMON_GENDERS and s_clean.lower() not in {k.lower() for k in genders}:
                    if r_lower in reliable_male:
                        genders[s_clean] = 'M'
                    elif r_lower in reliable_female:
                        genders[s_clean] = 'F'

    for _ in range(3):
        for s, r, o in facts:
            s_clean = s.strip()
            o_clean = o.strip()
            r_lower = r.lower()
            
            if not (s_clean.lower() in FIRST_PERSON_PRONOUNS or s_clean.lower() in THIRD_PERSON_PRONOUNS or s == "__LAST_ENTITY__"):
                if s_clean.lower() not in COMMON_GENDERS and s_clean.lower() not in {k.lower() for k in genders}:
                    if r_lower == "husband":
                        genders[s_clean] = 'M'
                    elif r_lower == "wife":
                        genders[s_clean] = 'F'
            
            if not (o_clean.lower() in FIRST_PERSON_PRONOUNS or o_clean.lower() in THIRD_PERSON_PRONOUNS or o == "__LAST_ENTITY__"):
                if o_clean.lower() not in COMMON_GENDERS and o_clean.lower() not in {k.lower() for k in genders}:
                    if r_lower == "husband":
                        genders[o_clean] = 'F'
                    elif r_lower == "wife":
                        genders[o_clean] = 'M'

    last_male = None
    last_female = None
    last_any = None
    resolved = []

    for subject, relation, obj in facts:
        s = subject
        o = obj

        s_lower = s.lower().strip()
        if s_lower in FIRST_PERSON_PRONOUNS:
            s = "Speaker"
        elif s_lower in MALE_PRONOUNS:
            s = last_male if last_male is not None else (last_any if last_any is not None else "Speaker")
        elif s_lower in FEMALE_PRONOUNS:
            s = last_female if last_female is not None else (last_any if last_any is not None else "Speaker")
        elif s == "__LAST_ENTITY__":
            s = last_any if last_any is not None else "Speaker"

        o_lower = o.lower().strip()
        if o_lower in FIRST_PERSON_PRONOUNS:
            o = "Speaker"
        elif o_lower in MALE_PRONOUNS:
            o = last_male if last_male is not None else (last_any if last_any is not None else "Speaker")
        elif o_lower in FEMALE_PRONOUNS:
            o = last_female if last_female is not None else (last_any if last_any is not None else "Speaker")
        elif o == "__LAST_ENTITY__":
            o = last_any if last_any is not None else "Speaker"

        for ent in [s, o]:
            ent_resolved_lower = ent.lower().strip()
            if ent_resolved_lower not in (FIRST_PERSON_PRONOUNS | THIRD_PERSON_PRONOUNS | {"__last_entity__", "speaker"}):
                last_any = ent
                gender = genders.get(ent)
                if gender == 'M':
                    last_male = ent
                elif gender == 'F':
                    last_female = ent
                else:
                    if ent == s:
                        if relation.lower() in male_relations:
                            last_male = ent
                        elif relation.lower() in female_relations:
                            last_female = ent
                        else:
                            if last_male is None:
                                last_male = ent
                            if last_female is None:
                                last_female = ent

        resolved.append((s, relation, o))

    speaker_name = None
    for s, r, o in resolved:
        if r == "identity":
            if s == "Speaker":
                speaker_name = o
            elif o == "Speaker":
                speaker_name = s

    if not speaker_name:
        speaker_parents = set()
        speaker_spouses = set()
        for s, r, o in resolved:
            s_clean = s.strip()
            o_clean = o.strip()
            r_lower = r.lower().strip()
            if s_clean == "Speaker":
                if r_lower in {"son", "daughter", "child"}:
                    speaker_parents.add(o_clean.lower())
                elif r_lower in {"husband", "wife", "spouse"}:
                    speaker_spouses.add(o_clean.lower())
            elif o_clean == "Speaker":
                if r_lower in {"father", "mother", "parent"}:
                    speaker_parents.add(s_clean.lower())
                elif r_lower in {"husband", "wife", "spouse"}:
                    speaker_spouses.add(s_clean.lower())

        if speaker_parents or speaker_spouses:
            candidates = []
            for ent in genders:
                if ent.lower() == "speaker":
                    continue
                ent_parents = set()
                ent_spouses = set()
                for s, r, o in resolved:
                    s_clean = s.strip()
                    o_clean = o.strip()
                    r_lower = r.lower().strip()
                    if s_clean.lower() == ent.lower():
                        if r_lower in {"son", "daughter", "child"}:
                            ent_parents.add(o_clean.lower())
                        elif r_lower in {"husband", "wife", "spouse"}:
                            ent_spouses.add(o_clean.lower())
                    elif o_clean.lower() == ent.lower():
                        if r_lower in {"father", "mother", "parent"}:
                            ent_parents.add(s_clean.lower())
                        elif r_lower in {"husband", "wife", "spouse"}:
                            ent_spouses.add(s_clean.lower())

                if (speaker_parents and ent_parents == speaker_parents) or (speaker_spouses and ent_spouses == speaker_spouses):
                    assert_diff = False
                    for s, r, o in resolved:
                        s_l = s.lower().strip()
                        o_l = o.lower().strip()
                        r_l = r.lower().strip()
                        if (s_l == ent.lower() and o_l == "speaker") or (s_l == "speaker" and o_l == ent.lower()):
                            if r_l in {"brother", "sister", "sibling", "husband", "wife", "spouse"}:
                                assert_diff = True
                                break
                    if not assert_diff:
                        candidates.append(ent)
            uniq_candidates = list(set(c.title() for c in candidates))
            if len(uniq_candidates) == 1:
                speaker_name = uniq_candidates[0]

    if speaker_name:
        final_resolved = []
        for s, r, o in resolved:
            if r == "identity":
                continue
            s_new = speaker_name if s == "Speaker" else s
            o_new = speaker_name if o == "Speaker" else o
            final_resolved.append((s_new, r, o_new))
        resolved = final_resolved

    resolved_genders = {}
    for k, v in COMMON_GENDERS.items():
        resolved_genders[k.title()] = v
        resolved_genders[k.lower()] = v

    for _ in range(3):
        for s, r, o in resolved:
            s_clean = s.strip()
            r_lower = r.lower()
            if s_clean.lower() not in COMMON_GENDERS and s_clean.lower() not in {k.lower() for k in resolved_genders}:
                if r_lower in reliable_male:
                    resolved_genders[s_clean] = 'M'
                elif r_lower in reliable_female:
                    resolved_genders[s_clean] = 'F'

    for _ in range(3):
        for s, r, o in resolved:
            s_clean = s.strip()
            o_clean = o.strip()
            r_lower = r.lower()
            if s_clean.lower() not in COMMON_GENDERS and s_clean.lower() not in {k.lower() for k in resolved_genders}:
                if r_lower == "husband":
                    resolved_genders[s_clean] = 'M'
                elif r_lower == "wife":
                    resolved_genders[s_clean] = 'F'
            if o_clean.lower() not in COMMON_GENDERS and o_clean.lower() not in {k.lower() for k in resolved_genders}:
                if r_lower == "husband":
                    resolved_genders[o_clean] = 'F'
                elif r_lower == "wife":
                    resolved_genders[o_clean] = 'M'

    final_resolved_spouse = []
    for s, r, o in resolved:
        r_lower = r.lower().strip()
        if r_lower in {"husband", "wife"}:
            s_gender = resolved_genders.get(s)
            o_gender = resolved_genders.get(o)
            
            if s_gender == 'F' and o_gender == 'M':
                final_resolved_spouse.append((s, "wife", o))
            elif s_gender == 'M' and o_gender == 'F':
                final_resolved_spouse.append((s, "husband", o))
            elif s_gender == 'M' and o_gender == 'M':
                final_resolved_spouse.append((s, r, o))
            elif s_gender == 'F' and o_gender == 'F':
                final_resolved_spouse.append((s, r, o))
            elif s_gender == 'F':
                final_resolved_spouse.append((s, "wife", o))
            elif s_gender == 'M':
                final_resolved_spouse.append((s, "husband", o))
            elif o_gender == 'M':
                final_resolved_spouse.append((s, "wife", o))
            elif o_gender == 'F':
                final_resolved_spouse.append((s, "husband", o))
            else:
                final_resolved_spouse.append((s, r, o))
        else:
            final_resolved_spouse.append((s, r, o))
    resolved = final_resolved_spouse

    final_seen = set()
    cleaned_resolved = []
    for s, r, o in resolved:
        key = (s.lower().strip(), r.lower().strip(), o.lower().strip())
        if key not in final_seen:
            final_seen.add(key)
            cleaned_resolved.append((s, r, o))

    return ResolvedFactsList(cleaned_resolved, speaker_name=speaker_name)

def symbolic_solve(
    narrative: str,
    query_subject: str,
    query_object: str,
) -> Dict[str, Any]:
    """
    Full symbolic reasoning pipeline:
      1. Extract facts from narrative using regex patterns
      2. Resolve pronoun coreferences (Algorithm 1)
      3. Run forward chaining inference (Algorithm 2)
      4. Return answer with proof trace
    """
    qs = query_subject.lower().strip()
    qo = query_object.lower().strip()
    if qs == qo and qs and qs != "?":
        proof_trace = [f"IDENTITY: {query_subject} is the same person as {query_object}"]
        return {
            "success": True,
            "relation": "self",
            "confidence": 1.0,
            "proof_trace": proof_trace,
        }

    raw_facts = extract_facts_from_text(narrative)
    logger.info(f"Extracted {len(raw_facts)} raw facts from narrative")

    resolved_facts = resolve_coreferences(raw_facts)
    logger.info(f"After coreference resolution: {len(resolved_facts)} facts")

    if not resolved_facts:
        return {
            "success": False,
            "relation": None,
            "confidence": 0.0,
            "proof_trace": ["No facts could be extracted from the narrative."],
        }

    speaker_name = getattr(resolved_facts, 'speaker_name', None)

    if speaker_name:
        qs_lower = query_subject.lower().strip()
        qo_lower = query_object.lower().strip()
        if qs_lower in FIRST_PERSON_PRONOUNS | {"speaker", "me"}:
            query_subject = speaker_name
        if qo_lower in FIRST_PERSON_PRONOUNS | {"speaker", "me"}:
            query_object = speaker_name

    result = run_forward_chaining(resolved_facts, query_subject, query_object)

    return result

if __name__ == "__main__":
    story = (
        "Apollo is Hermes' father. "
        "Zeus is Apollo's father. "
        "Zeus's father is Cronus. "
        "Cronus's father is Uranus. "
        "Gaia is the mother of Uranus."
    )
    print("Facts extracted:")
    facts = extract_facts_from_text(story)
    for f in facts:
        print(f"  {f}")

    print()
    result = symbolic_solve(story, "Gaia", "Hermes")
    print(f"\nResult: {result['relation']} (confidence={result['confidence']})")
    print("\nProof trace:")
    for step in result["proof_trace"]:
        print(f"  {step}")
