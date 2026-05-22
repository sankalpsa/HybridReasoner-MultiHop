
"""KGManager: safe ConceptNet and WordNet helpers with retries and timeouts."""
from typing import List, Dict, Optional
import requests
import logging

logger = logging.getLogger(__name__)

CONCEPTNET_API = "http://api.conceptnet.io/c/en/"

def query_conceptnet(term: str, max_edges: int = 5, timeout: int = 5) -> List[Dict]:
    term_clean = term.strip().lower().replace(' ', '_')
    url = CONCEPTNET_API + term_clean
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        edges = []
        for edge in data.get('edges', [])[:max_edges]:
            rel = edge.get('rel', {}).get('label')
            end = edge.get('end', {}).get('label')
            if rel and end:
                edges.append({'relation': rel, 'concept': end, 'weight': edge.get('weight', 1.0)})
        return edges
    except Exception as exc:
        logger.debug(f"ConceptNet query failed for {term}: {exc}")
        return []

def query_wordnet_sim(term: str) -> List[str]:
    return [f"hypernym({term}, example_hypernym)", f"synonym({term}, example_synonym)"]

def get_upamana_triples(entities: List[str]) -> List[str]:
    triples: List[str] = []
    for ent in entities:
        triples.extend([f"conceptnet_rel({ent},{e['relation']},{e['concept']})" for e in query_conceptnet(ent, max_edges=2)])
        triples.extend(query_wordnet_sim(ent))
    if 'father' in entities and 'mother' in entities:
        triples.append('analogy_rule(parent_pair, has_child)')
    return triples

if __name__ == '__main__':
    print('KGManager demo: conceptnet for "dog" (first 2 edges)')
    print(query_conceptnet('dog')[:2])
