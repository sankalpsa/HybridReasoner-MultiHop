
"""LogicLearner: extract facts and rules from CSV training data.

Improvements:
- clearer parsing, robust to missing columns
- returns facts as sorted list and rules as list of tuples
"""
from typing import List, Dict, Tuple, Set, Any
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

class LogicLearner:
    @staticmethod
    def _extract_facts_from_story(story: str) -> Set[str]:
        facts: Set[str] = set()
        if not story or not isinstance(story, str):
            return facts
        patterns = [
            (r"(\w+)'s\s+(\w+)\s+(?:is|are)\s+(\w+)", lambda m: f"{m.group(2).lower()}({m.group(1)},{m.group(3)})"),
            (r"(\w+)\s+(?:is|are)\s+(\w+)'s\s+(\w+)", lambda m: f"{m.group(3).lower()}({m.group(2)},{m.group(1)})"),
            (r"(\w+)\s+has\s+a\s+(\w+)\s+named\s+(\w+)", lambda m: f"{m.group(2).lower()}({m.group(1)},{m.group(3)})"),
            (r"(\w+)\s+and\s+(\w+)\s+(?:are|is)\s+(\w+)", lambda m: f"{m.group(3).rstrip('s').lower()}({m.group(1)},{m.group(2)})"),
        ]
        for pat, fn in patterns:
            for match in re.finditer(pat, story, re.IGNORECASE):
                try:
                    f = fn(match)
                    if ',' in f:
                        facts.add(f)
                except Exception:
                    continue
        return facts

    @staticmethod
    def _extract_gender_facts(gstr: str) -> Set[str]:
        facts: Set[str] = set()
        if not gstr or not isinstance(gstr, str):
            return facts
        for entry in gstr.split(','):
            entry = entry.strip()
            if ':' in entry:
                name, gen = entry.split(':', 1)
                gen = gen.strip().lower()
                if gen in ('male', 'female'):
                    facts.add(f"gender({name.strip()},{gen})")
            else:
                for g in ('male', 'female'):
                    if entry.lower().endswith(g):
                        name = entry[:-len(g)].strip()
                        if name:
                            facts.add(f"gender({name},{g})")
        return facts

    @staticmethod
    def extract_logic_from_csv(path: str = 'clutrr_train.csv') -> Dict[str, Any]:
        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {len(df)} rows from {path}")
        except Exception as exc:
            logger.error(f"Could not open CSV: {exc}")
            return {'facts': [], 'rules': []}

        all_facts: Set[str] = set()
        learned_rules: Dict[str, Set[Tuple[str, ...]]] = {}
        processed = 0

        for _, row in df.iterrows():
            target = None
            for c in ('target_text','target','relation','label'):
                if c in row and pd.notna(row[c]):
                    target = str(row[c]).strip().lower()
                    if target and target != 'nan':
                        break
            if not target or target == 'nan':
                continue

            for gc in ('gender','genders','gender_info'):
                if gc in row and pd.notna(row[gc]):
                    all_facts.update(LogicLearner._extract_gender_facts(str(row[gc])))

            for sc in ('story','clean_story','text','context'):
                if sc in row and pd.notna(row[sc]):
                    all_facts.update(LogicLearner._extract_facts_from_story(str(row[sc])))
                    break

            for ec in ('edge_types','edges','path_type','rule_path'):
                if ec in row and pd.notna(row[ec]):
                    edge_str = str(row[ec])
                    if '-' in edge_str:
                        bodies = [e.strip().lower() for e in edge_str.split('-') if e.strip()]
                    elif ',' in edge_str and ec != 'edges':
                        bodies = [e.strip().lower() for e in edge_str.split(',') if e.strip()]
                    else:
                        bodies = [edge_str.strip().lower()]
                    if bodies:
                        learned_rules.setdefault(target, set()).add(tuple(bodies))
                    break

            processed += 1
            if processed % 100 == 0:
                logger.info(f"Processed {processed} rows... facts={len(all_facts)} rules={len(learned_rules)}")

        final_rules = [(h, list(b)) for h, bodies in learned_rules.items() for b in bodies]
        return {'facts': sorted(list(all_facts)), 'rules': final_rules}
