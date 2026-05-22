# inference_wrapper.py  (fast prefix-based fallback)
import numpy as np
import json
from pathlib import Path
import torch
from collections import defaultdict

MODELS = Path("models")
ENTITY_NPY = MODELS / "entity_emb.npy"
REL_NPY = MODELS / "relation_emb.npy"
ENT_MAP = Path("entity2idx.json")
REL_MAP = Path("rel2idx.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load maps with prefix index -----------------
def load_maps_and_prefix(klen=3):
    ent2idx = {}
    rel2idx = {}
    if ENT_MAP.exists():
        with open(ENT_MAP, "r", encoding="utf8") as fh:
            ent2idx = json.load(fh)
    if REL_MAP.exists():
        with open(REL_MAP, "r", encoding="utf8") as fh:
            rel2idx = json.load(fh)
    # normalize keys to lowercase strings and ensure int values
    ent2idx = {str(k).strip().lower(): int(v) for k, v in ent2idx.items()}
    rel2idx = {str(k).strip().lower(): int(v) for k, v in rel2idx.items()}
    idx2rel = {int(v): str(k) for k, v in rel2idx.items()}

    # build simple prefix map -> list of entity keys sharing first klen chars
    prefix_map = defaultdict(list)
    for key in ent2idx.keys():
        pref = key[:klen]
        prefix_map[pref].append(key)
    return ent2idx, rel2idx, idx2rel, prefix_map

_ent2idx, _rel2idx, _idx2rel, _prefix_map = load_maps_and_prefix(klen=3)

# ----------------- Load embeddings -----------------
_entity_emb = None
_relation_emb = None

if ENTITY_NPY.exists():
    _entity_emb = np.load(str(ENTITY_NPY))
if REL_NPY.exists():
    _relation_emb = np.load(str(REL_NPY))

# fallback: try to read from checkpoint if not found
if (_entity_emb is None) or (_relation_emb is None):
    CKPT = MODELS / "neuro_symbolic_joint.pt"
    if CKPT.exists():
        try:
            ck = torch.load(str(CKPT), map_location="cpu")
            # best-effort extraction as before (kept small)
            if _entity_emb is None:
                ent = None
                if isinstance(ck, dict):
                    ent = ck.get("entity_emb") or ck.get("entity_embeddings")
                    if ent is None and "transE" in ck and isinstance(ck["transE"], dict):
                        trans = ck["transE"]
                        ent = trans.get("ent") or trans.get("ent.weight")
                    if hasattr(ent, "detach"):
                        _entity_emb = ent.detach().cpu().numpy()
                # final fallback None
            if _relation_emb is None:
                rel = None
                if isinstance(ck, dict):
                    rel = ck.get("relation_emb") or ck.get("relation_embeddings")
                    if rel is None and "transE" in ck and isinstance(ck["transE"], dict):
                        trans = ck["transE"]
                        rel = trans.get("rel") or trans.get("rel.weight")
                    if hasattr(rel, "detach"):
                        _relation_emb = rel.detach().cpu().numpy()
        except Exception:
            pass

# convert to torch tensors on device for scoring
_entity_emb_t = None
_relation_emb_t = None
if _entity_emb is not None:
    _entity_emb_t = torch.tensor(_entity_emb, dtype=torch.float32).to(device)
if _relation_emb is not None:
    _relation_emb_t = torch.tensor(_relation_emb, dtype=torch.float32).to(device)

# ----------------- Helpers & scoring -----------------
def normalize(s: str) -> str:
    return str(s).strip().lower()

def _score_transE(h_idx: int, t_idx: int):
    global _entity_emb_t, _relation_emb_t
    if _entity_emb_t is None or _relation_emb_t is None:
        raise RuntimeError("Embeddings not loaded. Run export or check models/ files.")
    h = _entity_emb_t[h_idx].unsqueeze(0)   # 1 x dim
    t = _entity_emb_t[t_idx].unsqueeze(0)   # 1 x dim
    r = _relation_emb_t                     # R x dim
    diff = h.unsqueeze(1) + r.unsqueeze(0) - t.unsqueeze(1)
    dist = torch.norm(diff, dim=-1).squeeze(0)
    scores = (-dist).cpu().numpy()
    return scores

# ----------------- Fast predict_relation -----------------
def predict_relation(subject: str, obj: str, top_k: int = 5):
    s = normalize(subject)
    o = normalize(obj)

    # 1) exact lookup
    if s in _ent2idx and o in _ent2idx:
        sid, oid = _ent2idx[s], _ent2idx[o]
    else:
        # 2) quick normalization attempts (strip punctuation)
        import re
        def simple_norm(x):
            return re.sub(r"[^a-z0-9\s\-]", "", x.lower()).strip()
        s2, o2 = simple_norm(s), simple_norm(o)
        if s2 in _ent2idx and o2 in _ent2idx:
            sid, oid = _ent2idx[s2], _ent2idx[o2]
        else:
            # 3) prefix-based candidate search (limit candidate pool strongly)
            pref_s = s[:3]
            pref_o = o[:3]
            cand_s = _prefix_map.get(pref_s, [])
            cand_o = _prefix_map.get(pref_o, [])
            # if either prefix empty, bail out
            if not cand_s or not cand_o:
                return []
            # reduce to candidates and perform substring membership within this small set
            found_s = None
            for k in cand_s:
                if s in k or s2 in k:
                    found_s = k; break
            found_o = None
            for k in cand_o:
                if o in k or o2 in k:
                    found_o = k; break
            if not found_s or not found_o:
                return []
            sid, oid = _ent2idx[found_s], _ent2idx[found_o]

    # compute scores
    scores = _score_transE(sid, oid)
    order = scores.argsort()[::-1]
    out = []
    for idx in order[:top_k]:
        rel_name = _idx2rel.get(int(idx), str(idx))
        out.append((rel_name, float(scores[int(idx)])))
    return out

# quick demo
if __name__ == "__main__":
    print("Inference wrapper loaded.")
    if _entity_emb is not None:
        print("Entity emb shape:", _entity_emb.shape)
    if _relation_emb is not None:
        print("Relation emb shape:", _relation_emb.shape)
