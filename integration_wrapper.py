# integration_wrapper.py
from typing import Optional, Tuple, List, Dict, Any
from kinship_symbolic import KinshipGraph, interpret_chain, solve_relationship_statement  # place your symbolic code in kinship_symbolic.py or import from app
from predict_with_clf import predict as clf_predict

# If you have a function that scores KGE (predict_relation), import it:
# from inference_wrapper import predict_relation as kge_predict

def combined_reason(a: str, b: str, context_story: str = "", top_k=5) -> Dict[str, Any]:
    """
    Try symbolic -> classifier -> KGE fallback.
    a,b: person names (lowercase ideally)
    context_story: optional story to feed symbolic parser
    """
    # 1) Symbolic attempt (very precise when patterns exist)
    try:
        sym = solve_relationship_statement(f'"{a}" is "{b}"' if not context_story else context_story)
        # The symbolic solver returns parsed_left/parsed_right and an 'answer' string.
        ans = sym.get("answer","")
        if ans and not ans.startswith("Unable") and not ans.startswith("Unsupported"):
            return {"source": "symbolic", "answer": ans, "debug": sym}
    except Exception:
        # ignore, move on
        pass

    # 2) Classifier (trained on CLUTRR)
    try:
        clf_preds = clf_predict(a, b, top_k=top_k)
        if clf_preds:
            return {"source": "classifier", "predictions": clf_preds}
    except Exception:
        pass

    # 3) KGE fallback (if you have one)
    try:
        from inference_wrapper import predict_relation as kge_predict  # lazy import
        kge_preds = kge_predict(a, b, top_k=top_k)
        if kge_preds:
            return {"source": "kge", "predictions": kge_preds}
    except Exception:
        pass

    return {"source": "none", "answer": "I could not determine the relation with confidence."}
