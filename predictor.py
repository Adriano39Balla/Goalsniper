# file: predictor.py
# Minimal model hook for adaptive scan(); swap with your real model.

from __future__ import annotations
import os
import json
import pickle
import logging
import threading
from typing import Dict, Any, Optional, Callable

log = logging.getLogger("predictor")

# ───────── Config ─────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")              # e.g., /data/model.pkl on Railway
MODEL_KIND = os.getenv("MODEL_KIND", "").strip().lower()       # "", "pickle", "json"
PREDICTOR_STRICT = os.getenv("PREDICTOR_STRICT", "0").lower() not in {"0", "false", "no", ""}

# Optional: allow a small per-fixture cache (seconds). 0 disables.
PREDICTION_CACHE_TTL = int(os.getenv("PREDICTION_CACHE_TTL", "0"))

# ───────── Internal State ─────────
_model: Any = None
_model_loaded = False
_model_lock = threading.RLock()

# very small TTL cache: { fid: (ts, dict) }
_pred_cache: Dict[int, tuple[float, Dict[str, float]]] = {}
_pred_cache_lock = threading.RLock()

def _now() -> float:
    import time
    return time.time()

def _load_model() -> Optional[Any]:
    """
    Lazy load model exactly once in a thread-safe way.
    Supports:
      - Pickle file (default / MODEL_KIND in {"", "pickle"})
      - JSON file (MODEL_KIND="json") returning a static mapping or config
    """
    global _model, _model_loaded
    if _model_loaded:
        return _model
    with _model_lock:
        if _model_loaded:
            return _model
        try:
            if not os.path.exists(MODEL_PATH):
                _model = None
                _model_loaded = True
                log.info("[predictor] MODEL_PATH not found: %s (fallback to odds de-vig)", MODEL_PATH)
                return _model

            # Determine kind
            kind = MODEL_KIND or ("json" if MODEL_PATH.endswith(".json") else "pickle")

            if kind == "json":
                with open(MODEL_PATH, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                # Accept either a static probability mapping or a config your real predictor might use
                if not isinstance(obj, dict):
                    log.warning("[predictor] JSON model is not a dict; ignoring")
                    obj = {}
                _model = {"kind": "json", "data": obj}
                log.info("[predictor] loaded JSON model: %s", MODEL_PATH)
            else:
                with open(MODEL_PATH, "rb") as f:
                    _model = pickle.load(f)
                log.info("[predictor] loaded PICKLE model: %s (%s)", MODEL_PATH, type(_model).__name__)
            _model_loaded = True
        except Exception as e:
            _model = None
            _model_loaded = True
            log.warning("[predictor] failed to load model (%s): %s — fallback to odds de-vig", MODEL_PATH, e, exc_info=False)
        return _model

def warm_model() -> bool:
    """Optionally call this at boot to load the model early."""
    return _load_model() is not None

def _predict_with_json_model(model_obj: dict, fid: int) -> Dict[str, float]:
    """
    If you ship a simple JSON with static probabilities or per-league defaults,
    adapt here. Current behavior: return the static mapping if present.
    """
    data = model_obj.get("data", {})
    if isinstance(data, dict):
        # NOTE: you could key by fixture/league here if your JSON contains that.
        return {str(k): float(v) for k, v in data.items() if v is not None}
    return {}

def _predict_with_pickle(model: Any, fid: int) -> Dict[str, float]:
    """
    Call into your real model. This stub simply tries common interfaces:
      - model.predict_proba(features) -> mapping or array you map yourself
      - model.predict(features) -> you map it to probabilities
    Replace 'features_for(fid)' with your own feature builder when ready.
    """
    try:
        # from features import features_for  # <- your own implementation
        # feats = features_for(fid)
        # if hasattr(model, "predict_proba"):
        #     probs = model.predict_proba(feats)
        #     return postprocess_to_market_probs(probs)
        # elif hasattr(model, "predict"):
        #     y = model.predict(feats)
        #     return postprocess_to_market_probs(y)
        # Placeholder: return empty to fallback to odds-based hints
        return {}
    except Exception as e:
        log.warning("[predictor] model prediction failed for fid=%s: %s", fid, e)
        return {}

def _get_cached_prediction(fid: int) -> Optional[Dict[str, float]]:
    if PREDICTION_CACHE_TTL <= 0:
        return None
    with _pred_cache_lock:
        entry = _pred_cache.get(fid)
        if not entry:
            return None
        ts, val = entry
        if (_now() - ts) <= PREDICTION_CACHE_TTL:
            return val
        _pred_cache.pop(fid, None)
        return None

def _put_cached_prediction(fid: int, val: Dict[str, float]) -> None:
    if PREDICTION_CACHE_TTL <= 0:
        return
    with _pred_cache_lock:
        # tiny cache: cap ~200 items by random eviction (simple & fast)
        if len(_pred_cache) >= 200:
            try:
                _pred_cache.pop(next(iter(_pred_cache)))
            except Exception:
                _pred_cache.clear()
        _pred_cache[fid] = (_now(), val)

def predict_for_fixture(fid: int) -> Dict[str, float]:
    """
    Return probabilities keyed by suggestion labels used in scan.py:
      "Over 2.5 Goals", "Under 2.5 Goals", "BTTS: Yes", "BTTS: No",
      "Home Win", "Away Win", etc.

    Contract:
      - On any failure or missing model, return {} to let scan.py fall back to odds.
      - Values must be in [0,1].
    """
    cached = _get_cached_prediction(fid)
    if cached is not None:
        return cached

    model = _load_model()
    if model is None:
        if PREDICTOR_STRICT:
            # In strict mode we want to notice missing model in logs, but still return {}
            log.debug("[predictor] strict mode: model missing (fid=%s)", fid)
        return {}

    if isinstance(model, dict) and model.get("kind") == "json":
        probs = _predict_with_json_model(model, fid)
    else:
        probs = _predict_with_pickle(model, fid)

    # Sanitize outputs
    out: Dict[str, float] = {}
    for k, v in (probs or {}).items():
        try:
            x = float(v)
            if not (0.0 <= x <= 1.0):
                continue
            out[str(k)] = x
        except Exception:
            continue

    _put_cached_prediction(fid, out)
    return out
