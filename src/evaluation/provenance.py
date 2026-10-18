"""Simplified provenance and fingerprinting metadata."""

import hashlib
import json
from typing import Any, Dict, Optional


def digest(value: Any) -> str:
    """Generates a stable sha256 digest of a serializable object."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def build_provenance(cfg: Any, checkpoint_path: Optional[str] = None, root: Optional[str] = None) -> Dict[str, Any]:
    """Builds lightweight metadata identity for an experiment run without heavy filesystem hashing."""
    scen = getattr(cfg, "scenario_name", "unknown")
    algo = getattr(cfg, "algorithm", "unknown")
    seed = getattr(cfg, "seed", 0)

    identity = {
        "scenario": scen,
        "algorithm": algo,
        "seed": seed,
        "checkpoint": checkpoint_path or getattr(cfg, "model_path", None),
    }
    return {
        "schema": "simple-v1",
        "identity": identity,
        "run_fingerprint": f"{scen}_{algo}_{seed}",
        "cohort_fingerprint": f"{scen}_{algo}",
    }


def reusable_record(record: Dict[str, Any], expected: Optional[str] = None) -> bool:
    """Determines if a previous record can be reused without re-running."""
    return record.get("status") == "SUCCESS" and record.get("is_valid", False) is True


def comparison_identity(provenance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns basic comparison identity (empty dict = fully compatible)."""
    return {}
