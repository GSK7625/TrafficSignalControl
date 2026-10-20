"""Reliable JSON persistence for checkpoints and experiment results."""

import json
from pathlib import Path
import tempfile


def write_json_atomic(path, data) -> None:
    """Serialize completely before replacing a file, preserving it on failure."""
    payload = json.dumps(data, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
