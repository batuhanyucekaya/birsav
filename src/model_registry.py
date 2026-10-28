"""Explicit model activation. Training never updates this manifest automatically."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_verified = {}


def manifest():
    return json.loads((ROOT / "config" / "models.json").read_text(encoding="utf-8"))[
        "models"
    ]


def model_path(name, required=True):
    item = manifest()[name]
    path = ROOT / item["path"]
    if not item["enabled"] or not path.is_file():
        if required:
            raise FileNotFoundError(f"Model unavailable: {name} ({item['status']})")
        return None
    stat = path.stat()
    signature = (str(path), stat.st_size, stat.st_mtime_ns, item.get("sha256"))
    if item.get("sha256") and signature not in _verified:
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Model checksum mismatch: {name}")
        _verified[signature] = True
    return path


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def version(name):
    item = manifest()[name]
    return {
        "name": name,
        "path": item["path"],
        "status": item["status"],
        "enabled": item["enabled"],
        "sha256": item.get("sha256"),
    }
