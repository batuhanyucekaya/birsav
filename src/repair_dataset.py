"""Version datasets without modifying originals: exact-label dedup and exact split-leak removal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import yaml
from src.audit_project import checksum, image_paths


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def repair(config, output):
    config, output = Path(config).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output exists; choose a new dataset version")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    base = Path(cfg.get("path", config.parent))
    if not base.is_absolute():
        base = config.parent / base
    result = {
        "source": str(config),
        "output": str(output),
        "changes": [],
        "independent_test": False,
        "group_provenance": "unknown; source-video/route/day manifest required",
        "warning": "Existing models may already have seen validation images. This does not retroactively create an independent test.",
    }
    seen = {}
    # Reserve held-out images first; remove overlapping TRAIN copies only.
    images = {}
    for split in ("test", "val", "train"):
        if cfg.get(split):
            path = Path(cfg[split])
            images[split] = image_paths(path if path.is_absolute() else base / path)
    import cv2

    for split, paths in images.items():
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Unreadable image: {path}")
            digest = hashlib.sha256(
                str(frame.shape).encode() + frame.tobytes()
            ).hexdigest()
            if digest in seen and seen[digest] != split:
                result["changes"].append(
                    {
                        "image": str(path),
                        "action": "excluded_cross_split_duplicate",
                        "reserved_split": seen[digest],
                    }
                )
                continue
            seen[digest] = split
            label = base / "labels" / split / f"{path.stem}.txt"
            if not label.is_file():
                raise ValueError(f"Missing label: {label}")
            rows = label.read_text(encoding="utf-8-sig").splitlines()
            unique = []
            keys = set()
            for row in rows:
                if not row.strip():
                    continue
                key = tuple(float(v) for v in row.split())
                if key not in keys:
                    unique.append(row)
                    keys.add(key)
            link_or_copy(path, output / "images" / split / path.name)
            target = output / "labels" / split / label.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "\n".join(unique) + ("\n" if unique else ""), encoding="utf-8"
            )
            if len(rows) != len(unique):
                result["changes"].append(
                    {
                        "label": str(label),
                        "sha256_before": checksum(label),
                        "removed_duplicate_rows": len(rows) - len(unique),
                    }
                )
    cfg["path"] = output.as_posix()
    for split in images:
        cfg[split] = f"images/{split}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    (output / "repair_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f'{output}: {len(result["changes"])} changes; original unchanged')
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repair(args.data, args.output)
