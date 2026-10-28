"""Rebase local dataset roots after a workspace move; keep historical run metadata intact."""

import argparse
import json
from pathlib import Path
import yaml


def relocate(root, apply=False):
    root = Path(root).resolve()
    changes = []
    for config in sorted((root / "data").rglob("data.yaml")):
        text = config.read_text(encoding="utf-8")
        cfg = yaml.safe_load(text)
        if (
            not cfg.get("path")
            or str(cfg["path"]).replace("\\", "/").rstrip("/")
            == config.parent.as_posix()
        ):
            continue
        # Only self-contained datasets; do not redirect external datasets silently.
        splits = [cfg.get(s) for s in ("train", "val", "test") if cfg.get(s)]
        if not splits or not all(
            isinstance(s, str)
            and not Path(s).is_absolute()
            and (config.parent / s).exists()
            for s in splits
        ):
            continue
        changes.append(
            {
                "config": str(config.relative_to(root)),
                "old_path": cfg["path"],
                "new_path": config.parent.as_posix(),
                "original_yaml": text,
            }
        )
        if apply:
            cfg["path"] = config.parent.as_posix()
            config.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
    if apply:
        out = root / "reports/workspace_path_migration.json"
        out.parent.mkdir(exist_ok=True)
        if out.exists():
            previous = json.loads(out.read_text(encoding="utf-8"))
            changes = previous["changes"] + changes
        out.write_text(
            json.dumps(
                {
                    "workspace": str(root),
                    "changes": changes,
                    "historical_training_metadata_modified": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return changes


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(
            relocate(Path(__file__).resolve().parent.parent, args.apply),
            indent=2,
            ensure_ascii=False,
        )
    )
