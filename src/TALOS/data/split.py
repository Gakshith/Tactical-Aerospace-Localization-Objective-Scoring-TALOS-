"""Frozen, by-video dataset splits.

Frames from one video are strongly correlated, so splitting by frame leaks
the test set into training. Splits are made once per dataset, written to
``configs/splits/<dataset>.json`` and committed; every later stage reads that
file and never re-splits.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SPLIT_DIR = Path("configs/splits")


def make_split(
    videos: list[str],
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
    fixed: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Split video names into train/val/test.

    ``fixed`` pins an official split (e.g. a published test list); only the
    remaining videos are shuffled into the other buckets, in proportion.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    names = sorted(set(videos))
    if len(names) != len(videos):
        raise ValueError("duplicate video names")
    split = {"train": [], "val": [], "test": []}
    fixed = fixed or {}
    pinned = set()
    for key, vids in fixed.items():
        if key not in split:
            raise ValueError(f"unknown split {key!r}")
        missing = set(vids) - set(names)
        if missing:
            raise ValueError(f"pinned videos not in dataset: {sorted(missing)}")
        split[key] = sorted(vids)
        pinned |= set(vids)

    free = [v for v in names if v not in pinned]
    random.Random(seed).shuffle(free)
    open_keys = [k for k in ("train", "val", "test") if k not in fixed]
    if open_keys:
        weights = [fractions[("train", "val", "test").index(k)] for k in open_keys]
        total = sum(weights)
        n = len(free)
        counts = [round(n * w / total) for w in weights]
        counts[0] += n - sum(counts)  # rounding remainder goes to the first bucket
        start = 0
        for key, c in zip(open_keys, counts):
            split[key] = sorted(free[start:start + c])
            start += c
    elif free:
        raise ValueError(f"videos left unassigned: {free}")

    _check_disjoint(split)
    return split


def _check_disjoint(split: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for key, vids in split.items():
        for v in vids:
            if v in seen:
                raise ValueError(f"{v} is in both {seen[v]} and {key}")
            seen[v] = key


def save_split(dataset: str, split: dict[str, list[str]], meta: dict, root: Path = SPLIT_DIR) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{dataset}.json"
    if path.exists():
        raise FileExistsError(f"{path} is frozen; delete it deliberately to re-split")
    path.write_text(json.dumps({"dataset": dataset, **meta, "split": split}, indent=2) + "\n")
    return path


def load_split(dataset: str, root: Path = SPLIT_DIR) -> dict[str, list[str]]:
    data = json.loads((root / f"{dataset}.json").read_text())
    split = data["split"]
    _check_disjoint(split)
    return split
