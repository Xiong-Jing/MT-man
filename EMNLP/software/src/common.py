from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must parse into a dictionary.")
    return cfg


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= int(max_samples):
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_dev_rows(
    rows: Sequence[Dict[str, Any]],
    dev_size: int,
    split_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    total = len(rows)
    dev_size = int(dev_size)
    if total <= 1:
        raise ValueError("Cannot split train/dev: dataset size must be greater than 1.")
    if dev_size <= 0:
        raise ValueError("dev_size must be positive when internal dev split is enabled.")
    if dev_size >= total:
        raise ValueError("dev_size={} must be smaller than total rows={}.".format(dev_size, total))

    indices = list(range(total))
    rng = random.Random(int(split_seed))
    rng.shuffle(indices)

    dev_idx = set(indices[:dev_size])
    train_rows = []
    dev_rows = []
    for idx, row in enumerate(rows):
        if idx in dev_idx:
            dev_rows.append(row)
        else:
            train_rows.append(row)

    manifest = {
        "total_rows": total,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "dev_size": dev_size,
        "split_seed": int(split_seed),
        "dev_indices_head": sorted(list(dev_idx))[:20],
    }
    return train_rows, dev_rows, manifest


def estimate_remaining_seconds(done_steps: int, total_steps: int, elapsed_sec: float) -> float:
    done_steps = int(done_steps)
    total_steps = int(total_steps)
    elapsed_sec = float(elapsed_sec)
    if done_steps <= 0 or total_steps <= 0 or elapsed_sec <= 0:
        return -1.0
    if done_steps >= total_steps:
        return 0.0
    steps_per_sec = done_steps / max(elapsed_sec, 1e-8)
    if steps_per_sec <= 0:
        return -1.0
    remain = (total_steps - done_steps) / steps_per_sec
    return float(remain)


def resolve_progress_cfg(progress_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = progress_cfg or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "dynamic_ncols": bool(cfg.get("dynamic_ncols", True)),
        "mininterval": float(cfg.get("mininterval", 0.5)),
        "leave": bool(cfg.get("leave", True)),
    }


def tqdm_kwargs(progress_cfg: Dict[str, Any], desc: str) -> Dict[str, Any]:
    cfg = resolve_progress_cfg(progress_cfg)
    return {
        "desc": desc,
        "disable": not cfg["enabled"],
        "dynamic_ncols": cfg["dynamic_ncols"],
        "mininterval": cfg["mininterval"],
        "leave": cfg["leave"],
    }
