from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

from common import ensure_dir, load_yaml_config, read_jsonl
from calm_trans import simple_source_tokens

try:
    import jieba  # type: ignore
except Exception:
    jieba = None


LEXICON_PATTERN = re.compile(r"\[([^:\]]+)\s*:\s*([^\]]+)\]")


def split_candidate_text(candidate_blob: str) -> List[str]:
    pieces = re.split(r"[\/,，;；、]", candidate_blob.strip())
    clean = []
    for item in pieces:
        item = item.strip()
        if not item:
            continue
        clean.append(item)
    return clean


def segment_target_text(text: str) -> List[str]:
    if not text:
        return []
    if jieba is not None:
        words = [w.strip() for w in jieba.lcut(text) if w.strip()]
        return words
    # Fallback: treat each Han character as a token.
    chars = [c.strip() for c in text if c.strip()]
    return chars


def all_source_ngrams(tokens: List[str], max_span_len: int) -> List[str]:
    spans = []
    max_span_len = max(1, int(max_span_len))
    for span_len in range(1, max_span_len + 1):
        for i in range(0, max(0, len(tokens) - span_len + 1)):
            spans.append(" ".join(tokens[i : i + span_len]))
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CALM-Trans lexical memory from training data.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths", {})
    calm_cfg = cfg.get("calm", {})
    memory_cfg = cfg.get("memory", {})

    train_path = paths.get("train_data")
    memory_path = paths.get("memory_path")
    coverage_path = paths.get("coverage_report_path")
    if not train_path or not memory_path or not coverage_path:
        raise ValueError("paths.train_data, paths.memory_path and paths.coverage_report_path are required in config.")

    ensure_dir(os.path.dirname(memory_path))
    ensure_dir(os.path.dirname(coverage_path))

    max_samples = memory_cfg.get("max_samples")
    max_span_len = int(calm_cfg.get("max_span_len", 4))
    max_candidates_per_span = int(memory_cfg.get("max_candidates_per_span", 16))
    min_span_freq = float(memory_cfg.get("min_span_freq", 1))
    heuristic_align = bool(memory_cfg.get("heuristic_align", True))
    heuristic_weight = float(memory_cfg.get("heuristic_weight", 0.2))
    tail_freq_threshold = float(memory_cfg.get("tail_freq_threshold", 2))

    rows = read_jsonl(train_path, max_samples=max_samples)
    if not rows:
        raise ValueError("No train samples loaded, please check train_data path.")

    span_freq = Counter()
    span_cand_freq = defaultdict(lambda: defaultdict(float))

    for row in rows:
        instruction = str(row.get("instruction", ""))
        source = str(row.get("input", row.get("query", ""))).strip()
        target = str(row.get("output", row.get("response", ""))).strip()

        src_tokens = simple_source_tokens(source)
        if not src_tokens:
            continue

        for span in all_source_ngrams(src_tokens, max_span_len=max_span_len):
            span_freq[span] += 1.0

        # 1) High-quality lexical candidates from provided prompt hints.
        for match in LEXICON_PATTERN.finditer(instruction):
            src_span = match.group(1).strip().lower()
            if not src_span:
                continue
            candidates = split_candidate_text(match.group(2))
            for cand in candidates:
                span_cand_freq[src_span][cand] += 1.0
                span_freq[src_span] += 1.0

        # 2) Heuristic fallback alignment for source spans with no explicit hints.
        if heuristic_align and target:
            tgt_terms = segment_target_text(target)
            if tgt_terms:
                denom = max(1, len(src_tokens) - 1)
                for idx, token in enumerate(src_tokens):
                    pos = int(round((idx / float(denom)) * (len(tgt_terms) - 1)))
                    cand = tgt_terms[pos].strip()
                    if cand:
                        span_cand_freq[token][cand] += heuristic_weight

    memory_rows = []
    for span, cand_map in span_cand_freq.items():
        total = float(sum(cand_map.values()))
        if total <= 0:
            continue
        if float(span_freq.get(span, 0.0)) < min_span_freq:
            continue

        cand_items = []
        for cand, freq in cand_map.items():
            freq = float(freq)
            p_val = float(freq / total)
            cand_items.append(
                {
                    "candidate_text": cand,
                    "freq": freq,
                    "p_cand_given_span": p_val,
                    "cand_len": float(max(1, len(cand))),
                    "span_len": float(max(1, len(span.split()))),
                    "align_score": p_val,
                }
            )
        cand_items.sort(key=lambda x: (x["p_cand_given_span"], x["freq"]), reverse=True)
        memory_rows.append(
            {
                "span": span,
                "total_freq": float(span_freq.get(span, total)),
                "candidates": cand_items[:max_candidates_per_span],
            }
        )

    memory_rows.sort(key=lambda x: x["total_freq"], reverse=True)

    with open(memory_path, "w", encoding="utf-8") as f:
        for row in memory_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    memory_span_set = set(row["span"] for row in memory_rows)
    covered_samples = 0
    candidate_counts = []
    for row in rows:
        source = str(row.get("input", row.get("query", ""))).strip()
        src_tokens = simple_source_tokens(source)
        ngrams = all_source_ngrams(src_tokens, max_span_len=max_span_len)
        hit = False
        for span in ngrams:
            if span in memory_span_set:
                hit = True
                break
        if hit:
            covered_samples += 1
    for row in memory_rows:
        candidate_counts.append(len(row.get("candidates", [])))

    tail_spans = [row for row in memory_rows if float(row.get("total_freq", 0.0)) <= tail_freq_threshold]
    coverage_report = {
        "num_train_samples": len(rows),
        "num_memory_spans": len(memory_rows),
        "sample_coverage_rate": float(covered_samples / max(1, len(rows))),
        "avg_candidates_per_span": float(sum(candidate_counts) / max(1, len(candidate_counts))),
        "tail_span_ratio": float(len(tail_spans) / max(1, len(memory_rows))),
        "max_span_len": max_span_len,
        "max_candidates_per_span": max_candidates_per_span,
        "heuristic_align": heuristic_align,
        "heuristic_weight": heuristic_weight,
    }

    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage_report, f, ensure_ascii=False, indent=2)

    print("Memory build complete.")
    print("memory_path =", memory_path)
    print("coverage_report_path =", coverage_path)
    print("num_memory_spans =", len(memory_rows))
    print("sample_coverage_rate =", round(coverage_report["sample_coverage_rate"], 4))


if __name__ == "__main__":
    main()

