from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from transformers import LogitsProcessor


SOURCE_SYSTEM_PROMPT = "You are a translation expert specializing in Qing dynasty history and Manchu. Translate the following romanized Manchu text into Classical/Literary Chinese."


def build_source_only_prompt(source_text: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + SOURCE_SYSTEM_PROMPT
        + "\n"
        + source_text.strip()
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def simple_source_tokens(text: str) -> List[str]:
    return [tok for tok in re.split(r"\s+", text.strip()) if tok]


def mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    masked = last_hidden * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def logits_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    entropy = logits_entropy(logits)
    vocab = float(logits.size(-1))
    return entropy / max(math.log(vocab), 1.0)


def encode_text_mean(tokenizer: Any, embedding_layer: nn.Embedding, text: str, device: torch.device) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return torch.zeros(embedding_layer.embedding_dim, device=device, dtype=embedding_layer.weight.dtype)
    token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
    embeds = embedding_layer(token_tensor)
    return embeds.mean(dim=0)


@dataclass
class SpanItem:
    text: str
    start: int
    end: int
    source: str


class SpanConstructor(object):
    def __init__(
        self,
        tokenizer: Any,
        max_span_len: int = 4,
        use_longest_match: bool = True,
        frequent_spans: Optional[Iterable[str]] = None,
        max_spans: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_span_len = max(1, int(max_span_len))
        self.use_longest_match = use_longest_match
        self.max_spans = max(1, int(max_spans))
        self.frequent_spans = set(frequent_spans or [])

    def _subword_score(self, span: str) -> int:
        pieces = self.tokenizer.tokenize(span)
        return len(pieces)

    def _match_from_vocab(self, tokens: Sequence[str], covered: List[bool]) -> List[SpanItem]:
        spans = []
        if not self.frequent_spans:
            return spans
        for span_len in range(self.max_span_len, 0, -1):
            upper = len(tokens) - span_len + 1
            for i in range(max(0, upper)):
                if any(covered[i : i + span_len]):
                    continue
                candidate = " ".join(tokens[i : i + span_len]).strip()
                if candidate in self.frequent_spans:
                    spans.append(SpanItem(text=candidate, start=i, end=i + span_len, source="freq"))
                    for j in range(i, i + span_len):
                        covered[j] = True
        return spans

    def build_spans(self, source_text: str) -> List[SpanItem]:
        tokens = simple_source_tokens(source_text)
        if not tokens:
            return []

        spans = []
        covered = [False for _ in tokens]

        if self.use_longest_match:
            for span_len in range(self.max_span_len, 0, -1):
                upper = len(tokens) - span_len + 1
                for i in range(max(0, upper)):
                    if any(covered[i : i + span_len]):
                        continue
                    candidate = " ".join(tokens[i : i + span_len]).strip()
                    if not candidate:
                        continue
                    if self._subword_score(candidate) <= 0:
                        continue
                    spans.append(SpanItem(text=candidate, start=i, end=i + span_len, source="subword"))
                    for j in range(i, i + span_len):
                        covered[j] = True

        spans.extend(self._match_from_vocab(tokens=tokens, covered=covered))
        spans.sort(key=lambda x: (x.start, x.end))
        return spans[: self.max_spans]


class LexicalMemory(object):
    def __init__(self, memory_path: str, topk: int = 8) -> None:
        self.memory_path = memory_path
        self.topk = max(1, int(topk))
        self.span_to_candidates = {}  # type: Dict[str, List[Dict[str, Any]]]
        self.span_total_freq = {}  # type: Dict[str, float]
        self.candidate_pool = []  # type: List[Dict[str, Any]]
        self._load(memory_path)

    def _load(self, memory_path: str) -> None:
        with open(memory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                span = row.get("span", "").strip()
                if not span:
                    continue
                candidates = row.get("candidates", [])
                if not isinstance(candidates, list):
                    continue

                normalized = []
                total_freq = float(row.get("total_freq", 0.0))
                for cand in candidates:
                    text = str(cand.get("candidate_text", "")).strip()
                    if not text:
                        continue
                    item = {
                        "candidate_text": text,
                        "freq": float(cand.get("freq", 0.0)),
                        "p_cand_given_span": float(cand.get("p_cand_given_span", 0.0)),
                        "cand_len": float(cand.get("cand_len", max(1, len(text)))),
                        "span_len": float(cand.get("span_len", max(1, len(span.split())))),
                        "align_score": float(cand.get("align_score", cand.get("p_cand_given_span", 0.0))),
                    }
                    normalized.append(item)
                    self.candidate_pool.append(item)

                normalized.sort(
                    key=lambda x: (x.get("p_cand_given_span", 0.0), x.get("freq", 0.0)),
                    reverse=True,
                )
                self.span_to_candidates[span] = normalized
                self.span_total_freq[span] = max(total_freq, sum(item["freq"] for item in normalized))

        self.candidate_pool.sort(key=lambda x: x.get("freq", 0.0), reverse=True)

    def get_frequent_spans(self, min_freq: float = 3.0, max_spans: int = 50000) -> Set[str]:
        filtered = [k for k, v in self.span_total_freq.items() if float(v) >= float(min_freq)]
        filtered.sort(key=lambda x: self.span_total_freq.get(x, 0.0), reverse=True)
        return set(filtered[: int(max_spans)])

    def retrieve(self, span_text: str, topk: Optional[int] = None) -> List[Dict[str, Any]]:
        k = self.topk if topk is None else max(1, int(topk))
        candidates = self.span_to_candidates.get(span_text, [])
        return candidates[:k]

    def retrieve_for_spans(self, spans: Sequence[SpanItem], topk: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        out = {}
        for span in spans:
            out[span.text] = self.retrieve(span.text, topk=topk)
        return out

    def sample_negative_candidates(
        self,
        exclude_texts: Set[str],
        num_samples: int = 4,
        hard_pool: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        seen = set(exclude_texts)

        if hard_pool:
            for item in hard_pool:
                text = str(item.get("candidate_text", "")).strip()
                if (not text) or (text in seen):
                    continue
                results.append(item)
                seen.add(text)
                if len(results) >= num_samples:
                    return results[:num_samples]

        if len(results) < num_samples:
            shuffled = list(self.candidate_pool[: max(256, num_samples * 16)])
            random.shuffle(shuffled)
            for item in shuffled:
                text = str(item.get("candidate_text", "")).strip()
                if (not text) or (text in seen):
                    continue
                results.append(item)
                seen.add(text)
                if len(results) >= num_samples:
                    break
        return results[:num_samples]


class ConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int, feat_dim: int, mlp_hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.feat_dim = int(feat_dim)
        input_dim = self.hidden_size * 3 + self.feat_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(mlp_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden), 1),
        )

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.net(feature_tensor).squeeze(-1)
        return torch.sigmoid(logits)


class BiasBuilder(object):
    def __init__(
        self,
        tokenizer: Any,
        phrase_mode: str = "first_token",
        min_conf: float = 0.0,
        confidence_power: float = 1.0,
        max_phrases: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.phrase_mode = phrase_mode
        self.min_conf = float(min_conf)
        self.confidence_power = float(confidence_power)
        self.max_phrases = int(max_phrases)

    def _candidate_to_token_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def build_bias(
        self,
        candidates: Sequence[Dict[str, Any]],
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, Dict[Tuple[int, ...], float]]:
        bias_vec = torch.zeros(vocab_size, device=device, dtype=dtype)
        sequence_bias = {}  # type: Dict[Tuple[int, ...], float]

        kept = 0
        for item in candidates:
            confidence = float(item.get("confidence", 0.0))
            if confidence < self.min_conf:
                continue

            candidate_text = str(item.get("candidate_text", "")).strip()
            token_ids = self._candidate_to_token_ids(candidate_text)
            if not token_ids:
                continue

            prior = float(item.get("p_cand_given_span", item.get("align_score", 1.0)))
            score = (confidence ** self.confidence_power) * max(prior, 1e-6)

            bias_vec[token_ids[0]] += score
            if self.phrase_mode != "first_token" and len(token_ids) > 1:
                sequence_bias[tuple(token_ids)] = sequence_bias.get(tuple(token_ids), 0.0) + score

            kept += 1
            if kept >= self.max_phrases:
                break

        return bias_vec, sequence_bias


class CalmLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        bias_vector: torch.Tensor,
        alpha: float = 1.0,
        dynamic_alpha: bool = False,
        w1: float = 1.0,
        w2: float = 0.0,
        bias_term: float = 0.0,
        coverage: float = 0.0,
        min_alpha: float = 0.0,
        max_alpha: float = 1.0,
    ) -> None:
        self.bias_vector = bias_vector
        self.alpha = float(alpha)
        self.dynamic_alpha = bool(dynamic_alpha)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.bias_term = float(bias_term)
        self.coverage = float(coverage)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.last_alpha = float(alpha)

    def _compute_alpha(self, scores: torch.Tensor) -> float:
        if not self.dynamic_alpha:
            return self.alpha
        ent = normalized_entropy(scores).mean().item()
        value = self.w1 * ent + self.w2 * self.coverage + self.bias_term
        value = min(max(value, self.min_alpha), self.max_alpha)
        return float(value)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.bias_vector.device != scores.device:
            self.bias_vector = self.bias_vector.to(scores.device)
        if self.bias_vector.dtype != scores.dtype:
            bias_vec = self.bias_vector.to(scores.dtype)
        else:
            bias_vec = self.bias_vector

        step_alpha = self._compute_alpha(scores)
        self.last_alpha = step_alpha
        return scores + step_alpha * bias_vec.unsqueeze(0)


def compute_sentence_entropy_from_logits(logits: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    # logits: [seq, vocab], label_mask: [seq]
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    ent = normalized_entropy(logits)
    if label_mask.numel() == ent.numel():
        valid = ent[label_mask]
        if valid.numel() > 0:
            return valid.mean()
    return ent.mean()

