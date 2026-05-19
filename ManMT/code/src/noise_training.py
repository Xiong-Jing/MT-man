from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F


class NARTSampler(object):
    def __init__(
        self,
        hard_negative_ratio: float = 0.5,
        random_negative_ratio: float = 0.5,
        min_noise_per_span: int = 2,
        random_seed: int = 42,
    ) -> None:
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.random_negative_ratio = float(random_negative_ratio)
        self.min_noise_per_span = int(min_noise_per_span)
        self.rng = random.Random(random_seed)

    def build_noisy_candidates(
        self,
        memory: Any,
        clean_candidates: Sequence[Dict[str, Any]],
        topk: int,
        hard_pool: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if topk <= 0:
            return []

        clean_texts = set(str(item.get("candidate_text", "")).strip() for item in clean_candidates if item.get("candidate_text"))
        num_hard = int(round(topk * self.hard_negative_ratio))
        num_random = max(self.min_noise_per_span, int(round(topk * self.random_negative_ratio)))
        target_size = max(self.min_noise_per_span, num_hard + num_random)

        negatives = memory.sample_negative_candidates(
            exclude_texts=clean_texts,
            num_samples=target_size,
            hard_pool=hard_pool,
        )

        noisy = []
        for item in negatives:
            cloned = dict(item)
            cloned["is_noisy"] = True
            noisy.append(cloned)
            if len(noisy) >= target_size:
                break
        return noisy


def calibration_bce_loss(clean_scores: torch.Tensor, noisy_scores: torch.Tensor) -> torch.Tensor:
    losses = []
    if clean_scores is not None and clean_scores.numel() > 0:
        pos_target = torch.ones_like(clean_scores)
        losses.append(F.binary_cross_entropy(clean_scores, pos_target))
    if noisy_scores is not None and noisy_scores.numel() > 0:
        neg_target = torch.zeros_like(noisy_scores)
        losses.append(F.binary_cross_entropy(noisy_scores, neg_target))
    if not losses:
        ref = clean_scores if clean_scores is not None else noisy_scores
        device = ref.device if ref is not None else "cpu"
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def margin_ranking_noise_loss(
    clean_scores: torch.Tensor,
    noisy_scores: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    if clean_scores is None or noisy_scores is None or clean_scores.numel() == 0 or noisy_scores.numel() == 0:
        ref = clean_scores if clean_scores is not None else noisy_scores
        device = ref.device if ref is not None else "cpu"
        return torch.tensor(0.0, device=device)

    clean_expand = clean_scores.unsqueeze(1)
    noisy_expand = noisy_scores.unsqueeze(0)
    diff = clean_expand - noisy_expand
    loss = torch.relu(float(margin) - diff)
    return loss.mean()


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    if not probabilities:
        return 0.0

    bins = [i / float(n_bins) for i in range(n_bins + 1)]
    total = float(len(probabilities))
    ece = 0.0

    for b_idx in range(n_bins):
        left = bins[b_idx]
        right = bins[b_idx + 1]
        idx = []
        for i, p in enumerate(probabilities):
            include = (p >= left and p < right) if b_idx < n_bins - 1 else (p >= left and p <= right)
            if include:
                idx.append(i)
        if not idx:
            continue

        acc = sum(float(labels[i]) for i in idx) / float(len(idx))
        conf = sum(float(probabilities[i]) for i in idx) / float(len(idx))
        ece += abs(acc - conf) * (len(idx) / total)
    return float(ece)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if not probabilities:
        return 0.0
    err = 0.0
    for p, y in zip(probabilities, labels):
        err += (float(p) - float(y)) ** 2
    return float(err / max(1, len(probabilities)))


def term_accuracy(
    high_conf_candidates: Sequence[str],
    prediction: str,
    reference: str,
) -> Dict[str, float]:
    predicted_terms = 0
    supported_terms = 0
    hits = 0

    for term in high_conf_candidates:
        if not term:
            continue
        appears_in_pred = term in prediction
        appears_in_ref = term in reference
        if appears_in_pred:
            predicted_terms += 1
        if appears_in_ref:
            supported_terms += 1
            if appears_in_pred:
                hits += 1

    precision = float(hits / predicted_terms) if predicted_terms > 0 else 0.0
    recall = float(hits / supported_terms) if supported_terms > 0 else 0.0
    return {
        "term_hits": float(hits),
        "term_precision": precision,
        "term_recall": recall,
    }

