from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from calm_trans import (
    ConfidenceHead,
    LexicalMemory,
    SpanConstructor,
    compute_sentence_entropy_from_logits,
)
from common import (
    append_jsonl,
    estimate_remaining_seconds,
    ensure_dir,
    load_yaml_config,
    read_jsonl,
    resolve_progress_cfg,
    set_seed,
    split_train_dev_rows,
    tqdm_kwargs,
)
from ioce_utils import (
    build_feature_tensor,
    choose_ckpt_root,
    collect_clean_candidates,
    load_ioce_backbone,
    save_confidence_head,
    validate_input_ckpt_root,
)
from lmsa_aps import EvalCollator, StageEvalDataset
from noise_training import NARTSampler, brier_score, calibration_bce_loss, margin_ranking_noise_loss


def _safe_logit(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return float(math.log(p / (1.0 - p)))


def _apply_temperature(prob: float, temperature: float) -> float:
    t = max(1e-6, float(temperature))
    z = _safe_logit(prob) / t
    return float(1.0 / (1.0 + math.exp(-z)))


def _fit_temperature_by_brier(probs: List[float], labels: List[int]) -> Dict[str, float]:
    if not probs:
        return {
            "temperature": 1.0,
            "brier_before": 0.0,
            "brier_after": 0.0,
        }
    b0 = brier_score(probs, labels)
    candidates = [0.6 + 0.05 * i for i in range(29)]
    best_t = 1.0
    best_b = b0
    for t in candidates:
        calibrated = [_apply_temperature(p, t) for p in probs]
        b = brier_score(calibrated, labels)
        if b < best_b:
            best_b = b
            best_t = float(t)
    return {
        "temperature": float(best_t),
        "brier_before": float(b0),
        "brier_after": float(best_b),
    }


def _collect_calibration_pairs(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    model: Any,
    model_device: torch.device,
    embedding_layer: Any,
    confidence_head: Any,
    head_dtype: torch.dtype,
    span_constructor: Any,
    memory: Any,
    calm_cfg: Dict[str, Any],
    max_length: int,
    batch_size: int,
) -> Dict[str, Any]:
    if not rows:
        return {"probs": [], "labels": [], "covered_rate": 0.0, "fallback_hit_rate": 0.0}

    dataset = StageEvalDataset(rows=rows, tokenizer=tokenizer, max_length=int(max_length))
    if len(dataset) == 0:
        return {"probs": [], "labels": [], "covered_rate": 0.0, "fallback_hit_rate": 0.0}
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=EvalCollator(int(tokenizer.pad_token_id)),
    )

    probs: List[float] = []
    labels: List[int] = []
    covered_total = 0
    fallback_total = 0
    sample_total = 0

    with torch.no_grad():
        for batch in loader:
            tensor_batch = {
                "input_ids": batch["input_ids"].to(model_device),
                "attention_mask": batch["attention_mask"].to(model_device),
                "lexicon_ids": batch["lexicon_ids"].to(model_device),
                "lexicon_mask": batch["lexicon_mask"].to(model_device),
                "morph_ids": batch["morph_ids"].to(model_device),
                "morph_mask": batch["morph_mask"].to(model_device),
            }
            inputs_embeds = model._inject(
                input_ids=tensor_batch["input_ids"],
                lexicon_ids=tensor_batch["lexicon_ids"],
                lexicon_mask=tensor_batch["lexicon_mask"],
                morph_ids=tensor_batch["morph_ids"],
                morph_mask=tensor_batch["morph_mask"],
            )
            outputs = model.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=tensor_batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            logits = outputs.logits
            last_hidden = outputs.hidden_states[-1]

            for b_idx, source_text in enumerate(batch["source"]):
                sample_total += 1
                clean_candidates, _, covered, used_fallback = collect_clean_candidates(
                    source_text=source_text,
                    span_constructor=span_constructor,
                    memory=memory,
                    topk=int(calm_cfg.get("topk", 8)),
                    retrieve_multiplier=int(calm_cfg.get("retrieve_multiplier", 3)),
                    topk_per_span=int(calm_cfg.get("topk_per_span", calm_cfg.get("topk", 8))),
                    enable_fallback_span=bool(calm_cfg.get("enable_fallback_span", False)),
                    fallback_max_unigram=int(calm_cfg.get("fallback_max_unigram", 6)),
                    min_candidate_pool=int(calm_cfg.get("min_candidate_pool", 1)),
                )
                covered_total += int(covered)
                fallback_total += int(used_fallback)
                if not clean_candidates:
                    continue
                attn_mask = tensor_batch["attention_mask"][b_idx].bool()
                valid_hidden = last_hidden[b_idx][attn_mask]
                if valid_hidden.numel() == 0:
                    continue
                h_x_pool = valid_hidden.mean(dim=0)
                entropy_x = compute_sentence_entropy_from_logits(logits[b_idx], attn_mask).item()
                features = build_feature_tensor(
                    tokenizer=tokenizer,
                    embedding_layer=embedding_layer,
                    device=model_device,
                    h_x_pool=h_x_pool,
                    entropy_x=entropy_x,
                    candidates=clean_candidates,
                )
                if features.numel() == 0:
                    continue
                scores = confidence_head(features.to(head_dtype))
                reference = str(batch["target"][b_idx])
                for i, cand in enumerate(clean_candidates):
                    prob = float(scores[i].item())
                    term = str(cand.get("candidate_text", ""))
                    probs.append(prob)
                    labels.append(int(term in reference))

    return {
        "probs": probs,
        "labels": labels,
        "covered_rate": float(covered_total / max(1, sample_total)),
        "fallback_hit_rate": float(fallback_total / max(1, sample_total)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IOCE output-side confidence head with NART auxiliary objective.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--ckpt", type=str, default="", help="Input-side checkpoint root containing dad_adapter and lmsa.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    lmsa_cfg = cfg.get("lmsa", {})
    calm_cfg = cfg.get("calm", {})
    nart_cfg = cfg.get("nart", {})
    head_cfg = cfg.get("ioce_head", {})
    eval_cfg = cfg.get("eval", {})
    protocol_cfg = cfg.get("protocol", {})
    progress_cfg = resolve_progress_cfg(cfg.get("progress", {}))

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    model_id = str(paths["base_model"])
    train_data = str(paths["train_data"])
    output_dir = str(paths.get("output_dir", "./outputs/ioce_lexical"))
    ioce_head_dir = os.path.join(output_dir, "ioce_head")
    ensure_dir(ioce_head_dir)
    log_file = os.path.join(ioce_head_dir, "head_train_log.jsonl")
    if os.path.exists(log_file):
        os.remove(log_file)

    ckpt_root = choose_ckpt_root(cfg, override=args.ckpt)
    validate_input_ckpt_root(ckpt_root, context="train_ioce_head")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model, model_device = load_ioce_backbone(
        model_id=model_id,
        model_cfg=model_cfg,
        lmsa_cfg=lmsa_cfg,
        ckpt_root=ckpt_root,
    )
    embedding_layer = model.base_model.get_input_embeddings()

    memory = LexicalMemory(memory_path=str(paths["memory_path"]), topk=int(calm_cfg.get("topk", 8)))
    frequent_spans = memory.get_frequent_spans(min_freq=float(calm_cfg.get("frequent_span_min_freq", 3.0)))
    span_constructor = SpanConstructor(
        tokenizer=tokenizer,
        max_span_len=int(calm_cfg.get("max_span_len", 4)),
        use_longest_match=True,
        frequent_spans=frequent_spans,
        max_spans=int(calm_cfg.get("max_spans", 64)),
    )
    nart_sampler = NARTSampler(
        hard_negative_ratio=float(nart_cfg.get("hard_negative_ratio", 0.5)),
        random_negative_ratio=float(nart_cfg.get("random_negative_ratio", 0.5)),
        min_noise_per_span=int(nart_cfg.get("min_noise_per_span", 2)),
        random_seed=seed,
    )
    topk = int(calm_cfg.get("topk", 8))
    retrieve_multiplier = int(calm_cfg.get("retrieve_multiplier", 3))
    topk_per_span = int(calm_cfg.get("topk_per_span", topk))
    enable_fallback_span = bool(calm_cfg.get("enable_fallback_span", False))
    fallback_max_unigram = int(calm_cfg.get("fallback_max_unigram", 6))
    min_candidate_pool = int(calm_cfg.get("min_candidate_pool", 1))

    rows_all = read_jsonl(train_data, max_samples=None)
    if not rows_all:
        raise ValueError("Head-training dataset is empty.")
    use_internal_dev_split = bool(protocol_cfg.get("use_internal_dev_split", True))
    if use_internal_dev_split:
        dev_size = int(protocol_cfg.get("dev_size", 1000))
        split_seed = int(protocol_cfg.get("split_seed", 20260426))
        train_rows, dev_rows_unused, split_manifest = split_train_dev_rows(
            rows=rows_all,
            dev_size=dev_size,
            split_seed=split_seed,
        )
        with open(os.path.join(ioce_head_dir, "internal_dev_split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, ensure_ascii=False, indent=2)
    else:
        train_rows = rows_all
        dev_rows_unused = []

    if head_cfg.get("max_samples") is not None:
        train_rows = train_rows[: int(head_cfg.get("max_samples"))]

    dataset = StageEvalDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        max_length=int(head_cfg.get("max_length", 512)),
    )
    if len(dataset) == 0:
        raise ValueError("Head-training dataset is empty after tokenization.")
    loader = DataLoader(
        dataset,
        batch_size=int(head_cfg.get("train_batch_size", 1)),
        shuffle=True,
        collate_fn=EvalCollator(int(tokenizer.pad_token_id)),
    )

    hidden_size = int(model.base_model.config.hidden_size)
    confidence_head = ConfidenceHead(
        hidden_size=hidden_size,
        feat_dim=4,
        mlp_hidden=int(calm_cfg.get("mlp_hidden", 64)),
        dropout=float(calm_cfg.get("mlp_dropout", 0.1)),
    ).to(model_device)
    head_dtype = next(confidence_head.parameters()).dtype

    params = [p for p in confidence_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params=params,
        lr=float(head_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(head_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(head_cfg.get("epochs", 2))
    grad_accum = int(head_cfg.get("grad_accum_steps", 8))
    grad_clip = float(head_cfg.get("grad_clip", 1.0))
    log_steps = int(head_cfg.get("log_steps", 20))
    lambda_cal = float(nart_cfg.get("lambda_cal", 0.2))
    lambda_noise = float(nart_cfg.get("lambda_noise", 0.2))
    margin = float(nart_cfg.get("margin", 0.2))
    total_opt_steps = epochs * int(math.ceil(len(loader) / float(max(1, grad_accum))))

    global_step = 0
    seen_samples = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        confidence_head.train()
        pbar = tqdm(loader, **tqdm_kwargs(progress_cfg, "IOCE-Head Epoch {}/{}".format(epoch, epochs)))
        running_loss = 0.0
        running_cov = 0
        running_fb = 0
        running_samples = 0

        for step, batch in enumerate(pbar, start=1):
            tensor_batch = {
                "input_ids": batch["input_ids"].to(model_device),
                "attention_mask": batch["attention_mask"].to(model_device),
                "lexicon_ids": batch["lexicon_ids"].to(model_device),
                "lexicon_mask": batch["lexicon_mask"].to(model_device),
                "morph_ids": batch["morph_ids"].to(model_device),
                "morph_mask": batch["morph_mask"].to(model_device),
            }
            seen_samples += int(tensor_batch["input_ids"].size(0))

            with torch.no_grad():
                inputs_embeds = model._inject(
                    input_ids=tensor_batch["input_ids"],
                    lexicon_ids=tensor_batch["lexicon_ids"],
                    lexicon_mask=tensor_batch["lexicon_mask"],
                    morph_ids=tensor_batch["morph_ids"],
                    morph_mask=tensor_batch["morph_mask"],
                )
                outputs = model.base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=tensor_batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                logits = outputs.logits
                last_hidden = outputs.hidden_states[-1]

            sample_losses = []
            covered_in_batch = 0

            for b_idx, source_text in enumerate(batch["source"]):
                clean_candidates, hard_pool, covered, used_fallback = collect_clean_candidates(
                    source_text=source_text,
                    span_constructor=span_constructor,
                    memory=memory,
                    topk=topk,
                    retrieve_multiplier=retrieve_multiplier,
                    topk_per_span=topk_per_span,
                    enable_fallback_span=enable_fallback_span,
                    fallback_max_unigram=fallback_max_unigram,
                    min_candidate_pool=min_candidate_pool,
                )
                covered_in_batch += covered
                running_fb += int(used_fallback)
                if not clean_candidates:
                    continue

                attn_mask = tensor_batch["attention_mask"][b_idx].bool()
                valid_hidden = last_hidden[b_idx][attn_mask]
                if valid_hidden.numel() == 0:
                    continue
                h_x_pool = valid_hidden.mean(dim=0)
                entropy_x = compute_sentence_entropy_from_logits(logits[b_idx], attn_mask).item()

                clean_features = build_feature_tensor(
                    tokenizer=tokenizer,
                    embedding_layer=embedding_layer,
                    device=model_device,
                    h_x_pool=h_x_pool,
                    entropy_x=entropy_x,
                    candidates=clean_candidates,
                )
                if clean_features.numel() == 0:
                    continue
                clean_scores = confidence_head(clean_features.to(head_dtype))

                noisy_candidates = nart_sampler.build_noisy_candidates(
                    memory=memory,
                    clean_candidates=clean_candidates,
                    topk=topk,
                    hard_pool=hard_pool,
                )
                default_span = str(clean_candidates[0].get("span", "")) if clean_candidates else ""
                for cand in noisy_candidates:
                    if "span" not in cand:
                        cand["span"] = default_span

                noisy_features = build_feature_tensor(
                    tokenizer=tokenizer,
                    embedding_layer=embedding_layer,
                    device=model_device,
                    h_x_pool=h_x_pool,
                    entropy_x=entropy_x,
                    candidates=noisy_candidates,
                )
                if noisy_features.numel() > 0:
                    noisy_scores = confidence_head(noisy_features.to(head_dtype))
                else:
                    noisy_scores = torch.empty(0, device=model_device, dtype=head_dtype)

                l_cal = calibration_bce_loss(clean_scores, noisy_scores)
                l_noise = margin_ranking_noise_loss(clean_scores, noisy_scores, margin=margin)
                sample_losses.append(lambda_cal * l_cal + lambda_noise * l_noise)

            if sample_losses:
                loss = torch.stack(sample_losses).mean()
                (loss / float(grad_accum)).backward()
                running_loss += float(loss.detach().item())
                running_cov += int(covered_in_batch)
                running_samples += int(tensor_batch["input_ids"].size(0))

            should_step = (step % grad_accum == 0) or (step == len(loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = running_loss / float(max(1, grad_accum))
                cov_rate = float(running_cov / max(1, running_samples))
                fb_rate = float(running_fb / max(1, running_samples))
                running_loss = 0.0
                running_cov = 0
                running_fb = 0
                running_samples = 0
                elapsed = time.time() - start_time
                eta_stage = estimate_remaining_seconds(global_step, total_opt_steps, elapsed)
                samples_per_sec = float(seen_samples / max(1e-8, elapsed))
                pbar.set_postfix(
                    {
                        "epoch": epoch,
                        "gstep": global_step,
                        "loss": round(avg_loss, 4),
                        "cov": round(cov_rate, 3),
                        "fb": round(fb_rate, 3),
                        "eta_s": int(eta_stage) if eta_stage >= 0 else -1,
                    },
                    refresh=False,
                )
                if global_step % max(1, log_steps) == 0:
                    row = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": round(avg_loss, 6),
                        "coverage_rate": round(cov_rate, 6),
                        "fallback_hit_rate": round(fb_rate, 6),
                        "elapsed_sec": round(elapsed, 2),
                        "eta_stage_sec": round(eta_stage, 2) if eta_stage >= 0 else -1.0,
                        "eta_total_sec": round(eta_stage, 2) if eta_stage >= 0 else -1.0,
                        "samples_per_sec": round(samples_per_sec, 4),
                        "steps_per_sec": round(global_step / max(1e-8, elapsed), 6),
                    }
                    append_jsonl(log_file, row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)

    calibration_info = {
        "temperature": 1.0,
        "brier_before": 0.0,
        "brier_after": 0.0,
        "num_pairs": 0,
        "num_dev_samples": len(dev_rows_unused),
        "covered_rate_dev": 0.0,
        "fallback_hit_rate_dev": 0.0,
    }
    if dev_rows_unused:
        calib_rows = list(dev_rows_unused)
        calib_max = eval_cfg.get("calibration_max_samples")
        if calib_max is not None:
            calib_rows = calib_rows[: int(calib_max)]
        pair_info = _collect_calibration_pairs(
            rows=calib_rows,
            tokenizer=tokenizer,
            model=model,
            model_device=model_device,
            embedding_layer=embedding_layer,
            confidence_head=confidence_head,
            head_dtype=head_dtype,
            span_constructor=span_constructor,
            memory=memory,
            calm_cfg=calm_cfg,
            max_length=int(eval_cfg.get("max_length", head_cfg.get("max_length", 512))),
            batch_size=int(eval_cfg.get("batch_size", 1)),
        )
        probs = pair_info["probs"]
        labels = pair_info["labels"]
        fit = _fit_temperature_by_brier(probs=probs, labels=labels)
        calibration_info = {
            "temperature": float(fit["temperature"]),
            "brier_before": float(fit["brier_before"]),
            "brier_after": float(fit["brier_after"]),
            "num_pairs": len(probs),
            "num_dev_samples": len(calib_rows),
            "covered_rate_dev": float(pair_info["covered_rate"]),
            "fallback_hit_rate_dev": float(pair_info["fallback_hit_rate"]),
        }
        print("[Calibration] {}".format(json.dumps(calibration_info, ensure_ascii=False)), flush=True)

    head_path = os.path.join(ioce_head_dir, "confidence_head.pt")
    save_confidence_head(head_path, confidence_head)
    calibration_path = os.path.join(ioce_head_dir, "calibration.json")
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration_info, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ioce_head_dir, "head_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "framework": "IOCE-Lexical",
                "input_ckpt_root": ckpt_root,
                "head_checkpoint": head_path,
                "num_train_samples": len(dataset),
                "seed": seed,
                "use_internal_dev_split": use_internal_dev_split,
                "num_unused_dev_rows": len(dev_rows_unused),
                "calibration_temperature": float(calibration_info["temperature"]),
                "calibration_path": calibration_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("IOCE head training complete.")
    print("input_ckpt_root =", ckpt_root)
    print("head_checkpoint =", head_path)


if __name__ == "__main__":
    main()
