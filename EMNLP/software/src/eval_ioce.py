from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LogitsProcessorList

from calm_trans import (
    BiasBuilder,
    CalmLogitsProcessor,
    LexicalMemory,
    SpanConstructor,
    compute_sentence_entropy_from_logits,
)
from common import ensure_dir, load_yaml_config, read_jsonl, resolve_progress_cfg, tqdm_kwargs
from ioce_utils import (
    build_feature_tensor,
    choose_ckpt_root,
    collect_clean_candidates,
    load_confidence_head,
    load_ioce_backbone,
    resolve_input_ckpt_paths,
    validate_input_ckpt_root,
)
from lmsa_aps import EvalCollator, StageEvalDataset, compute_metrics, decode_generated_prediction, dump_predictions
from noise_training import brier_score, expected_calibration_error, term_accuracy
from train_lmsa_aps import load_base_model


def resolve_head_path(cfg: Dict[str, Any], override: str = "") -> str:
    if override:
        return override
    paths = cfg.get("paths", {})
    output_dir = str(paths.get("output_dir", "./outputs/ioce_lexical"))
    return os.path.join(output_dir, "ioce_head", "confidence_head.pt")


def validate_dad_only_ckpt_root(ckpt_root: str) -> str:
    root = str(ckpt_root).strip()
    if not root:
        raise ValueError("Input checkpoint root is empty for prompt_dad mode.")
    if not os.path.isdir(root):
        raise FileNotFoundError("Input checkpoint root does not exist: {}".format(root))
    ckpt_paths = resolve_input_ckpt_paths(root)
    dad_adapter_dir = ckpt_paths["dad_adapter_dir"]
    if not os.path.isdir(dad_adapter_dir):
        raise FileNotFoundError("DAD adapter checkpoint not found: {}".format(dad_adapter_dir))
    return dad_adapter_dir


def _safe_logit(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return float(math.log(p / (1.0 - p)))


def _apply_temperature(prob: float, temperature: float) -> float:
    t = max(1e-6, float(temperature))
    z = _safe_logit(prob) / t
    return float(1.0 / (1.0 + math.exp(-z)))


def _load_calibration_temperature(head_path: str) -> float:
    if not head_path:
        return 1.0
    head_dir = os.path.dirname(head_path)
    for filename in ["calibration.json", "head_meta.json"]:
        p = os.path.join(head_dir, filename)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if "temperature" in obj:
                return float(obj["temperature"])
            if "calibration_temperature" in obj:
                return float(obj["calibration_temperature"])
        except Exception:
            continue
    return 1.0


def _quantile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    q = min(max(float(q), 0.0), 1.0)
    idx = int(round(q * (len(arr) - 1)))
    return float(arr[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IOCE-Lexical model (LMSA input modulation + CALM decode bias).")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--ckpt", type=str, default="", help="Input-side checkpoint root containing dad_adapter and lmsa.")
    parser.add_argument("--head", type=str, default="", help="Path to confidence_head.pt.")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "prompt_dad", "static_bias"],
        help="Evaluation mode: full | prompt_dad | static_bias.",
    )
    parser.add_argument("--data_path", type=str, default="", help="Override evaluation data path.")
    parser.add_argument("--disable_bias", action="store_true", help="Disable decoder-time lexical bias.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    lmsa_cfg = cfg.get("lmsa", {})
    calm_cfg = cfg.get("calm", {})
    eval_cfg = cfg.get("eval", {})
    progress_cfg = resolve_progress_cfg(cfg.get("progress", {}))

    model_id = str(paths["base_model"])
    val_data = str(args.data_path).strip() or str(paths["val_data"])
    output_dir = str(paths.get("eval_output_dir", os.path.join(str(paths.get("output_dir", "./outputs/ioce_lexical")), "eval")))
    ensure_dir(output_dir)

    mode = str(args.mode).strip()
    ckpt_root = choose_ckpt_root(cfg, override=args.ckpt)
    head_path = resolve_head_path(cfg, override=args.head)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompt_dad_model = None
    confidence_head = None
    head_dtype = None

    if mode == "prompt_dad":
        dad_adapter_dir = validate_dad_only_ckpt_root(ckpt_root)
        base_model = load_base_model(model_id, model_cfg)
        prompt_dad_model = PeftModel.from_pretrained(base_model, dad_adapter_dir, is_trainable=False)
        for p in prompt_dad_model.parameters():
            p.requires_grad = False
        model_device = next(prompt_dad_model.parameters()).device
        prompt_dad_model = prompt_dad_model.to(model_device)
        prompt_dad_model.eval()
        model = None
        embedding_layer = None
    else:
        validate_input_ckpt_root(ckpt_root, context="eval_ioce")
        model, model_device = load_ioce_backbone(
            model_id=model_id,
            model_cfg=model_cfg,
            lmsa_cfg=lmsa_cfg,
            ckpt_root=ckpt_root,
        )
        embedding_layer = model.base_model.get_input_embeddings()
        if mode == "full":
            if not os.path.isfile(head_path):
                raise FileNotFoundError("confidence head checkpoint not found: {}".format(head_path))
            confidence_head = load_confidence_head(head_path, device=model_device)
            confidence_head.eval()
            head_dtype = next(confidence_head.parameters()).dtype

    memory = LexicalMemory(memory_path=str(paths["memory_path"]), topk=int(calm_cfg.get("topk", 8)))
    frequent_spans = memory.get_frequent_spans(min_freq=float(calm_cfg.get("frequent_span_min_freq", 3.0)))
    span_constructor = SpanConstructor(
        tokenizer=tokenizer,
        max_span_len=int(calm_cfg.get("max_span_len", 4)),
        use_longest_match=True,
        frequent_spans=frequent_spans,
        max_spans=int(calm_cfg.get("max_spans", 64)),
    )
    bias_builder = BiasBuilder(
        tokenizer=tokenizer,
        phrase_mode=str(calm_cfg.get("phrase_mode", "first_token")),
        min_conf=float(calm_cfg.get("min_confidence", 0.0)),
        confidence_power=float(calm_cfg.get("confidence_power", 1.0)),
        max_phrases=int(calm_cfg.get("max_phrases", 64)),
    )

    rows = read_jsonl(val_data, max_samples=eval_cfg.get("max_samples"))
    dataset = StageEvalDataset(
        rows=rows,
        tokenizer=tokenizer,
        max_length=int(eval_cfg.get("max_length", 512)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 1)),
        shuffle=False,
        collate_fn=EvalCollator(int(tokenizer.pad_token_id)),
    )

    use_bias = (mode != "prompt_dad") and (not args.disable_bias)
    dynamic_alpha = bool(calm_cfg.get("dynamic_alpha", False)) if mode == "full" else False
    alpha_static = float(calm_cfg.get("alpha_eval", 1.0))
    high_conf_threshold = float(eval_cfg.get("high_conf_threshold", 0.5))
    topk = int(calm_cfg.get("topk", 8))
    retrieve_multiplier = int(calm_cfg.get("retrieve_multiplier", 3))
    topk_per_span = int(calm_cfg.get("topk_per_span", topk))
    enable_fallback_span = bool(calm_cfg.get("enable_fallback_span", False))
    fallback_max_unigram = int(calm_cfg.get("fallback_max_unigram", 6))
    min_candidate_pool = int(calm_cfg.get("min_candidate_pool", 1))
    report_candidate_quantiles = bool(eval_cfg.get("report_candidate_quantiles", True))
    calib_cfg = eval_cfg.get("calibration_temperature", "auto")
    if isinstance(calib_cfg, str) and calib_cfg.strip().lower() == "auto":
        calibration_temperature = _load_calibration_temperature(head_path if mode == "full" else "")
    else:
        calibration_temperature = float(calib_cfg)

    pred_rows: List[Dict[str, Any]] = []
    preds: List[str] = []
    refs: List[str] = []
    confidence_probs: List[float] = []
    confidence_labels: List[int] = []

    total_sec = 0.0
    total_gen_tokens = 0
    covered_samples = 0
    fallback_hit_samples = 0
    candidate_counts: List[int] = []
    total_samples = 0
    total_term_hits = 0.0
    total_term_precision_denom = 0.0
    total_term_recall_denom = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, **tqdm_kwargs(progress_cfg, "IOCE Eval")):
            tensor_batch = {
                "input_ids": batch["input_ids"].to(model_device),
                "attention_mask": batch["attention_mask"].to(model_device),
                "lexicon_ids": batch["lexicon_ids"].to(model_device),
                "lexicon_mask": batch["lexicon_mask"].to(model_device),
                "morph_ids": batch["morph_ids"].to(model_device),
                "morph_mask": batch["morph_mask"].to(model_device),
            }
            if mode == "prompt_dad":
                prompt_out = prompt_dad_model(
                    input_ids=tensor_batch["input_ids"],
                    attention_mask=tensor_batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                prompt_logits = prompt_out.logits
                last_hidden = prompt_out.hidden_states[-1]
            else:
                inputs_embeds = model._inject(
                    input_ids=tensor_batch["input_ids"],
                    lexicon_ids=tensor_batch["lexicon_ids"],
                    lexicon_mask=tensor_batch["lexicon_mask"],
                    morph_ids=tensor_batch["morph_ids"],
                    morph_mask=tensor_batch["morph_mask"],
                )
                prompt_out = model.base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=tensor_batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                prompt_logits = prompt_out.logits
                last_hidden = prompt_out.hidden_states[-1]

            bsz = int(tensor_batch["input_ids"].size(0))
            for b_idx in range(bsz):
                source = str(batch["source"][b_idx])
                reference = str(batch["target"][b_idx])
                total_samples += 1
                prediction = "Пе"
                alpha_used = 0.0
                covered = 0
                used_fallback = 0
                clean_candidates: List[Dict[str, Any]] = []
                err_msg = ""

                try:
                    attn_mask = tensor_batch["attention_mask"][b_idx].bool()
                    valid_hidden = last_hidden[b_idx][attn_mask]
                    if valid_hidden.numel() == 0:
                        h_x_pool = last_hidden[b_idx].mean(dim=0)
                    else:
                        h_x_pool = valid_hidden.mean(dim=0)
                    entropy_x = compute_sentence_entropy_from_logits(prompt_logits[b_idx], attn_mask).item()

                    clean_candidates, _, covered, used_fallback = collect_clean_candidates(
                        source_text=source,
                        span_constructor=span_constructor,
                        memory=memory,
                        topk=topk,
                        retrieve_multiplier=retrieve_multiplier,
                        topk_per_span=topk_per_span,
                        enable_fallback_span=enable_fallback_span,
                        fallback_max_unigram=fallback_max_unigram,
                        min_candidate_pool=min_candidate_pool,
                    )
                    covered_samples += covered
                    fallback_hit_samples += int(used_fallback)
                    candidate_counts.append(len(clean_candidates))

                    if mode == "full" and clean_candidates:
                        features = build_feature_tensor(
                            tokenizer=tokenizer,
                            embedding_layer=embedding_layer,
                            device=model_device,
                            h_x_pool=h_x_pool,
                            entropy_x=entropy_x,
                            candidates=clean_candidates,
                        )
                        if features.numel() > 0:
                            scores = confidence_head(features.to(head_dtype))
                            for i, cand in enumerate(clean_candidates):
                                prob_raw = float(scores[i].item())
                                prob = _apply_temperature(prob_raw, calibration_temperature)
                                cand["confidence"] = prob
                                text = str(cand.get("candidate_text", ""))
                                confidence_probs.append(prob)
                                confidence_labels.append(int(text in reference))
                    elif mode == "static_bias" and clean_candidates:
                        for cand in clean_candidates:
                            cand["confidence"] = 1.0

                    if mode == "prompt_dad":
                        vocab_size = int(prompt_dad_model.config.vocab_size)
                    else:
                        vocab_size = int(model.base_model.config.vocab_size)
                    bias_vec, _ = bias_builder.build_bias(
                        candidates=clean_candidates,
                        vocab_size=vocab_size,
                        device=model_device,
                        dtype=prompt_logits.dtype,
                    )
                    logits_processors = LogitsProcessorList()
                    processor = None
                    if use_bias and bias_vec.abs().sum().item() > 0:
                        processor = CalmLogitsProcessor(
                            bias_vector=bias_vec,
                            alpha=alpha_static,
                            dynamic_alpha=dynamic_alpha,
                            w1=float(calm_cfg.get("dynamic_alpha_w1", 1.0)),
                            w2=float(calm_cfg.get("dynamic_alpha_w2", 0.0)),
                            bias_term=float(calm_cfg.get("dynamic_alpha_b", 0.0)),
                            coverage=float(covered),
                            min_alpha=float(calm_cfg.get("alpha_min", 0.0)),
                            max_alpha=float(calm_cfg.get("alpha_max", 1.0)),
                        )
                        logits_processors.append(processor)

                    gen_kwargs: Dict[str, Any] = {
                        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 128)),
                        "do_sample": bool(eval_cfg.get("do_sample", False)),
                        "num_beams": int(eval_cfg.get("num_beams", 1)),
                        "repetition_penalty": float(eval_cfg.get("repetition_penalty", 1.15)),
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                    if logits_processors:
                        gen_kwargs["logits_processor"] = logits_processors

                    t0 = time.perf_counter()
                    if mode == "prompt_dad":
                        generated = prompt_dad_model.generate(
                            input_ids=tensor_batch["input_ids"][b_idx : b_idx + 1],
                            attention_mask=tensor_batch["attention_mask"][b_idx : b_idx + 1],
                            **gen_kwargs,
                        )
                    else:
                        generated = model.generate(
                            input_ids=tensor_batch["input_ids"][b_idx : b_idx + 1],
                            attention_mask=tensor_batch["attention_mask"][b_idx : b_idx + 1],
                            lexicon_ids=tensor_batch["lexicon_ids"][b_idx : b_idx + 1],
                            lexicon_mask=tensor_batch["lexicon_mask"][b_idx : b_idx + 1],
                            morph_ids=tensor_batch["morph_ids"][b_idx : b_idx + 1],
                            morph_mask=tensor_batch["morph_mask"][b_idx : b_idx + 1],
                            **gen_kwargs,
                        )
                    t1 = time.perf_counter()
                    total_sec += (t1 - t0)
                    if processor is not None:
                        alpha_used = float(processor.last_alpha)

                    prediction = decode_generated_prediction(
                        output_ids=generated[0],
                        attention_mask=tensor_batch["attention_mask"][b_idx],
                        tokenizer=tokenizer,
                        empty_fallback="Пе",
                    )
                    input_len = int(tensor_batch["attention_mask"][b_idx].sum().item())
                    total_gen_tokens += int(max(1, int(generated[0].shape[0]) - input_len))

                    if mode == "full":
                        high_conf_terms = []
                        for cand in clean_candidates:
                            if float(cand.get("confidence", 0.0)) >= high_conf_threshold:
                                high_conf_terms.append(str(cand.get("candidate_text", "")))
                        term_stat = term_accuracy(
                            high_conf_candidates=high_conf_terms,
                            prediction=prediction,
                            reference=reference,
                        )
                        total_term_hits += float(term_stat["term_hits"])
                        total_term_precision_denom += float(sum(1 for term in high_conf_terms if term and term in prediction))
                        total_term_recall_denom += float(sum(1 for term in high_conf_terms if term and term in reference))
                except Exception as exc:
                    err_msg = "{}: {}".format(type(exc).__name__, str(exc))
                    prediction = "[GEN_ERROR]"

                preds.append(prediction)
                refs.append(reference)
                pred_rows.append(
                    {
                        "source": source,
                        "predict": prediction,
                        "label": reference,
                        "num_candidates": len(clean_candidates),
                        "covered": int(covered),
                        "used_fallback": int(used_fallback),
                        "alpha_used": float(alpha_used),
                        "error": err_msg,
                    }
                )

    core_metrics = compute_metrics(preds, refs)
    coverage_rate = float(covered_samples / max(1, total_samples))
    fallback_hit_rate = float(fallback_hit_samples / max(1, total_samples))
    latency_ms_per_token = float((total_sec / max(1, total_gen_tokens)) * 1000.0)
    if mode == "full":
        ece = expected_calibration_error(confidence_probs, confidence_labels, n_bins=int(eval_cfg.get("ece_bins", 10)))
        brier = brier_score(confidence_probs, confidence_labels)
        term_precision = float(total_term_hits / max(1.0, total_term_precision_denom))
        term_recall = float(total_term_hits / max(1.0, total_term_recall_denom))
    else:
        ece = 0.0
        brier = 0.0
        term_precision = 0.0
        term_recall = 0.0

    metrics = {
        "num_samples": total_samples,
        "word_bleu4": float(core_metrics.get("word_bleu4", 0.0)),
        "char_bleu4": float(core_metrics.get("char_bleu4", 0.0)),
        "chrfpp": float(core_metrics.get("chrfpp", 0.0)),
        "coverage_rate": coverage_rate,
        "fallback_hit_rate": fallback_hit_rate,
        "latency_ms_per_token": latency_ms_per_token,
        "ece": ece,
        "brier": brier,
        "term_precision": term_precision,
        "term_recall": term_recall,
        "bias_enabled": bool(use_bias),
        "mode": mode,
        "input_ckpt_root": ckpt_root,
        "confidence_head": head_path if mode == "full" else "",
        "calibration_temperature_used": float(calibration_temperature),
    }
    if report_candidate_quantiles:
        metrics["candidate_count_p50"] = _quantile(candidate_counts, 0.5)
        metrics["candidate_count_p90"] = _quantile(candidate_counts, 0.9)

    pred_path = os.path.join(output_dir, "generated_predictions.jsonl")
    metric_path = os.path.join(output_dir, "metrics.json")
    dump_predictions(pred_path, pred_rows)
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("IOCE evaluation complete.")
    print(
        "Final Metrics | word_bleu4={:.4f} | char_bleu4={:.4f} | chrfpp={:.4f}".format(
            metrics["word_bleu4"], metrics["char_bleu4"], metrics["chrfpp"]
        )
    )
    print("Prediction file:", pred_path)
    print("Metric file:", metric_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
