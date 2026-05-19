from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from peft import PeftModel

from calm_trans import ConfidenceHead, encode_text_mean, simple_source_tokens
from lmsa_aps import LMSAModel, load_lmsa_checkpoint
from train_lmsa_aps import load_base_model


def resolve_input_ckpt_paths(ckpt_root: str) -> Dict[str, str]:
    dad_adapter_dir = os.path.join(ckpt_root, "dad_adapter")
    lmsa_path = os.path.join(ckpt_root, "lmsa", "best_lmsa.pt")
    if not os.path.exists(lmsa_path):
        alt = os.path.join(ckpt_root, "best_lmsa.pt")
        if os.path.exists(alt):
            lmsa_path = alt
    return {
        "dad_adapter_dir": dad_adapter_dir,
        "lmsa_path": lmsa_path,
    }


def validate_input_ckpt_root(ckpt_root: str, context: str = "") -> Dict[str, str]:
    root = str(ckpt_root).strip()
    if not root:
        hint = " Set paths.input_ckpt_root or pass --ckpt when skipping input-side training."
        if context:
            hint = " [{}]{}".format(context, hint)
        raise ValueError("Input checkpoint root is empty.{}".format(hint))
    if not os.path.isdir(root):
        raise FileNotFoundError("Input checkpoint root does not exist: {}".format(root))

    ckpt_paths = resolve_input_ckpt_paths(root)
    dad_adapter_dir = ckpt_paths["dad_adapter_dir"]
    lmsa_path = ckpt_paths["lmsa_path"]

    missing = []
    if not os.path.isdir(dad_adapter_dir):
        missing.append(dad_adapter_dir)
    if not os.path.isfile(lmsa_path):
        missing.append(lmsa_path)
    if missing:
        raise FileNotFoundError(
            "Input checkpoint root is invalid: {}. Missing required artifact(s): {}".format(
                root, ", ".join(missing)
            )
        )
    return ckpt_paths


def load_ioce_backbone(
    model_id: str,
    model_cfg: Dict[str, Any],
    lmsa_cfg: Dict[str, Any],
    ckpt_root: str,
) -> Tuple[LMSAModel, torch.device]:
    ckpt_paths = validate_input_ckpt_root(ckpt_root, context="load_ioce_backbone")
    dad_adapter_dir = ckpt_paths["dad_adapter_dir"]
    lmsa_path = ckpt_paths["lmsa_path"]

    base_model = load_base_model(model_id, model_cfg)
    base_with_dad = PeftModel.from_pretrained(base_model, dad_adapter_dir, is_trainable=False)
    for p in base_with_dad.parameters():
        p.requires_grad = False

    model = LMSAModel(
        base_model=base_with_dad,
        hidden_size=int(base_with_dad.config.hidden_size),
        dropout=float(lmsa_cfg.get("dropout", 0.1)),
    )
    load_lmsa_checkpoint(lmsa_path, model)
    model_device = next(base_with_dad.parameters()).device
    model = model.to(model_device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, model_device


def collect_clean_candidates(
    source_text: str,
    span_constructor: Any,
    memory: Any,
    topk: int,
    retrieve_multiplier: int = 3,
    topk_per_span: int = 0,
    enable_fallback_span: bool = False,
    fallback_max_unigram: int = 6,
    min_candidate_pool: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    spans = span_constructor.build_spans(source_text)
    if not spans:
        return [], [], 0, 0

    clean = []
    hard_pool = []
    merged = {}
    used_fallback = 0
    retrieve_multiplier = max(1, int(retrieve_multiplier))
    topk_per_span = int(topk_per_span) if int(topk_per_span) > 0 else int(topk)
    min_candidate_pool = max(1, int(min_candidate_pool))
    retrieve_k = max(1, int(topk_per_span) * int(retrieve_multiplier))

    def _merge_span_candidates(span_text: str, span_candidates: Sequence[Dict[str, Any]]) -> None:
        for rank, cand in enumerate(span_candidates):
            item = dict(cand)
            item["span"] = span_text
            if rank < topk_per_span:
                key = item.get("candidate_text", "")
                score = float(item.get("p_cand_given_span", 0.0))
                if key not in merged or score > float(merged[key].get("p_cand_given_span", 0.0)):
                    merged[key] = item
            else:
                hard_pool.append(item)

    for span in spans:
        span_candidates = memory.retrieve(span.text, topk=retrieve_k)
        _merge_span_candidates(span.text, span_candidates)

    clean = list(merged.values())
    clean.sort(key=lambda x: (x.get("p_cand_given_span", 0.0), x.get("freq", 0.0)), reverse=True)
    clean = clean[:topk]
    main_clean_size = len(clean)

    if bool(enable_fallback_span) and len(clean) < min_candidate_pool:
        tokens = simple_source_tokens(source_text)
        fallback_tokens = tokens[: max(1, int(fallback_max_unigram))]
        for token in fallback_tokens:
            span_candidates = memory.retrieve(token, topk=retrieve_k)
            _merge_span_candidates(token, span_candidates)
        clean = list(merged.values())
        clean.sort(key=lambda x: (x.get("p_cand_given_span", 0.0), x.get("freq", 0.0)), reverse=True)
        clean = clean[:topk]
        if len(clean) > main_clean_size:
            used_fallback = 1

    return clean, hard_pool, int(len(clean) > 0), int(used_fallback)


def build_feature_tensor(
    tokenizer: Any,
    embedding_layer: torch.nn.Embedding,
    device: torch.device,
    h_x_pool: torch.Tensor,
    entropy_x: float,
    candidates: List[Dict[str, Any]],
) -> torch.Tensor:
    rows = []
    for cand in candidates:
        span_text = str(cand.get("span", ""))
        cand_text = str(cand.get("candidate_text", ""))
        h_span = encode_text_mean(tokenizer, embedding_layer, span_text, device=device)
        h_cand = encode_text_mean(tokenizer, embedding_layer, cand_text, device=device)

        sim_i = float(cand.get("p_cand_given_span", cand.get("align_score", 0.0)))
        freq_i = float(cand.get("freq", 0.0))
        len_i = float(cand.get("cand_len", max(1, len(cand_text))))
        extra = torch.tensor(
            [sim_i, float(torch.log1p(torch.tensor(freq_i)).item()), len_i / 8.0, float(entropy_x)],
            device=device,
            dtype=h_x_pool.dtype,
        )
        row = torch.cat([h_x_pool, h_span, h_cand, extra], dim=0)
        rows.append(row)

    if not rows:
        return torch.empty(0, device=device, dtype=h_x_pool.dtype)
    return torch.stack(rows, dim=0)


def save_confidence_head(path: str, head: ConfidenceHead) -> None:
    state = {
        "hidden_size": int(head.hidden_size),
        "feat_dim": int(head.feat_dim),
        "mlp_hidden": int(head.net[0].out_features),
        "mlp_dropout": float(head.net[2].p),
        "state_dict": head.state_dict(),
    }
    torch.save(state, path)


def load_confidence_head(path: str, device: torch.device) -> ConfidenceHead:
    state = torch.load(path, map_location=device)
    head = ConfidenceHead(
        hidden_size=int(state["hidden_size"]),
        feat_dim=int(state["feat_dim"]),
        mlp_hidden=int(state.get("mlp_hidden", 64)),
        dropout=float(state.get("mlp_dropout", 0.1)),
    ).to(device)
    head.load_state_dict(state["state_dict"])
    head.eval()
    return head


def choose_ckpt_root(cfg: Dict[str, Any], override: str = "") -> str:
    if override:
        return str(override)
    paths = cfg.get("paths", {})
    input_ckpt_root = str(paths.get("input_ckpt_root", "")).strip()
    if input_ckpt_root:
        return input_ckpt_root
    return str(paths.get("output_dir", "./outputs/ioce_lexical"))


def merge_term_set(clean_candidates: Sequence[Dict[str, Any]]) -> Set[str]:
    terms = set()
    for cand in clean_candidates:
        text = str(cand.get("candidate_text", "")).strip()
        if text:
            terms.add(text)
    return terms
