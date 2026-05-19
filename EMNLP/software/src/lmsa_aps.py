from __future__ import annotations

import json
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    import jieba  # type: ignore
except Exception:
    jieba = None

try:
    import sacrebleu  # type: ignore
except Exception:
    sacrebleu = None


def extract_lexicon_text(instruction: str) -> str:
    instruction = instruction or ""
    blocks = re.findall(r"\[(.*?)\]", instruction)
    if not blocks:
        return "[NONE]"
    text = " ; ".join([b.strip() for b in blocks if b.strip()])
    return text if text else "[NONE]"


def extract_morph_text(source: str) -> str:
    source = (source or "").lower()
    source = re.sub(r"[^a-zA-Z\-\s']", " ", source)
    tokens = source.split()

    suffixes = [
        "-i",
        "-ni",
        "-de",
        "-ci",
        "-be",
        "-ngge",
        "-ha",
        "-he",
        "-ho",
        "-mbi",
        "-fi",
        "-me",
        "-qi",
        "-rafi",
        "-se",
        "-tala",
        "-dari",
    ]
    tags = []
    for tok in tokens:
        for suf in suffixes:
            raw = suf[1:]
            if tok.endswith(raw) and len(tok) > len(raw) + 1:
                tags.append("SUF_{}".format(suf))

    seen = set()
    uniq = []
    for item in tags:
        if item not in seen:
            uniq.append(item)
            seen.add(item)
    return " ".join(uniq[:20]) if uniq else "[NO_MORPH]"


def extract_instruction_candidates(instruction: str) -> List[str]:
    instruction = instruction or ""
    blocks = re.findall(r"\[(.*?)\]", instruction)
    terms: List[str] = []
    seen = set()

    for block in blocks:
        for seg in block.split(","):
            seg = seg.strip()
            if not seg or ":" not in seg:
                continue
            rhs = seg.split(":", 1)[1].strip()
            if not rhs:
                continue
            for item in rhs.split("/"):
                term = item.strip()
                if not term:
                    continue
                if term not in seen:
                    seen.add(term)
                    terms.append(term)
    return terms


def candidate_stats(instruction: str, prediction: str) -> Dict[str, int]:
    terms = extract_instruction_candidates(instruction)
    pred = prediction or ""
    covered = 1 if any((t and t in pred) for t in terms) else 0
    return {
        "num_candidates": len(terms),
        "covered": int(covered),
    }


def decode_generated_prediction(
    output_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    empty_fallback: str = "EMPTY",
) -> str:
    # `generate` may return either:
    # 1) prompt + generated ids, or
    # 2) generated ids only (when driven by inputs_embeds).
    prompt_len = int(attention_mask.sum().item())
    if int(output_ids.shape[0]) > prompt_len:
        gen_ids = output_ids[prompt_len:]
    else:
        gen_ids = output_ids

    pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if not pred:
        return empty_fallback
    return pred


def build_eval_prediction_row(
    source: str,
    prediction: str,
    label: str,
    instruction: str,
    alpha_used: float,
) -> Dict[str, Any]:
    stats = candidate_stats(instruction=instruction, prediction=prediction)
    return {
        "source": source,
        "predict": prediction,
        "label": label,
        "num_candidates": int(stats["num_candidates"]),
        "covered": int(stats["covered"]),
        "alpha_used": float(alpha_used),
    }


def build_user_content(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if instruction and input_text:
        return instruction + "\n" + input_text
    return instruction or input_text


def build_prompt_text(tokenizer: Any, instruction: str, input_text: str) -> str:
    user_content = build_user_content(instruction, input_text)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": user_content}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return user_content


def build_full_text(tokenizer: Any, instruction: str, input_text: str, output_text: str) -> str:
    user_content = build_user_content(instruction, input_text)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output_text},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return user_content + "\n" + str(output_text)


def normalize_record(row: Dict[str, Any]) -> Dict[str, str]:
    instruction = str(row.get("instruction", "")).strip()
    source = str(row.get("input", row.get("query", ""))).strip()
    target = str(row.get("output", row.get("response", ""))).strip()
    return {
        "instruction": instruction,
        "source": source,
        "target": target,
    }


class StageTrainDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512,
        include_features: bool = False,
    ) -> None:
        self.samples = []
        for row in rows:
            rec = normalize_record(row)
            prompt_text = build_prompt_text(tokenizer, rec["instruction"], rec["source"])
            full_text = build_full_text(tokenizer, rec["instruction"], rec["source"], rec["target"])

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_enc = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            input_ids = list(full_enc["input_ids"])
            attention_mask = list(full_enc["attention_mask"])
            if len(input_ids) < 2:
                continue

            labels = list(input_ids)
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

            item = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            if include_features:
                lex_text = extract_lexicon_text(rec["instruction"])
                morph_text = extract_morph_text(rec["source"])
                lex_enc = tokenizer(
                    lex_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=48,
                )
                morph_enc = tokenizer(
                    morph_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=48,
                )
                item["lexicon_ids"] = list(lex_enc["input_ids"]) or [int(tokenizer.eos_token_id)]
                item["lexicon_mask"] = list(lex_enc["attention_mask"]) or [1]
                item["morph_ids"] = list(morph_enc["input_ids"]) or [int(tokenizer.eos_token_id)]
                item["morph_mask"] = list(morph_enc["attention_mask"]) or [1]

            self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class StageEvalDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.samples = []
        for row in rows:
            rec = normalize_record(row)
            prompt_text = build_prompt_text(tokenizer, rec["instruction"], rec["source"])
            prompt_enc = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            lex_text = extract_lexicon_text(rec["instruction"])
            morph_text = extract_morph_text(rec["source"])
            lex_enc = tokenizer(
                lex_text,
                add_special_tokens=False,
                truncation=True,
                max_length=48,
            )
            morph_enc = tokenizer(
                morph_text,
                add_special_tokens=False,
                truncation=True,
                max_length=48,
            )

            self.samples.append(
                {
                    "instruction": rec["instruction"],
                    "source": rec["source"],
                    "target": rec["target"],
                    "input_ids": list(prompt_enc["input_ids"]),
                    "attention_mask": list(prompt_enc["attention_mask"]),
                    "lexicon_ids": list(lex_enc["input_ids"]) or [int(tokenizer.eos_token_id)],
                    "lexicon_mask": list(lex_enc["attention_mask"]) or [1],
                    "morph_ids": list(morph_enc["input_ids"]) or [int(tokenizer.eos_token_id)],
                    "morph_mask": list(morph_enc["attention_mask"]) or [1],
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


@dataclass
class Stage1Collator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        def _pad_2d(key: str, pad_value: int) -> torch.Tensor:
            max_len = max(len(x[key]) for x in features)
            out = []
            for x in features:
                seq = x[key]
                out.append(seq + [pad_value] * (max_len - len(seq)))
            return torch.tensor(out, dtype=torch.long)

        return {
            "input_ids": _pad_2d("input_ids", self.pad_token_id),
            "attention_mask": _pad_2d("attention_mask", 0),
            "labels": _pad_2d("labels", -100),
        }


@dataclass
class Stage2Collator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        def _pad_2d(key: str, pad_value: int) -> torch.Tensor:
            max_len = max(len(x[key]) for x in features)
            out = []
            for x in features:
                seq = x[key]
                out.append(seq + [pad_value] * (max_len - len(seq)))
            return torch.tensor(out, dtype=torch.long)

        return {
            "input_ids": _pad_2d("input_ids", self.pad_token_id),
            "attention_mask": _pad_2d("attention_mask", 0),
            "labels": _pad_2d("labels", -100),
            "lexicon_ids": _pad_2d("lexicon_ids", self.pad_token_id),
            "lexicon_mask": _pad_2d("lexicon_mask", 0),
            "morph_ids": _pad_2d("morph_ids", self.pad_token_id),
            "morph_mask": _pad_2d("morph_mask", 0),
        }


@dataclass
class EvalCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        def _pad_2d(key: str, pad_value: int) -> torch.Tensor:
            max_len = max(len(x[key]) for x in features)
            out = []
            for x in features:
                seq = x[key]
                out.append(seq + [pad_value] * (max_len - len(seq)))
            return torch.tensor(out, dtype=torch.long)

        return {
            "instruction": [x["instruction"] for x in features],
            "source": [x["source"] for x in features],
            "target": [x["target"] for x in features],
            "input_ids": _pad_2d("input_ids", self.pad_token_id),
            "attention_mask": _pad_2d("attention_mask", 0),
            "lexicon_ids": _pad_2d("lexicon_ids", self.pad_token_id),
            "lexicon_mask": _pad_2d("lexicon_mask", 0),
            "morph_ids": _pad_2d("morph_ids", self.pad_token_id),
            "morph_mask": _pad_2d("morph_mask", 0),
        }


class LMSAModel(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.base_model = base_model
        self.lmsa = nn.ModuleDict(
            {
                "lex_proj": nn.Linear(hidden_size, hidden_size, bias=False),
                "morph_proj": nn.Linear(hidden_size, hidden_size, bias=False),
                "gate": nn.Linear(hidden_size * 2, hidden_size, bias=True),
                "dropout": nn.Dropout(float(dropout)),
            }
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _mean_pool(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb_layer = self.base_model.get_input_embeddings()
        emb = emb_layer(ids)
        mask = mask.unsqueeze(-1).to(emb.dtype)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _inject(
        self,
        input_ids: torch.Tensor,
        lexicon_ids: torch.Tensor,
        lexicon_mask: torch.Tensor,
        morph_ids: torch.Tensor,
        morph_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb_layer = self.base_model.get_input_embeddings()
        input_embeds = emb_layer(input_ids)

        lex_vec = self._mean_pool(lexicon_ids, lexicon_mask)
        morph_vec = self._mean_pool(morph_ids, morph_mask)

        proj_dtype = self.lmsa["lex_proj"].weight.dtype
        lex_vec = lex_vec.to(proj_dtype)
        morph_vec = morph_vec.to(proj_dtype)

        lex_vec = torch.tanh(self.lmsa["lex_proj"](lex_vec))
        morph_vec = torch.tanh(self.lmsa["morph_proj"](morph_vec))

        gate_in = torch.cat([lex_vec, morph_vec], dim=-1).to(self.lmsa["gate"].weight.dtype)
        gate = torch.sigmoid(self.lmsa["gate"](gate_in))
        fused = gate * lex_vec + (1.0 - gate) * morph_vec
        fused = self.lmsa["dropout"](fused)

        alpha = torch.sigmoid(self.scale).view(1, 1, 1).to(input_embeds.dtype)
        return input_embeds + alpha * fused.unsqueeze(1).to(input_embeds.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        lexicon_ids: Optional[torch.Tensor] = None,
        lexicon_mask: Optional[torch.Tensor] = None,
        morph_ids: Optional[torch.Tensor] = None,
        morph_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        if lexicon_ids is None or lexicon_mask is None or morph_ids is None or morph_mask is None:
            raise ValueError("LMSAModel forward requires lexicon/morph feature tensors.")
        inputs_embeds = self._inject(input_ids, lexicon_ids, lexicon_mask, morph_ids, morph_mask)
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lexicon_ids: torch.Tensor,
        lexicon_mask: torch.Tensor,
        morph_ids: torch.Tensor,
        morph_mask: torch.Tensor,
        **gen_kwargs: Any,
    ) -> torch.Tensor:
        inputs_embeds = self._inject(input_ids, lexicon_ids, lexicon_mask, morph_ids, morph_mask)
        return self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )


def save_lmsa_checkpoint(path: str, model: LMSAModel) -> None:
    torch.save(
        {
            "lmsa": model.lmsa.state_dict(),
            "scale": model.scale.detach().cpu(),
        },
        path,
    )


def load_lmsa_checkpoint(path: str, model: LMSAModel) -> Dict[str, str]:
    state = torch.load(path, map_location="cpu")
    meta = {"format": "legacy_lmsa_only"}
    if isinstance(state, dict) and "lmsa" in state:
        model.lmsa.load_state_dict(state["lmsa"])
        if "scale" in state:
            with torch.no_grad():
                model.scale.copy_(state["scale"].to(model.scale.dtype))
        meta["format"] = "full_lmsa_checkpoint"
        return meta
    model.lmsa.load_state_dict(state)
    return meta


def get_amp_context(device: torch.device, amp_dtype: torch.dtype = torch.bfloat16) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    if sacrebleu is None:
        char_bleu4 = 0.0
        word_bleu4 = 0.0
        chrfpp = 0.0
    else:
        char_bleu4 = float(sacrebleu.corpus_bleu(preds, [refs], tokenize="zh").score)
        if jieba is not None:
            preds_word = [" ".join(jieba.lcut(x)) for x in preds]
            refs_word = [" ".join(jieba.lcut(x)) for x in refs]
        else:
            preds_word = [" ".join(list(x.strip())) for x in preds]
            refs_word = [" ".join(list(x.strip())) for x in refs]
        word_bleu4 = float(sacrebleu.corpus_bleu(preds_word, [refs_word]).score)
        chrfpp = float(sacrebleu.corpus_chrf(preds, [refs], word_order=2).score)

    return {
        "char_bleu4": round(char_bleu4, 4),
        "word_bleu4": round(word_bleu4, 4),
        "chrfpp": round(chrfpp, 4),
    }


def dump_predictions(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
