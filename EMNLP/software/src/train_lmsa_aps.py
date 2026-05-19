from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
from lmsa_aps import (
    EvalCollator,
    LMSAModel,
    Stage1Collator,
    Stage2Collator,
    StageEvalDataset,
    StageTrainDataset,
    build_eval_prediction_row,
    compute_metrics,
    decode_generated_prediction,
    dump_predictions,
    get_amp_context,
    save_lmsa_checkpoint,
)


def maybe_build_qconfig(model_cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=bool(model_cfg.get("use_4bit", True)),
        bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=getattr(torch, str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))),
    )


def load_base_model(model_id: str, model_cfg: Dict[str, Any]) -> torch.nn.Module:
    use_4bit = bool(model_cfg.get("use_4bit", True))
    if use_4bit:
        qconfig = maybe_build_qconfig(model_cfg)
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=qconfig,
            device_map=model_cfg.get("device_map", "auto"),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, str(model_cfg.get("torch_dtype", "bfloat16"))),
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )


def build_lora_model(base_model: torch.nn.Module, model_cfg: Dict[str, Any]) -> torch.nn.Module:
    if bool(model_cfg.get("use_4bit", True)):
        base_model = prepare_model_for_kbit_training(base_model)
    lora_cfg = LoraConfig(
        r=int(model_cfg.get("lora_rank", 16)),
        lora_alpha=int(model_cfg.get("lora_alpha", 32)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        target_modules=list(
            model_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model


def _resume_path(output_dir: str, stage_name: str) -> str:
    resume_dir = os.path.join(output_dir, "resume_ckpt")
    ensure_dir(resume_dir)
    return os.path.join(resume_dir, "{}_latest.pt".format(stage_name))


def _to_cpu_obj(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_obj(v) for v in obj)
    return obj


def _get_trainable_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = param.detach().cpu().clone()
    return state


def _load_trainable_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, saved_tensor in state.items():
            param = named_params.get(name)
            if param is None:
                continue
            param.copy_(saved_tensor.to(device=param.device, dtype=param.dtype))


def _move_optimizer_state_to_param_device(optimizer: torch.optim.Optimizer) -> None:
    for param, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(param.device)


def _save_resume_state(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _load_resume_state(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu")


def _build_epoch_sampler(dataset: Any, seed: int, epoch: int, offset: int = 0) -> RandomSampler:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset) + int(epoch) * 1009)
    return RandomSampler(dataset, generator=generator)


def _clear_resume_state(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def run_stage1_dad_training(
    model_id: str,
    train_rows: List[Dict[str, Any]],
    tokenizer: Any,
    model_cfg: Dict[str, Any],
    stage1_cfg: Dict[str, Any],
    progress_cfg: Dict[str, Any],
    output_dir: str,
    seed: int = 42,
    resume: bool = False,
    resume_save_steps: int = 50,
    pipeline_total_opt_steps: int = 0,
    pipeline_completed_before: int = 0,
    pipeline_start_time: float = 0.0,
) -> str:
    dad_adapter_dir = os.path.join(output_dir, "dad_adapter")
    ensure_dir(dad_adapter_dir)
    log_file = os.path.join(output_dir, "stage1_dad_train_log.jsonl")
    resume_file = _resume_path(output_dir, "stage1")
    resume_state = _load_resume_state(resume_file) if resume else None
    if os.path.exists(log_file) and resume_state is None:
        os.remove(log_file)

    base_model = load_base_model(model_id, model_cfg)
    model = build_lora_model(base_model, model_cfg)
    model_device = next(model.parameters()).device
    model.train()

    dataset = StageTrainDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        max_length=int(stage1_cfg.get("max_length", 512)),
        include_features=False,
    )
    if len(dataset) == 0:
        raise ValueError("Stage-1 dataset is empty after tokenization.")

    batch_size = int(stage1_cfg.get("train_batch_size", 1))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params=params,
        lr=float(stage1_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(stage1_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(stage1_cfg.get("epochs", 1))
    grad_accum = int(stage1_cfg.get("grad_accum_steps", 8))
    grad_clip = float(stage1_cfg.get("grad_clip", 1.0))
    log_steps = int(stage1_cfg.get("log_steps", 10))
    steps_per_epoch = int(math.ceil(len(dataset) / float(max(1, batch_size))))
    stage_total_opt_steps = epochs * int(math.ceil(steps_per_epoch / float(max(1, grad_accum))))

    global_step = 0
    stage_seen_samples = 0
    start_epoch = 1
    resume_step_in_epoch = 0
    if resume_state is not None:
        saved_trainable = resume_state.get("trainable_state", {})
        if isinstance(saved_trainable, dict) and saved_trainable:
            _load_trainable_state(model, saved_trainable)
        saved_opt = resume_state.get("optimizer_state")
        if saved_opt is not None:
            try:
                optimizer.load_state_dict(saved_opt)
                _move_optimizer_state_to_param_device(optimizer)
            except Exception as exc:
                print("[Resume][Stage1] optimizer state ignored due to mismatch: {}".format(exc), flush=True)
        start_epoch = max(1, int(resume_state.get("epoch", 1)))
        resume_step_in_epoch = max(0, int(resume_state.get("step_in_epoch", 0)))
        global_step = max(0, int(resume_state.get("global_step", 0)))
        stage_seen_samples = max(0, int(resume_state.get("stage_seen_samples", 0)))
        print(
            "[Resume][Stage1] epoch={}, step_in_epoch={}, global_step={}".format(
                start_epoch, resume_step_in_epoch, global_step
            ),
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    if pipeline_start_time <= 0:
        pipeline_start_time = start_time

    for epoch in range(start_epoch, epochs + 1):
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=_build_epoch_sampler(dataset, seed=seed, epoch=epoch, offset=17),
            collate_fn=Stage1Collator(int(tokenizer.pad_token_id)),
        )
        pbar = tqdm(train_loader, **tqdm_kwargs(progress_cfg, "Stage1-DAD Epoch {}/{}".format(epoch, epochs)))
        running_loss = 0.0
        skip_until = resume_step_in_epoch if epoch == start_epoch else 0
        for step, batch in enumerate(pbar, start=1):
            if step <= skip_until:
                continue
            batch = {k: v.to(model_device) for k, v in batch.items()}
            stage_seen_samples += int(batch["input_ids"].size(0))
            with get_amp_context(model_device):
                outputs = model(**batch)
                loss = outputs.loss
            (loss / float(grad_accum)).backward()
            running_loss += float(loss.detach().item())

            should_step = (step % grad_accum == 0) or (step == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = running_loss / float(max(1, grad_accum))
                running_loss = 0.0
                elapsed_stage = time.time() - start_time
                eta_stage = estimate_remaining_seconds(global_step, stage_total_opt_steps, elapsed_stage)
                done_total = int(pipeline_completed_before + global_step)
                elapsed_total = time.time() - pipeline_start_time
                eta_total = estimate_remaining_seconds(done_total, int(max(1, pipeline_total_opt_steps)), elapsed_total)
                samples_per_sec = float(stage_seen_samples / max(1e-8, elapsed_stage))
                pbar.set_postfix(
                    {
                        "epoch": epoch,
                        "gstep": global_step,
                        "loss": round(avg_loss, 4),
                        "eta_stage_s": int(eta_stage) if eta_stage >= 0 else -1,
                    },
                    refresh=False,
                )
                if global_step % max(1, log_steps) == 0:
                    row = {
                        "stage": "stage1_dad",
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "global_step": global_step,
                        "loss": round(avg_loss, 6),
                        "elapsed_sec": round(elapsed_stage, 2),
                        "eta_stage_sec": round(eta_stage, 2) if eta_stage >= 0 else -1.0,
                        "eta_total_sec": round(eta_total, 2) if eta_total >= 0 else -1.0,
                        "samples_per_sec": round(samples_per_sec, 4),
                        "steps_per_sec": round(global_step / max(1e-8, elapsed_stage), 6),
                    }
                    append_jsonl(log_file, row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)

                if resume and int(max(1, resume_save_steps)) > 0:
                    if (global_step % int(max(1, resume_save_steps)) == 0) or (step == len(train_loader)):
                        payload = {
                            "epoch": epoch,
                            "step_in_epoch": step,
                            "global_step": global_step,
                            "stage_seen_samples": stage_seen_samples,
                            "trainable_state": _get_trainable_state(model),
                            "optimizer_state": _to_cpu_obj(optimizer.state_dict()),
                        }
                        _save_resume_state(resume_file, payload)
        resume_step_in_epoch = 0

    model.save_pretrained(dad_adapter_dir)
    tokenizer.save_pretrained(dad_adapter_dir)
    with open(os.path.join(dad_adapter_dir, "stage1_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": "stage1_dad",
                "base_model": model_id,
                "num_train_samples": len(dataset),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _clear_resume_state(resume_file)
    return dad_adapter_dir


def evaluate_lmsa(
    model: LMSAModel,
    tokenizer: Any,
    eval_loader: DataLoader,
    model_device: torch.device,
    gen_cfg: Dict[str, Any],
    progress_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    preds = []
    refs = []
    pred_rows = []
    alpha_used = float(torch.sigmoid(model.scale.detach()).item())

    with torch.no_grad():
        for batch in tqdm(eval_loader, **tqdm_kwargs(progress_cfg, "Stage2 Eval")):
            tensor_batch = {
                "input_ids": batch["input_ids"].to(model_device),
                "attention_mask": batch["attention_mask"].to(model_device),
                "lexicon_ids": batch["lexicon_ids"].to(model_device),
                "lexicon_mask": batch["lexicon_mask"].to(model_device),
                "morph_ids": batch["morph_ids"].to(model_device),
                "morph_mask": batch["morph_mask"].to(model_device),
            }

            outputs = model.generate(
                **tensor_batch,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 256)),
                do_sample=bool(gen_cfg.get("do_sample", False)),
                num_beams=int(gen_cfg.get("num_beams", 1)),
                repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.15)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for i in range(outputs.size(0)):
                pred = decode_generated_prediction(
                    output_ids=outputs[i],
                    attention_mask=tensor_batch["attention_mask"][i],
                    tokenizer=tokenizer,
                    empty_fallback="EMPTY",
                )
                ref = str(batch["target"][i])
                instruction = str(batch["instruction"][i])
                source = str(batch["source"][i])

                preds.append(pred)
                refs.append(ref)
                pred_rows.append(
                    build_eval_prediction_row(
                        source=source,
                        prediction=pred,
                        label=ref,
                        instruction=instruction,
                        alpha_used=alpha_used,
                    )
                )

    metrics = compute_metrics(preds, refs)
    metrics["num_samples"] = len(preds)
    return metrics, pred_rows


def run_stage2_lmsa_training(
    model_id: str,
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    tokenizer: Any,
    model_cfg: Dict[str, Any],
    lmsa_cfg: Dict[str, Any],
    stage2_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    progress_cfg: Dict[str, Any],
    output_dir: str,
    dad_adapter_dir: str,
    seed: int = 42,
    resume: bool = False,
    resume_save_steps: int = 50,
    pipeline_total_opt_steps: int = 0,
    pipeline_completed_before: int = 0,
    pipeline_start_time: float = 0.0,
) -> str:
    lmsa_dir = os.path.join(output_dir, "lmsa")
    ensure_dir(lmsa_dir)
    log_file = os.path.join(output_dir, "stage2_lmsa_train_log.jsonl")
    resume_file = _resume_path(output_dir, "stage2")
    resume_state = _load_resume_state(resume_file) if resume else None
    if os.path.exists(log_file) and resume_state is None:
        os.remove(log_file)

    base_model = load_base_model(model_id, model_cfg)
    base_with_dad = PeftModel.from_pretrained(base_model, dad_adapter_dir, is_trainable=False)
    for p in base_with_dad.parameters():
        p.requires_grad = False

    model_device = next(base_with_dad.parameters()).device
    model = LMSAModel(
        base_model=base_with_dad,
        hidden_size=int(base_with_dad.config.hidden_size),
        dropout=float(lmsa_cfg.get("dropout", 0.1)),
    ).to(model_device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.lmsa.parameters():
        p.requires_grad = True
    model.scale.requires_grad = True

    if model_device.type == "cuda" and bool(stage2_cfg.get("use_bfloat16_for_lmsa", True)):
        model.lmsa.to(torch.bfloat16)

    train_set = StageTrainDataset(
        rows=train_rows,
        tokenizer=tokenizer,
        max_length=int(stage2_cfg.get("max_length", 512)),
        include_features=True,
    )
    val_set = StageEvalDataset(
        rows=val_rows,
        tokenizer=tokenizer,
        max_length=int(stage2_cfg.get("max_length", 512)),
    )
    if len(train_set) == 0:
        raise ValueError("Stage-2 train dataset is empty after tokenization.")
    if len(val_set) == 0:
        raise ValueError("Stage-2 val dataset is empty after tokenization.")

    train_batch_size = int(stage2_cfg.get("train_batch_size", 1))
    eval_loader = DataLoader(
        val_set,
        batch_size=int(stage2_cfg.get("eval_batch_size", 1)),
        shuffle=False,
        collate_fn=EvalCollator(int(tokenizer.pad_token_id)),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params=params,
        lr=float(stage2_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(stage2_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(stage2_cfg.get("epochs", 2))
    grad_accum = int(stage2_cfg.get("grad_accum_steps", 8))
    grad_clip = float(stage2_cfg.get("grad_clip", 1.0))
    log_steps = int(stage2_cfg.get("log_steps", 10))
    steps_per_epoch = int(math.ceil(len(train_set) / float(max(1, train_batch_size))))
    stage_total_opt_steps = epochs * int(math.ceil(steps_per_epoch / float(max(1, grad_accum))))

    best_word_bleu = -1.0
    best_path = os.path.join(lmsa_dir, "best_lmsa.pt")
    global_step = 0
    stage_seen_samples = 0
    start_epoch = 1
    resume_step_in_epoch = 0
    if resume_state is not None:
        saved_trainable = resume_state.get("trainable_state", {})
        if isinstance(saved_trainable, dict) and saved_trainable:
            _load_trainable_state(model, saved_trainable)
        saved_opt = resume_state.get("optimizer_state")
        if saved_opt is not None:
            try:
                optimizer.load_state_dict(saved_opt)
                _move_optimizer_state_to_param_device(optimizer)
            except Exception as exc:
                print("[Resume][Stage2] optimizer state ignored due to mismatch: {}".format(exc), flush=True)
        start_epoch = max(1, int(resume_state.get("epoch", 1)))
        resume_step_in_epoch = max(0, int(resume_state.get("step_in_epoch", 0)))
        global_step = max(0, int(resume_state.get("global_step", 0)))
        stage_seen_samples = max(0, int(resume_state.get("stage_seen_samples", 0)))
        best_word_bleu = float(resume_state.get("best_word_bleu", best_word_bleu))
        print(
            "[Resume][Stage2] epoch={}, step_in_epoch={}, global_step={}".format(
                start_epoch, resume_step_in_epoch, global_step
            ),
            flush=True,
        )
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    if pipeline_start_time <= 0:
        pipeline_start_time = start_time

    for epoch in range(start_epoch, epochs + 1):
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            sampler=_build_epoch_sampler(train_set, seed=seed, epoch=epoch, offset=97),
            collate_fn=Stage2Collator(int(tokenizer.pad_token_id)),
        )
        model.train()
        pbar = tqdm(train_loader, **tqdm_kwargs(progress_cfg, "Stage2-LMSA Epoch {}/{}".format(epoch, epochs)))
        running_loss = 0.0
        skip_until = resume_step_in_epoch if epoch == start_epoch else 0

        for step, batch in enumerate(pbar, start=1):
            if step <= skip_until:
                continue
            batch = {k: v.to(model_device) for k, v in batch.items()}
            stage_seen_samples += int(batch["input_ids"].size(0))
            with get_amp_context(model_device):
                outputs = model(**batch)
                loss = outputs.loss
            (loss / float(grad_accum)).backward()
            running_loss += float(loss.detach().item())

            should_step = (step % grad_accum == 0) or (step == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = running_loss / float(max(1, grad_accum))
                running_loss = 0.0
                elapsed_stage = time.time() - start_time
                eta_stage = estimate_remaining_seconds(global_step, stage_total_opt_steps, elapsed_stage)
                done_total = int(pipeline_completed_before + global_step)
                elapsed_total = time.time() - pipeline_start_time
                eta_total = estimate_remaining_seconds(done_total, int(max(1, pipeline_total_opt_steps)), elapsed_total)
                samples_per_sec = float(stage_seen_samples / max(1e-8, elapsed_stage))
                pbar.set_postfix(
                    {
                        "epoch": epoch,
                        "gstep": global_step,
                        "loss": round(avg_loss, 4),
                        "eta_stage_s": int(eta_stage) if eta_stage >= 0 else -1,
                    },
                    refresh=False,
                )
                if global_step % max(1, log_steps) == 0:
                    row = {
                        "stage": "stage2_lmsa",
                        "epoch": epoch,
                        "step_in_epoch": step,
                        "global_step": global_step,
                        "loss": round(avg_loss, 6),
                        "elapsed_sec": round(elapsed_stage, 2),
                        "eta_stage_sec": round(eta_stage, 2) if eta_stage >= 0 else -1.0,
                        "eta_total_sec": round(eta_total, 2) if eta_total >= 0 else -1.0,
                        "samples_per_sec": round(samples_per_sec, 4),
                        "steps_per_sec": round(global_step / max(1e-8, elapsed_stage), 6),
                    }
                    append_jsonl(log_file, row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)

                if resume and int(max(1, resume_save_steps)) > 0:
                    if (global_step % int(max(1, resume_save_steps)) == 0) or (step == len(train_loader)):
                        payload = {
                            "epoch": epoch,
                            "step_in_epoch": step,
                            "global_step": global_step,
                            "stage_seen_samples": stage_seen_samples,
                            "best_word_bleu": float(best_word_bleu),
                            "trainable_state": _get_trainable_state(model),
                            "optimizer_state": _to_cpu_obj(optimizer.state_dict()),
                        }
                        _save_resume_state(resume_file, payload)

        metrics, pred_rows = evaluate_lmsa(
            model=model,
            tokenizer=tokenizer,
            eval_loader=eval_loader,
            model_device=model_device,
            gen_cfg=eval_cfg,
            progress_cfg=progress_cfg,
        )
        metrics["epoch"] = epoch
        metrics["stage"] = "stage2_lmsa"

        epoch_pred_path = os.path.join(lmsa_dir, "val_predictions_epoch{}.jsonl".format(epoch))
        dump_predictions(epoch_pred_path, pred_rows)
        with open(os.path.join(lmsa_dir, "metrics_epoch{}.json".format(epoch)), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("[Stage-2 Validation Metrics]", flush=True)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

        current_word_bleu = float(metrics.get("word_bleu4", 0.0))
        if current_word_bleu > best_word_bleu:
            best_word_bleu = current_word_bleu
            save_lmsa_checkpoint(best_path, model)
            with open(os.path.join(lmsa_dir, "best_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": epoch,
                        "best_word_bleu4": round(best_word_bleu, 4),
                        "checkpoint_format": "lmsa+scale",
                        "dad_adapter_dir": dad_adapter_dir,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        resume_step_in_epoch = 0

    final_path = os.path.join(lmsa_dir, "last_lmsa.pt")
    save_lmsa_checkpoint(final_path, model)
    _clear_resume_state(resume_file)
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train input-side modules in two stages (DAD -> LMSA).")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--resume", action="store_true", help="Resume stage training from latest checkpoint if present.")
    parser.add_argument("--resume_save_steps", type=int, default=50, help="Save resume checkpoint every N optimizer steps.")
    parser.add_argument("--skip_stage1", action="store_true", help="Skip stage-1 and reuse existing dad_adapter.")
    parser.add_argument("--skip_stage2", action="store_true", help="Skip stage-2 and reuse existing lmsa checkpoint.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    stage1_cfg = cfg.get("stage1", {})
    stage2_cfg = cfg.get("stage2", {})
    lmsa_cfg = cfg.get("lmsa", {})
    eval_cfg = cfg.get("eval", {})
    protocol_cfg = cfg.get("protocol", {})
    progress_cfg = resolve_progress_cfg(cfg.get("progress", {}))

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    model_id = str(paths["base_model"])
    train_data = str(paths["train_data"])
    val_data = str(paths["val_data"])
    output_dir = str(paths.get("output_dir", "./outputs/ioce_lexical"))
    ensure_dir(output_dir)

    with open(os.path.join(output_dir, "train_config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_pool = read_jsonl(train_data, max_samples=None)
    if not train_pool:
        raise ValueError("No train samples loaded.")

    use_internal_dev_split = bool(protocol_cfg.get("use_internal_dev_split", True))
    if use_internal_dev_split:
        dev_size = int(protocol_cfg.get("dev_size", 1000))
        split_seed = int(protocol_cfg.get("split_seed", 20260426))
        split_train_rows, split_dev_rows, split_manifest = split_train_dev_rows(
            rows=train_pool,
            dev_size=dev_size,
            split_seed=split_seed,
        )
        with open(os.path.join(output_dir, "internal_dev_split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, ensure_ascii=False, indent=2)
    else:
        split_train_rows = list(train_pool)
        split_dev_rows = read_jsonl(val_data, max_samples=stage2_cfg.get("max_eval_samples", eval_cfg.get("max_samples")))
        if not split_dev_rows:
            raise ValueError("No val samples loaded.")

    stage1_max = stage1_cfg.get("max_samples")
    stage2_max = stage2_cfg.get("max_samples")
    stage2_eval_max = stage2_cfg.get("max_eval_samples", eval_cfg.get("max_samples"))
    stage1_train_rows = split_train_rows[: int(stage1_max)] if stage1_max is not None else split_train_rows
    stage2_train_rows = split_train_rows[: int(stage2_max)] if stage2_max is not None else split_train_rows
    val_rows = split_dev_rows[: int(stage2_eval_max)] if stage2_eval_max is not None else split_dev_rows

    if not stage1_train_rows:
        raise ValueError("No stage-1 train samples loaded.")
    if not stage2_train_rows:
        raise ValueError("No stage-2 train samples loaded.")
    if not val_rows:
        raise ValueError("No stage-2 val samples loaded.")

    stage1_est_steps = int(math.ceil(len(stage1_train_rows) / max(1, int(stage1_cfg.get("train_batch_size", 1)))))
    stage1_est_steps = int(math.ceil(stage1_est_steps / max(1, int(stage1_cfg.get("grad_accum_steps", 8)))))
    stage1_est_steps *= int(stage1_cfg.get("epochs", 1))
    stage2_est_steps = int(math.ceil(len(stage2_train_rows) / max(1, int(stage2_cfg.get("train_batch_size", 1)))))
    stage2_est_steps = int(math.ceil(stage2_est_steps / max(1, int(stage2_cfg.get("grad_accum_steps", 8)))))
    stage2_est_steps *= int(stage2_cfg.get("epochs", 2))
    pipeline_total_est_steps = max(1, stage1_est_steps + stage2_est_steps)
    pipeline_start_time = time.time()

    dad_adapter_dir = os.path.join(output_dir, "dad_adapter")
    if args.skip_stage1:
        if not os.path.isdir(dad_adapter_dir):
            raise FileNotFoundError("skip_stage1 requested but dad_adapter dir not found: {}".format(dad_adapter_dir))
        print("Stage-1 skipped. Reusing DAD adapter:", dad_adapter_dir, flush=True)
    else:
        print("Stage-1: training DAD adapter...", flush=True)
        dad_adapter_dir = run_stage1_dad_training(
            model_id=model_id,
            train_rows=stage1_train_rows,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            stage1_cfg=stage1_cfg,
            progress_cfg=progress_cfg,
            output_dir=output_dir,
            seed=seed,
            resume=bool(args.resume),
            resume_save_steps=int(max(1, args.resume_save_steps)),
            pipeline_total_opt_steps=pipeline_total_est_steps,
            pipeline_completed_before=0,
            pipeline_start_time=pipeline_start_time,
        )
        print("Stage-1 done. DAD adapter:", dad_adapter_dir, flush=True)

    if args.skip_stage2:
        best_path = os.path.join(output_dir, "lmsa", "best_lmsa.pt")
        alt_best_path = os.path.join(output_dir, "best_lmsa.pt")
        if not os.path.isfile(best_path) and not os.path.isfile(alt_best_path):
            raise FileNotFoundError(
                "skip_stage2 requested but best LMSA checkpoint not found: {} (or {}).".format(best_path, alt_best_path)
            )
        if not os.path.isfile(best_path):
            best_path = alt_best_path
        print("Stage-2 skipped. Reusing LMSA checkpoint:", best_path, flush=True)
    else:
        print("Stage-2: training LMSA on top of frozen Base + DAD...", flush=True)
        best_path = run_stage2_lmsa_training(
            model_id=model_id,
            train_rows=stage2_train_rows,
            val_rows=val_rows,
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            lmsa_cfg=lmsa_cfg,
            stage2_cfg=stage2_cfg,
            eval_cfg=eval_cfg,
            progress_cfg=progress_cfg,
            output_dir=output_dir,
            dad_adapter_dir=dad_adapter_dir,
            seed=seed,
            resume=bool(args.resume),
            resume_save_steps=int(max(1, args.resume_save_steps)),
            pipeline_total_opt_steps=pipeline_total_est_steps,
            pipeline_completed_before=stage1_est_steps,
            pipeline_start_time=pipeline_start_time,
        )
        print("Stage-2 done. Best LMSA checkpoint:", best_path, flush=True)
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
