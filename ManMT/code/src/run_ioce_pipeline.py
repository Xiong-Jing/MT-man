from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

from common import load_yaml_config


def run_cmd(cmd: List[str]) -> None:
    print("[RUN] {}".format(" ".join(cmd)), flush=True)
    subprocess.run(cmd, check=True)


def resolve_ckpt_root(cfg: dict, ckpt_override: str = "") -> str:
    if ckpt_override:
        return str(ckpt_override)
    paths = cfg.get("paths", {})
    input_ckpt_root = str(paths.get("input_ckpt_root", "")).strip()
    if input_ckpt_root:
        return input_ckpt_root
    return str(paths.get("output_dir", "./outputs/ioce_lexical"))


def validate_ckpt_root_for_output_side(ckpt_root: str) -> None:
    root = str(ckpt_root).strip()
    if not root:
        raise ValueError(
            "Input checkpoint root is empty. Set paths.input_ckpt_root or pass --ckpt when input-side training is skipped."
        )
    if not os.path.isdir(root):
        raise FileNotFoundError("Input checkpoint root does not exist: {}".format(root))

    dad_adapter_dir = os.path.join(root, "dad_adapter")
    lmsa_path = os.path.join(root, "lmsa", "best_lmsa.pt")
    alt_lmsa_path = os.path.join(root, "best_lmsa.pt")
    if not os.path.isdir(dad_adapter_dir):
        raise FileNotFoundError("Required checkpoint artifact missing: {}".format(dad_adapter_dir))
    if (not os.path.isfile(lmsa_path)) and (not os.path.isfile(alt_lmsa_path)):
        raise FileNotFoundError(
            "Required checkpoint artifact missing: {} (or fallback path {}).".format(lmsa_path, alt_lmsa_path)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IOCE-Lexical end-to-end pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--skip_memory", action="store_true", help="Skip lexical memory build.")
    parser.add_argument("--skip_input", action="store_true", help="Skip input-side training script.")
    parser.add_argument("--skip_head", action="store_true", help="Skip IOCE head training.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip IOCE evaluation.")
    parser.add_argument("--resume_input", action="store_true", help="Resume input-side training from latest checkpoint.")
    parser.add_argument("--resume_save_steps", type=int, default=50, help="Input-side resume checkpoint save interval.")
    parser.add_argument("--skip_stage1", action="store_true", help="Pass through: skip stage-1 input-side training.")
    parser.add_argument("--skip_stage2", action="store_true", help="Pass through: skip stage-2 input-side training.")
    parser.add_argument("--ckpt", type=str, default="", help="Input checkpoint root used by head/eval steps.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    pipeline_cfg = cfg.get("pipeline", {})
    train_input_side = bool(pipeline_cfg.get("train_input_side", True))

    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    should_run_input = (not args.skip_input) and train_input_side

    if not args.skip_memory:
        run_cmd([py, os.path.join(here, "build_memory.py"), "--config", args.config])

    if should_run_input:
        cmd = [py, os.path.join(here, "train_input_side.py"), "--config", args.config]
        if args.resume_input:
            cmd.append("--resume")
            cmd.extend(["--resume_save_steps", str(max(1, int(args.resume_save_steps)))])
        if args.skip_stage1:
            cmd.append("--skip_stage1")
        if args.skip_stage2:
            cmd.append("--skip_stage2")
        run_cmd(cmd)
    elif (not args.skip_head) or (not args.skip_eval):
        ckpt_root = resolve_ckpt_root(cfg, ckpt_override=args.ckpt)
        validate_ckpt_root_for_output_side(ckpt_root)

    if not args.skip_head:
        cmd = [py, os.path.join(here, "train_ioce_head.py"), "--config", args.config]
        if args.ckpt:
            cmd.extend(["--ckpt", args.ckpt])
        run_cmd(cmd)

    if not args.skip_eval:
        cmd = [py, os.path.join(here, "eval_ioce.py"), "--config", args.config]
        if args.ckpt:
            cmd.extend(["--ckpt", args.ckpt])
        run_cmd(cmd)

    print("IOCE pipeline finished.", flush=True)


if __name__ == "__main__":
    main()
