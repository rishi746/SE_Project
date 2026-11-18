"""Kaggle-only completion-masked QLoRA for the two SMS.AC tasks.

Importing this module or running --help does not import torch or load a model.
The full report is retained: oversized examples cause an error by default.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
IGNORE_INDEX = -100


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_kaggle() -> None:
    """Fail before importing torch, downloading weights, or touching a GPU."""
    marker = any(os.environ.get(name) for name in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DATA_PROXY_TOKEN"))
    if platform.system() != "Linux" or not marker or not Path("/kaggle/working").is_dir():
        raise RuntimeError("GPU/model execution is allowed only inside a Kaggle notebook. Upload this project and run kaggle/train_dual_models.ipynb; no laptop GPU fallback exists.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use ordinary python, not torchrun/accelerate launch. This memory-budgeted recipe uses one GPU per task.")
    # Must happen before torch import, including on a Kaggle T4 x2 runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("SMS_AC_GPU", "0")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def encode_completion(row: dict, tokenizer, max_length: int | None = None) -> dict:
    """Use the tokenizer's chat template and mask every prompt token.

    No textual delimiter search and no truncation: the prefix must match exactly.
    This is intentionally independent of torch/datasets for CPU unit tests.
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 != 1:
        raise ValueError(f"{row.get('id', '?')}: expected system followed by user/assistant pairs")
    expected = ['system'] + ['user', 'assistant'] * ((len(messages) - 1) // 2)
    if any(not isinstance(m, dict) for m in messages) or [m.get('role') for m in messages] != expected:
        raise ValueError('Conversation must alternate user/assistant after the system message')
    if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
        raise ValueError(f"{row.get('id', '?')}: message content must be nonempty text")
    def token_ids(value):
        if hasattr(value, "data") and isinstance(value.data, dict) and "input_ids" in value.data:
            value = value.data["input_ids"]
        elif isinstance(value, dict) and "input_ids" in value:
            value = value["input_ids"]
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        return value

    prompt = token_ids(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True))
    full = token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    if not isinstance(prompt, list) or not isinstance(full, list) or any(isinstance(token, list) for token in prompt + full):
        raise TypeError("Chat template must return a flat token ID list")
    if full[: len(prompt)] != prompt:
        raise ValueError("Chat template prompt is not a prefix of the completed example; refusing incorrect loss masking")
    if len(full) <= len(prompt):
        raise ValueError("No assistant tokens remain for supervised learning")
    if max_length is not None and len(full) > max_length:
        raise ValueError(f"{row.get('id', '?')}: {len(full)} tokens exceeds {max_length}; full reports are never truncated")
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [IGNORE_INDEX] * len(prompt) + full[len(prompt):],
        "length": len(full),
        "completion_tokens": len(full) - len(prompt),
    }


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        width = max(len(row["input_ids"]) for row in features)
        # Dynamic right padding; ignore both prompt and padding in the loss.
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in features:
            padding = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [IGNORE_INDEX] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def check_split_integrity(raw: dict, task: str) -> None:
    groups: dict[str, set] = {}
    ids: dict[str, set] = {}
    for split, dataset in raw.items():
        if not len(dataset):
            raise ValueError(f"{task}/{split} is empty")
        if set(dataset["task"]) != {task}:
            raise ValueError(f"Unexpected task label in {task}/{split}")
        ids[split] = set(dataset["id"])
        groups[split] = set(dataset["group_id"])
        if len(ids[split]) != len(dataset) or "" in groups[split] or None in groups[split]:
            raise ValueError(f"Duplicate IDs or missing group_id in {task}/{split}")
    splits = list(raw)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            if groups[left] & groups[right] or ids[left] & ids[right]:
                raise ValueError(f"Data leakage: {task}/{left} and {task}/{right} overlap by ID or scenario group")


def length_summary(lengths: list[int]) -> dict:
    values = sorted(lengths)
    if not values:
        return {"count": 0}
    def quantile(fraction: float) -> int:
        return values[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]
    return {"count": len(values), "min": values[0], "p50": quantile(.5), "p95": quantile(.95), "max": values[-1], "total_tokens": sum(values)}


def tokenize_split(dataset, tokenizer, max_length: int, overflow_policy: str, split: str):
    tokenized = dataset.map(
        lambda row: encode_completion(row, tokenizer),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing complete {split} reports (no truncation)",
        writer_batch_size=64,
    )
    lengths = list(tokenized["length"])
    too_long = [index for index, length in enumerate(lengths) if length > max_length]
    stats = {**length_summary(lengths), "oversized": len(too_long), "excluded": 0}
    if too_long and overflow_policy == "error":
        raise ValueError(f"{split}: {len(too_long)}/{len(lengths)} complete examples exceed {max_length} tokens (max {max(lengths)}). Increase --max-seq-length after reviewing GPU memory, or explicitly use --overflow-policy exclude. No text was truncated.")
    if too_long:
        tokenized = tokenized.filter(lambda row: row["length"] <= max_length, desc="Explicitly excluding complete oversized rows")
        stats["excluded"] = len(too_long)
        stats["excluded_ids"] = [dataset[index]["id"] for index in too_long]
    if not len(tokenized):
        raise ValueError(f"No usable examples in {split}")
    return tokenized.remove_columns(["length", "completion_tokens"]), stats


def gpu_configuration():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is disabled or unavailable. Enable the T4 accelerator; CPU fallback is intentionally disabled.")
    # Check real arithmetic: CUDA discovery alone misses incompatible P100 wheels.
    try:
        test = torch.ones((8, 8), device="cuda", dtype=torch.float16)
        _ = (test @ test).sum().item()
        del test
    except RuntimeError as error:
        raise RuntimeError("Kaggle CUDA arithmetic failed. Select T4 or use the notebook's isolated PyTorch 2.6 CUDA 12.4 environment (P100-compatible).") from error
    major, minor = torch.cuda.get_device_capability(0)
    use_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return dtype, {
        "name": torch.cuda.get_device_name(0), "compute_capability": f"{major}.{minor}",
        "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "compute_dtype": str(dtype), "visible_gpu_count": torch.cuda.device_count(),
        "device_map": {"": 0}, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }


def load_quantized_base(model_id: str, revision: str, dtype):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    return AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        device_map={"": 0}, torch_dtype=dtype, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["design", "sdlc"], required=True)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision", default="main", help="Resolved to a commit and recorded before weights load")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "dual")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=150, help="Pilot optimizer steps; ignored with --full")
    parser.add_argument("--full", action="store_true", help="Train all eligible training rows for --epochs; may exceed one Kaggle session")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", "--seq-length", type=int, default=4096)
    parser.add_argument("--overflow-policy", choices=["error", "exclude"], default="error")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-samples", type=int, default=16, help="Fixed validation subset; test set remains held out")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", help="Path or 'auto' for the most recent checkpoint in output-dir")
    parser.add_argument("--preflight-only", action="store_true", help="Kaggle tokenizer, data, length and GPU checks only; does not download model weights")
    args = parser.parse_args()
    for key in ("max_steps", "max_seq_length", "gradient_accumulation", "lora_r", "eval_samples", "save_steps"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("--epochs must be at least 1 and --learning-rate must be positive")
    return args


def main() -> None:
    args = parse_args()
    require_kaggle()
    import torch
    from datasets import load_dataset
    from huggingface_hub import model_info
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(args.seed)
    output = args.output_dir or ROOT / "models" / f"{args.task}-adapter"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "model_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if not args.resume_from_checkpoint and any(output.glob("adapter_model.*")):
        raise RuntimeError(f"{output} already contains adapter weights. Select a new output directory or explicitly resume a checkpoint.")
    revision = model_info(args.model, revision=args.revision).sha
    files = {split: args.data_dir / args.task / f"{split}.jsonl" for split in ("train", "validation", "test")}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset split: {path}")
    fingerprints = {split: sha256_file(path) for split, path in files.items()}
    if args.resume_from_checkpoint and previous:
        if previous.get("base_model") != args.model or previous.get("base_revision") != revision or previous.get("data_sha256") != fingerprints:
            raise ValueError("Resume base model/revision or dataset hashes differ from the original manifest")
        if previous.get("hyperparameters", {}).get("max_seq_length") != args.max_seq_length:
            raise ValueError("Resume must preserve max-seq-length so the eligible training data remains identical")
    dtype, gpu = gpu_configuration()
    manifest = {
        "schema_version": 1, "task": args.task, "base_model": args.model, "base_revision": revision,
        "training_completed": False, "training_status": "preparing", "started_at": utc_now(),
        "run_tier": "full" if args.full else ("smoke" if args.max_steps <= 5 else "pilot"),
        "full_dataset_epochs_completed": False, "evaluation_status": "not_run",
        "data_sha256": fingerprints, "training_data_sha256": fingerprints["train"],
        "hyperparameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gpu": gpu, "output_format": "markdown", "method": "4-bit NF4 QLoRA; assistant-only loss",
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes")},
        "limitations": ["Training targets may be synthetic; review the dataset manifest.", "Pilot/smoke runs do not mean the full dataset was trained.", "Held-out synthetic checks cannot establish real-world correctness or a universally best lifecycle."],
    }
    write_json(manifest_path, manifest)
    try:
        raw = load_dataset("json", data_files={key: str(value) for key, value in files.items()})
        check_split_integrity(raw, args.task)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        prepared = {}
        stats = {}
        for split in ("train", "validation"):
            prepared[split], stats[split] = tokenize_split(raw[split], tokenizer, args.max_seq_length, args.overflow_policy, split)
        manifest["sequence_statistics"] = stats
        manifest["eligible_train_examples"] = len(prepared["train"])
        manifest["held_out_test_examples"] = len(raw["test"])
        manifest["training_status"] = "preflight_complete" if args.preflight_only else "training"
        write_json(manifest_path, manifest)
        print(json.dumps({"gpu": gpu, "sequence_statistics": stats, "run_tier": manifest["run_tier"]}, indent=2))
        if args.preflight_only:
            return
        validation = prepared["validation"].shuffle(seed=args.seed).select(range(min(args.eval_samples, len(prepared["validation"]))))
        model = load_quantized_base(args.model, revision, dtype)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        model.print_trainable_parameters()
        training = TrainingArguments(
            output_dir=str(output), per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate, num_train_epochs=args.epochs, max_steps=-1 if args.full else args.max_steps,
            warmup_ratio=.03, lr_scheduler_type="cosine", weight_decay=.01, max_grad_norm=.3,
            logging_steps=5, save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
            eval_strategy="steps", eval_steps=args.save_steps, prediction_loss_only=True,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, tf32=False,
            optim="paged_adamw_8bit", report_to="none", seed=args.seed, data_seed=args.seed,
            remove_unused_columns=False, dataloader_num_workers=0, label_names=["labels"],
            save_safetensors=True,
        )
        trainer = Trainer(model=model, args=training, train_dataset=prepared["train"], eval_dataset=validation, data_collator=CompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer)
        resume = args.resume_from_checkpoint
        if resume == "auto":
            resume = get_last_checkpoint(str(output))
            if not resume:
                raise FileNotFoundError(f"No checkpoint-* exists in {output}")
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        final_validation = trainer.evaluate()
        trainer.save_metrics("validation", final_validation)
        if not (output / "adapter_model.safetensors").is_file():
            raise RuntimeError("Training returned but adapter weights were not saved")
        trained_epochs = float(trainer.state.epoch or 0)
        manifest.update({
            "training_completed": True, "training_status": "completed", "completed_at": utc_now(),
            "global_step": trainer.state.global_step, "trained_epochs": trained_epochs,
            "full_dataset_epochs_completed": bool(args.full and trained_epochs + 1e-6 >= args.epochs and not stats["train"]["excluded"]),
            "validation_metrics": final_validation, "train_metrics": result.metrics,
            "adapter_sha256": sha256_file(output / "adapter_model.safetensors"),
        })
        write_json(manifest_path, manifest)
        print(f"Saved {args.task} {manifest['run_tier']} adapter to {output}. Held-out base/adapted comparison still needs evaluate_dual_models.py.")
    except BaseException as error:
        manifest.update({"training_completed": False, "training_status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "failure_type": type(error).__name__, "failure_message": str(error), "updated_at": utc_now()})
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
