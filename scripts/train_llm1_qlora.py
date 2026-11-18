"""Kaggle-only QLoRA trainer for the SMS AC LLM1 candidate dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sms_ac.prompts import DESIGN_SYSTEM_PROMPT, CLARIFICATION_SYSTEM_PROMPT
from train_dual_qlora import (
    BASE_MODEL,
    CompletionCollator,
    encode_completion,
    gpu_configuration,
    length_summary,
    load_quantized_base,
    require_kaggle,
    sha256_file,
    utc_now,
    write_json,
)


TASK_SYSTEMS = {
    "llm1_clarification": CLARIFICATION_SYSTEM_PROMPT,
    "llm1_section_generation": (
        "You are the SMS AC LLM1 section writer. Generate exactly the requested section "
        "for a networking or cybersecurity SRS/preliminary design report. Use the supplied "
        "facts, IDs, and evidence ledger. Keep unknowns explicit and do not select the SDLC."
    ),
    "llm1_full_report": DESIGN_SYSTEM_PROMPT,
    "llm1_consistency_review": (
        "You are the SMS AC LLM1 report reviewer. Identify unsupported claims, unresolved "
        "assumptions, broken IDs, missing sections, role conflicts, domain contamination, "
        "and readiness gaps before a networking/cybersecurity report is used for training."
    ),
}


def normalize_row(row: dict) -> dict:
    task = row.get("task")
    if task in {"llm1_clarification", "llm1_section_generation", "llm1_consistency_review"}:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{row.get('id')}: missing messages")
        return {**row, "messages": messages}
    if task == "llm1_full_report":
        prompt = {
            "request": row.get("prompt", ""),
            "evidence": row.get("evidence", []),
            "output_contract": "Return the complete 28-section SRS/preliminary design report in Markdown for Word export. Keep unresolved facts explicit.",
        }
        return {
            **row,
            "messages": [
                {"role": "system", "content": TASK_SYSTEMS[task]},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                {"role": "assistant", "content": row.get("report_markdown", "")},
            ],
        }
    raise ValueError(f"{row.get('id')}: unknown LLM1 task {task!r}")


def check_split_integrity(raw: dict) -> None:
    groups: dict[str, set[str]] = {}
    ids: dict[str, set[str]] = {}
    for split, dataset in raw.items():
        if not len(dataset):
            raise ValueError(f"{split} split is empty")
        groups[split] = set(dataset["group_id"])
        ids[split] = set(dataset["id"])
        if len(ids[split]) != len(dataset) or "" in groups[split] or None in groups[split]:
            raise ValueError(f"{split}: duplicate IDs or missing group_id")
        unexpected = set(dataset["task"]) - set(TASK_SYSTEMS)
        if unexpected:
            raise ValueError(f"{split}: unexpected tasks {sorted(unexpected)}")
    names = list(raw)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if groups[left] & groups[right] or ids[left] & ids[right]:
                raise ValueError(f"Data leakage between {left} and {right}")


def tokenize_split(dataset, tokenizer, max_length: int, overflow_policy: str, split: str):
    tokenized = dataset.map(
        lambda row: encode_completion(normalize_row(row), tokenizer),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing LLM1 {split}",
        writer_batch_size=32,
    )
    lengths = list(tokenized["length"])
    too_long = [index for index, length in enumerate(lengths) if length > max_length]
    stats = {**length_summary(lengths), "oversized": len(too_long), "excluded": 0}
    if too_long and overflow_policy == "error":
        raise ValueError(f"{split}: {len(too_long)} examples exceed {max_length} tokens; max={max(lengths)}")
    if too_long:
        tokenized = tokenized.filter(lambda row: row["length"] <= max_length, desc=f"Excluding oversized {split} rows")
        stats["excluded"] = len(too_long)
        stats["excluded_ids"] = [dataset[index]["id"] for index in too_long]
    if not len(tokenized):
        raise ValueError(f"No usable LLM1 examples in {split}")
    return tokenized.remove_columns(["length", "completion_tokens"]), stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "llm1_candidate_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "llm1-report-adapter")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--overflow-policy", choices=["error", "exclude"], default="error")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--eval-samples", type=int, default=24)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_seq_length <= 0 or args.gradient_accumulation <= 0:
        parser.error("positive integer hyperparameters are required")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "model_manifest.json"
    revision = model_info(args.model, revision=args.revision).sha
    files = {split: args.data_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    fingerprints = {split: sha256_file(path) for split, path in files.items()}
    dtype, gpu = gpu_configuration()
    manifest = {
        "schema_version": 1,
        "task": "llm1_report_specialist",
        "base_model": args.model,
        "base_revision": revision,
        "training_completed": False,
        "training_status": "preparing",
        "started_at": utc_now(),
        "run_tier": "full" if args.full else ("smoke" if args.max_steps <= 5 else "pilot"),
        "data_dir": str(args.data_dir),
        "data_sha256": fingerprints,
        "hyperparameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gpu": gpu,
        "method": "4-bit NF4 QLoRA; assistant-only loss; LLM1 mixed objective",
        "limitations": [
            "The v1 data is assistant-generated candidate supervision, not expert-approved gold.",
            "Full-report rows are long assembly exemplars; section rows provide most detailed supervised signal.",
            "Challenge evaluation and sampled human review are required before claiming project quality.",
        ],
    }
    write_json(manifest_path, manifest)
    try:
        raw = load_dataset("json", data_files={key: str(value) for key, value in files.items()})
        check_split_integrity(raw)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        prepared = {}
        stats = {}
        for split in ("train", "validation"):
            prepared[split], stats[split] = tokenize_split(raw[split], tokenizer, args.max_seq_length, args.overflow_policy, split)
        manifest.update({
            "training_status": "preflight_complete" if args.preflight_only else "training",
            "sequence_statistics": stats,
            "eligible_train_examples": len(prepared["train"]),
            "held_out_test_examples": len(raw["test"]),
        })
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
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        training = TrainingArguments(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            max_steps=-1 if args.full else args.max_steps,
            warmup_ratio=.03,
            lr_scheduler_type="cosine",
            weight_decay=.01,
            max_grad_norm=.3,
            logging_steps=5,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=args.save_steps,
            prediction_loss_only=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            optim="paged_adamw_8bit",
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            label_names=["labels"],
            save_safetensors=True,
        )
        trainer = Trainer(
            model=model,
            args=training,
            train_dataset=prepared["train"],
            eval_dataset=validation,
            data_collator=CompletionCollator(tokenizer.pad_token_id),
            processing_class=tokenizer,
        )
        resume = args.resume_from_checkpoint
        if resume == "auto":
            resume = get_last_checkpoint(str(args.output_dir))
            if not resume:
                raise FileNotFoundError(f"No checkpoint exists in {args.output_dir}")
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        final_validation = trainer.evaluate()
        trainer.save_metrics("validation", final_validation)
        manifest.update({
            "training_completed": True,
            "training_status": "completed",
            "completed_at": utc_now(),
            "global_step": trainer.state.global_step,
            "trained_epochs": float(trainer.state.epoch or 0),
            "validation_metrics": final_validation,
            "train_metrics": result.metrics,
            "adapter_sha256": sha256_file(args.output_dir / "adapter_model.safetensors"),
        })
        write_json(manifest_path, manifest)
        print(f"Saved LLM1 adapter to {args.output_dir}")
    except BaseException as error:
        manifest.update({
            "training_completed": False,
            "training_status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "updated_at": utc_now(),
        })
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
