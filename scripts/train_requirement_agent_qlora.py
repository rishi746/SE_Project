from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "generated" / "networking_requirements_sft.jsonl"


SYSTEM_PROMPT = (
    "You are a security-aware software requirements engineering agent for networking applications. "
    "Extract complete, testable, traceable requirements and identify ambiguities, clarification questions, "
    "human approval points, and source-grounding notes. Return only valid JSON."
)


def format_example(example: dict) -> str:
    user_payload = {
        "instruction": example["instruction"],
        "scenario": example["input"],
    }
    assistant_payload = example["output"]
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{json.dumps(user_payload, ensure_ascii=False)}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{json.dumps(assistant_payload, ensure_ascii=False)}\n"
        "<|im_end|>"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(ROOT / "models" / "networking-requirement-agent-qwen3-8b-lora"))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading for local smoke tests.")
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules. Use c_attn,c_proj for GPT-2-style smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.map(lambda row: {"text": format_example(row)})

    quantization = None
    if not args.no_4bit:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise SystemExit(
                "4-bit QLoRA requires bitsandbytes. On Windows, run with --no-4bit for a tiny smoke test, "
                "or run full Qwen3-8B QLoRA in Linux/WSL/Colab."
            )
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if quantization is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=50,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=not args.no_4bit,
        fp16=False,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        report_to="none",
    )

    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=args.seq_length,
            peft_config=lora,
            args=training_args,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=lora,
            args=training_args,
        )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
