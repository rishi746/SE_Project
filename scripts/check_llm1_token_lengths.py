"""CPU tokenizer audit for the LLM1 candidate dataset.

Downloads tokenizer assets only. It never imports torch or model weights.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(ROOT / "scripts"))

from train_dual_qlora import BASE_MODEL, encode_completion, length_summary
from train_llm1_qlora import normalize_row


def main() -> None:
    from transformers import AutoTokenizer

    model_path = BASE_MODEL
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        cache = ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507"
        revision = (cache / "refs/main").read_text().strip()
        model_path = str(cache / "snapshots" / revision)
    print(f"Loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=offline)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    report = {"model": BASE_MODEL, "weights_downloaded": False, "splits": {}}
    base = ROOT / "data/llm1_candidate_v1"
    for split in ("train", "validation", "test"):
        lengths = []
        tasks = {}
        with (base / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for index, line in enumerate(stream, 1):
                row = json.loads(line)
                tasks[row["task"]] = tasks.get(row["task"], 0) + 1
                lengths.append(encode_completion(normalize_row(row), tokenizer)["length"])
                if index % 500 == 0:
                    print(f"{split}: tokenized {index} rows", flush=True)
        stats = {**length_summary(lengths), "over_6144": sum(length > 6144 for length in lengths), "tasks": tasks}
        report["splits"][split] = stats
        print(split, json.dumps(stats), flush=True)

    out = ROOT / "reports/llm1_token_length_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if any(item["over_6144"] for item in report["splits"].values()):
        raise SystemExit("Some LLM1 examples exceed the 6144-token budget; review before Kaggle training.")


if __name__ == "__main__":
    main()
