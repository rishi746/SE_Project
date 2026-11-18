from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "vector_index"
DEFAULT_ADAPTER = ROOT / "models" / "networking-requirement-agent-qwen3-8b-lora"

SYSTEM_PROMPT = (
    "You are a security-aware software requirements engineering agent for networking applications. "
    "Use the retrieved trusted source context where relevant. Extract complete, testable, traceable "
    "requirements and identify ambiguities, clarification questions, human approval points, and source "
    "grounding notes. Do not include reasoning, markdown, or <think> tags. Return only valid JSON."
)


def retrieve_context(scenario: str, top_k: int = 4) -> list[dict]:
    metadata = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_DIR / "networking_sources.faiss"))
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    query_vector = embedder.encode([scenario], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(query_vector, dtype="float32"), top_k)
    results = []
    for score, idx in zip(scores[0], ids[0], strict=False):
        if idx >= 0:
            item = dict(metadata[int(idx)])
            item["score"] = float(score)
            results.append(item)
    return results


def build_prompt(scenario: str, context: list[dict]) -> str:
    payload = {
        "instruction": "Identify and classify software requirements for the given networking project scenario.",
        "scenario": scenario,
        "retrieved_context": [
            {
                "source_id": item["source_id"],
                "title": item["title"],
                "publisher": item["publisher"],
                "use": item["use_in_dataset"],
                "url": item["url"],
            }
            for item in context
        ],
    }
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = retrieve_context(args.scenario, top_k=args.top_k)
    prompt = build_prompt(args.scenario, context)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    if "</think>" in output:
        output = output.split("</think>", 1)[1]
    elif "<think>" in output:
        output = output.split("<think>", 1)[0]
    print(output.strip())


if __name__ == "__main__":
    main()
