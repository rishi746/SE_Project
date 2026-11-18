"""Paired Kaggle evaluation of a held-out split with the base and trained adapter.

Scores are diagnostics, not proof of design correctness. Markdown generations
and a human review sheet are saved beside machine-readable measurements.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from contextlib import nullcontext
from pathlib import Path

from train_dual_qlora import (
    ROOT, CompletionCollator, check_split_integrity, encode_completion,
    gpu_configuration, load_quantized_base, require_kaggle, sha256_file, utc_now, write_json,
)

LIFECYCLE_ALIASES = {
    "waterfall": ["waterfall"],
    "v-model": ["v-model", "v model", "v-shaped", "v shaped"],
    "spiral": ["spiral"],
    "iterative": ["iterative"],
    "incremental": ["incremental"],
    "agile": ["agile", "scrum"],
    "prototyping": ["prototyping", "prototype", "prototype-driven", "evolutionary prototype", "rapid prototype"],
    "rad": ["rapid application development", "rad"],
    "hybrid": ["hybrid"],
    "devops": ["devops", "continuous delivery"],
}


def lifecycle_label(text: str | None) -> str | None:
    """Identify only the explicit recommendation, never alternatives later on."""
    if not text:
        return None
    clean = text.lower().replace("**", "").replace("__", "")
    # Start with the recommendation field or the leading heading/paragraph.
    match = re.search(r"(?:recommended(?:\s+sdlc)?(?:\s+(?:model|lifecycle))?|selected\s+(?:model|lifecycle)|primary\s+(?:model|lifecycle))\s*[:\-]\s*([^\n]+)", clean)
    section = re.search(r"(?m)^#{1,6}\s+(?:recommendation|recommended lifecycle)[^\n]*\n", clean)
    if match:
        candidate = match.group(1)
    elif section:
        body = clean[section.end():].lstrip()
        candidate = re.split(r"\n\s*\n|\n#{1,6}\s", body, maxsplit=1)[0][:350]
    else:
        candidate = " ".join(clean.strip().splitlines()[:3])[:350]
    found = []
    for label, aliases in LIFECYCLE_ALIASES.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", candidate):
                found.append(label)
                break
    if "hybrid" in found:
        return "hybrid"
    return found[0] if len(found) == 1 else None


def reference_label(row: dict) -> str | None:
    metadata = row.get("metadata") or {}
    for key in ("recommended_model", "gold_model", "lifecycle_model", "sdlc_model", "label"):
        value = metadata.get(key)
        if isinstance(value, str):
            parsed = lifecycle_label(value)
            if parsed:
                return parsed
    return lifecycle_label(row["messages"][-1]["content"])


def diagnostic_scores(task: str, prediction: str, row: dict) -> dict:
    lower = prediction.lower()
    if (row.get('metadata') or {}).get('stage') == 'clarification':
        return {'stage':'clarification', 'question_count':prediction.count('?'),
                'requires_human_review':True}
    if task == "sdlc":
        expected, predicted = reference_label(row), lifecycle_label(prediction)
        return {
            "reference_label": expected, "predicted_label": predicted,
            "label_match": None if expected is None else int(predicted == expected),
            "mentions_alternatives": bool(re.search(r"alternative|waterfall|v-model|spiral|agile|iterative", lower)),
            "mentions_uncertainty": bool(re.search(r"assum|uncertain|missing|clarif|confidence|unknown", lower)),
        }
    sections = {
        "architecture": r"architect|component|deployment", "requirements": r"requirement|functional",
        "data_and_interfaces": r"data model|schema|interface|api\b", "security": r"secur|authentic|authoriz|threat",
        "network_resilience": r"timeout|retry|idempot|failover|packet loss|circuit breaker",
        "validation": r"acceptance|verification|test plan|testing", "operations": r"observ|monitor|roll.?back|telemetry",
        "assumptions": r"assum|clarif|unresolved|unknown",
    }
    coverage = {name: bool(re.search(pattern, lower)) for name, pattern in sections.items()}
    return {"section_keyword_coverage": sum(coverage.values()) / len(coverage), "section_keywords": coverage}


def evaluate_loss(model, dataset, collator) -> dict:
    import torch
    weighted_loss = 0.0
    tokens = 0
    model.eval()
    model.config.use_cache = False
    with torch.inference_mode():
        for row in dataset:
            batch = {key: value.to("cuda:0") for key, value in collator([row]).items()}
            # Causal LM shifts labels once, so token zero is not predicted.
            count = int((batch["labels"][:, 1:] != -100).sum().item())
            loss = float(model(**batch).loss.item())
            if count <= 0 or not math.isfinite(loss):
                raise RuntimeError("Evaluation produced no supervised tokens or non-finite loss")
            weighted_loss += loss * count
            tokens += count
    mean_loss = weighted_loss / tokens
    return {"assistant_token_nll": mean_loss, "assistant_token_perplexity": math.exp(mean_loss) if mean_loss < 700 else None, "supervised_tokens": tokens, "examples": len(dataset)}


def generate_sample(model, tokenizer, messages: list[dict], max_new_tokens: int, max_context: int) -> dict:
    import torch
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    if len(prompt) + max_new_tokens > max_context:
        raise ValueError(f"Generation needs {len(prompt) + max_new_tokens} tokens, exceeding --max-generation-context={max_context}. Raise the explicit budget; input is never silently cut.")
    input_ids = torch.tensor([prompt], dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )[0, input_ids.shape[-1]:].tolist()
    return {
        "text": tokenizer.decode(output, skip_special_tokens=True),
        "generated_tokens": len(output), "prompt_tokens": len(prompt),
        "hit_output_limit": len(output) >= max_new_tokens and (not output or output[-1] != tokenizer.eos_token_id),
    }


def load_challenges(path: Path, task: str, limit: int) -> list[dict]:
    if not limit or not path.exists():
        return []
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("task") == task:
                cases.append(row)
    return cases[:limit]


def ablation_messages(row: dict, task: str, reference_material: str) -> list[dict]:
    """Change only retrieval context; preserve the held-out request, history and target."""
    messages=copy.deepcopy(row['messages'])
    if len(messages)!=3 or (row.get('metadata') or {}).get('stage')=='clarification':
        raise ValueError('RAG ablation requires a report example')
    payload=json.loads(messages[1]['content'])
    field='request' if task=='design' else 'software_design'
    if field not in payload: raise ValueError('Missing task input')
    payload['reference_material']=reference_material
    messages[1]['content']=json.dumps(payload,ensure_ascii=False)
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["design", "sdlc"], required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "dual")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-limit", type=int, default=24, help="0 evaluates every held-out row")
    parser.add_argument("--generation-limit", type=int, default=4)
    parser.add_argument('--rag-ablation-limit',type=int,default=0,help='Report cases to compare across base/adapter and RAG on/off')
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--max-generation-context", type=int, default=8192)
    parser.add_argument("--challenge-file", type=Path, default=ROOT / "data" / "evaluation" / "challenge_cases.jsonl")
    parser.add_argument("--challenge-limit", type=int, default=0, help="Optional independently authored challenge samples per task; no expert-gold claims")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.test_limit < 0 or args.generation_limit < 0 or args.challenge_limit < 0 or args.rag_ablation_limit < 0 or args.max_new_tokens < 1 or args.max_generation_context < 1:
        parser.error("Limits must be nonnegative; token/context budgets must be positive")
    require_kaggle()
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoTokenizer, set_seed

    set_seed(args.seed)
    adapter = args.adapter or ROOT / "models" / f"{args.task}-adapter"
    manifest_path = adapter / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != args.task or not manifest.get("training_completed"):
        raise ValueError("Only a completed adapter for the requested task can be evaluated")
    weights = adapter / "adapter_model.safetensors"
    if sha256_file(weights) != manifest.get("adapter_sha256"):
        raise ValueError("Adapter checksum differs from its completed training manifest")
    files = {split: args.data_dir / args.task / f"{split}.jsonl" for split in ("train", "validation", "test")}
    if {split: sha256_file(path) for split, path in files.items()} != manifest.get("data_sha256"):
        raise ValueError("Evaluation dataset differs from the training manifest; refusing invalid held-out claims")
    raw = load_dataset("json", data_files={split: str(path) for split, path in files.items()})
    check_split_integrity(raw, args.task)
    test = raw["test"].shuffle(seed=args.seed)
    if args.test_limit:
        test = test.select(range(min(args.test_limit, len(test))))
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=False)
    max_length = manifest["hyperparameters"]["max_seq_length"]
    encoded = [encode_completion(row, tokenizer, max_length=max_length) for row in test]
    dtype, gpu = gpu_configuration()
    base = load_quantized_base(manifest["base_model"], manifest["base_revision"], dtype)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model.eval()
    output = args.output_dir or ROOT / "reports" / "evaluation" / args.task
    output.mkdir(parents=True, exist_ok=True)
    challenges = load_challenges(args.challenge_file, args.task, args.challenge_limit)
    if challenges:
        sys.path.insert(0, str(ROOT / "src"))
        from sms_ac.prompts import TASK_PROMPTS
    collator = CompletionCollator(tokenizer.pad_token_id)
    report = {
        "schema_version": 1, "task": args.task, "base_model": manifest["base_model"], "base_revision": manifest["base_revision"],
        "adapter_sha256": manifest["adapter_sha256"], "run_tier": manifest["run_tier"], "evaluated_at": utc_now(),
        "test_data_sha256": manifest["data_sha256"]["test"], "seed": args.seed, "gpu": gpu,
        "held_out_examples_available": len(raw["test"]), "selected_test_ids": list(test["id"]),
        "generation_method": "deterministic greedy decoding; same prompts and budgets for both models",
        "limitations": ["Section keywords check coverage, not technical correctness.", "Lifecycle labels follow synthetic reference decisions, not a universal optimum.", "Real document blind review and subject-matter validation remain required.", "Output-limit hits are reported as incomplete, not silently accepted."],
        "variants": {}, "generations": [], "challenge_generations": [],
        "challenge_provenance": "Independently authored challenge prompts, not expert-validated ground truth",
    }
    for variant in ("base", "adapter"):
        context = model.disable_adapter() if variant == "base" else nullcontext()
        with context:
            metrics = evaluate_loss(model, encoded, collator)
            variant_scores = []
            for row in list(test)[:args.generation_limit]:
                result = generate_sample(model, tokenizer, row["messages"], args.max_new_tokens, args.max_generation_context)
                scores = diagnostic_scores(args.task, result["text"], row)
                record = {"id": row["id"], "group_id": row["group_id"], "variant": variant, **result, "diagnostics": scores}
                report["generations"].append(record)
                variant_scores.append(scores)
                filename = f"{variant}-{len(variant_scores):03d}.md"
                (output / filename).write_text(result["text"] + "\n", encoding="utf-8")
                write_json(output / "evaluation_report.partial.json", report)
            metrics["generation_count"] = len(variant_scores)
            if args.task == "sdlc":
                labeled = [score["label_match"] for score in variant_scores if score["label_match"] is not None]
                metrics["synthetic_reference_label_accuracy"] = sum(labeled) / len(labeled) if labeled else None
                metrics["labeled_generations"] = len(labeled)
            elif variant_scores:
                coverage = [score['section_keyword_coverage'] for score in variant_scores if 'section_keyword_coverage' in score]
                metrics["mean_section_keyword_coverage"] = sum(coverage) / len(coverage) if coverage else None
            report["variants"][variant] = metrics
            for index, case in enumerate(challenges, start=1):
                field = "request" if args.task == "design" else "software_design"
                payload = case["input"] if isinstance(case["input"], dict) else {field: case["input"]}
                messages = [
                    {"role": "system", "content": TASK_PROMPTS[args.task]},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    {"role": "assistant", "content": "Reference output is intentionally absent for a human-reviewed challenge."},
                ]
                result = generate_sample(model, tokenizer, messages, args.max_new_tokens, args.max_generation_context)
                report["challenge_generations"].append({"id": case["id"], "variant": variant, **result, "review_checklist": {key: case.get(key, []) for key in ("acceptable_lifecycles", "required_considerations", "red_flags")}})
                (output / f"challenge-{variant}-{index:03d}.md").write_text(result["text"] + "\n", encoding="utf-8")
    report['rag_ablation_generations']=[]
    if args.rag_ablation_limit:
        sys.path.insert(0,str(ROOT/'src'))
        from sms_ac.retrieval import retrieve_context
        from sms_ac.lifecycle_references import retrieve_lifecycle_context
        ablation_rows=[row for row in raw['test'].shuffle(seed=args.seed)
                       if (row.get('metadata') or {}).get('stage')!='clarification'][:args.rag_ablation_limit]
        for index,row in enumerate(ablation_rows):
            payload=json.loads(row['messages'][1]['content'])
            references=(retrieve_context(payload['request'],max_chars=2000) if args.task=='design'
                        else retrieve_lifecycle_context(payload['software_design']))
            for adapted in (False,True):
                for use_rag in (False,True):
                    variant=('adapter' if adapted else 'base')+('_rag' if use_rag else '_no_rag')
                    messages=ablation_messages(row,args.task,references if use_rag else '')
                    with (nullcontext() if adapted else model.disable_adapter()):
                        result=generate_sample(model,tokenizer,messages,args.max_new_tokens,args.max_generation_context)
                    record={'id':row['id'],'variant':variant,'reference_material':references if use_rag else '',**result}
                    report['rag_ablation_generations'].append(record)
                    (output/f'ablation-{index:03d}-{variant}.md').write_text(result['text'],encoding='utf-8')
                    write_json(output/'evaluation_report.partial.json',report)
        report['rag_ablation_note']='Same held-out report inputs and generation settings; human review required. Design retrieval uses a 2000-character evidence budget. No quality improvement is inferred from generation alone.'
    base_nll = report["variants"]["base"]["assistant_token_nll"]
    adapter_nll = report["variants"]["adapter"]["assistant_token_nll"]
    report["adapter_minus_base_nll"] = adapter_nll - base_nll
    report["adapter_lower_heldout_nll"] = adapter_nll < base_nll
    report["incomplete_generations"] = sum(record["hit_output_limit"] for record in report["generations"] + report["challenge_generations"] + report['rag_ablation_generations'])
    write_json(output / "evaluation_report.json", report)
    summary = [
        f"# {args.task.upper()} paired evaluation", "",
        f"Run tier: **{manifest['run_tier']}**. Evaluated {len(test)} of {len(raw['test'])} held-out rows.", "",
        "| Variant | Assistant-token NLL (lower is better) | Perplexity |",
        "| --- | ---: | ---: |",
    ]
    for variant, metrics in report["variants"].items():
        summary.append(f"| {variant} | {metrics['assistant_token_nll']:.4f} | {metrics['assistant_token_perplexity']:.3f} |")
    summary += ["", f"Adapter minus base NLL: {report['adapter_minus_base_nll']:+.4f}.", f"Incomplete generations at the configured output limit: {report['incomplete_generations']}.", "", "## Human review still required", "", "Blindly compare the saved base and adapter Markdown outputs against the input and sources. Score correctness, network failure handling, security, completeness, traceability, and explicit assumptions from 1–5. For lifecycle decisions, score evidence, alternatives, limitations, and the practical phase plan. Record reviewer, date, and reasons. Do not interpret keyword or synthetic-label scores as expert validation.", "", "| Example | Blind candidate | Correctness | Completeness | Evidence | Assumptions | Reviewer notes |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for index in range(min(args.generation_limit, len(test))):
        summary.append(f"| {test[index]['id']} | A / B | | | | | |")
    for case in challenges:
        summary.append(f"| Challenge: {case['id']} | A / B | | | | | |")
    (output / "evaluation_report.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    manifest.update({"evaluation_status": "paired_heldout_complete", "evaluation_report": str(output / "evaluation_report.json"), "adapter_lower_heldout_nll": report["adapter_lower_heldout_nll"], "evaluated_at": utc_now()})
    write_json(manifest_path, manifest)
    print(json.dumps({"task": args.task, "variants": report["variants"], "adapter_minus_base_nll": report["adapter_minus_base_nll"], "incomplete_generations": report["incomplete_generations"], "report": str(output / "evaluation_report.md")}, indent=2))
    del model, base
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
