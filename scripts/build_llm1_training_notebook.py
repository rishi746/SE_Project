"""Build the Kaggle notebook for LLM1-only training."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def main() -> None:
    cells = [
        markdown("""# SMS AC LLM1 QLoRA training

This notebook trains only the LLM1 networking/cybersecurity requirements and preliminary design report specialist on `data/llm1_candidate_v1`. Run it on Kaggle with GPU and Internet enabled.
"""),
        code("""import os, platform, shutil, subprocess, sys
from pathlib import Path

assert platform.system() == "Linux" and Path("/kaggle/working").is_dir(), "Run this notebook on Kaggle."
PROJECT = Path("/kaggle/working/SEproject")
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RUN_FULL = False
MAX_STEPS = 250
MAX_SEQ_LENGTH = 3072
EPOCHS = 1.0
LORA_R = 8
GRADIENT_ACCUMULATION = 16
EVAL_SAMPLES = 8
RESUME = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""),
        code("""if not (PROJECT / "scripts/train_llm1_qlora.py").exists():
    unpacked = [p.parent.parent for p in Path("/kaggle/input").rglob("scripts/train_llm1_qlora.py")]
    if unpacked:
        shutil.copytree(unpacked[0], PROJECT, dirs_exist_ok=True)
if not (PROJECT / "scripts/train_llm1_qlora.py").exists():
    archives = [p for p in Path("/kaggle/input").rglob("*.zip") if p.name.lower() in {"project.zip", "llm1-candidate-v1.zip"}]
    assert archives, "Attach the SMS AC project package or llm1 candidate zip."
    shutil.unpack_archive(str(archives[0]), "/kaggle/working/unpacked")
    roots = [p for p in Path("/kaggle/working/unpacked").rglob("scripts/train_llm1_qlora.py")]
    assert roots, "train_llm1_qlora.py not found after unpacking."
    shutil.copytree(roots[0].parent.parent, PROJECT, dirs_exist_ok=True)
assert (PROJECT / "data/llm1_candidate_v1/train.jsonl").exists(), "LLM1 dataset missing."
print("Project ready", PROJECT)
"""),
        code("""import venv

VENV = Path("/tmp/sms-ac-llm1-venv")
PYTHON = VENV / "bin/python"
if not PYTHON.exists():
    venv.EnvBuilder(with_pip=False).create(VENV)
subprocess.run([sys.executable, "-m", "pip", "--python", str(PYTHON), "install", "pip==25.3"], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
                "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
                "-r", str(PROJECT / "requirements-kaggle.txt")], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "check"], check=True)
(PROJECT / "reports").mkdir(exist_ok=True)
lock = subprocess.check_output([str(PYTHON), "-m", "pip", "freeze"], text=True)
(PROJECT / "reports/kaggle-environment.txt").write_text(lock)
print("Isolated training environment ready:", PYTHON)
"""),
        markdown("""## Preflight

This checks dataset structure, split leakage, chat-template masking, GPU access, and sequence lengths. It does not train the model while `--preflight-only` is used.
"""),
        code("""args = [
    str(PYTHON), str(PROJECT / "scripts/train_llm1_qlora.py"),
    "--model", BASE_MODEL,
    "--data-dir", str(PROJECT / "data/llm1_candidate_v1"),
    "--output-dir", str(PROJECT / "models/llm1-report-adapter"),
    "--max-seq-length", str(MAX_SEQ_LENGTH),
    "--overflow-policy", "exclude",
    "--lora-r", str(LORA_R),
    "--gradient-accumulation", str(GRADIENT_ACCUMULATION),
    "--eval-samples", str(EVAL_SAMPLES),
    "--preflight-only",
]
subprocess.run(args, check=True)
"""),
        markdown("""## Train

Pilot mode uses `MAX_STEPS`. Set `RUN_FULL = True` only after reviewing preflight and available Kaggle quota.
"""),
        code("""args = [
    str(PYTHON), str(PROJECT / "scripts/train_llm1_qlora.py"),
    "--model", BASE_MODEL,
    "--data-dir", str(PROJECT / "data/llm1_candidate_v1"),
    "--output-dir", str(PROJECT / "models/llm1-report-adapter"),
    "--max-seq-length", str(MAX_SEQ_LENGTH),
    "--overflow-policy", "exclude",
    "--lora-r", str(LORA_R),
    "--gradient-accumulation", str(GRADIENT_ACCUMULATION),
    "--eval-samples", str(EVAL_SAMPLES),
    "--epochs", str(EPOCHS),
    "--max-steps", str(MAX_STEPS),
]
if RUN_FULL:
    args.append("--full")
if RESUME:
    args += ["--resume-from-checkpoint", RESUME]
subprocess.run(args, check=True)
"""),
        code("""import zipfile

export_path = Path("/kaggle/working/sms-ac-llm1-report-adapter.zip")
with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as z:
    for path in (PROJECT / "models/llm1-report-adapter").rglob("*"):
        if path.is_file():
            z.write(path, "llm1-report-adapter/" + path.relative_to(PROJECT / "models/llm1-report-adapter").as_posix())
    for extra in [
        PROJECT / "data/llm1_candidate_v1/manifest.json",
        PROJECT / "reports/LLM1_CANDIDATE_V1_REVIEW.md",
        PROJECT / "reports/llm1_token_length_audit.json",
    ]:
        if extra.exists():
            z.write(extra, extra.relative_to(PROJECT).as_posix())
print(export_path, export_path.stat().st_size)
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "kaggle" / "train_llm1_report_model.ipynb"
    out.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
