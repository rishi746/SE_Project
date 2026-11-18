"""Package, submit, and inspect the private Kaggle job; never runs model code locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = ["app.py", "requirements.txt", "requirements-local.txt", "requirements-kaggle.txt", "README.md",
             "ONLINE_GPU_RUN.md", "docs_methodology.md", "MODEL_AND_ARCHITECTURE.md", "PROJECT_HANDOFF.md"]
DIRECTORIES = ["src/sms_ac", "configs", "docs", "tests", "data/dual", "data/llm1_candidate_v1", "data/llm2_training", "data/corpus", "data/evaluation", "kaggle"]
SCRIPT_FILES = ["prepare_corpus.py", "build_dual_dataset.py", "validate_dual_dataset.py",
                "train_dual_qlora.py", "train_llm1_qlora.py", "evaluate_dual_models.py", "run_sms_ac.py",
                "kaggle_job.py", "create_challenge_cases.py", "planning_examples.py", "check_token_lengths.py",
                "check_llm1_token_lengths.py", "build_llm1_candidate_dataset.py", "validate_llm1_candidate_dataset.py",
                "build_llm1_training_notebook.py", "discuss_project.py"]
FORBIDDEN_PARTS = {".git", ".venv", ".cache", "__pycache__", "models", "node_modules"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_files(root: Path = ROOT, llm2_only: bool = False) -> list[Path]:
    candidates = [root / item for item in TOP_FILES]
    candidates += [root / "scripts" / name for name in SCRIPT_FILES]
    directories = ["src/sms_ac", "configs", "data/llm2_training", "data/evaluation", "kaggle"] if llm2_only else DIRECTORIES
    for name in directories:
        folder = root / name
        if folder.exists():
            candidates.extend(folder.rglob("*"))
    result = []
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"File escapes project root: {relative}")
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            continue
        if path.name.startswith(".") or path.suffix in {".pyc", ".env", ".pem", ".key"}:
            continue
        if path.name.lower() in {"kaggle.json", "access_token", "credentials.json"}:
            continue
        result.append(path)
    return sorted(set(result))


def check_username(username: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,50}", username):
        raise ValueError("Use your Kaggle username, without a URL or email address.")
    return username


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(output: Path, username: str, slug: str, dataset_slug: str | None = None, llm1_only: bool = False, llm2_only: bool = False) -> dict:
    check_username(username)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,49}", slug):
        raise ValueError("Use a 3-50 character lowercase Kaggle slug.")
    if llm1_only and llm2_only:
        raise ValueError("Select only one training task")
    notebook = ROOT / ("kaggle/train_llm2_detailed.ipynb" if llm2_only else "kaggle/train_llm1_report_model.ipynb" if llm1_only else "kaggle/train_dual_models.ipynb")
    required_inputs = [notebook]
    if llm2_only:
        required_inputs += [ROOT / "data/llm2_training/sdlc" / f"{split}.jsonl" for split in ("train", "validation", "test")]
    elif llm1_only:
        required_inputs += [ROOT / "data/llm1_candidate_v1/train.jsonl", ROOT / "data/llm1_candidate_v1/validation.jsonl", ROOT / "data/llm1_candidate_v1/test.jsonl"]
    else:
        required_inputs += [ROOT / "data/dual/design/train.jsonl", ROOT / "data/dual/sdlc/train.jsonl"]
    for required in required_inputs:
        if not required.is_file():
            raise ValueError(f"Required project input is missing: {required}")
    output = output.resolve()
    if not output.is_relative_to((ROOT / "build").resolve()):
        raise ValueError("Package output must stay under the project's build directory.")
    if output.exists():
        raise ValueError("Use a new output directory to preserve any previously submitted job.")
    dataset_dir, kernel_dir = output / "dataset", output / "kernel"
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir()
    entries = []
    with zipfile.ZipFile(dataset_dir / "project.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in package_files(llm2_only=llm2_only):
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, "SEproject/" + relative)
            entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(dataset_dir / "package_manifest.json", {"files": entries, "file_count": len(entries)})
    dataset_slug = dataset_slug or slug+'-data'
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,49}',dataset_slug):
        raise ValueError('Use a valid lowercase dataset slug')
    dataset_ref = f"{username}/{dataset_slug}"
    if len(slug + "-data") > 50:
        raise ValueError("Dataset slug with '-data' suffix exceeds Kaggle's limit.")
    description = ("Private SMS AC research package. Real reference documents retain their original "
                   "licenses and attributions in data/corpus and configs/corpus_sources.json. "
                   "Generated task examples are explicitly labeled candidate/weak supervision, not expert gold labels. "
                   "This mixed collection is not relicensed: inspect the source manifest and license files. "
                   "Code, corpus preparation, separate design/SDLC chat datasets, and evaluation are included.")
    write_json(dataset_dir / "dataset-metadata.json", {
        "title": "SMS AC LLM2 detailed SDLC training" if llm2_only else "SMS AC LLM1 networking SRS training" if llm1_only else "SMS AC networking design and SDLC training",
        "id": dataset_ref, "licenses": [{"name": "other"}], "description": description,
    })
    shutil.copyfile(notebook, kernel_dir / notebook.name)
    write_json(kernel_dir / "kernel-metadata.json", {
        "id": f"{username}/{slug}", "title": "SMS AC LLM2 detailed QLoRA training" if llm2_only else "SMS AC LLM1 report QLoRA training" if llm1_only else "SMS AC dual LLM QLoRA training",
        "code_file": notebook.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [dataset_ref], "competition_sources": [], "kernel_sources": [],
    })
    state = {"status": "prepared_not_submitted", "created_at": datetime.now(timezone.utc).isoformat(),
             "kernel": f"{username}/{slug}", "dataset": dataset_ref, "accelerator": "NvidiaTeslaT4",
             "package_sha256": sha256(dataset_dir / "project.zip"),
             "notebook_sha256": sha256(kernel_dir / notebook.name), "mode": "llm2_only" if llm2_only else "llm1_only" if llm1_only else "dual",
             "trained_models_exist": False}
    write_json(output / "job_state.json", state)
    return state


def cli_prefix() -> list[str]:
    executable = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if executable.is_file():
        return [str(executable)]
    located = shutil.which("kaggle")
    if not located:
        raise ValueError("Install the CPU Kaggle client: python -m pip install kaggle==2.2.4")
    return [located]


def run_cli(arguments: list[str]) -> None:
    # Keep config creation inside this project when no existing config directory is available.
    env = os.environ.copy()
    if not env.get("KAGGLE_CONFIG_DIR") and not (Path.home() / ".kaggle").is_dir():
        env["KAGGLE_CONFIG_DIR"] = str(ROOT / ".cache/kaggle")
    subprocess.run(cli_prefix() + arguments, check=True, env=env)


def load_job(output: Path) -> dict:
    if not output.resolve().is_relative_to((ROOT / "build").resolve()):
        raise ValueError("Job directory must be inside build/.")
    state = json.loads((output / "job_state.json").read_text(encoding="utf-8"))
    if "YOUR_KAGGLE" in state["kernel"]:
        raise ValueError("Prepare a new package with your real Kaggle username before submission.")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--username", default="YOUR_KAGGLE_USERNAME")
    p.add_argument("--slug", default="sms-ac-dual-llm")
    p.add_argument('--dataset-slug',help='Existing dataset slug when preparing a version update')
    p.add_argument("--output", type=Path, default=ROOT / "build" / datetime.now().strftime("kaggle-%Y%m%d-%H%M%S"))
    p.add_argument("--llm1-only", action="store_true", help="Package the LLM1 candidate dataset and LLM1-only notebook")
    p.add_argument("--llm2-only", action="store_true", help="Package the fresh detailed LLM2 dataset and LLM2-only notebook")
    for action in ["submit", "push", "status", "download"]:
        p = sub.add_parser(action)
        p.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            state = prepare(args.output, args.username, args.slug, args.dataset_slug, args.llm1_only, args.llm2_only)
            print(json.dumps({"directory": str(args.output), **state}, indent=2))
            return
        state = load_job(args.job)
        if args.command == "submit":
            if state["status"] != "prepared_not_submitted":
                raise ValueError("Dataset upload already attempted; use push to run an existing upload.")
            expected = {"project.zip", "package_manifest.json", "dataset-metadata.json"}
            actual = {path.name for path in (args.job / "dataset").iterdir()}
            if actual != expected or sha256(args.job / "dataset/project.zip") != state["package_sha256"]:
                raise ValueError("Package changed after preparation; prepare a fresh reviewed package.")
            run_cli(["datasets", "create", "-p", str(args.job / "dataset")])
            state["status"] = "dataset_uploaded"
            write_json(args.job / "job_state.json", state)
            print("Private dataset uploaded. Wait until Kaggle finishes processing it, then run 'push'.")
        elif args.command == "push":
            if state["status"] != "dataset_uploaded":
                raise ValueError("Upload the dataset first. A submitted run is not pushed a second time.")
            notebook_name = "train_llm2_detailed.ipynb" if state.get("mode") == "llm2_only" else "train_llm1_report_model.ipynb" if state.get("mode") == "llm1_only" else "train_dual_models.ipynb"
            if sha256(args.job / "kernel" / notebook_name) != state["notebook_sha256"]:
                raise ValueError("Notebook changed after preparation; prepare a new package.")
            run_cli(["kernels", "push", "-p", str(args.job / "kernel"), "--accelerator", "NvidiaTeslaT4", "--timeout", "39600"])
            state["status"] = "submitted_not_verified"
            state["submitted_at"] = datetime.now(timezone.utc).isoformat()
            write_json(args.job / "job_state.json", state)
            print(f"Submitted: https://www.kaggle.com/code/{state['kernel']}")
        elif args.command == "status":
            run_cli(["kernels", "status", state["kernel"]])
        elif args.command == "download":
            run_cli(["kernels", "output", state["kernel"], "-p", str(args.job / "outputs"),
                     "--file-pattern", r".*(\.zip|\.json|\.jsonl|\.log)$"])
            print("Outputs downloaded; inspect manifests and evaluation before claiming training success.")
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
