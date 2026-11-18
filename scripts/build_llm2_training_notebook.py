"""Build an LLM2-only Kaggle notebook from the maintained dual-run setup cells."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    original = json.loads((ROOT / "kaggle/train_dual_models.ipynb").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in original["cells"]}
    cells = []

    def markdown(identifier: str, value: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": value.splitlines(keepends=True), "id": identifier})

    def code(identifier: str, value: str) -> None:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": value.splitlines(keepends=True), "id": identifier})

    markdown("llm2-intro", "# SMS.AC LLM 2 — fresh detailed SDLC training\n\n"
             "This notebook starts from Qwen3-4B-Instruct-2507 and trains **only** the SDLC adapter on the 1,008-record detailed v3 dataset. "
             "The old LLM 2 adapter and dataset are not loaded. Select a Kaggle GPU and enable Internet. "
             "Outputs remain synthetic-data research results until separately reviewed on real design documents.\n")
    setup = "".join(by_id["sms-ac-02"]["source"])
    setup = setup.replace("RUN_FULL = False", "RUN_FULL = True")
    setup = setup.replace("MAX_SEQ_LENGTH = 4096", "MAX_SEQ_LENGTH = 4096")
    setup = setup.replace('RESUME = {"design": None, "sdlc": None}', 'RESUME = {"sdlc": None}')
    setup = setup.replace("TEST_LIMIT = 12", "TEST_LIMIT = 24")
    setup = setup.replace("GENERATION_LIMIT = 3", "GENERATION_LIMIT = 3")
    setup = setup.replace("CHALLENGE_LIMIT = 1", "CHALLENGE_LIMIT = 0")
    code("llm2-setup", setup)
    markdown("llm2-load-heading", "## Load the private project package\n")
    loader = "".join(by_id["sms-ac-04"]["source"])
    loader = loader.replace('for task in ("design", "sdlc"):\n    for split in ("train", "validation", "test"):\n        assert (PROJECT / "data/dual" / task / f"{split}.jsonl").is_file(), f"Missing {task}/{split}"',
                            'for split in ("train", "validation", "test"):\n    assert (PROJECT / "data/llm2_training/sdlc" / f"{split}.jsonl").is_file(), f"Missing detailed LLM2/{split}"')
    code("llm2-load", loader)
    markdown("llm2-env-heading", "## Install isolated Kaggle training environment\n")
    code("llm2-env", "".join(by_id["sms-ac-06"]["source"]))
    markdown("llm2-preflight-heading", "## Validate all examples and full token lengths before training\n")
    code("llm2-preflight", '''def run_script(name, *arguments):
    command = [str(PYTHON), str(PROJECT / "scripts" / name), *map(str, arguments)]
    subprocess.run(command, cwd=str(PROJECT), env=os.environ.copy(), check=True)

run_script("train_dual_qlora.py", "--task", "sdlc", "--data-dir", PROJECT / "data/llm2_training",
           "--model", BASE_MODEL, "--revision", BASE_REVISION, "--max-seq-length", MAX_SEQ_LENGTH,
           "--output-dir", PROJECT / "models/preflight/llm2-detailed", "--preflight-only")
''')
    markdown("llm2-train-heading", "## Train fresh LLM 2 adapter for one complete epoch\n\n"
             "The run uses only the new detailed dataset. It writes checkpoints and a manifest containing dataset hashes.\n")
    code("llm2-train", '''arguments = [
    "--task", "sdlc", "--data-dir", PROJECT / "data/llm2_training",
    "--model", BASE_MODEL, "--revision", BASE_REVISION,
    "--max-seq-length", MAX_SEQ_LENGTH, "--gradient-accumulation", GRADIENT_ACCUMULATION,
    "--output-dir", PROJECT / "models/llm2-detailed-v3-adapter",
    "--save-steps", 25, "--eval-samples", 24,
]
arguments += ["--full", "--epochs", FULL_EPOCHS] if RUN_FULL else ["--max-steps", PILOT_STEPS]
if RESUME["sdlc"]:
    arguments += ["--resume-from-checkpoint", RESUME["sdlc"]]
run_script("train_dual_qlora.py", *arguments)
manifest = json.loads((PROJECT / "models/llm2-detailed-v3-adapter/model_manifest.json").read_text())
assert manifest["training_completed"] and manifest["full_dataset_epochs_completed"]
print("Completed:", manifest["global_step"], "steps;", manifest["trained_epochs"], "epochs")
''')
    markdown("llm2-eval-heading", "## Compare base and adapter on held-out project families\n")
    code("llm2-eval", '''run_script("evaluate_dual_models.py", "--task", "sdlc",
           "--data-dir", PROJECT / "data/llm2_training",
           "--adapter", PROJECT / "models/llm2-detailed-v3-adapter",
           "--output-dir", PROJECT / "reports/llm2-detailed-v3-evaluation",
           "--test-limit", TEST_LIMIT, "--generation-limit", GENERATION_LIMIT,
           "--challenge-limit", 0, "--rag-ablation-limit", 0,
           "--max-new-tokens", 2048, "--max-generation-context", 8192)
report = json.loads((PROJECT / "reports/llm2-detailed-v3-evaluation/evaluation_report.json").read_text())
print("Adapter minus base held-out NLL:", report["adapter_minus_base_nll"])
print("Incomplete generations:", report["incomplete_generations"])
''')
    markdown("llm2-export-heading", "## Export the new adapter, evaluation, and data provenance\n")
    code("llm2-export", '''export_path = Path("/kaggle/working/sms-ac-llm2-detailed-v3.zip")
adapter = PROJECT / "models/llm2-detailed-v3-adapter"
manifest = json.loads((adapter / "model_manifest.json").read_text())
assert manifest["training_completed"] and manifest["evaluation_status"] == "paired_heldout_complete"
files = [p for p in adapter.iterdir() if p.is_file()]
files += [p for p in (PROJECT / "reports/llm2-detailed-v3-evaluation").rglob("*") if p.is_file()]
files += [PROJECT / "data/llm2_training/conversion_manifest.json"]
with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        archive.write(path, path.relative_to(PROJECT))
print("Saved:", export_path, "MiB:", round(export_path.stat().st_size / 1024**2, 1))
from IPython.display import FileLink, display
display(FileLink(str(export_path)))
''')
    original["cells"] = cells
    (ROOT / "kaggle/train_llm2_detailed.ipynb").write_text(json.dumps(original, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote Kaggle LLM2-only notebook")


if __name__ == "__main__":
    main()
