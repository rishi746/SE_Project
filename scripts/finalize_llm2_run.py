"""Download and verify the fresh Kaggle LLM2 adapter before starting its chat app."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "build/kaggle-llm2-detailed-v3-clean"
KERNEL = "rishinandandr/sms-ac-llm2-detailed-qlora-training"
EXPORT = "sms-ac-llm2-detailed-v3.zip"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str) -> None:
    subprocess.run(["kaggle", *args], cwd=ROOT, check=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        target = JOB / "outputs"
        target.mkdir(parents=True, exist_ok=True)
        run("kernels", "output", KERNEL, "-p", str(target), "--file-pattern", r".*\.zip$")
    archive_path = JOB / "outputs" / EXPORT
    if not archive_path.is_file():
        raise FileNotFoundError(f"Completed training export missing: {archive_path}")
    target = ROOT / "models/llm2-detailed-v3-adapter"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        prefix = "models/llm2-detailed-v3-adapter/"
        if not any(name == prefix + "adapter_model.safetensors" for name in names):
            raise ValueError("Export lacks the new adapter weights")
        for item in archive.infolist():
            if not item.filename.startswith(prefix) or item.is_dir():
                continue
            relative = Path(item.filename.removeprefix(prefix))
            if len(relative.parts) != 1 or relative.name in ("", ".", ".."):
                raise ValueError("Unsafe adapter export path")
            destination = target / relative
            if destination.exists() and destination.read_bytes() != archive.read(item):
                raise ValueError(f"Existing file differs; refusing to overwrite {destination}")
            destination.write_bytes(archive.read(item))
        report_name = "reports/llm2-detailed-v3-evaluation/evaluation_report.json"
        report = json.loads(archive.read(report_name))
    manifest = json.loads((target / "model_manifest.json").read_text(encoding="utf-8"))
    if not (manifest["training_completed"] and manifest["full_dataset_epochs_completed"]):
        raise ValueError("Fresh LLM2 did not finish a full epoch")
    if sha(target / "adapter_model.safetensors") != manifest["adapter_sha256"]:
        raise ValueError("Adapter SHA-256 does not match training manifest")
    for split in ("train", "validation", "test"):
        local = ROOT / "data/llm2_training/sdlc" / f"{split}.jsonl"
        if sha(local) != manifest["data_sha256"][split]:
            raise ValueError(f"Dataset hash differs for {split}")
    if report["adapter_sha256"] != manifest["adapter_sha256"]:
        raise ValueError("Evaluation report belongs to a different adapter")
    print(json.dumps({"adapter": str(target), "steps": manifest["global_step"],
                      "epochs": manifest["trained_epochs"],
                      "train_examples": manifest["eligible_train_examples"],
                      "held_out_examples": manifest["held_out_test_examples"],
                      "adapter_minus_base_nll": report["adapter_minus_base_nll"],
                      "incomplete_generations": report["incomplete_generations"]}, indent=2))


if __name__ == "__main__":
    main()
