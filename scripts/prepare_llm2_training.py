"""Convert the validated detailed LLM2 cases to the existing Kaggle trainer schema."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/llm2_sdlc_detailed"
TARGET = ROOT / "data/llm2_training/sdlc"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    summary = {}
    groups = {}
    for split in ("train", "validation", "test"):
        source = SOURCE / f"{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        output = TARGET / f"{split}.jsonl"
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                assert row["task"] == "sdlc_selection" and row["split"] == split
                assert row["messages"][1]["content"] == row["question"]
                assert row["messages"][2]["content"] == row["answer"]
                group = row["metadata"]["project_family"]
                assert group not in groups or groups[group] == split
                groups[group] = split
                adapted = {"id": row["id"], "task": "sdlc", "group_id": group,
                           "label": row["label"], "messages": row["messages"],
                           "source_dataset": "llm2_sdlc_detailed_v3"}
                stream.write(json.dumps(adapted, ensure_ascii=False) + "\n")
        summary[split] = {"count": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
                          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    assert sum(s["count"] for s in summary.values()) == 1008
    assert len(groups) == 24
    (TARGET.parent / "conversion_manifest.json").write_text(
        json.dumps({"source": str(SOURCE.relative_to(ROOT)), "task": "sdlc",
                    "conversion": "metadata-only trainer schema; question and answer bytes unchanged",
                    "splits": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
