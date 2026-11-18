"""Validate every serialized detailed LLM2 case independently of its builder."""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(os.environ.get("SMS_AC_LLM2_OUTPUT", str(Path(__file__).resolve().parents[1] / "data" / "llm2_sdlc_detailed")))
MODELS = ("Waterfall", "V-shaped", "RAD", "Prototype", "Spiral", "Iterative")
SECTIONS = (
    "## Why this model is appropriate",
    "## Applying",
    "## Ten advantages and their relevance to this project",
    "## Why close alternatives are less suitable",
    "## Acceptance and conclusion",
)

def main() -> None:
    all_rows = [json.loads(line) for line in (DATA / "all.jsonl").read_text(encoding="utf-8").splitlines()]
    split_rows = []
    for split in ("train", "validation", "test"):
        for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["split"] == split
            split_rows.append(row)
    assert len(all_rows) == len(split_rows) == 1008
    assert {r["id"] for r in all_rows} == {r["id"] for r in split_rows}
    assert Counter(r["label"] for r in all_rows) == {model: 168 for model in MODELS}
    for key in ("id", "question", "answer"):
        assert len({r[key] for r in all_rows}) == 1008, f"duplicate {key}"
    families = defaultdict(set)
    for row in all_rows:
        label, q, a = row["label"], row["question"], row["answer"]
        assert row["task"] == "sdlc_selection"
        assert a.startswith(f"# Most Appropriate Life Cycle Model: {label}\n")
        assert all(s in a for s in SECTIONS)
        assert len(re.findall(r"(?m)^\d+\. \*\*", a)) == 16, row["id"]
        benefits = a.split(SECTIONS[2] + "\n\n", 1)[1].split("\n\n" + SECTIONS[3], 1)[0].split("\n\n")
        assert len(benefits) == len(set(benefits)) == 10
        assert "**Acceptance:**" in a and "**Conclusion:**" in a
        assert row["messages"][1]["content"] == q and row["messages"][2]["content"] == a
        assert len(q.split()) >= 160 and len(a.split()) >= 600
        assert row["metadata"]["provenance"] == "original_synthetic_case"
        assert row["metadata"]["review_status"] == "rule_and_structure_validated_not_independent_expert_gold"
        bad_phrase = re.search(r"\ba (?:IoT|intrusion|API)\b|\ba (?:five|six|seven|eight|nine) months\b", q + " " + a, re.I)
        assert bad_phrase is None, (row["id"], bad_phrase.group() if bad_phrase else "")
        families[row["metadata"]["project_family"]].add(row["split"])
    assert len(families) == 24 and all(len(splits) == 1 for splits in families.values())
    result = {
        "valid": True, "rows_checked": len(all_rows),
        "per_model": dict(Counter(r["label"] for r in all_rows)),
        "per_split": dict(Counter(r["split"] for r in all_rows)),
        "unique_project_families": len(families),
        "min_question_words": min(len(r["question"].split()) for r in all_rows),
        "min_answer_words": min(len(r["answer"].split()) for r in all_rows),
        "review_limit": "structural and rule consistency only; independent subject-matter review not yet performed",
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
