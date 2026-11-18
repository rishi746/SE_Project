from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from network_req_agent.schema import SftExample


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/validate_dataset.py <dataset.jsonl>")

    path = Path(sys.argv[1])
    topic_counts: Counter[str] = Counter()
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                example = SftExample.model_validate(json.loads(line))
            except Exception as exc:
                raise SystemExit(f"Invalid record at line {line_number}: {exc}") from exc
            topic_counts[example.output.project_type] += 1
            total += 1

    print(f"validated_records={total}")
    for topic, count in sorted(topic_counts.items()):
        print(f"{topic}={count}")


if __name__ == "__main__":
    main()
