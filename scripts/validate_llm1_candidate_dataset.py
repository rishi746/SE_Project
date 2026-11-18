"""Validate LLM1 candidate data before upload or training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path


DOMAIN = re.compile(r"\b(network|cybersecurity|security|firewall|vpn|dns|dhcp|routing|switch|wireless|telemetry|packet|tls|identity|intrusion|zero-trust)\b", re.I)
FINANCE = re.compile(r"\b(loan|banking|credit scoring|payment processor|kyc|insurance underwriting)\b", re.I)
URL = re.compile(r"https?://[^\s)\]>]+")
ID = re.compile(r"\b(?:EVID|REQ|FR|NFR|SEC|UC|RISK|TEST|DEC)-\d{3,}\b")
REQUIRED_TASKS = {
    "llm1_clarification",
    "llm1_section_generation",
    "llm1_full_report",
    "llm1_consistency_review",
}
FULL_SECTION_HINTS = [
    "purpose", "problem", "objectives", "scope", "stakeholders", "role",
    "discovery", "network", "requirements", "nonfunctional", "security",
    "user stories", "use cases", "workflow", "architecture", "protocols",
    "data model", "risk", "feasibility", "priorities", "phases",
    "verification", "deployment", "traceability", "standards", "open issues",
    "glossary", "references",
]


def text_of(row: dict) -> str:
    bits = []
    for msg in row.get("messages", []):
        bits.append(str(msg.get("content", "")))
    bits.append(str(row.get("prompt", "")))
    bits.append(str(row.get("report_markdown", "")))
    return "\n".join(bits)


def validate_row(row: dict) -> list[str]:
    issues = []
    if row.get("task") not in REQUIRED_TASKS:
        issues.append("unknown_task")
    if row.get("split") not in {"train", "validation", "test"}:
        issues.append("bad_split")
    if not row.get("group_id"):
        issues.append("missing_group_id")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("training_approved") is not False:
        issues.append("training_approval_not_withheld")
    if metadata.get("review_status") != "assistant_reviewed_candidate":
        issues.append("bad_review_status")
    if metadata.get("domain") not in {"networking", "cybersecurity"}:
        issues.append("bad_domain")
    text = text_of(row)
    if not DOMAIN.search(text):
        issues.append("no_domain_signal")
    if FINANCE.search(text):
        issues.append("financial_domain_contamination")
    if row.get("task") in {"llm1_clarification", "llm1_section_generation", "llm1_consistency_review"}:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            issues.append("bad_message_shape")
    if row.get("task") == "llm1_full_report":
        report = str(row.get("report_markdown", ""))
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or len(evidence) < 5:
            issues.append("missing_evidence")
            evidence = []
        urls = {item.get("url") for item in evidence if isinstance(item, dict)}
        for url in URL.findall(report):
            if url.rstrip(".,;") not in urls:
                issues.append("unlisted_url")
                break
        lower = report.lower()
        for hint in FULL_SECTION_HINTS:
            if hint not in lower:
                issues.append("missing_full_section:" + hint)
        ids = set(ID.findall(report))
        for prefix in ("FR-", "NFR-", "SEC-", "UC-", "RISK-", "TEST-", "DEC-", "EVID-"):
            if not any(value.startswith(prefix) for value in ids):
                issues.append("missing_id_prefix:" + prefix)
        if len(report.split()) < 1800:
            issues.append("full_report_too_short_for_candidate")
    return sorted(set(issues))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=7000)
    args = parser.parse_args()

    rows = []
    for line_no, line in enumerate(args.input.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line"] = line_no
        rows.append(row)

    decisions = []
    counts = Counter()
    family_splits: dict[str, set[str]] = defaultdict(set)
    ids = set()
    duplicate_ids = []
    for row in rows:
        issues = validate_row(row)
        rid = row.get("id")
        if rid in ids:
            issues.append("duplicate_id")
            duplicate_ids.append(rid)
        ids.add(rid)
        family_splits[row.get("group_id")].add(row.get("split"))
        counts["checked"] += 1
        counts[row.get("task", "unknown")] += 1
        counts["eligible_candidate" if not issues else "rejected"] += 1
        decisions.append({"line": row["_line"], "id": row.get("id"), "task": row.get("task"), "issues": issues})

    for family, splits in sorted(family_splits.items()):
        if len(splits) > 1:
            counts["family_split_leakage"] += 1
            for decision in decisions:
                if next((r for r in rows if r.get("id") == decision["id"]), {}).get("group_id") == family:
                    decision["issues"].append("family_split_leakage")

    if len(rows) < args.min_rows:
        counts["below_min_rows"] = args.min_rows - len(rows)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "counts": dict(counts),
        "unique_ids": len(ids),
        "unique_families": len(family_splits),
        "duplicate_ids": duplicate_ids[:20],
        "decisions": decisions,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(counts), indent=2))
    if counts.get("rejected") or counts.get("family_split_leakage") or counts.get("below_min_rows"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
