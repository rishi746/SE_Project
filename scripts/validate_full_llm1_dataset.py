"""Gate complete LLM 1 SRS records before they enter any training run."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path

REQUIRED_SECTIONS = (
    'purpose', 'problem', 'objectives', 'scope', 'stakeholders', 'roles',
    'discovery', 'network', 'requirements', 'nonfunctional', 'security',
    'user stories', 'use cases', 'workflows', 'architecture', 'interfaces',
    'data model', 'risks', 'feasibility', 'priorities', 'phases', 'verification',
    'deployment', 'traceability', 'standards', 'open issues', 'glossary',
    'references',
)
ID = re.compile(r'\b(?:EVID|REQ|FR|NFR|SEC|UC|RISK|TEST|DEC)-\d{3,}\b')
URL = re.compile(r'https?://[^\s)\]>]+')
HEADING = re.compile(r'(?m)^#{1,4}\s+(.+?)\s*$')
DOMAIN = re.compile(r'\b(network|cybersecurity|security|firewall|vpn|dns|dhcp|routing|switch|wireless|telemetry|observability|packet|tls|identity|intrusion)\b', re.I)
FINANCE = re.compile(r'\b(loan|banking|credit scoring|payment processor|kyc|insurance underwriting)\b', re.I)


def validate(row: dict) -> list[str]:
    issues = []
    if row.get('kind') != 'complete_llm1_report': issues.append('wrong_record_kind')
    if row.get('domain') not in {'networking', 'cybersecurity'}: issues.append('wrong_domain')
    if row.get('review_status') != 'expert_approved': issues.append('no_expert_approval')
    evidence = row.get('evidence')
    if not isinstance(evidence, list) or not evidence: issues.append('missing_evidence_ledger'); evidence = []
    if not row.get('group_id'): issues.append('missing_scenario_family')
    if row.get('split') not in {'train', 'validation', 'test'}: issues.append('missing_split')
    text = row.get('report_markdown', '')
    if not isinstance(text, str): text = ''
    if len(text.split()) < 3000: issues.append('report_under_3000_words')
    if not DOMAIN.search(text): issues.append('no_domain_signal')
    if FINANCE.search(text): issues.append('financial_domain_contamination')
    headings = [h.lower() for h in HEADING.findall(text)]
    for section in REQUIRED_SECTIONS:
        if not any(section in h for h in headings): issues.append('missing_section:' + section)
    evidence_ids = {x.get('id') for x in evidence if isinstance(x, dict)}
    if not evidence_ids or any(not x or not re.fullmatch(r'EVID-\d{3,}', x) for x in evidence_ids):
        issues.append('invalid_evidence_ids')
    sources = {x.get('url') for x in evidence if isinstance(x, dict) and x.get('url')}
    for url in URL.findall(text):
        if url.rstrip('.,;') not in sources: issues.append('unlisted_url:' + url[:120])
    ids = ID.findall(text)
    if len({x for x in ids if x.startswith(('REQ-', 'FR-', 'NFR-', 'SEC-'))}) < 8:
        issues.append('too_few_requirement_ids')
    for kind in ('UC-', 'RISK-', 'TEST-'):
        if not any(x.startswith(kind) for x in ids): issues.append('missing_id_family:' + kind)
    # Machine checks cannot determine whether an assertion follows from source
    # evidence. Expert approval and factual review remain separate requirements.
    return sorted(set(issues))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    decisions, counts, families = [], Counter(), {}
    for number, line in enumerate(args.jsonl.open(encoding='utf-8'), 1):
        row = json.loads(line)
        issues = validate(row)
        family, split = row.get('group_id'), row.get('split')
        if family in families and families[family] != split: issues.append('family_split_leakage')
        families[family] = split
        counts['checked'] += 1
        counts['eligible' if not issues else 'rejected'] += 1
        decisions.append({'line': number, 'id': row.get('id'), 'eligible': not issues, 'issues': issues})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'counts': counts, 'decisions': decisions}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(counts)))
    if counts['rejected']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
