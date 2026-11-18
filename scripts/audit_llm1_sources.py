"""Exhaustive structural and suitability audit of collected LLM 1 sources.

This is a screening tool. Passing a rule is not expert approval or a complete
networking SRS. The output records a decision for every input row and spec.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/llm1_collected'
OUT = ROOT / 'reports/llm1_source_audit'
NETWORK = re.compile(r'\b(network|routing|router|switch|firewall|dns|dhcp|vpn|bgp|ospf|ipv[46]|subnet|packet|traffic|load.balan|proxy|tls|authentication|authorization|metadata.agent|security|cyber|policy|endpoint|service.discovery|telemetry)\b', re.I)
CORE_DOMAIN = re.compile(r'\b(network|routing|router|switch|firewall|dns|dhcp|vpn|bgp|ospf|ipv[46]|subnet|packet|traffic|load.balan|proxy|tls|security|cyber|service.discovery|telemetry|observability|zero.trust|identity|authenticat|authoriz)\b', re.I)
OFF_DOMAIN = re.compile(r'(^|/)(ai-ml|data|databases|analytics|media|web|integration|mobile|sap|mainframe)(/|$)', re.I)
FINANCE = re.compile(r'\b(loan|banking|credit.score|insurance|payment.transaction|kyc)\b', re.I)
HEADINGS = ('problem description', 'proposed change', 'alternatives', 'impact', 'testing', 'implementation')


def normalize_text(value: str) -> str:
    return ' '.join(value.casefold().split())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')


def audit_policy() -> tuple[list[dict], dict]:
    results, seen, counts = [], set(), Counter()
    metadata = json.loads((BASE / 'netconfeval_metadata.json').read_text(encoding='utf-8'))
    expected = {x['rfilename'] for x in metadata['siblings'] if x['rfilename'].endswith('.jsonl')}
    for name in sorted(expected):
        path = BASE / 'netconfeval' / name
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            issues = []
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {}; issues.append('invalid_json')
            request = row.get('human_language') or row.get('prompt') or ''
            if not isinstance(request, str) or not request.strip(): issues.append('missing_requirement_text')
            if name.startswith('step_1'):
                try:
                    target = json.loads(row['expected'])
                    if not isinstance(target, dict): issues.append('expected_not_object')
                except (KeyError, ValueError, TypeError): issues.append('invalid_expected_policy')
            else:
                issues.append('configuration_or_code_task_not_llm1_srs')
            if name == 'step_1_spec_conflict.jsonl' and not isinstance(row.get('conflict_exist'), bool):
                issues.append('missing_boolean_conflict_label')
            key = sha256(normalize_text(request).encode()).hexdigest() if isinstance(request, str) else ''
            if key in seen: issues.append('duplicate_requirement_text_across_sources')
            else: seen.add(key)
            if isinstance(request, str) and FINANCE.search(request): issues.append('financial_domain_language')
            status = 'auxiliary_policy_candidate' if not issues and name.startswith('step_1') else 'excluded'
            counts[status] += 1
            for issue in issues: counts['issue:' + issue] += 1
            results.append({'source': 'NetConfEval/NetConfEval', 'file': name, 'line': number,
                'record_id': sha256(f'{name}:{number}'.encode()).hexdigest()[:20],
                'decision': status, 'issues': issues, 'complete_srs': False,
                'sft_approved': False, 'revision': metadata['sha']})
    return results, dict(counts)


def audit_neutron() -> tuple[list[dict], dict]:
    results, seen, counts = [], set(), Counter()
    repo = BASE / 'neutron-specs'
    for path in sorted((repo / 'specs').rglob('*.rst')):
        text = path.read_text(encoding='utf-8', errors='replace')
        relative = path.relative_to(repo).as_posix()
        issues = []
        if path.name in {'template.rst', 'postmortem_template.rst', 'scorecard_template.rst', 'index.rst'}:
            issues.append('template_or_index')
        if len(text.split()) < 250: issues.append('too_short_for_design_source')
        if not NETWORK.search(text): issues.append('no_network_or_security_signal')
        if FINANCE.search(text[:2000]): issues.append('financial_domain_signal')
        if '\ufffd' in text: issues.append('decoding_replacement_character')
        canonical = sha256(normalize_text(text).encode()).hexdigest()
        if canonical in seen: issues.append('duplicate_document_text')
        else: seen.add(canonical)
        present = [h for h in HEADINGS if re.search(r'(?im)^' + re.escape(h) + r'\s*$', text)]
        if len(present) < 2: issues.append('insufficient_spec_sections')
        status = 'design_source_candidate' if not issues else 'excluded'
        counts[status] += 1
        for issue in issues: counts['issue:' + issue] += 1
        results.append({'source': 'openstack/neutron-specs', 'file': relative,
            'decision': status, 'issues': issues, 'words': len(text.split()),
            'detected_sections': present, 'complete_srs': False,
            'sft_approved': False, 'license': 'CC-BY-3.0 per repository LICENSE',
            'content_sha256': sha256(text.encode()).hexdigest()})
    return results, dict(counts)


def audit_corpus() -> tuple[list[dict], dict]:
    results, counts = [], Counter()
    for number, line in enumerate((ROOT / 'data/corpus/documents.jsonl').open(encoding='utf-8'), 1):
        row = json.loads(line)
        issues = []
        text = row.get('text', '')
        title_path = row.get('title', '') + '\n' + row.get('source_path', '')
        if OFF_DOMAIN.search(row.get('source_path', '')):
            issues.append('off_domain_source_path')
        if not CORE_DOMAIN.search(title_path):
            issues.append('no_core_domain_signal_in_title_or_path')
        if FINANCE.search(row.get('title', '')): issues.append('financial_domain_title')
        if row.get('review_status') != 'license_file_verified_content_unreviewed':
            issues.append('unexpected_review_status')
        status = 'retrieval_reference_candidate' if not issues else 'excluded_from_llm1_domain_pool'
        counts[status] += 1
        for issue in issues: counts['issue:' + issue] += 1
        results.append({'source': row.get('source_id'), 'source_url': row.get('source_url'),
            'record_id': row.get('id'), 'line': number, 'decision': status, 'issues': issues,
            'complete_srs': False, 'sft_approved': False, 'existing_split': row.get('split'),
            'existing_review_status': row.get('review_status')})
    return results, dict(counts)


def audit_cilium() -> tuple[list[dict], dict]:
    repo = BASE / 'cilium-design-cfps'
    results, counts, seen = [], Counter(), set()
    for path in sorted(repo.rglob('*.md')):
        text = path.read_text(encoding='utf-8', errors='replace')
        rel = path.relative_to(repo).as_posix()
        issues = []
        if path.name.lower() in {'readme.md', 'template.md'} or 'template' in path.stem.lower():
            issues.append('template_or_index')
        if len(text.split()) < 250: issues.append('too_short_for_design_source')
        if not CORE_DOMAIN.search(rel + '\n' + text[:1000]): issues.append('no_core_domain_signal')
        if '\ufffd' in text: issues.append('decoding_replacement_character')
        digest = sha256(normalize_text(text).encode()).hexdigest()
        if digest in seen: issues.append('duplicate_document_text')
        else: seen.add(digest)
        headings = re.findall(r'(?m)^#{1,5}\s+(.+)$', text)
        if len(headings) < 3: issues.append('insufficient_spec_sections')
        status = 'design_source_candidate' if not issues else 'excluded'
        counts[status] += 1
        for issue in issues: counts['issue:' + issue] += 1
        results.append({'source': 'cilium/design-cfps', 'file': rel,
                        'decision': status, 'issues': issues, 'words': len(text.split()),
                        'headings': headings[:30], 'complete_srs': False,
                        'sft_approved': False, 'license': 'Apache-2.0 per repository LICENSE',
                        'content_sha256': sha256(text.encode()).hexdigest()})
    return results, dict(counts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {'method': 'automatic_exhaustive_screening_not_expert_validation', 'sources': {}}
    for name, audit in [('netconfeval', audit_policy), ('neutron_specs', audit_neutron),
                        ('cilium_cfps', audit_cilium), ('existing_corpus', audit_corpus)]:
        rows, counts = audit()
        write_jsonl(OUT / f'{name}_decisions.jsonl', rows)
        summary['sources'][name] = {'total': len(rows), 'decisions': counts}
    summary['full_report_sft_approved'] = 0
    summary['warning'] = 'Auxiliary policy and design references are not paired, complete, reviewed SRS examples.'
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
