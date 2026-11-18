"""Verify every transformed candidate against its pinned source and task fit."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/llm1_collected'
OUT = ROOT / 'reports/llm1_source_audit'


def each(path):
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            yield number, json.loads(line)


def policy():
    result, counts = [], Counter()
    source_cache = {}
    for number, row in each(BASE / 'normalized_candidates.jsonl'):
        src = row['provenance']; file = src['file']; line = src['line']
        if file not in source_cache:
            source_cache[file] = [json.loads(x) for x in (BASE / 'netconfeval' / file).read_text(encoding='utf-8').splitlines()]
        original = source_cache[file][line - 1]
        issues = []
        prompt = json.loads(row['messages'][1]['content'])
        answer = json.loads(row['messages'][2]['content'])
        if prompt['requirements'] != original['human_language'].strip(): issues.append('input_mismatch')
        if row['task'] == 'network_policy_extraction':
            if answer != json.loads(original['expected']): issues.append('target_mismatch')
        elif row['task'] == 'network_policy_conflict_detection':
            if answer['conflict_detected'] is not original['conflict_exist']: issues.append('label_mismatch')
        else: issues.append('wrong_task')
        if row['training_approved'] or row['split'] != 'unassigned': issues.append('premature_training_approval')
        counts['checked'] += 1
        if issues: counts['bad'] += 1
        result.append({'candidate_id': row['id'], 'source_file': file, 'source_line': line,
                       'source_pair_exact': not issues, 'issues': issues,
                       'srs_status': 'auxiliary_only', 'training_approved': False})
    return result, counts


def sections():
    result, counts = [], Counter()
    for number, row in each(BASE / 'neutron_section_candidates.jsonl'):
        src = row['provenance']
        original = (BASE / 'neutron-specs' / src['path']).read_text(encoding='utf-8')
        prompt = json.loads(row['messages'][0]['content'])
        target = row['messages'][1]['content']
        body = target.rsplit('\n\nSource: ', 1)[0].split('\n\n', 1)[1]
        issues = []
        if body not in original: issues.append('target_not_in_source')
        if prompt['source_problem_description'] not in original: issues.append('problem_not_in_source')
        if row['training_approved'] or row['split'] != 'unassigned': issues.append('premature_training_approval')
        # A problem description alone does not guarantee its implementation,
        # performance, or security impact section. This is a task-fit blocker.
        counts['checked'] += 1
        counts['pair_context_insufficient_for_faithful_generation'] += 1
        if issues: counts['bad_provenance'] += 1
        result.append({'candidate_id': row['id'], 'source_path': src['path'], 'source_exact': not issues,
                       'issues': issues + ['pair_context_insufficient_for_faithful_generation'],
                       'training_approved': False})
    return result, counts


def main():
    summary = {}
    for name, fn in [('normalized_policy', policy), ('neutron_sections', sections)]:
        rows, counts = fn()
        (OUT / f'{name}_validation.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows), encoding='utf-8')
        summary[name] = dict(counts)
    (OUT / 'candidate_validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
