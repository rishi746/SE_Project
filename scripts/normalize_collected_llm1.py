"""Normalize collected network policy pairs as candidates, never approved SRS targets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/llm1_collected'


def main():
    revision = json.loads((BASE / 'netconfeval_metadata.json').read_text())['sha']
    records, seen = [], set()
    for filename, task in [('step_1_spec_translation.jsonl', 'network_policy_extraction'),
                           ('step_1_spec_conflict.jsonl', 'network_policy_conflict_detection')]:
        for number, line in enumerate((BASE / 'netconfeval' / filename).read_text(encoding='utf-8').splitlines(), 1):
            row = json.loads(line)
            request = row['human_language'].strip()
            # The conflict corpus includes an expected policy even for contradictions.
            # Never teach its policy as a resolved answer to a contradictory request.
            if task.endswith('conflict_detection'):
                answer = {'conflict_detected': row['conflict_exist'],
                          'policy_resolution': 'requires_clarification' if row['conflict_exist'] else 'no_conflict_labeled_by_source'}
            else:
                answer = json.loads(row['expected'])
            key = hashlib.sha256(json.dumps([task, request, answer], sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            records.append({'id': key, 'task': task, 'messages': [
                {'role': 'system', 'content': row['description'] + '\nExtract network policies or identify conflicts as requested; do not add project facts.'},
                {'role': 'user', 'content': json.dumps({'task': task, 'requirements': request})},
                {'role': 'assistant', 'content': json.dumps(answer, sort_keys=True)}],
                'provenance': {'kind': 'collected_transformed', 'dataset': 'NetConfEval/NetConfEval',
                    'revision': revision, 'file': filename, 'line': number, 'license': 'MIT per dataset card'},
                'review_status': 'pending_content_review', 'training_approved': False,
                'split': 'unassigned', 'family': 'netconfeval_shared_policy_pool',
                'limitations': ['Auxiliary task, not a full report.', 'Shared policy pools require overlap-aware splitting.',
                                 'Published benchmark data; cannot claim independent evaluation on this benchmark after training.']})
    target = BASE / 'normalized_candidates.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    print(json.dumps({'normalized_unique_candidates': len(records), 'training_approved': 0,
                      'complete_report_instances': 0, 'target': str(target)}))


if __name__ == '__main__':
    main()
