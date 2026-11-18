"""Create attributed source-grounded section reconstruction candidates.

These teach report-section structure and technical vocabulary. They are not
complete, stakeholder-approved SRS pairs and are never silently marked so.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/llm1_collected'
REPO = BASE / 'neutron-specs'
AUDIT = ROOT / 'reports/llm1_source_audit/neutron_specs_decisions.jsonl'
HEADING = re.compile(r'(?m)^([^\n]{4,80})\n([=~^-])\2{3,}\s*$')
NONCONTENT = re.compile(r'^(references|contributors|implementation|work items|assignee|other contributor|history|launchpad|dependencies|release note|documentation impact)', re.I)


def sections(text: str):
    matches = list(HEADING.finditer(text))
    return [(m.group(1).strip(), text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip())
            for i, m in enumerate(matches)]


def main():
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    accepted = [json.loads(x) for x in AUDIT.read_text(encoding='utf-8').splitlines()]
    accepted = [x for x in accepted if x['decision'] == 'design_source_candidate']
    result = []
    for item in accepted:
        path = REPO / item['file']
        text = path.read_text(encoding='utf-8')
        parts = sections(text)
        problem = next((body for title, body in parts if title.casefold() == 'problem description'), '')
        if len(problem.split()) < 35:
            continue
        title = path.stem.replace('-', ' ').title()
        for heading, body in parts:
            if NONCONTENT.search(heading) or heading.casefold() == 'problem description':
                continue
            words = len(body.split())
            if not 45 <= words <= 900 or len(body) > 5000:
                continue
            # Do not let a section target see itself in the prompt.
            context = problem[:5000]
            target = f'## {heading}\n\n{body}\n\nSource: https://github.com/openstack/neutron-specs/blob/{commit}/{item["file"]}'
            prompt = {'project_title': title, 'source_problem_description': context,
                      'requested_section': heading, 'scope': 'networking feature design',
                      'source_url': f'https://github.com/openstack/neutron-specs/blob/{commit}/{item["file"]}',
                      'instruction': 'Draft only the requested section using the supplied problem description. Identify unresolved details; do not invent measurements or approvals.'}
            uid = hashlib.sha256((item['file'] + '\n' + heading + '\n' + body).encode()).hexdigest()
            result.append({'id': uid, 'task': 'llm1_source_section_candidate',
                'messages': [{'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False)},
                             {'role': 'assistant', 'content': target}],
                'provenance': {'kind': 'collected_source_section', 'source': 'openstack/neutron-specs',
                    'commit': commit, 'path': item['file'], 'license': 'CC-BY-3.0'},
                'group_id': item['file'], 'split': 'unassigned', 'review_status': 'rejected_insufficient_input_evidence',
                'training_approved': False, 'complete_srs': False,
                'limitations': ['The source problem may not entail every design detail in the target section.',
                                'A section excerpt is not a complete SRS or validated implementation.']})
    dest = BASE / 'neutron_section_candidates.jsonl'
    dest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result), encoding='utf-8')
    print(json.dumps({'section_candidates': len(result), 'complete_srs': 0, 'training_approved': 0,
                      'target': str(dest)}))


if __name__ == '__main__':
    main()
