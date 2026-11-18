"""Validate task schema, provenance and paired family split isolation on CPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(root):
    seen_ids, seen_inputs, group_splits, counts = set(), set(), {}, Counter()
    for task in ('design','sdlc'):
        for split in ('train','validation','test'):
            path = root/task/(split+'.jsonl')
            count = 0
            with path.open(encoding='utf-8') as stream:
                for line_number,line in enumerate(stream,1):
                    row=json.loads(line)
                    assert row['task']==task, f'{path}:{line_number}: wrong task'
                    assert row['id'] not in seen_ids, 'Duplicate record ID'
                    seen_ids.add(row['id'])
                    messages=row['messages']
                    assert len(messages)>=3 and len(messages)%2==1
                    assert [m['role'] for m in messages]==['system']+['user','assistant']*((len(messages)-1)//2)
                    assert all(isinstance(m['content'],str) and m['content'].strip() for m in messages)
                    if row.get('metadata',{}).get('stage') != 'clarification':
                        payload=json.loads(messages[1]['content'])
                        assert ('request' if task=='design' else 'software_design') in payload
                    fingerprint=hashlib.sha256((task+json.dumps(messages[:-1],sort_keys=True)).encode()).hexdigest()
                    assert fingerprint not in seen_inputs, 'Duplicate task input'
                    seen_inputs.add(fingerprint)
                    group=row['group_id']
                    assert group and group_splits.setdefault(group,split)==split, 'Scenario-family leakage across splits/tasks'
                    metadata=row['metadata']
                    assert metadata.get('supervision') and metadata.get('review_status')
                    assert len(metadata['source_ids'])==len(metadata['source_urls'])==len(metadata['source_licenses'])
                    assert len(messages[-1]['content'])>(40 if metadata.get('stage')=='clarification' else 500), 'Incomplete target'
                    count+=1
            assert count>0, f'Empty split {path}'
            counts[task+'/'+split]=count
    return {'valid':True,'rows':sum(counts.values()),'families':len(group_splits),'counts':dict(counts),'note':'Structural validation; not expert content validation or semantic deduplication.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,default=ROOT/'data/dual')
    args=parser.parse_args()
    print(json.dumps(validate(args.directory),indent=2))
