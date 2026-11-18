"""CPU tokenizer audit. Downloads tokenizer assets only, never model weights."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(ROOT/'scripts'))
from train_dual_qlora import encode_completion,length_summary


def main():
    from transformers import AutoTokenizer
    model_path = 'Qwen/Qwen3-4B-Instruct-2507'
    if os.environ.get('HF_HUB_OFFLINE') == '1':
        cache = ROOT/'.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507'
        revision = (cache/'refs/main').read_text().strip()
        model_path = str(cache/'snapshots'/revision)
    tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=False,
        local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1')
    report={'model':'Qwen/Qwen3-4B-Instruct-2507','weights_downloaded':False,'splits':{}}
    for task in ['design','sdlc']:
        for split in ['train','validation','test']:
            lengths=[]
            with (ROOT/'data/dual'/task/(split+'.jsonl')).open(encoding='utf-8') as stream:
                for line in stream:
                    lengths.append(encode_completion(json.loads(line),tokenizer)['length'])
            stats={**length_summary(lengths),'over_4096':sum(n>4096 for n in lengths)}
            report['splits'][task+'/'+split]=stats
            print(task,split,json.dumps(stats),flush=True)
    out=ROOT/'reports/token_length_audit.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if any(item['over_4096'] for item in report['splits'].values()):
        raise SystemExit('Some examples exceed the notebook length budget; resolve before training.')

if __name__=='__main__':main()
