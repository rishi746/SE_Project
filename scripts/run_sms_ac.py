"""CPU demo/document extraction; actual model inference restricted to Kaggle."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sms_ac.documents import read_design_docx
from sms_ac.pipeline import SMSACPipeline, DemoBackend, KaggleAdapterBackend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['demo','clarify','design','sdlc','all','extract'])
    p.add_argument('--conversation', type=Path, help='JSON list of alternating user/assistant turns')
    p.add_argument('--provisional', action='store_true', help='Explicitly draft with unresolved requirements')
    p.add_argument('--request')
    p.add_argument('--docx',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--max-context-tokens',type=int,default=8192)
    p.add_argument('--max-new-tokens',type=int,default=4096)
    a=p.parse_args()
    if a.command=='extract':
        if not a.docx:p.error('--docx is required')
        extracted=read_design_docx(a.docx)
        print(json.dumps({'text':extracted.text,'warnings':extracted.warnings},ensure_ascii=True,indent=2))
        return
    if a.command in ('design','all') and not a.request:p.error('--request is required')
    if a.command=='sdlc' and not a.docx:p.error('--docx is required')
    backend=DemoBackend() if a.command=='demo' else KaggleAdapterBackend(a.models,max_context_tokens=a.max_context_tokens,max_new_tokens=a.max_new_tokens)
    pipeline=SMSACPipeline(backend,output_dir=a.output_dir)
    conversation=json.loads(a.conversation.read_text(encoding='utf-8')) if a.conversation else []
    if a.command=='clarify':
        if a.request: conversation.append({'role':'user','content':a.request})
        print(pipeline.clarify(conversation))
        return
    if a.command=='demo':results=pipeline.run_demo()
    elif a.command=='design':results=[pipeline.design(a.request,conversation=conversation,provisional=a.provisional)]
    elif a.command=='sdlc':results=[pipeline.recommend(a.docx)]
    else:
        first=pipeline.design(a.request,conversation=conversation,provisional=a.provisional)
        results=[first,pipeline.recommend(first.docx_path)]
    print(json.dumps([{'task':r.task,'docx':str(r.docx_path),'markdown':str(r.markdown_path),'warnings':r.warnings} for r in results],indent=2))


if __name__=='__main__':
    main()
