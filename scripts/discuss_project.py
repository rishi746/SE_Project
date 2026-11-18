"""Interactive Kaggle-only discussion -> planning DOCX -> explained SDLC DOCX."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sms_ac.pipeline import SMSACPipeline, KaggleAdapterBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=ROOT/'outputs/conversation.json')
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/project')
    args = parser.parse_args()
    backend = KaggleAdapterBackend(args.models)
    pipeline = SMSACPipeline(backend, output_dir=args.output_dir)
    conversation = json.loads(args.session.read_text(encoding='utf-8')) if args.session.exists() else []
    pipeline._validate_conversation(conversation)
    print('Describe your networking project. /report creates both documents; /draft explicitly allows unresolved gaps; /quit saves.')
    if conversation:
        print('Resumed conversation:', conversation[-1]['content'])
    while True:
        user = input('\nYou: ').strip()
        if user == '/quit': break
        if user in ('/report', '/draft'):
            if not conversation:
                print('Describe the project first.'); continue
            design = pipeline.design(conversation[0]['content'], conversation=conversation, provisional=user=='/draft')
            lifecycle = pipeline.recommend(design.docx_path)
            print('Saved:', design.docx_path, lifecycle.docx_path)
            continue
        if not user: continue
        if conversation and conversation[-1]['role']=='user':
            conversation[-1]['content'] += '\n'+user
        else:
            conversation.append({'role':'user','content':user})
        args.session.parent.mkdir(parents=True, exist_ok=True)
        args.session.write_text(json.dumps(conversation,ensure_ascii=False,indent=2),encoding='utf-8')
        reply = pipeline.clarify(conversation)
        conversation.append({'role':'assistant','content':reply})
        args.session.write_text(json.dumps(conversation,ensure_ascii=False,indent=2),encoding='utf-8')
        print('\nPlanning specialist:', reply)


if __name__=='__main__': main()
