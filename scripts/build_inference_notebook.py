"""Build the interactive Kaggle notebook using the same isolated environment as training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def cell(kind, source):
    value={'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code': value.update(execution_count=None,outputs=[])
    return value

training=json.loads((ROOT/'kaggle/train_dual_models.ipynb').read_text(encoding='utf-8'))
cells=[cell('markdown', '''# SMS AC interactive project discussion
Run this notebook interactively on Kaggle with GPU and Internet enabled. Attach the current project dataset and the completed training notebook outputs. This notebook never trains or uses a laptop GPU. Execute the discussion cell repeatedly, then the report cell. Model loading occurs once per command; this simple isolated process interface prioritizes reproducibility over latency. Do not Save and Run All: the discussion needs your input.
''')]
for index in (1,3,5): cells.append(training['cells'][index])
cells.append(cell('code','''import shutil
from pathlib import Path
for task in ('design','sdlc'):
    target=PROJECT/'models'/f'{task}-adapter'
    if not (target/'model_manifest.json').exists():
        candidates=[p.parent for p in Path('/kaggle/input').rglob('model_manifest.json') if p.parent.name==f'{task}-adapter']
        if len(candidates)!=1:
            raise RuntimeError(f'Attach exactly one completed {task} adapter output: {candidates}')
        shutil.copytree(candidates[0],target,dirs_exist_ok=True)
SESSION=PROJECT/'outputs/interactive_conversation.json'
SESSION.parent.mkdir(parents=True,exist_ok=True)
conversation=json.loads(SESSION.read_text()) if SESSION.exists() else []
print('Loaded discussion turns:',len(conversation))
'''))
cells.append(cell('markdown','## Discussion\nRun the following cell once per answer. Conversation text stays in the working output folder.'))
cells.append(cell('code','''answer=input('Describe your networking project or answer the latest question: ').strip()
if not answer: raise ValueError('Enter a project description or answer')
if conversation and conversation[-1]['role']=='user': conversation[-1]['content']+='\\n'+answer
else: conversation.append({'role':'user','content':answer})
SESSION.write_text(json.dumps(conversation,ensure_ascii=False,indent=2))
command=[str(PYTHON),str(PROJECT/'scripts/run_sms_ac.py'),'clarify','--conversation',str(SESSION),'--models',str(PROJECT/'models')]
result=subprocess.run(command,cwd=PROJECT,text=True,capture_output=True)
if result.returncode:
    print(result.stderr)
    raise RuntimeError('Clarification failed; your answer was saved for retry')
reply=result.stdout.strip()
conversation.append({'role':'assistant','content':reply})
SESSION.write_text(json.dumps(conversation,ensure_ascii=False,indent=2))
print(reply)
'''))
cells.append(cell('markdown','## Generate both Word reports\nRun after discussion. Set PROVISIONAL to True only when you want a draft with unresolved requirements.'))
cells.append(cell('code','''PROVISIONAL=False
if not conversation: raise ValueError('Discuss your project first')
output=PROJECT/'outputs/my-project'
command=[str(PYTHON),str(PROJECT/'scripts/run_sms_ac.py'),'all','--request',conversation[0]['content'],
         '--conversation',str(SESSION),'--models',str(PROJECT/'models'),'--output-dir',str(output)]
if PROVISIONAL: command.append('--provisional')
subprocess.run(command,cwd=PROJECT,check=True)
from IPython.display import FileLink,display
for path in output.glob('*.docx'): display(FileLink(str(path)))
'''))
cells.append(cell('markdown','## Analyze another Word document\nUpload it as a private notebook input and set DOCUMENT_PATH to its visible Kaggle path.'))
cells.append(cell('code','''DOCUMENT_PATH=''  # Explicit path to your uploaded .docx
if DOCUMENT_PATH:
    subprocess.run([str(PYTHON),str(PROJECT/'scripts/run_sms_ac.py'),'sdlc','--docx',DOCUMENT_PATH,
                    '--models',str(PROJECT/'models'),'--output-dir',str(PROJECT/'outputs/uploaded-document')],cwd=PROJECT,check=True)
    display(FileLink(str(PROJECT/'outputs/uploaded-document/sdlc_recommendation.docx')))
'''))
notebook={'cells':cells,'metadata':training['metadata'],'nbformat':4,'nbformat_minor':5}
for index,item in enumerate(cells):
    item['id']=f'sms-ac-inference-{index}'
    if item['cell_type']=='code': compile(''.join(item['source']),f'cell-{index}','exec')
(ROOT/'kaggle/use_dual_models.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print('Built and syntax-checked interactive Kaggle notebook')
