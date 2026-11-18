"""Package a private, fresh LLM2 Kaggle job without old datasets or weights."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
JOB=ROOT/'build/kaggle-llm2-approved315-v1'
SLUG='sms-ac-llm2-approved315-v1'

def main():
    if JOB.exists():raise RuntimeError('Preserve the existing submission package')
    dataset=JOB/'dataset';kernel=JOB/'kernel'
    dataset.mkdir(parents=True);kernel.mkdir()
    files=[ROOT/'requirements-kaggle.txt',ROOT/'scripts/train_dual_qlora.py',
        ROOT/'scripts/train_llm2_approved.py']
    files.extend(p for p in (ROOT/'data/llm2_approved315_v1').rglob('*') if p.is_file())
    entries=[]
    with zipfile.ZipFile(dataset/'project.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            relative=path.relative_to(ROOT).as_posix()
            archive.write(path,'SEproject/'+relative)
            entries.append({'path':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
    (dataset/'package_manifest.json').write_text(json.dumps(entries,indent=2))
    (dataset/'dataset-metadata.json').write_text(json.dumps({'title':'SMS AC LLM2 approved315 v1 data',
        'id':'rishinandandr/'+SLUG+'-data','licenses':[{'name':'other'}],
        'description':'Private user-approved synthetic teacher dataset, exact 315-row snapshot; 252 training and 63 held-out. Not expert gold. Fresh adapter from pinned Qwen base.'},indent=2))
    original=json.loads((ROOT/'kaggle/train_dual_models.ipynb').read_text(encoding='utf-8'))
    env=''.join(next(c['source'] for c in original['cells'] if c.get('id')=='sms-ac-06'))
    setup='''from pathlib import Path
import json, os, shutil, subprocess, sys, zipfile
assert Path('/kaggle/working').is_dir(), 'Kaggle only'
PROJECT=Path('/kaggle/working/SEproject')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['SMS_AC_GPU']='0'
os.environ['HF_HOME']='/tmp/sms-ac-huggingface'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['PYTHONUNBUFFERED']='1'
roots=[p.parent.parent for p in Path('/kaggle/input').rglob('train_llm2_approved.py')]
if roots:
    assert len(roots)==1, roots
    shutil.copytree(roots[0],PROJECT,dirs_exist_ok=True)
else:
    archives=list(Path('/kaggle/input').rglob('project.zip'))
    assert len(archives)==1, archives
    with zipfile.ZipFile(archives[0]) as z:
        for item in z.infolist():
            parts=Path(item.filename).parts
            assert parts and parts[0]=='SEproject' and '..' not in parts and not item.filename.startswith('/')
            assert not ((item.external_attr >> 16) & 0o170000 == 0o120000)
        z.extractall('/kaggle/working')
assert (PROJECT/'data/llm2_approved315_v1/manifest.json').is_file()
print(json.loads((PROJECT/'data/llm2_approved315_v1/manifest.json').read_text()))
'''
    train='''subprocess.run([str(PYTHON),'-u',str(PROJECT/'scripts/train_llm2_approved.py')],cwd=PROJECT,check=True)
adapter=PROJECT/'models/llm2-approved315-v1-adapter'
manifest=json.loads((adapter/'model_manifest.json').read_text())
assert manifest['training_completed']
destination=Path('/kaggle/working/sms-ac-llm2-approved315-v1.zip')
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in adapter.iterdir():
        if path.is_file():archive.write(path,path.relative_to(PROJECT))
    for name in ['manifest.json']:
        path=PROJECT/'data/llm2_approved315_v1'/name
        archive.write(path,path.relative_to(PROJECT))
print('Training finished; no evaluation campaign performed. Adapter export:',destination)
'''
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Fresh LLM2 approved315 training\nOne epoch on 252 approved training rows; 63 held out. Fresh pinned original base only, no prior adapters. No evaluation campaign.']}]
    for i,source in enumerate((setup,env,train)):
        compile(source,'kaggle-cell','exec')
        cells.append({'id':f'run-{i}','cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'source':source.splitlines(True)})
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells}
    (kernel/'train.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    (kernel/'kernel-metadata.json').write_text(json.dumps({'id':'rishinandandr/'+SLUG,'title':'SMS AC LLM2 approved315 v1',
        'code_file':'train.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,
        'dataset_sources':['rishinandandr/'+SLUG+'-data'],'competition_sources':[],'kernel_sources':[]},indent=2))
    state={'state':'prepared_not_submitted','kernel':'rishinandandr/'+SLUG,'dataset':'rishinandandr/'+SLUG+'-data',
        'package_sha256':hashlib.sha256((dataset/'project.zip').read_bytes()).hexdigest(),'files':len(entries)}
    (JOB/'submission.json').write_text(json.dumps(state,indent=2))
    print(json.dumps({'job':str(JOB),**state},indent=2))

if __name__=='__main__':main()
