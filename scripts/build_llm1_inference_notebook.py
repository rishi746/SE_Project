"""Build Kaggle inference notebook/package for the completed LLM1 adapter."""
from __future__ import annotations
import json, zipfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build' / 'kaggle-llm1-inference-v1'

def nb_code(src: str) -> dict:
    compile(src, 'cell', 'exec')
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':src.splitlines(True)}

def nb_md(src: str) -> dict:
    return {'cell_type':'markdown','metadata':{},'source':src.splitlines(True)}

def build_assets() -> None:
    for sub in ['dataset','kernel']:
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    project_zip = OUT/'dataset'/'project.zip'
    with zipfile.ZipFile(project_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ['src','scripts']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:
                    z.write(p, p.relative_to(ROOT).as_posix())
        for name in ['requirements-kaggle.txt','LLM1_REPORT_SPECIFICATION.md','MODEL_AND_ARCHITECTURE.md']:
            p=ROOT/name
            if p.exists(): z.write(p, p.relative_to(ROOT).as_posix())
    shutil.copy2(ROOT/'build/kaggle-llm1-candidate-v5/adapter-download/sms-ac-llm1-report-adapter.zip', OUT/'dataset'/'sms-ac-llm1-report-adapter.zip')
    (OUT/'dataset'/'dataset-metadata.json').write_text(json.dumps({
        'title':'SMS AC LLM1 inference test assets',
        'id':'rishinandandr/sms-ac-llm1-inference-test-assets',
        'licenses':[{'name':'other'}],
        'description':'Private SMS AC LLM1 inference test assets: minimal project code and completed LLM1 report-specialist adapter zip.'
    }, indent=2), encoding='utf-8')

INFERENCE_SCRIPT = r"""
import json, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path('/kaggle/working/SEproject/src')))
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

PROJECT=Path('/kaggle/working/SEproject')
ADAPTER=PROJECT/'models/llm1-report-adapter'
OUT=PROJECT/'outputs/llm1-inference-test'
OUT.mkdir(parents=True, exist_ok=True)
manifest=json.loads((ADAPTER/'model_manifest.json').read_text())
request = ('Create a cybersecurity and networking software design report for a Zero Trust Network Access and VPN replacement platform for a mid-size university. '
           'The system must support students, faculty, IT admins and security analysts, integrate with SSO/MFA, discover managed/unmanaged devices, enforce role-based access to internal applications, '
           'log security events, provide incident response workflows, and support phased migration from the old VPN. Budget is limited, exact device count is uncertain, and stakeholders can review a prototype every two weeks.')
SECTION_SYSTEM='Generate one section of a detailed networking/cybersecurity SRS report using supplied facts, IDs, and evidence ledger. Keep assumptions explicit.'
sections=['Document Purpose And Control','Problem Context And Objectives','Scope Boundaries And Exclusions','Stakeholders And Role Model','Clarification Summary And Open Questions','Current Environment And Network Assumptions','Business Requirements And Success Criteria','Functional Requirements','Nonfunctional Requirements','Security Privacy And Compliance Requirements','User Stories','Use Cases And Main Workflows','Architecture Overview','Component Responsibilities','Protocols Interfaces And Integration Contracts','Data Model Logging And Retention','Threat Model And Risk Register','Feasibility Analysis','Priority Classification','Planning Phases Timeline And Milestones','Verification And Acceptance Strategy','Deployment Migration And Rollback Plan','Operations Support And Maintenance','Traceability Matrix','Important And Non Important Factors','Alternatives Tradeoffs And Prototype Needs','Glossary','References And Evidence Ledger']
facts={'project':'Zero Trust Network Access and VPN replacement platform','domain':'cybersecurity/networking','capability':'role-based access to internal applications with SSO/MFA, device posture, logging, and incident workflows','environment':'mid-size university with students, faculty, IT admins, and security analysts','scale':'exact device count uncertain','integration':'SSO/MFA, legacy VPN, internal applications, device discovery, security event logging','main_risk':'migration disruption, unmanaged device visibility gaps, access-policy mistakes, privacy concerns, budget limits','profile':'Stakeholders can review a prototype every two weeks; budget is limited; unresolved scale requires discovery and staged rollout.','variant':1}
ids={'fr':['FR-1101','FR-1102','FR-1103','FR-1104','FR-1105','FR-1106','FR-1107','FR-1108'], 'nfr':['NFR-1101','NFR-1102','NFR-1103','NFR-1104','NFR-1105'], 'sec':['SEC-1101','SEC-1102','SEC-1103','SEC-1104','SEC-1105'], 'uc':['UC-1101','UC-1102','UC-1103','UC-1104','UC-1105'], 'risk':['RISK-1101','RISK-1102','RISK-1103','RISK-1104','RISK-1105'], 'test':['TEST-1101','TEST-1102','TEST-1103','TEST-1104','TEST-1105','TEST-1106'], 'dec':['DEC-1101','DEC-1102','DEC-1103','DEC-1104']}
evidence=[{'id':'EVID-1101','title':'User supplied sample inference request','url':'local://sample-request','license':'project-local','use':'Used as the scenario source for this inference smoke test.'},{'id':'EVID-1102','title':'NIST Secure Software Development Framework 1.1','url':'https://www.nist.gov/publications/secure-software-development-framework-ssdf-version-11-recommendations-mitigating-risk','license':'Public domain unless otherwise noted by NIST','use':'Used for secure planning and verification categories.'}]

tokenizer=AutoTokenizer.from_pretrained(str(ADAPTER), trust_remote_code=False)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
base=AutoModelForCausalLM.from_pretrained(manifest['base_model'], revision=manifest.get('base_revision') or 'main', trust_remote_code=False, device_map={'':0}, torch_dtype=torch.float16, quantization_config=quant)
model=PeftModel.from_pretrained(base, str(ADAPTER))
model.eval()

def generate_section(section):
    payload={'section':section,'project_request':request,'facts':facts,'ids':ids,'evidence':evidence}
    messages=[{'role':'system','content':SECTION_SYSTEM},{'role':'user','content':json.dumps(payload, ensure_ascii=False)}]
    prompt=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded=tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        output=model.generate(**encoded, do_sample=False, max_new_tokens=350, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.03)
    text=tokenizer.decode(output[0][encoded['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    if '</think>' in text: text=text.split('</think>',1)[1].strip()
    if not text.startswith('###'):
        text='### '+section+'\\n'+text
    return text

parts=['# Zero Trust Network Access and VPN Replacement Platform - Software Design Report','Generated by the fine-tuned LLM1 report specialist using section-by-section inference. The result is a smoke-test sample and still needs human review before project submission.']
section_outputs=[]
for section in sections:
    print('GENERATING_SECTION', section, flush=True)
    out=generate_section(section)
    section_outputs.append({'section':section,'chars':len(out),'preview':out[:240]})
    parts.append(out)
text='\n\n'.join(parts).strip()
md_path=OUT/'llm1_sample_design_report.md'
md_path.write_text(text, encoding='utf-8')
DOCX=OUT/'llm1_sample_design_report.docx'
doc=Document()
doc.styles['Normal'].font.name='Aptos'
doc.styles['Normal'].font.size=Pt(10.5)
title=doc.add_heading('LLM1 Sample Networking/Cybersecurity Software Design Report', 0)
title.alignment=WD_ALIGN_PARAGRAPH.CENTER
for line in text.splitlines():
    raw=line.rstrip()
    if not raw: continue
    if raw.startswith('# '): doc.add_heading(raw[2:].strip(), level=1)
    elif raw.startswith('## '): doc.add_heading(raw[3:].strip(), level=2)
    elif raw.startswith('### '): doc.add_heading(raw[4:].strip(), level=3)
    elif raw.startswith('- '): doc.add_paragraph(raw[2:].strip(), style='List Bullet')
    else: doc.add_paragraph(raw)
doc.save(DOCX)
summary={'request': request, 'markdown': str(md_path), 'docx': str(DOCX), 'chars': len(text), 'section_count': len(sections), 'section_outputs': section_outputs, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'manifest_training_completed': manifest.get('training_completed'), 'global_step': manifest.get('global_step'), 'adapter_sha256': manifest.get('adapter_sha256'), 'preview': text[:3000]}
(OUT/'inference_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
"""

def build_notebook() -> None:
    cells=[nb_md('# SMS AC LLM1 trained adapter inference test\n\nLoads the completed LLM1 report-specialist adapter and generates one networking/cybersecurity software design report as Markdown and DOCX.')]
    cells.append(nb_code("import os, platform, shutil, subprocess, sys, zipfile\nfrom pathlib import Path\nassert platform.system() == 'Linux' and Path('/kaggle/working').is_dir(), 'Run on Kaggle.'\nos.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'\nos.environ['TOKENIZERS_PARALLELISM']='false'\nPROJECT=Path('/kaggle/working/SEproject')\n"))
    cells.append(nb_code("project_roots=[p.parents[2] for p in Path('/kaggle/input').rglob('src/sms_ac/prompts.py')]\nif not project_roots:\n    archives=[p for p in Path('/kaggle/input').rglob('project.zip')]\n    assert archives, 'Attach project asset dataset.'\n    shutil.unpack_archive(str(archives[0]), '/kaggle/working/unpacked-project')\n    project_roots=[p.parents[2] for p in Path('/kaggle/working/unpacked-project').rglob('src/sms_ac/prompts.py')]\nassert project_roots, 'project source not found'\nshutil.copytree(project_roots[0], PROJECT, dirs_exist_ok=True)\nadapter_dirs=[p.parent for p in Path('/kaggle/input').rglob('llm1-report-adapter/model_manifest.json')]\nif not adapter_dirs:\n    adapters=[p for p in Path('/kaggle/input').rglob('sms-ac-llm1-report-adapter.zip')]\n    assert adapters, 'Attach trained LLM1 adapter zip or unpacked adapter dataset.'\n    shutil.unpack_archive(str(adapters[0]), PROJECT/'models')\nelse:\n    (PROJECT/'models').mkdir(parents=True, exist_ok=True)\n    shutil.copytree(adapter_dirs[0], PROJECT/'models/llm1-report-adapter', dirs_exist_ok=True)\nassert (PROJECT/'models/llm1-report-adapter/model_manifest.json').exists()\nprint('Project and adapter ready', PROJECT)\n"))
    cells.append(nb_code("import venv\nVENV=Path('/tmp/sms-ac-llm1-infer-venv')\nPYTHON=VENV/'bin/python'\nif not PYTHON.exists():\n    venv.EnvBuilder(with_pip=False).create(VENV)\nsubprocess.run([sys.executable,'-m','pip','--python',str(PYTHON),'install','pip==25.3'],check=True)\nsubprocess.run([str(PYTHON),'-m','pip','install','--disable-pip-version-check','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124'],check=True)\nsubprocess.run([str(PYTHON),'-m','pip','install','--disable-pip-version-check','-r',str(PROJECT/'requirements-kaggle.txt')],check=True)\nprint('Inference environment ready')\n"))
    cells.append(nb_code(f"script = {INFERENCE_SCRIPT!r}\nscript_path=PROJECT/'scripts/run_llm1_inference_test.py'\nscript_path.write_text(script, encoding='utf-8')\nsubprocess.run([str(PYTHON), str(script_path)], cwd=PROJECT, check=True)\n"))
    cells.append(nb_code("import zipfile\nexport_path=Path('/kaggle/working/sms-ac-llm1-inference-output.zip')\nbase=PROJECT/'outputs/llm1-inference-test'\nwith zipfile.ZipFile(export_path,'w',zipfile.ZIP_DEFLATED) as z:\n    for p in base.rglob('*'):\n        if p.is_file(): z.write(p, p.relative_to(PROJECT).as_posix())\n    z.write(PROJECT/'models/llm1-report-adapter/model_manifest.json','models/llm1-report-adapter/model_manifest.json')\nprint(export_path, export_path.stat().st_size)\n"))
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.x'}},'nbformat':4,'nbformat_minor':5}
    nb_path=OUT/'kernel'/'llm1_inference_test.ipynb'
    nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
    (OUT/'kernel'/'kernel-metadata.json').write_text(json.dumps({'id':'rishinandandr/sms-ac-llm1-inference-test','title':'SMS AC LLM1 inference test','code_file':'llm1_inference_test.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':['rishinandandr/sms-ac-llm1-inference-test-assets'],'competition_sources':[],'kernel_sources':[]}, indent=2), encoding='utf-8')
    print(json.dumps({'notebook':str(nb_path),'cells':len(cells),'dataset':str(OUT/'dataset')}, indent=2))

if __name__ == '__main__':
    build_assets(); build_notebook()
