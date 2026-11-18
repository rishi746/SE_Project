"""Build Kaggle Gradio chat notebook for LLM1 clarification-first report generation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "kaggle-llm1-chat-app-v1"
KERNEL = OUT / "kernel"
KERNEL.mkdir(parents=True, exist_ok=True)


def nb_code(src: str) -> dict:
    compile(src, "cell", "exec")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}


def nb_md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}

APP_CODE = r'''
import os, sys, json, shutil, subprocess, zipfile, hashlib
from pathlib import Path

assert Path('/kaggle/working').is_dir(), 'Run this notebook on Kaggle.'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
PROJECT = Path('/kaggle/working/SEproject')
PROJECT.mkdir(parents=True, exist_ok=True)

project_roots=[p.parents[2] for p in Path('/kaggle/input').rglob('src/sms_ac/prompts.py')]
if not project_roots:
    archives=[p for p in Path('/kaggle/input').rglob('project.zip')]
    assert archives, 'Attach dataset rishinandandr/sms-ac-llm1-inference-test-assets.'
    shutil.unpack_archive(str(archives[0]), '/kaggle/working/unpacked-project')
    project_roots=[p.parents[2] for p in Path('/kaggle/working/unpacked-project').rglob('src/sms_ac/prompts.py')]
assert project_roots, 'Project source not found in attached dataset.'
shutil.copytree(project_roots[0], PROJECT, dirs_exist_ok=True)

adapter_dirs=[p.parent for p in Path('/kaggle/input').rglob('llm1-report-adapter/model_manifest.json')]
if not adapter_dirs:
    adapters=[p for p in Path('/kaggle/input').rglob('sms-ac-llm1-report-adapter.zip')]
    assert adapters, 'Attach trained LLM1 adapter zip or unpacked adapter dataset.'
    shutil.unpack_archive(str(adapters[0]), PROJECT/'models')
else:
    (PROJECT/'models').mkdir(parents=True, exist_ok=True)
    shutil.copytree(adapter_dirs[0], PROJECT/'models/llm1-report-adapter', dirs_exist_ok=True)
ADAPTER = PROJECT/'models/llm1-report-adapter'
assert (ADAPTER/'model_manifest.json').exists(), 'LLM1 adapter manifest missing.'
print('Project and LLM1 adapter ready:', PROJECT, ADAPTER)

subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                'transformers==4.57.3', 'peft==0.17.1', 'accelerate==1.11.0', 'bitsandbytes==0.48.2',
                'safetensors==0.6.2', 'python-docx==1.2.0', 'gradio==5.49.1'], check=True)
print('Dependencies ready')
'''

CHAT_CODE = r"""
import json, re, time, hashlib, sys
from uuid import uuid4
from pathlib import Path
import torch
import gradio as gr
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

PROJECT = Path('/kaggle/working/SEproject')
ADAPTER = PROJECT/'models/llm1-report-adapter'
OUT = PROJECT/'outputs/llm1-chat-app'
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((ADAPTER/'model_manifest.json').read_text())

CLARIFY_SYSTEM = '''You are LLM1, a networking and cybersecurity software-design consultant.
Your first job is discussion, not document generation.
Ask adaptive clarification questions before creating the final Word report.
Do not dump a full report in chat unless the user explicitly says to generate the report.
Ask 4 to 8 focused questions when important details are missing.
Cover: project objective, users/roles, network environment/topology, cybersecurity controls, integrations, data/logging, feasibility constraints, timeline, risks, approval workflow, and prototype needs.
Keep each chat reply concise and useful. If the user answers questions, summarize what is known and ask only the next missing critical questions.
Domain scope: networking and cybersecurity applications only.
Never recommend the SDLC model; that belongs to LLM2.
Do not include <think> tags.'''

REPORT_SYSTEM = '''You are LLM1, a fine-tuned report specialist for networking/cybersecurity software design.
Use the entire conversation to create a large Word-report-ready software design document.
The report must be detailed and structured, with role-based details, phases, important and non-important factors, feasibility, timeline, scope, risks, assumptions, traceability, architecture, functional requirements, nonfunctional requirements, security requirements, deployment, operations, acceptance criteria, and open questions.
This output is after planning and defining phase. It is not an SDLC recommendation.
Keep assumptions explicit and do not invent final approvals.'''

sys.path.insert(0, str(PROJECT / 'src'))
from sms_ac.srs import SRS_SECTIONS, SRS_INSTRUCTIONS
SECTION_SYSTEM = SRS_INSTRUCTIONS
SECTIONS = SRS_SECTIONS

print('Loading tokenizer and LLM1 adapter...')
tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER), trust_remote_code=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(manifest['base_model'], revision=manifest.get('base_revision') or 'main', trust_remote_code=False, device_map={'':0}, torch_dtype=torch.float16, quantization_config=quant)
model = PeftModel.from_pretrained(base, str(ADAPTER))
model.eval()
print('LLM1 chat model loaded.')


def clean(text: str) -> str:
    text = (text or '').strip()
    if '</think>' in text:
        text = text.split('</think>', 1)[1].strip()
    text = text.replace('<think>', '').strip()
    return text


def run_messages(messages, max_new_tokens=700):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors='pt').to(model.device)
    context_limit = min(int(getattr(model.config, 'max_position_embeddings', 16384)), 16384)
    if encoded['input_ids'].shape[1] + max_new_tokens > context_limit:
        raise ValueError('The complete discussion exceeds the model context. Start a shorter discussion; no evidence was silently dropped.')
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.03)
    return clean(tokenizer.decode(output[0][encoded['input_ids'].shape[1]:], skip_special_tokens=True))


def chat_reply(message, history, state):
    state = state or []
    if not message or not message.strip():
        return history, state, ''
    state.append({'role':'user','content':message.strip()})
    messages = [{'role':'system','content':CLARIFY_SYSTEM}] + state
    reply = run_messages(messages, max_new_tokens=650)
    if len(reply) < 20:
        reply = 'I need a bit more detail. What is the network/cybersecurity project, who will use it, and what problem should it solve?'
    state.append({'role':'assistant','content':reply})
    history = history + [(message, reply)]
    return history, state, ''


def conversation_text(state):
    if not state:
        return ''
    return '\n\n'.join(f"{turn['role'].upper()}: {turn['content']}" for turn in state)


def generate_section(section, state):
    convo = conversation_text(state)
    payload = {
        'section': section,
        'conversation': convo,
        'domain': 'networking/cybersecurity',
        'ids': {
            'fr':['FR-1001','FR-1002','FR-1003','FR-1004','FR-1005','FR-1006','FR-1007','FR-1008'],
            'nfr':['NFR-1001','NFR-1002','NFR-1003','NFR-1004','NFR-1005'],
            'sec':['SEC-1001','SEC-1002','SEC-1003','SEC-1004','SEC-1005'],
            'risk':['RISK-1001','RISK-1002','RISK-1003','RISK-1004','RISK-1005'],
            'test':['TEST-1001','TEST-1002','TEST-1003','TEST-1004','TEST-1005']
        },
        'evidence': [{'id':'CONV-1001','title':'User clarification conversation','url':'local://conversation','license':'project-local','use':'Primary source for this generated report.'}]
    }
    messages = [{'role':'system','content':SECTION_SYSTEM},{'role':'user','content':json.dumps(payload, ensure_ascii=False)}]
    text = run_messages(messages, max_new_tokens=1000)
    text = re.sub(r'^#{1,6}[^\n]*\n?', '', text).strip()
    level = 1 if section.startswith(('Appendix', 'Index')) else min(3, len(section.split()[0].split('.')))
    text = '#' * level + ' ' + section + '\n' + text
    return text


def markdown_to_docx(md: str, path: Path):
    doc = Document()
    doc.styles['Normal'].font.name = 'Aptos'
    doc.styles['Normal'].font.size = Pt(10.5)
    title = doc.add_heading('Networking Software Requirements Specification', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in md.splitlines():
        raw = line.rstrip()
        if not raw:
            continue
        if raw.startswith('# '):
            doc.add_heading(raw[2:].strip(), level=1)
        elif raw.startswith('## '):
            doc.add_heading(raw[3:].strip(), level=2)
        elif raw.startswith('### '):
            doc.add_heading(raw[4:].strip(), level=3)
        elif raw.startswith('- '):
            doc.add_paragraph(raw[2:].strip(), style='List Bullet')
        else:
            doc.add_paragraph(raw)
    doc.save(path)


def generate_report(state):
    if not state:
        return None, None, 'Start the chat first. Describe your networking/cybersecurity project and answer the clarifying questions.'
    parts = ['IEEE 830 structured SRS draft for the networking or cybersecurity project discussed below. Assumptions and unresolved decisions require stakeholder review.']
    section_meta = []
    for i, section in enumerate(SECTIONS, 1):
        print(f'Generating section {i}/{len(SECTIONS)}: {section}', flush=True)
        s = generate_section(section, state)
        parts.append(s)
        section_meta.append({'section': section, 'chars': len(s), 'preview': s[:180]})
    md = '\n\n'.join(parts).strip()
    stamp = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:10]
    md_path = OUT/f'llm1_chat_design_report_{stamp}.md'
    docx_path = OUT/f'llm1_chat_design_report_{stamp}.docx'
    summary_path = OUT/f'llm1_chat_design_report_{stamp}.json'
    md_path.write_text(md, encoding='utf-8')
    markdown_to_docx(md, docx_path)
    summary = {'chars': len(md), 'section_count': len(SECTIONS), 'section_outputs': section_meta, 'sha256': hashlib.sha256(md.encode()).hexdigest(), 'conversation_turns': len(state), 'adapter_sha256': manifest.get('adapter_sha256')}
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return str(docx_path), str(md_path), f"Generated report. Characters: {len(md):,}. Sections: {len(SECTIONS)}. Summary: {summary_path}"


def reset_chat():
    return [], [], ''

def website_chat(message, conversation):
    _, updated, _ = chat_reply(message, [], list(conversation or []))
    return updated[-1]['content']

with gr.Blocks(title='SMS AC LLM1 Clarification Chat') as demo:
    gr.Markdown('# SMS AC LLM1 · Clarification Chat → Word Design Report')
    gr.Markdown('Chat with LLM1 first. It should ask clarification questions. When enough details are collected, click **Generate Word report**.')
    with gr.Row(visible=False):
        web_message = gr.Textbox()
        web_conversation = gr.JSON()
        web_reply = gr.Textbox()
        web_chat = gr.Button('Website chat')
        web_report = gr.Button('Website report')
    state = gr.State([])
    chatbot = gr.Chatbot(label='LLM1 conversation', height=440)
    msg = gr.Textbox(label='Your message', placeholder='Example: I want to build a campus VPN replacement / ZTNA platform...', lines=3)
    with gr.Row():
        send = gr.Button('Send')
        clear = gr.Button('Reset chat')
    with gr.Row():
        generate = gr.Button('Generate Word report', variant='primary')
    status = gr.Textbox(label='Status', lines=3)
    docx_file = gr.File(label='Download Word document')
    md_file = gr.File(label='Download Markdown')
    web_chat.click(website_chat, inputs=[web_message, web_conversation], outputs=[web_reply], api_name='website_chat')
    web_report.click(generate_report, inputs=[web_conversation], outputs=[docx_file, md_file, status], api_name='website_report')
    send.click(chat_reply, inputs=[msg, chatbot, state], outputs=[chatbot, state, msg])
    msg.submit(chat_reply, inputs=[msg, chatbot, state], outputs=[chatbot, state, msg])
    clear.click(reset_chat, outputs=[chatbot, state, status])
    generate.click(generate_report, inputs=[state], outputs=[docx_file, md_file, status])

demo.queue(max_size=8).launch(share=True, debug=False, prevent_thread_lock=False)
print('LLM1 Gradio app is live. Keep this Kaggle notebook running while you chat.', flush=True)
while True:
    time.sleep(60)
"""

# Embed the small SRS policy so existing project asset archives remain usable.
APP_CODE += '\n(PROJECT / "src/sms_ac/srs.py").write_text(' + repr((ROOT / 'src/sms_ac/srs.py').read_text(encoding='utf-8')) + ', encoding="utf-8")\n'
cells = [
    nb_md('# SMS AC LLM1 chat app\n\nRun this Kaggle notebook with GPU enabled. It opens a Gradio link where LLM1 asks clarification questions first, then generates the Word report.'),
    nb_code(APP_CODE),
    nb_code(CHAT_CODE),
]
nb = {
    'cells': cells,
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.x'}},
    'nbformat': 4,
    'nbformat_minor': 5,
}
nb_path = KERNEL / 'llm1_chat_app.ipynb'
nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
(KERNEL/'kernel-metadata.json').write_text(json.dumps({
    'id':'rishinandandr/sms-ac-llm1-chat-app',
    'title':'SMS AC LLM1 clarification chat app',
    'code_file':'llm1_chat_app.ipynb',
    'language':'python',
    'kernel_type':'notebook',
    'is_private':True,
    'enable_gpu':True,
    'enable_internet':True,
    'dataset_sources':['rishinandandr/sms-ac-llm1-inference-test-assets'],
    'competition_sources':[],
    'kernel_sources':[],
}, indent=2), encoding='utf-8')
(ROOT/'kaggle'/'llm1_chat_app.ipynb').write_text(json.dumps(nb, indent=2), encoding='utf-8')
print(json.dumps({'notebook': str(nb_path), 'local_copy': str(ROOT/'kaggle'/'llm1_chat_app.ipynb'), 'kernel_dir': str(KERNEL)}, indent=2))
