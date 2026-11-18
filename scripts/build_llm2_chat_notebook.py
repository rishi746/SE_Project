"""Create a Kaggle Gradio notebook that mounts the fresh LLM2 training output."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, identifier: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "id": identifier,
             "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


def main() -> None:
    module = (ROOT / "src/sms_ac/llm2_chat.py").read_text(encoding="utf-8")
    documents = (ROOT / "src/sms_ac/documents.py").read_text(encoding="utf-8")
    server = (ROOT / "scripts/serve_llm2_chat.py").read_text(encoding="utf-8")
    setup = '''import json, os, sys, zipfile, subprocess
from pathlib import Path

assert Path('/kaggle/working').is_dir(), 'Run on Kaggle with a GPU accelerator.'
PROJECT = Path('/kaggle/working/SEproject')
PROJECT.mkdir(parents=True, exist_ok=True)
exports = list(Path('/kaggle/input').rglob('sms-ac-llm2-detailed-v3.zip'))
assert len(exports) == 1, f'Attach the completed LLM2 training notebook output exactly once: {exports}'
with zipfile.ZipFile(exports[0]) as archive:
    for item in archive.infolist():
        parts = Path(item.filename.replace('\\\\', '/')).parts
        assert parts and '..' not in parts and not item.filename.startswith('/')
    archive.extractall(PROJECT)
ADAPTER = PROJECT / 'models/llm2-detailed-v3-adapter'
manifest = json.loads((ADAPTER / 'model_manifest.json').read_text())
assert manifest['training_completed'] and manifest['full_dataset_epochs_completed']
print('Fresh LLM2 adapter:', ADAPTER, 'data:', manifest['training_data_sha256'])
'''
    install = '''subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                'transformers==4.57.3', 'peft==0.17.1', 'accelerate==1.11.0',
                'bitsandbytes==0.48.2', 'safetensors==0.6.2',
                'python-docx==1.2.0', 'gradio==5.49.1'], check=True)
module_path = PROJECT / 'src/sms_ac/llm2_chat.py'
module_path.parent.mkdir(parents=True, exist_ok=True)
module_path.write_text(MODULE_SOURCE, encoding='utf-8')
server_path = PROJECT / 'scripts/serve_llm2_chat.py'
server_path.parent.mkdir(parents=True, exist_ok=True)
server_path.write_text(SERVER_SOURCE, encoding='utf-8')
print('Chat application ready')
'''
    launch = '''os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
subprocess.run([sys.executable, '-u', str(PROJECT / 'scripts/serve_llm2_chat.py'),
                '--adapter', str(ADAPTER), '--share'], cwd=PROJECT, check=True)
'''
    notebook = {
        "cells": [
            cell("markdown", "llm2-chat-intro", "# SMS AC — LLM 2 chat and Word report\n\nAttach the private detailed LLM2 project dataset and the completed training notebook output. Enable a Kaggle GPU, then Run All. The last cell opens a Gradio chat link. Upload the LLM 1 `.docx`, discuss decision factors, and download LLM 2's detailed `.docx`.\n"),
            cell("code", "llm2-chat-setup", setup),
            cell("code", "llm2-chat-source", "MODULE_SOURCE = " + repr(module) + "\nDOCUMENTS_SOURCE = " + repr(documents) + "\nSERVER_SOURCE = " + repr(server) + "\n" + install.replace("module_path =", "documents_path = PROJECT / 'src/sms_ac/documents.py'\ndocuments_path.parent.mkdir(parents=True, exist_ok=True)\ndocuments_path.write_text(DOCUMENTS_SOURCE, encoding='utf-8')\n(PROJECT / 'src/sms_ac/__init__.py').write_text('', encoding='utf-8')\nmodule_path =")),
            cell("code", "llm2-chat-launch", launch),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    target = ROOT / "kaggle/llm2_chat_app.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
