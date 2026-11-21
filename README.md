# SMS AC

SMS AC is a software engineering project for networking and cybersecurity applications. It has two local assistants: one turns a project discussion into a software requirements document, and the other reads that document and recommends a development approach.

The website runs on the laptop. Kaggle was used for training and downloading the trained adapters; it is not needed to keep the website running.

## What each model does

| Assistant | Input | Output |
| --- | --- | --- |
| LLM1 — SRS studio | A networking project description and the user's answers | A Word document following the requested IEEE 830-1998 SRS outline, with requirements, roles, constraints, assumptions and supporting sections |
| LLM2 — lifecycle review | An uploaded `.docx` SRS/design document or a problem statement | A development approach recommendation with reasons, phases, advantages, disadvantages, alternatives and missing information, exported to Word |

LLM1 asks questions to establish the project requirements. LLM2 recommends directly from the supplied evidence; extra project context is optional. Its current taxonomy covers 21 approaches, including lifecycle models, Agile frameworks, flow approaches and hybrids.

Both assistants use **Qwen/Qwen3-4B-Instruct-2507**, with separate QLoRA adapters. The local runtime loads one quantized base model and switches adapters under a shared lock. Model requests run one at a time, so a long document can delay another request.

The current website supplies the uploaded text and conversation to the model. The repository also contains corpus and retrieval experiments, but retrieval is not connected to the local website's inference path.

## Before running it

The tested machine has Windows, an NVIDIA RTX 5050 Laptop GPU with 8 GB VRAM, and a CUDA-capable NVIDIA driver. The tested Python packages include PyTorch 2.9.1 with CUDA 12.8, Transformers 4.57.3, PEFT 0.17.1, bitsandbytes 0.48.2 and Django 5.2.19. Other machines have not been verified.

Use Python 3.11 or 3.12 for a fresh environment. The local model runtime requires CUDA; it does not silently fall back to CPU inference.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt -r requirements-inference-local.txt
```

Run these commands from the cloned repository directory. Installation downloads packages; inference can run offline once the packages and all model files are present.

## Model files are a separate download

This repository contains code, not trained weights. A fresh clone will not generate model responses until the following folders are supplied:

```text
models/
  qwen3-4b-base/                 # Full pinned base snapshot, including weight shards
  llm1-report-adapter/           # Trained LLM1 adapter, tokenizer and model_manifest.json
  llm2-approved315-v1-adapter/   # Current trained LLM2 adapter, tokenizer and model_manifest.json
```

Download the public base snapshot with the installed Hugging Face client:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-4B-Instruct-2507', revision='cdbee75f17c01a7cc42f958dc650907174af0554', local_dir='models/qwen3-4b-base')"
```

The base download is about 8 GB. Check the base model's own license and model card before redistributing or using it.

Obtain the trained adapters from the project owner or the original training outputs. The current LLM2 archive is named `sms-ac-llm2-approved315-v1.zip` from the owner's `rishinandandr/sms-ac-llm2-approved315-v1` Kaggle run. That run and its dataset may require owner access. There is no public adapter download bundled with this repository.

Keep the adapter tokenizers and completed `model_manifest.json` files alongside the weights. The runtime checks that both adapters use the same base model and revision. The expected identities and checksums are in [docs/model-provenance.json](docs/model-provenance.json). An adapter alone does not contain the base model.

## Start the websites

```powershell
.\Start-Website.ps1
```

The launcher applies Django migrations, loads the models and starts both sites:

- LLM1: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- LLM2: [http://127.0.0.1:8001](http://127.0.0.1:8001)

Create an account, then use the same account on either site. Username/email and password login are available locally. The sites share authentication but keep their project histories and generated downloads separate.

For LLM1, describe the project, answer the requirements questions and generate the SRS. For LLM2, upload the Word document and select **Generate detailed SDLC report**. Additional context is optional. The report is generated in bounded sections and checked before it is offered for download.

Uploads must be `.docx` files up to 8 MiB. Images and diagrams are not interpreted; put important design details in document paragraphs or tables. Unsupported embedded content and oversized inputs produce an error rather than silently removing evidence.

Generated files and logs stay under `outputs/`. Runtime logs are in `outputs/local-runtime/runtime.log`. Downloads require the authenticated session that generated them.

The header includes a light/dark toggle. It follows the system preference initially and remembers a manual choice in the browser.

After 19 minutes without human activity, the site shows a warning with **Stay signed in**. At 20 minutes it ends the session and returns to the public home page. Keyboard, pointer, touch and input activity renew the server clock; background polling does not. The latest saved project, uploaded document, draft and report links are stored privately by account and restored after login. A report already running can finish and be recovered after signing back in, provided the local server process remains running. Refresh the workspace after it finishes. Only the latest workspace is restored; this is not a versioned project archive.

The laptop must remain on and awake. Optional Windows login startup is available through `Enable-Local-Autostart.ps1`. The launcher exposes loopback addresses only; this is not a public hosting setup.
