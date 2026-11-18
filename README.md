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

The laptop must remain on and awake. Optional Windows login startup is available through `Enable-Local-Autostart.ps1`. The launcher exposes loopback addresses only; this is not a public hosting setup.

### Optional Google login

Google login needs a Google OAuth web client. Set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in the server environment before startup. Register the callback URLs for the sites you use:

```text
http://127.0.0.1:8000/accounts/google/login/callback/
http://127.0.0.1:8001/accounts/google/login/callback/
```

Google consent has not been verified in this project. Password-based login works without Google credentials. The application does not automatically load a `.env` file; set environment variables in the launching shell. Never commit the client secret.

## Training history

Training scripts are kept in `scripts/`, with supporting notebooks in `kaggle/`. They use completion-masked QLoRA and refuse to run model training outside Kaggle. The local website performs inference only.

- **LLM1:** the installed report adapter is a completed 250-step pilot. It is not a full-dataset training claim.
- **Current LLM2:** initialized from the original pinned base, independently of all earlier adapters. It completed 32 optimizer steps and one epoch on 252 examples from the approved 315-example snapshot. The other 63 examples were held out. No held-out evaluation campaign was performed.
- The approved LLM2 examples are synthetic teacher examples with shared blueprints, not independent expert labels or measured project outcomes. Finishing training does not establish recommendation accuracy.

`scripts/train_llm2_approved.py` and `scripts/prepare_llm2_approved_job.py` describe the current LLM2 run. Earlier notebooks and dataset builders remain as research history; their dataset counts and adapters should not be confused with the active website model.

Private datasets, downloaded corpora, training packages and generated reports are excluded from Git. Reproducing the original run requires obtaining the approved dataset separately. Running a builder does not automatically recreate an approved training snapshot.

## Checks

These checks do not load the full local model:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_llm2_chat.py tests/test_local_inference.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe website/manage.py check
.\.venv\Scripts\python.exe website/manage.py test workspace --verbosity 0
```

`scripts/check_local_llm2_approved_website.py` is a separate live GPU check. It needs the running sites, creates a disposable account and generates a real report. It also checks a reference model choice, so it can fail even when upload and Word download work. Avoid running extra model processes on the same GPU while the websites are active.

## Current limitations

The latest local integration check verified upload, GPU inference, Word export, private downloads and switching back to LLM1. It also found substantive problems: LLM2 chose Waterfall instead of the expected V-model for a verification-focused migration, invented review deadlines and did not compare both requested alternatives. The model should be treated as a draft assistant, not a reliable decision engine.

Long-document performance has not been fully revalidated after switching to section-by-section generation. Responses can take several minutes. Input plus output must fit the configured context budget: 8,192 tokens for LLM1 and 16,384 for LLM2 by default. Generated Word documents need review for both content and layout.

The website uses Django's development server, local SQLite sessions and synchronous model calls. Public production hosting would need a separate deployment and security review, HTTPS, production configuration, email verification/recovery and a file-retention policy. Publishing this source repository does not deploy the website.

## Repository layout

```text
src/sms_ac/        Local inference, prompts, SRS generation, document handling and research pipeline
website/          Django authentication, templates and model-backed endpoints
scripts/          Launchers, data preparation, training and integration checks
kaggle/           Training and earlier inference notebooks
configs/          Source catalog metadata for research tooling
tests/            CPU tests and model-independent checks
docs/             Non-sensitive model provenance
```

Older requirements-agent experiments are retained under `src/network_req_agent`. The Streamlit demo and older Kaggle/Gradio tools are separate from the current Django websites. Start with `Start-Website.ps1` for the current application.
