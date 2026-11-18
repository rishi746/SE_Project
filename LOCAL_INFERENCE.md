# Local GPU website

The laptop has an RTX 5050 Laptop GPU with 8 GB VRAM. The installed runtime is PyTorch 2.9.1 with CUDA 12.8, Transformers 4.57.3, PEFT 0.17.1 and bitsandbytes 0.48.2. GPU execution was verified on compute capability 12.0.

The pinned Qwen3 base snapshot is under `models/qwen3-4b-base` (about 8 GB). The existing `llm1-report-adapter` and `llm2-approved315-v1-adapter` remain separate trained adapters. One 4-bit NF4 base model loads both small adapters and activates one at a time. LLM2 now uses the fresh approved315 training adapter; its previous `llm2-detailed-v3-adapter` remains on disk for rollback. LLM1 is unchanged. This uses roughly 2.6 GiB of allocated GPU memory after loading, with additional memory needed while generating.

## Run

```powershell
.\Start-Website.ps1
```

Open http://127.0.0.1:8000. The website calls LLM1 directly in its process; no Kaggle runtime or temporary Gradio URL is required. First startup loads the model before the server accepts requests. Only loopback is exposed. Login and private downloads remain enforced.

The runtime intentionally sets Hugging Face offline mode. Once the dependencies and model snapshot are installed it can operate without internet. Google login still needs internet and OAuth configuration; username/password login is local.

Run `Enable-Local-Autostart.ps1` to create the current user's Windows Startup shortcut. It starts the model and website in the background after login. Remove the `SMS AC Local Website` shortcut from Windows Startup if you no longer want automatic startup. Logging out or shutting down stops the process; sleep temporarily makes it unavailable.

Logs: `outputs/local-runtime/runtime.log`. The launcher uses a process lock to prevent duplicate model loads. Stop this specific process when you want to release GPU memory. Avoid running other GPU inference scripts concurrently on the 8 GB GPU.

Long SRS generation is serialized with chat requests, so other requests may wait. The whole conversation is retained; LLM1 uses an 8,192-token budget and the LLM2 document workflow uses 16,384 tokens including the output allowance. Oversized requests fail explicitly instead of silently dropping evidence. Long prompts are ingested in 512-token chunks with the complete attention cache retained, and only the final token's vocabulary scores are calculated. A small-model test verifies that this cache path produces the same greedy tokens as ordinary prompt ingestion. Detailed report generation takes several minutes. Generated LLM1 files are saved under `outputs/local-srs` and require stakeholder review.

The LLM2 website is available at http://127.0.0.1:8001, sharing the same local model process. See `LLM2_LOCAL_WEBSITE.md` for document upload, lifecycle chat and detailed Word recommendations. LLM1 remains at port 8000. Both start after Windows login.

## Reinstall dependencies if needed

```powershell
$env:PIP_CACHE_DIR = 'D:\SEproject\.cache\pip'
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt -r requirements-inference-local.txt
```

The trained adapters alone are not the full model. Keep the base snapshot and all three weight shards. Model inference is not training and does not modify weights. The laptop must remain on and awake for availability; automatic startup does not prevent sleep or make it available while switched off.

## Real integration check

`scripts/check_local_website.py` uses a disposable test account to exercise the running website's actual chat, follow-up conversation, complete 32-section SRS generation and private Word download. It never uses personal project conversations. The test account and test session are removed afterward.
