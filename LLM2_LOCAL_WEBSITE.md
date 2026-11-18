# LLM2 lifecycle studio

Open http://127.0.0.1:8001. LLM1 remains at http://127.0.0.1:8000. Start both with `Start-Website.ps1`; the existing Windows login shortcut also starts both.

Both sites share the existing sign-up and username/email/password login system. An existing LLM1 account works for LLM2. Their document uploads, conversations and report-download session keys are separate. Google sign-in still requires the Google OAuth credentials described in WEBSITE.md, plus an authorised callback URI for port 8001.

Upload a `.docx` requirements/design document or enter a problem statement. Discuss the evidence or supply additional project facts. The new model recommends directly and states missing information instead of requiring clarification answers. Click **Generate detailed SDLC report** for a recommendation and private Word download. New uploads start a fresh LLM2 discussion so facts from different projects are not mixed.

The active adapter is `models/llm2-approved315-v1-adapter`, freshly trained from the pinned Qwen3-4B-Instruct-2507 base on the user-approved 252-row training split (63 held out). Its approved dataset covers 21 approaches, including lifecycle families, Agile frameworks, flow approaches and hybrids. The report requires at least six numbered reasons and ten numbered explained advantages, plus applied phases, disadvantages, closest alternatives, missing information and a conclusion. The model selects a canonical approach, then generates seven bounded sections; an incomplete section is retried once and export fails explicitly if still incomplete. This avoids a single overlong completion consuming the whole response budget. Outputs are drafts requiring review of reasoning and assumptions.

## Runtime

The two loopback web servers run in one Python process, using one quantized Qwen3 base model and the two separately trained adapters. A shared lock serializes generation and adapter selection. An LLM1 export may therefore delay an LLM2 reply. No Kaggle or Gradio connection is used. The new adapter completed 32 optimizer steps and one epoch on Kaggle; inference is entirely local. The prior `models/llm2-detailed-v3-adapter` remains untouched for rollback. Authentication and LLM1 are unchanged.

The LLM2 document plus conversation and generation budget must fit its 16,384-token local limit. Oversized inputs fail rather than being silently truncated. Files are parsed with the existing bounded DOCX reader; unsupported embedded content is rejected or reported as an extraction limitation.

Uploads and generated reports are stored in `outputs/local-llm2`. They are not exposed through a public media directory. Downloads require the authenticated session that generated them. Unreferenced old files are retained locally; there is no automatic retention cleanup yet.

## Verification

`website/manage.py test workspace` covers authentication, CSRF, invalid uploads, private downloads and separation between users and between LLM1 and LLM2 histories.

`scripts/check_local_llm2_approved_website.py` tests a real LLM1 SRS upload, actual GPU chat, a verification-focused network migration recommendation, complete report generation and private Word download through port 8001. It uses a disposable test account and does not use personal conversations.

## Installed adapter integrity

The downloaded adapter safetensors SHA-256 is `e7b7525a1778a32fe97e3817a364ada4723f1ff1b65ef01dbbb9f96b0adcead2`, matching its completed training manifest. The approved snapshot SHA-256 is `6cf3c6e643314a2fe3d49ac69cc27ebd754baa3e2da10a93edc982d182c8b3be`. Startup logs record the loaded adapter path and digest. Training completion and successful export do not establish recommendation accuracy; review outputs.

## Integration check findings on 15 November 2026

The new adapter is loaded locally; actual GPU chat and section-by-section report generation returned HTTP 200, and the private Word download contained 20 numbered points. LLM1 generation after switching back also passed. The live authenticated UI no longer contains the clarification guidance or Your message label. See `outputs/local-llm2-approved315-check/result.json`.

Recommendation quality is not validated: the verification-focused network migration reference case selected Waterfall instead of the expected V-model, invented 48-hour and 72-hour review constraints, and compared only one close alternative. These findings are retained rather than counted as a quality pass. Full-size SRS performance has not been revalidated after replacing the monolithic completion with bounded sections.
