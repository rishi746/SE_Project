"""Kaggle Gradio chat for the fresh LLM2 adapter and downloadable Word report."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sms_ac.llm2_chat")


class KaggleSDLCGenerator:
    def __init__(self, adapter: Path, max_context_tokens: int = 24576):
        if os.name == "nt" or not Path("/kaggle/working").is_dir():
            raise RuntimeError("Run LLM2 inference on Kaggle GPU")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        manifest = json.loads((adapter / "model_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("task") != "sdlc" or not manifest.get("training_completed"):
            raise RuntimeError("This is not a completed LLM2 adapter")
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Kaggle GPU accelerator")
        model_id = manifest["base_model"]
        revision = manifest["base_revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=False, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                  bnb_4bit_use_double_quant=True,
                                                  bnb_4bit_compute_dtype=torch.float16),
            torch_dtype=torch.float16, attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        self.model.eval()
        model_context = getattr(self.model.config, "max_position_embeddings", max_context_tokens)
        self.max_context_tokens = min(max_context_tokens, int(model_context))

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
        import torch

        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                     add_generation_prompt=True,
                                                     return_tensors="pt", return_dict=True)
        prompt_tokens = encoded["input_ids"].shape[-1]
        available = self.max_context_tokens - prompt_tokens
        if available < 1800:
            raise ValueError(f"The discussion and design use {prompt_tokens:,} tokens, leaving only "
                             f"{available:,} output tokens in the {self.max_context_tokens:,}-token model context. "
                             "Shorten the uploaded document or start a fresh chat; no source content was silently dropped.")
        generation_limit = min(max_new_tokens, available)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(**encoded, do_sample=False,
                                         max_new_tokens=generation_limit,
                                         pad_token_id=self.tokenizer.pad_token_id)
        completion = output[0, prompt_tokens:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


def launch(adapter: Path, share: bool) -> None:
    import gradio as gr
    from sms_ac.llm2_chat import discuss, create_report

    generator = KaggleSDLCGenerator(adapter)
    # Gradio only serves files from the app's working directory, its temp
    # directory, or explicitly allowed paths. Keep exports beneath ROOT so the
    # File component can safely cache and return the generated DOCX.
    output_dir = ROOT / "outputs" / "llm2-chat-reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    def source_bytes(path):
        return Path(path).read_bytes() if path else None

    def on_message(message, chat, history, design_path):
        history = list(history or [])
        chat = list(chat or [])
        try:
            answer = discuss(generator, history, message, design_docx=source_bytes(design_path),
                             filename=Path(design_path).name if design_path else "software_design.docx")
        except Exception as exc:
            answer = "Unable to answer: " + str(exc)
            chat.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
            return "", chat, history
        history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        chat.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        return "", chat, history

    def on_report(history, design_path):
        request_id = uuid4().hex[:10]
        try:
            logger.info("Report request %s started (uploaded_docx=%s, history_turns=%d)",
                        request_id, bool(design_path), len(history or []))
            result = create_report(generator, history=list(history or []),
                                   design_docx=source_bytes(design_path),
                                   filename=Path(design_path).name if design_path else "software_design.docx",
                                   output=output_dir / f"sdlc_recommendation_{request_id}.docx")
            if not result.docx_path.is_file() or result.docx_path.stat().st_size == 0:
                raise RuntimeError("Report generation finished without a readable Word file")
            logger.info("Report request %s completed: %s (%d bytes)",
                        request_id, result.docx_path.name, result.docx_path.stat().st_size)
            return result.markdown, str(result.docx_path)
        except Exception as exc:
            logger.exception("Report request %s failed", request_id)
            return f"### Report could not be generated (reference `{request_id}`)\n\n{exc}", None

    with gr.Blocks(title="SMS AC — SDLC specialist") as demo:
        gr.Markdown("# SMS AC — LLM 2 SDLC specialist\nUpload the software-design Word document, "
                    "discuss missing decision factors, then create a detailed Word recommendation.")
        design = gr.File(label="Software design (.docx)", file_types=[".docx"], type="filepath")
        conversation = gr.Chatbot(label="Project discussion", type="messages", height=430)
        history = gr.State([])
        entry = gr.Textbox(label="Your message", placeholder="Tell me about your project or answer a clarification question")
        send = gr.Button("Send")
        send.click(on_message, [entry, conversation, history, design], [entry, conversation, history])
        entry.submit(on_message, [entry, conversation, history, design], [entry, conversation, history])
        produce = gr.Button("Generate detailed SDLC Word document", variant="primary")
        report = gr.Markdown(label="Recommendation")
        download = gr.File(label="Download Word document")
        produce.click(on_report, [history, design], [report, download])
    demo.queue(default_concurrency_limit=1).launch(share=share, server_name="0.0.0.0", show_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    launch(args.adapter, args.share)
