"""One quantized GPU base model with the two existing trained adapters.

No network access is needed after setup. All inference is serialized because
the active adapter and CUDA memory are shared within the website process.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from uuid import uuid4

from .documents import markdown_to_docx
from .prompts import CLARIFICATION_SYSTEM_PROMPT
from .srs import SRS_INSTRUCTIONS, SRS_SECTIONS, section_body, section_index

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)
_engine = None
_initialization_lock = threading.Lock()


def prefill_in_chunks(model, encoded, chunk_size=512):
    """Cache the full prompt except its final token with bounded activation memory.

    generate() receives the original full input and this cache; it processes only
    the remaining final prompt token before generating. No source text is removed.
    """
    cache = None
    prefix_length = encoded['input_ids'].shape[-1] - 1
    for start in range(0, prefix_length, chunk_size):
        end = min(start + chunk_size, prefix_length)
        result = model(input_ids=encoded['input_ids'][:, start:end],
            attention_mask=encoded['attention_mask'][:, :end], past_key_values=cache,
            use_cache=True, logits_to_keep=1, return_dict=True)
        cache = result.past_key_values
    return cache


class LocalEngine:
    def __init__(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError('The local CUDA GPU is unavailable. Run the local inference setup first.')
        self.lock = threading.RLock()
        self.torch = torch
        self.context_limit = int(os.environ.get('SMS_CONTEXT_TOKENS', '8192'))
        base_path = ROOT / 'models/qwen3-4b-base'
        llm1 = ROOT / 'models/llm1-report-adapter'
        llm2 = ROOT / 'models/llm2-approved315-v1-adapter'
        self.adapter_paths = {'llm1': str(llm1), 'llm2': str(llm2)}
        manifests = [json.loads((p / 'model_manifest.json').read_text(encoding='utf-8')) for p in (llm1, llm2)]
        if not all(m.get('training_completed') for m in manifests):
            raise RuntimeError('A trained adapter is missing its completed training manifest.')
        if len({(m['base_model'], m['base_revision']) for m in manifests}) != 1:
            raise RuntimeError('The adapters require different base models.')
        logger.info('Loading local Qwen3 base model on %s', torch.cuda.get_device_name(0))
        self.tokenizer = AutoTokenizer.from_pretrained(str(llm1), local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            str(base_path), local_files_only=True, trust_remote_code=False,
            device_map={'': 0}, torch_dtype=torch.float16, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
        self.model = PeftModel.from_pretrained(base, str(llm1), adapter_name='llm1', is_trainable=False)
        self.model.load_adapter(str(llm2), adapter_name='llm2', is_trainable=False)
        self.model.eval()
        logger.info('Loaded local LLM2 adapter: %s (sha256 %s)', llm2, manifests[1].get('adapter_sha256', 'unrecorded'))
        logger.info('Both trained adapters ready; GPU allocated %.2f GiB', torch.cuda.memory_allocated() / 2**30)

    def generate(self, messages, *, max_new_tokens, task='llm1', context_limit=None):
        with self.lock:
            self.model.set_adapter(task)
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_tensors='pt', return_dict=True)
            count = encoded['input_ids'].shape[-1]
            limit = context_limit or self.context_limit
            if count + max_new_tokens > limit:
                raise ValueError(f'The complete document and discussion exceed the {limit:,}-token local context. Shorten the source or discussion; no evidence was silently dropped.')
            encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
            started = time.monotonic()
            logger.info('Local %s generation started: %s prompt tokens, budget %s', task, count, max_new_tokens)
            with self.torch.inference_mode():
                cache = prefill_in_chunks(self.model, encoded) if count > 1024 else None
                output = self.model.generate(**encoded, do_sample=False,
                    max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.03, use_cache=True, logits_to_keep=1,
                    past_key_values=cache)
            completion = output[0, count:]
            logger.info('Local %s generation completed: %s tokens in %.1fs', task, len(completion), time.monotonic() - started)
            text = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
            if len(completion) >= max_new_tokens and int(completion[-1]) != self.tokenizer.eos_token_id:
                raise ValueError('The model reached the response length limit. Shorten the request or retry; an incomplete response was not published.')
            if not text:
                raise ValueError('The local model returned an empty response.')
            return text

    def chat(self, message, history):
        return self.generate([{'role': 'system', 'content': CLARIFICATION_SYSTEM_PROMPT}]
            + list(history) + [{'role': 'user', 'content': message}], max_new_tokens=900)

    def report(self, history, *, progress=None):
        if not history:
            raise ValueError('Discuss your project before generating the SRS.')
        with self.lock:
            started = time.monotonic()
            parts = ['# Software Requirements Specification']
            # Keep the requirements already drafted available to traceability sections.
            requirements = ''
            for index, section in enumerate(SRS_SECTIONS, 1):
                if progress:
                    progress(index, len(SRS_SECTIONS), section)
                logger.info('SRS section %s/%s: %s', index, len(SRS_SECTIONS), section)
                payload = {'section': section, 'discussion': history,
                    'requirements_already_drafted': requirements}
                text = self.generate([{'role': 'system', 'content': SRS_INSTRUCTIONS},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                    max_new_tokens=1800)
                text = section_body(section, text)
                level = 2 if section.startswith(('Appendix', 'Index')) else min(4, 1 + len(section.split()[0].split('.')))
                parts.append('#' * level + ' ' + section + '\n\n' + text)
                if section.startswith(('3.2 ', '3.3 ', '3.6.')):
                    requirements += '\n' + text
            output = ROOT / 'outputs/local-srs' / uuid4().hex
            output.mkdir(parents=True)
            parts.append('## Index\n\n' + section_index())
            md = '\n\n'.join(parts)
            (output / 'srs.md').write_text(md, encoding='utf-8')
            docx = markdown_to_docx(md, output / 'srs.docx', title='Networking Software Requirements Specification',
                preserve_heading_numbers=True, include_toc=True)
            summary = {'sections': len(SRS_SECTIONS), 'characters': len(md),
                'elapsed_seconds': round(time.monotonic() - started, 1), 'runtime': 'local GPU',
                'adapter': 'llm1-report-adapter', 'visual_review': 'pending'}
            (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            return str(docx), str(output / 'srs.md'), "SRS ready."


def get_engine():
    global _engine
    with _initialization_lock:
        if _engine is None:
            _engine = LocalEngine()
    return _engine


class LocalSDLCGenerator:
    """Bind the existing detailed-report workflow to the trained LLM2 adapter."""
    def generate(self, messages, *, max_new_tokens):
        budget = max_new_tokens
        return get_engine().generate(messages, task='llm2', max_new_tokens=budget,
            context_limit=int(os.environ.get('SMS_SDLC_CONTEXT_TOKENS', '16384')))
