"""Check actual GPU adapter switching without downloading or training anything."""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
logging.basicConfig(level=logging.INFO)
from sms_ac.local_inference import get_engine

if __name__ == '__main__':
    engine = get_engine()
    answer = engine.generate([
        {'role': 'system', 'content': 'Select one SDLC model from Waterfall, V-shaped, RAD, Prototype, Spiral or Iterative. Reply in one short sentence with the model and the decisive reason.'},
        {'role': 'user', 'content': 'An existing network access system is being migrated as-is. Requirements are stable, no new functionality is added, and formal verification and corresponding acceptance tests are the central priority. Which lifecycle fits?'}],
        task='llm2', max_new_tokens=256)
    assert 'v-model' in answer.lower() or 'v-shaped' in answer.lower(), answer
    print('LLM2 on GPU:', answer, flush=True)
    reply = engine.chat('I need a VPN access management application for my office.', [])
    assert '?' in reply, reply
    result = {'llm2': answer, 'llm1_after_switching_back': reply,
        'gpu': engine.torch.cuda.get_device_name(0),
        'peak_allocated_gib': round(engine.torch.cuda.max_memory_allocated() / 2**30, 2)}
    output = ROOT / 'outputs/local-runtime-check/adapter-switch.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('LLM1 switch-back passed; results:', output, flush=True)
