"""Compare chunked prompt ingestion with ordinary causal-model generation."""
import sys
from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sms_ac.local_inference import prefill_in_chunks

def test_chunked_prefill_preserves_greedy_generation():
    torch.manual_seed(4)
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = transformers.Qwen3ForCausalLM(config).eval()
    inputs = {'input_ids': torch.tensor([[1, 7, 8, 14, 13, 19, 25, 4, 6]]),
        'attention_mask': torch.ones((1, 9), dtype=torch.long)}
    with torch.inference_mode():
        normal = model.generate(**inputs, max_new_tokens=5, do_sample=False, logits_to_keep=1)
        cache = prefill_in_chunks(model, inputs, chunk_size=3)
        chunked = model.generate(**inputs, past_key_values=cache,
            max_new_tokens=5, do_sample=False, logits_to_keep=1)
    assert torch.equal(normal, chunked)
