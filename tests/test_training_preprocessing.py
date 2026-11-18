"""Meaningful CPU-only invariants; no weights, torch, or GPU are required."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from train_dual_qlora import encode_completion, length_summary, require_kaggle, check_split_integrity
from evaluate_dual_models import diagnostic_scores, lifecycle_label, ablation_messages


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}>\n{m['content']}<end>\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>\n"
        return [ord(char) for char in text]


ROW = {"id": "design-1", "task": "design", "group_id": "g1", "messages": [
    {"role": "system", "content": "Write software designs."},
    {"role": "user", "content": "Design a monitoring network."},
    {"role": "assistant", "content": "# Architecture\nUse authenticated agents.\n# Tests\nInject packet loss."},
]}


class ColumnDataset(list):
    def __getitem__(self, key):
        return [row[key] for row in self] if isinstance(key, str) else super().__getitem__(key)


class TrainingPreprocessingTests(unittest.TestCase):
    def test_rag_ablation_changes_only_reference_context(self):
        import json
        row=copy.deepcopy(ROW)
        row['messages'][1]['content']=json.dumps({'request':'DNS service','reference_material':'old','clarification_transcript':[]})
        changed=ablation_messages(row,'design','new evidence')
        self.assertEqual(json.loads(changed[1]['content'])['request'],'DNS service')
        self.assertEqual(json.loads(changed[1]['content'])['reference_material'],'new evidence')
        self.assertEqual(json.loads(row['messages'][1]['content'])['reference_material'],'old')
        self.assertEqual(changed[-1],row['messages'][-1])
    def test_multiturn_masks_previous_assistant_answers(self):
        row = copy.deepcopy(ROW)
        row['messages'] += [{'role':'user','content':'Budget is unknown'},
                            {'role':'assistant','content':'Who can establish a budget range?'}]
        encoded = encode_completion(row, FakeTokenizer())
        supervised = ''.join(chr(t) for t in encoded['labels'] if t != -100)
        self.assertIn('Who can establish', supervised)
        self.assertNotIn('authenticated agents', supervised)

    def test_prompt_masked_and_entire_answer_supervised(self):
        tokenizer = FakeTokenizer()
        encoded = encode_completion(ROW, tokenizer)
        prompt = tokenizer.apply_chat_template(ROW["messages"][:-1], True, True)
        self.assertEqual(encoded["labels"][:len(prompt)], [-100] * len(prompt))
        self.assertEqual(encoded["labels"][len(prompt):], encoded["input_ids"][len(prompt):])
        answer = "".join(chr(token) for token in encoded["labels"] if token != -100)
        self.assertIn("Inject packet loss.", answer)

    def test_oversized_report_raises_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "never truncated"):
            encode_completion(ROW, FakeTokenizer(), 20)

    def test_bad_chat_boundary_fails_closed(self):
        class Broken(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                value = super().apply_chat_template(messages, tokenize, add_generation_prompt)
                return [999] + value if add_generation_prompt else value
        with self.assertRaisesRegex(ValueError, "not a prefix"):
            encode_completion(ROW, Broken())

    def test_empty_response_rejected(self):
        row = copy.deepcopy(ROW)
        row["messages"][-1]["content"] = " "
        with self.assertRaises(ValueError):
            encode_completion(row, FakeTokenizer())

    def test_wrong_roles_rejected(self):
        row = copy.deepcopy(ROW)
        row["messages"][-1]["role"] = "user"
        with self.assertRaises(ValueError):
            encode_completion(row, FakeTokenizer())

    def test_laptop_gpu_guard_does_not_import_torch(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "only inside a Kaggle"):
                require_kaggle()

    def test_group_leakage_rejected_even_when_ids_differ(self):
        train = ColumnDataset([{"id": "a", "task": "design", "group_id": "same"}])
        validation = ColumnDataset([{"id": "b", "task": "design", "group_id": "same"}])
        with self.assertRaisesRegex(ValueError, "Data leakage"):
            check_split_integrity({"train": train, "validation": validation}, "design")

    def test_length_stats(self):
        self.assertEqual(length_summary([40, 10, 30, 20]), {"count": 4, "min": 10, "p50": 20, "p95": 40, "max": 40, "total_tokens": 100})

    def test_sdlc_parser_does_not_select_alternative(self):
        self.assertEqual(lifecycle_label("# Selection\nRecommended model: Spiral\n\nWaterfall is less suitable."), "spiral")
        self.assertIsNone(lifecycle_label("# Alternatives\nAgile and waterfall each have strengths."))

    def test_hybrid_label(self):
        self.assertEqual(lifecycle_label("Hybrid Agile + V-model"), "hybrid")

    def test_coverage_is_explicitly_a_heuristic(self):
        score = diagnostic_scores("design", "Architecture security tests", ROW)
        self.assertIn("section_keyword_coverage", score)
        self.assertNotIn("correctness", score)


if __name__ == "__main__":
    unittest.main()
