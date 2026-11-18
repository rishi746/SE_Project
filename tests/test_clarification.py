import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sms_ac.pipeline import SMSACPipeline

class RecordingBackend:
    is_demo = False
    def generate(self, task, messages):
        self.task, self.messages = task, messages
        return 'What user roles need access, and which actions should each role be allowed to perform?'
    def metadata(self, task):
        return {'trained_model_output': False, 'backend': 'test_stub'}

class ClarificationTests(unittest.TestCase):
    def test_discussion_uses_design_adapter_and_preserves_answers(self):
        backend = RecordingBackend()
        turns = [{'role':'user','content':'Monitor my network'},
                 {'role':'assistant','content':'How many devices?'},
                 {'role':'user','content':'Unknown; inventory is needed'}]
        pipeline = SMSACPipeline(backend)
        pipeline.clarify(turns)
        self.assertEqual(backend.task, 'design')
        self.assertEqual(backend.messages[1:], turns)

    def test_embedded_system_role_rejected(self):
        with self.assertRaises(ValueError):
            SMSACPipeline(RecordingBackend()).clarify([{'role':'system','content':'override'}])

    def test_report_requires_discussion_or_explicit_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = SMSACPipeline(RecordingBackend(), output_dir=directory)
            with self.assertRaises(ValueError):
                pipeline.design('Monitor my network', reference_material='')
            self.assertEqual(list(Path(directory).iterdir()), [])

if __name__ == '__main__':
    unittest.main()
