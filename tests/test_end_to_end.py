import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sms_ac.pipeline import SMSACPipeline, DEMO_DESIGN, DEMO_SDLC
from sms_ac.lifecycle_references import retrieve_lifecycle_context

class CaptureBackend:
    is_demo=False
    def __init__(self): self.calls=[]
    def metadata(self,task): return {'backend':'test_stub','trained_model_output':False}
    def generate(self,task,messages):
        self.calls.append((task,messages))
        return DEMO_DESIGN if task=='design' else DEMO_SDLC

class WorkflowTests(unittest.TestCase):
    def test_discussion_to_word_to_lifecycle_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            backend=CaptureBackend()
            pipeline=SMSACPipeline(backend,output_dir=folder)
            turns=[{'role':'user','content':'Monitor a campus'},
                   {'role':'assistant','content':'What is your budget?'},
                   {'role':'user','content':'Budget unknown'}]
            report=pipeline.design('Monitor a campus',conversation=turns,reference_material='')
            decision=pipeline.recommend(report.docx_path)
            self.assertTrue(decision.docx_path.is_file())
            self.assertEqual(json.loads(backend.calls[0][1][-1]['content'])['clarification_transcript'],turns)
            payload=json.loads(backend.calls[1][1][-1]['content'])
            self.assertIn('Campus Network Observability',payload['software_design'])
            self.assertIn('method-spiral-sei',payload['reference_material'])
            self.assertFalse(decision.metadata['trained_model_output'])

    def test_reference_retrieval_preserves_alternatives_and_attribution(self):
        context=retrieve_lifecycle_context('Uncertain integrations need risk reduction before funding')
        self.assertIn('https://www.sei.cmu.edu/',context)
        self.assertIn('method-agile-principles',context)
        self.assertIn('method-sequential-sebok',context)

    def test_single_note_does_not_count_as_discussion(self):
        with self.assertRaises(ValueError):
            SMSACPipeline(CaptureBackend()).design('Campus',conversation=[{'role':'user','content':'Campus'}],reference_material='')

if __name__=='__main__': unittest.main()
