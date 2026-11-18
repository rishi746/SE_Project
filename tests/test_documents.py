import io
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from docx import Document
from sms_ac.documents import read_design_docx, DocumentError
from sms_ac.pipeline import SMSACPipeline, DemoBackend, KaggleAdapterBackend, InferenceError

class DocumentTests(unittest.TestCase):
    def test_body_table_order(self):
        doc=Document();doc.add_paragraph('Before')
        doc.add_table(rows=1,cols=1).cell(0,0).text='Table evidence'
        doc.add_paragraph('After'); stream=io.BytesIO();doc.save(stream)
        text=read_design_docx(stream.getvalue(),filename='design.docx').text
        self.assertLess(text.index('Before'),text.index('Table evidence'))
        self.assertLess(text.index('Table evidence'),text.index('After'))

    def test_invalid_document_rejected(self):
        with self.assertRaises(DocumentError):read_design_docx(b'not a zip',filename='design.docx')

    def test_demo_round_trip_keeps_title_and_discloses_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            first,second=SMSACPipeline(DemoBackend(),output_dir=folder).run_demo()
            text=read_design_docx(first.docx_path).text
            self.assertIn('Campus Network Observability',text)
            self.assertIn('DEMONSTRATION FIXTURE',text)
            self.assertFalse(second.metadata['trained_model_output'])
            self.assertIn('iterative and incremental',second.markdown)

    def test_local_backend_refuses_before_model_import(self):
        with self.assertRaises(InferenceError):KaggleAdapterBackend()

if __name__=='__main__':unittest.main()
