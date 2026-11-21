import ast
import runpy
from pathlib import Path

def test_section_body_keeps_requested_content_and_excludes_sibling_sections():
    policy = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/sms_ac/srs.py'))
    body = policy['section_body']('1.1 Purpose', '## 1.1 Purpose\nDefine the VPN application requirements.\n\n1.2 Scope\nOther section content.')
    assert body == 'Define the VPN application requirements.'
    assert policy['section_body']('3.2 Functional Requirements', '3.2 Functional Requirements\n1. Request access.\n2. Approve access.') == '1. Request access.\n2. Approve access.'

def test_section_body_preserves_numbered_descendants_from_trained_adapter():
    policy = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/sms_ac/srs.py'))
    raw = ('2.1 Purpose\nDescribe VPN access.\n\n'
           '2.2 Service Boundary\nProtect client traffic.\n\n'
           '3 Specific Requirements\nNot part of this section.')
    assert policy['section_body']('2 Overall Description', raw) == (
        'Purpose\nDescribe VPN access.\nService Boundary\nProtect client traffic.')

def test_srs_notebook_uses_policy_and_explicit_website_endpoints(tmp_path):
    root = Path(__file__).resolve().parents[1]
    policy = runpy.run_path(str(root / 'src/sms_ac/srs.py'))
    sections = policy['SRS_SECTIONS']
    assert len(sections) == len(set(sections))
    assert sections.index('1 Introduction') < sections.index('2 Overall Description') < sections.index('3 Specific Requirements')
    tree = ast.parse((root / 'scripts/build_llm1_chat_app_notebook.py').read_text(encoding='utf-8'))
    code = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CHAT_CODE' for t in n.targets))
    compile(code, 'chat_app', 'exec')
    assert "api_name='website_chat'" in code and "api_name='website_report'" in code
    assert 'from sms_ac.srs import SRS_SECTIONS, SRS_INSTRUCTIONS' in code
    assert 'state[-16:]' not in code


def test_srs_export_has_front_contents_explicit_nfr_and_no_format_commentary(tmp_path, monkeypatch):
    import sys, json, threading
    from docx import Document
    from docx.oxml.ns import qn
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import sms_ac.local_inference as runtime
    from sms_ac.srs import SRS_SECTIONS, section_body
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    engine = object.__new__(runtime.LocalEngine)
    engine.lock = threading.RLock()
    def generate(messages, **kwargs):
        section = json.loads(messages[-1]['content'])['section']
        return section + '\nIEEE 830 structured draft based on the project discussion.\nThe application shall enforce the confirmed requirement.'
    engine.generate = generate
    path, markdown, _ = engine.report([{'role': 'user', 'content': 'Office VPN access management.'}])
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    text = '\n'.join(paragraphs)
    assert paragraphs.count('Table of Contents') == 1
    first_body = next(i for i,p in enumerate(doc.paragraphs) if p.style.name == 'Heading 1')
    assert paragraphs.index('Table of Contents') < first_body
    assert doc.paragraphs[first_body].paragraph_format.page_break_before is True
    assert '3.6 Non Functional Requirements and Software System Quality Attributes' in text
    assert not any('References' in title for title in SRS_SECTIONS)
    assert 'IEEE 830' not in text and 'structured draft based on' not in text
    assert 'Feasibility Resources and Proposed Timeline' not in text
    assert doc.element.body.findall('.//' + qn('w:hyperlink'))
    assert doc.element.body.findall('.//' + qn('w:bookmarkStart'))
    assert any('TOC ' in n.text for n in doc.element.body.findall('.//' + qn('w:instrText')))
    assert section_body('3.5 Design Constraints', 'The network uses IEEE 802.1X. [7]') == 'The network uses IEEE 802.1X.'
