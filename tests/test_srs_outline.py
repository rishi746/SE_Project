import ast
import runpy
from pathlib import Path

def test_section_body_keeps_requested_content_and_excludes_sibling_sections():
    policy = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/sms_ac/srs.py'))
    body = policy['section_body']('1.1 Purpose', '## 1.1 Purpose\nDefine the VPN application requirements.\n\n1.2 Scope\nOther section content.')
    assert body == 'Define the VPN application requirements.'
    assert policy['section_body']('3.2 Functional Requirements', '3.2 Functional Requirements\n1. Request access.\n2. Approve access.') == '1. Request access.\n2. Approve access.'

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
