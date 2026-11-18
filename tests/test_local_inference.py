import sys
import threading
from pathlib import Path
from unittest.mock import Mock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sms_ac.local_inference import LocalEngine

def test_local_chat_preserves_full_history_and_uses_clarification_role():
    engine = object.__new__(LocalEngine)
    engine.generate = Mock(return_value='Which office roles approve VPN access?')
    history = [{'role': 'user', 'content': 'Office VPN'}] + [
        {'role': 'assistant' if n % 2 == 0 else 'user', 'content': f'project fact {n}'} for n in range(20)]
    assert 'VPN' in engine.chat('What next?', history)
    supplied = engine.generate.call_args.args[0]
    assert supplied[1:-1] == history
    assert supplied[-1]['content'] == 'What next?'
    assert 'requirements discussion' in supplied[0]['content']

def test_local_report_rejects_empty_conversation():
    engine = object.__new__(LocalEngine)
    with pytest.raises(ValueError, match='Discuss your project'):
        engine.report([])
