"""CPU checks for the LLM2 chat-to-Word workflow."""
from pathlib import Path
import sys

import pytest
from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sms_ac.llm2_chat import TRAINED_SYSTEM, REQUIRED_SECTIONS, create_report, discuss


COMPLETE = ("# Most Appropriate Development Approach: V-shaped\n\n"
            "## Why this approach is selected\n\n" +
            "\n".join(f"{i}. Requirement {i} needs verification." for i in range(1, 7)) +
            "\n\n## Applied phases and deliverables\n1. Requirements and acceptance tests.\n2. Architecture and system tests.\n3. Detailed design and component tests.\n4. Implementation and verification.\n\n"
            "## Advantages and their relevance to this project\n\n" +
            "\n".join(f"{i}. Advantage {i} supports acceptance." for i in range(1, 11)) +
            "\n\n## Disadvantages, safeguards and limits\nReview verification cost.\n\n## Closest alternatives and tradeoffs\nWaterfall foregrounds testing less.\n\n"
            "## Missing information and decision sensitivity\nReview unknowns.\n\n## Conclusion\nVerify every requirement.\n")


class FakeGenerator:
    def __init__(self, answer=COMPLETE):
        self.answer = answer
        self.messages = None

    def generate(self, messages, *, max_new_tokens):
        self.messages = messages
        if "Return exactly one approach name" in messages[0]["content"]:
            return "V-model"
        if self.answer == COMPLETE:
            for heading in REQUIRED_SECTIONS:
                if "Requested section: " + heading in messages[-1]["content"]:
                    return COMPLETE.split(heading, 1)[1].split("\n## ", 1)[0].strip()
        return self.answer


class RepairingGenerator(FakeGenerator):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.all_messages = []

    def generate(self, messages, *, max_new_tokens):
        self.calls += 1
        self.all_messages.append(messages)
        if self.calls == 2:
            return "Incomplete draft"
        return super().generate(messages, max_new_tokens=max_new_tokens)


def test_discussion_asks_without_export():
    fake = FakeGenerator("What verification evidence is required?")
    reply = discuss(fake, [], "I need to migrate a network controller")
    assert "verification" in reply
    assert "Do not ask follow-up questions" in fake.messages[0]["content"]


def test_docx_input_and_full_word_export(tmp_path):
    source = Document()
    source.add_paragraph("The router interface is fixed; every requirement needs a corresponding test.")
    path = tmp_path / "design.docx"
    source.save(path)
    fake = FakeGenerator()
    result = create_report(fake, history=[{"role": "user", "content": "Two-person team."}],
                           design_docx=path.read_bytes(), filename="design.docx",
                           output=tmp_path / "recommendation.docx")
    assert result.docx_path.is_file()
    assert "router interface is fixed" in fake.messages[1]["content"]
    assert "Write ONLY the requested section body" in fake.messages[0]["content"]
    assert "V-model" in "\n".join(p.text for p in Document(result.docx_path).paragraphs)


def test_incomplete_report_not_exported(tmp_path):
    fake = FakeGenerator("V-shaped")
    output = tmp_path / "bad.docx"
    with pytest.raises(ValueError, match="complete"):
        create_report(fake, history=[{"role": "user", "content": "Fixed network migration"}], output=output)
    assert not output.exists()


def test_incomplete_first_draft_is_repaired_once(tmp_path):
    fake = RepairingGenerator()
    output = tmp_path / "repaired.docx"
    result = create_report(fake, history=[{"role": "user", "content": "Fixed network migration"}], output=output)
    assert fake.calls == 9
    assert result.docx_path.is_file()
    retry_prompt = fake.all_messages[2][1]["content"]
    assert "Previous attempt was incomplete" in retry_prompt
    assert "DRAFT TO REPAIR" not in retry_prompt
