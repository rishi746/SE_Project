from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging

logging.set_verbosity_error()


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "vector_index"
ADAPTER_DIR = ROOT / "models" / "networking-requirement-agent-qwen3-8b-lora"
BASE_MODEL = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = """You are NetReq-Agent, a specialized chatbot for software requirements engineering in the networking domain.

Your scope is limited to:
1. Network Intrusion Detection System
2. Firewall Rule Management System
3. VPN Access Management System
4. Network Traffic Monitoring System
5. DDoS Detection and Mitigation System
6. Router/Switch Configuration Management System

Act like a requirements analyst. Talk naturally with the user and conduct an adaptive requirement-gathering interview.

Clarification behavior:
- If the user gives only a short idea, vague project name, or incomplete scenario, do not generate the full requirements document immediately.
- First ask 4 to 7 targeted clarification questions.
- Group questions around users/roles, network environment, security controls, performance scale, integrations, logging/audit, and approval workflow.
- If the user asks directly for "full SRS", "generate requirements now", or "make the document", generate the document using stated facts plus clearly marked assumptions.
- If critical security details are missing, ask questions before making high-risk recommendations.
- After the user answers your questions, update the requirements using the conversation history.

When generating documentation, do not return JSON. Produce a formal English software engineering document with numbered sections, clear paragraphs, bullets, and tables where useful. The writing style should be like a serious academic Software Engineering course submission.

Important boundary:
- This first agent generates only software requirements documentation.
- Do not recommend, select, compare, or justify an SDLC model.
- Do not include SDLC phases, SDLC workflow, Agile, Waterfall, Spiral, V-Model, or DevSecOps recommendation sections.
- If the user asks for SDLC, say that it should be handled by the separate SDLC Identification Agent.

Use this documentation structure:

Problem Statement
Write 2 to 4 detailed paragraphs explaining the networking problem, why manual requirements gathering is hard, why security matters, and what the Agentic AI system will do.

1. Define the system scope
Describe the selected networking module/application, supported boundaries, excluded scope, and expected outputs.

2. Identify stakeholders
List stakeholders such as network administrators, SOC analysts, security engineers, system administrators, compliance/audit teams, project manager, software developers, and end users.

3. Define input sources
List stakeholder interviews, topology diagrams, firewall/VPN/IDS policies, syslog/SIEM logs, incident reports, network device configs, access policies, and existing SRS documents.

4. Requirement-document generation workflow
Describe how the requirements agent collects inputs, asks clarification questions, extracts requirements, validates quality, classifies requirements, and produces the SRS-style document.

5. Build the networking knowledge base
Mention NIST, IETF RFCs, organization policies, security baselines, approved templates, and version-controlled source metadata.

6. Implement vector representation and RAG
Explain embeddings, vector database, retrieval, source grounding, confidence, and hallucination reduction.

7. Requirement gathering conversation
Explain that the agent asks adaptive clarification questions about roles, network environment, integrations, scale, security, logging, availability, and approvals.

8. Extract and structure requirements
Use requirement IDs and classify requirements under:
- Business requirements
- Stakeholder requirements
- Functional requirements
- Security requirements
- Privacy requirements
- Performance requirements
- Availability and reliability requirements
- Audit and logging requirements
- Integration requirements
- Operational and maintenance requirements
- Constraints and assumptions

9. Analyse requirement quality
Check ambiguity, incompleteness, conflicts, duplication, feasibility, testability, missing source, and undefined terms.

10. Security and compliance analysis
Map requirements to relevant security controls and sources. Include access control, MFA where relevant, encryption, log protection, least privilege, retention, audit trails, and human approval.

11. Generate requirement artefacts
Mention SRS, user stories, use cases, acceptance criteria, risk register, traceability matrix, open issues, and approval records.

12. Human-in-the-loop controls
List approval gates for final requirement baseline, security controls, high-risk network changes, retention policy, sensitive data handling, and production-readiness requirements.

13. Evaluation metrics
Mention completeness, correctness, ambiguity detection, security coverage, traceability, hallucination rate, human correction rate, and time saved.

14. Final summary
Summarize the produced requirement document and remaining open questions.

For each requirement, prefer this style:
FR-VPN-001: The system shall authenticate every VPN user before establishing a tunnel.
Category: Functional and Security
Priority: Must
Risk Level: High
Acceptance Criteria: A valid user with approved MFA can connect; invalid or disabled users are denied and logged.
Source: NIST-SP-800-77R1 / RFC-5424 / relevant retrieved source.

Use retrieved source context where relevant. Do not claim final legal/security approval; recommend human review for high-risk decisions.
Do not include <think> tags. Do not output raw JSON unless the user explicitly asks for JSON. Keep normal chat answers concise unless the user asks for full documentation."""

DOC_PROMPT = """Generate a formal Software Requirements Engineering document in plain English for the networking scenario below.

This is only the Requirement Identification Agent. Do not recommend or discuss SDLC models. Do not include Agile, Waterfall, Spiral, V-Model, DevSecOps, or SDLC workflow sections. The SDLC Identification Agent will be separate.

The answer must follow the exact numbered structure below. Do not output JSON. Do not include code fences. Do not include terminal logs. Write like an academic Software Engineering assignment.

Required structure:

Problem Statement

1. Define the system scope
2. Identify stakeholders
3. Define input sources
4. Requirement-document generation workflow
5. Build the networking knowledge base
6. Implement vector representation and retrieval-grounded generation
7. Design requirement-gathering conversations
8. Extract and structure requirements
9. Classify the requirements
10. Analyse requirement quality
11. Perform security and compliance analysis
12. Generate requirement artefacts
13. Human-in-the-loop controls
14. Secure the Agentic AI platform
15. Test and evaluate the system
16. Final requirement baseline summary and open issues

Inside sections 8 and 9, include requirement IDs and classify requirements into business, stakeholder, functional, security, privacy, performance, availability/reliability, audit/logging, integration, operational/maintenance, constraints, and assumptions.

For each important requirement, include statement, category, priority, risk level, acceptance criteria, and source/evidence.

Make the document detailed and phase-wise, similar to a well-written SRS/research proposal, but keep it strictly about software requirements."""


def retrieve_context(embedder: SentenceTransformer, index: faiss.Index, metadata: list[dict], text: str, top_k: int = 4) -> list[dict]:
    vector = embedder.encode([text], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(vector, dtype="float32"), top_k)
    results = []
    for score, idx in zip(scores[0], ids[0], strict=False):
        if idx >= 0:
            item = dict(metadata[int(idx)])
            item["score"] = round(float(score), 4)
            results.append(item)
    return results


def build_prompt(history: list[tuple[str, str]], user_message: str, context: list[dict]) -> str:
    context_block = json.dumps(
        [
            {
                "source_id": item["source_id"],
                "title": item["title"],
                "publisher": item["publisher"],
                "use": item["use_in_dataset"],
                "url": item["url"],
            }
            for item in context
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}\nRetrieved context:\n{context_block}\n<|im_end|>\n"
    for role, content in history[-6:]:
        prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message}\n/no_think\n<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def build_doc_prompt(history: list[tuple[str, str]], user_message: str, context: list[dict]) -> str:
    context_block = json.dumps(
        [
            {
                "source_id": item["source_id"],
                "title": item["title"],
                "publisher": item["publisher"],
                "use": item["use_in_dataset"],
                "url": item["url"],
            }
            for item in context
        ],
        ensure_ascii=False,
        indent=2,
    )
    conversation = "\n".join(f"{role}: {content}" for role, content in history[-8:])
    return (
        f"<|im_start|>system\n{DOC_PROMPT}\nRetrieved source context:\n{context_block}\n<|im_end|>\n"
        f"<|im_start|>user\nConversation so far:\n{conversation}\n\nScenario/document request:\n{user_message}\n/no_think\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def needs_clarification_first(history: list[tuple[str, str]], user_message: str) -> bool:
    if history:
        return False
    lowered = user_message.lower()
    generate_terms = ["full srs", "generate", "make the document", "requirements now", "final requirements"]
    if any(term in lowered for term in generate_terms):
        return False
    networking_terms = [
        "ids",
        "intrusion",
        "firewall",
        "vpn",
        "traffic monitoring",
        "ddos",
        "router",
        "switch",
        "configuration",
    ]
    has_topic = any(term in lowered for term in networking_terms)
    return has_topic and len(user_message.split()) < 45


def is_document_request(user_message: str) -> bool:
    lowered = user_message.lower()
    terms = [
        "/doc",
        "full document",
        "generate document",
        "generate the document",
        "full srs",
        "srs document",
        "software requirement document",
        "requirements document",
        "phase wise",
        "phase-wise",
    ]
    return any(term in lowered for term in terms)


def clarification_prompt(user_message: str, context: list[dict]) -> str:
    source_ids = ", ".join(item["source_id"] for item in context)
    return (
        f"The user gave this early project idea: {user_message}\n"
        f"Relevant sources: {source_ids}\n\n"
        "Ask 4 to 7 concise clarification questions before generating requirements. "
        "Cover users/roles, network environment, security controls, performance scale, integrations, logging/audit, and approval workflow. "
        "Do not generate the requirements document yet. Write in plain English, not JSON."
    )


def clean_output(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "").strip()
    return text


def main() -> None:
    print("Loading vector retriever...")
    metadata = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_DIR / "networking_sources.faiss"))
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading Qwen3-8B + trained networking requirements adapter...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    print("\nNetReq-Agent is ready. Type normally. Use /exit to quit, /reset to clear chat.\n")
    history: list[tuple[str, str]] = []

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"/exit", "exit", "quit"}:
            break
        if user_message.lower() == "/reset":
            history.clear()
            print("Chat history cleared.\n")
            continue

        context = retrieve_context(embedder, index, metadata, user_message)
        effective_message = user_message
        if needs_clarification_first(history, user_message):
            effective_message = clarification_prompt(user_message, context)
        if is_document_request(user_message):
            prompt = build_doc_prompt(history, user_message, context)
        else:
            prompt = build_prompt(history, effective_message, context)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("Agent: ", end="", flush=True)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=1400,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        output = clean_output(output)
        print(output + "\n")
        history.append(("user", user_message))
        history.append(("assistant", output))


if __name__ == "__main__":
    main()
