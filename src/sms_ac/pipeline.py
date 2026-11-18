"""Two separately fine-tuned task adapters with explicit demo and Kaggle execution."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .documents import ExtractedDocument, markdown_to_docx, read_design_docx
from .prompts import TASK_PROMPTS, CLARIFICATION_SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEMO_REQUEST = (
    "Design a campus network observability application for 2,000 managed devices. "
    "A four-person team needs a read-only dashboard and incident alerts, with weekly "
    "feedback from campus network operators. Device integration behavior and useful "
    "alert thresholds are initially uncertain. Deliver a pilot in twelve weeks."
)

DEMO_DESIGN = """# Campus Network Observability Software Design

## Executive summary and scope
This demonstration proposes a read-only observability application for a campus with 2,000 managed devices. Four developers will deliver a twelve-week pilot, with weekly feedback from network operators. The application collects authorized telemetry, normalizes device health, and provides an operator dashboard and incident alerts. It does not change device configuration or automatically block traffic.

## Stakeholders and requirements evidence
The request supplies device count, team size, pilot deadline and weekly operator availability. Device integration behavior and useful alert thresholds are uncertain. Campus network operators are the primary users. The identity administrator and service owner are proposed supporting roles; their participation has not been confirmed.

## Assumptions and open questions
Assume an existing campus identity provider and a Linux container platform are available. Confirm device protocol support, credential ownership, maintenance windows, telemetry sensitivity, deployment capacity and the pilot's representative device sample. No compliance classification, budget, incident escalation roster or current measured traffic rate was supplied. All numerical service targets below are proposals for operator review and load testing.

## Functional requirements
FR1 Inventory: an authorized operator can list device identity, site, capabilities and last observation. Acceptance: a provisioned pilot device appears once with a stable identifier and the correct site; an unauthorized user receives an access error.
FR2 Collection: collectors ingest supported telemetry without issuing configuration changes. Acceptance: replay a fixture from each supported device family, reject invalid fields, and record ingestion errors with no exposed credentials.
FR3 Health and alerts: the rules service correlates telemetry into incidents with severity, evidence, rule version and acknowledgement history. Acceptance: a threshold test opens one incident, repeated samples update it, and recovery closes it according to the configured rule.
FR4 Dashboard: operators filter current status and recent incidents by site and device. Acceptance: a filter cannot reveal devices outside the user's authorized sites.
FR5 Audit: record administrative rule changes and incident acknowledgements. Acceptance: each event identifies actor, timestamp, object and outcome and is searchable by authorized reviewers.

## Nonfunctional requirements and proposed targets
NFR1 Capacity: plan for one normalized sample per device per minute, about 33 samples per second at the nominal 2,000-device population. Test a fivefold burst of about 167 samples per second; device payload sizes and burst duration remain to be measured.
NFR2 Freshness: propose 95 percent of valid observations visible within two minutes of receipt under nominal load. Measure collector, queue and API timestamps end to end.
NFR3 Availability: propose 99.5 percent monthly dashboard availability excluding explicitly agreed maintenance. This is a target, not a measured result.
NFR4 Security: require encrypted collection where supported, scoped service accounts and site-based authorization. Unsupported legacy devices need an approved isolation and credential plan.
NFR5 Recovery: propose a one-hour restoration target and at most fifteen minutes of lost configuration history. Verify backups before agreeing these values.

## Architecture and component responsibilities
Use independently deployable collectors, a durable message queue, a normalization worker, a rules worker, PostgreSQL for inventory and incident state, a time-series store for observations, and a dashboard API. A modular API service is adequate for the small team; workers isolate integration failures and backpressure.

| Component | Responsibility | Failure behavior |
| --- | --- | --- |
| Collector | Read authorized device telemetry | Retry with bounded backoff and mark devices stale |
| Queue and normalizer | Buffer and validate samples | Reject malformed events and expose queue lag |
| Rules worker | Correlate incidents using versioned rules | Resume from acknowledged events without duplicate incident creation |
| API and dashboard | Enforce site access and expose status | Return explicit stale-data and dependency errors |

## Request and data flows
An operator signs in through the identity provider and receives a scoped session. The API authorizes each inventory or incident query. Collectors poll a configured device capability, attach device ID, observation time and a stable event ID, and publish after validation. A normalizer writes observations, then the rules worker updates incident state and emits notifications through an approved channel. Acknowledgement writes commit before notification side effects. A notification failure cannot erase the incident.

## Interfaces and protocol contracts
GET /v1/devices supports site and cursor filters; GET /v1/incidents supports status and device filters. POST /v1/incidents/{id}/acknowledgements requires authorization, an idempotency key and the current incident version. Return 400 for invalid input, 401 for absent authentication, 403 for denied scope, 409 for a stale version and 503 for an unavailable dependency. Retried requests use the same idempotency key.
Collectors initially support one validated protocol adapter per pilot device family. Prefer authenticated encrypted telemetry where the device supports it; SNMPv3 and HTTPS-based vendor APIs are candidates that must be confirmed, not guaranteed inventory facts. Schema v1 requires event_id, device_id, observed_at, received_at, metric_name, value and source_adapter_version. Reject future timestamps beyond an agreed clock-skew tolerance and route unsupported schema versions to a quarantine queue.

## Data design and retention
Device stores stable device ID, site, capability profile and last_seen. Observation stores event ID, device ID, metric, value and timestamps. Incident stores incident ID, rule version, affected device, evidence references, status and lifecycle timestamps. AuditEvent records actor, action and outcome. Unique event and idempotency keys prevent duplicate state writes. Keep raw observations for a proposed seven days and aggregated metrics for ninety days; confirm operational and privacy needs before implementation. Store secret references rather than secret values in configuration records.

## Deployment and network boundaries
Deploy collectors in restricted management-network segments and place the API behind an authenticated ingress. Permit explicit device-to-collector and collector-to-queue paths; prohibit inbound access from user devices to management credentials. Separate pilot and production configurations. Use immutable container versions, encrypted secret injection and a staged configuration rollout. Begin with a representative pilot segment; expand only after accuracy, capacity and recovery checks pass.

## Security and privacy design
Threats include stolen polling credentials, forged telemetry, excessive API access and sensitive identifiers in logs. Mitigate with scoped accounts, short-lived application credentials where supported, schema validation, authenticated transport, per-site authorization and rate limiting. Log event IDs and sanitized error classes rather than raw credentials or payloads. Audit rule changes. Confirm who may view device location and address metadata and how deletion and access requests are handled.

## Performance reliability and recovery
Size the queue after measuring payload distribution and desired outage buffer; do not equate sample count with byte throughput. Use bounded exponential backoff with jitter and a retry budget. Apply at-least-once delivery with idempotent observation writes and incident updates; do not promise exactly-once delivery. Use observation time and watermarks to identify late data. Display stale status during collector outages. Back up inventory, rule versions and incidents; test restore in an isolated environment. Alert on queue lag, rejected events, device coverage and backup failures.

## Alternatives and design tradeoffs
A single synchronous polling-and-dashboard process is easy to deploy but allows a slow device family to delay other work. An extensive microservice platform adds operational overhead for four developers. The proposed modular API and asynchronous workers keep the initial service count modest while isolating collection and processing. Validate the queue and time-series store choice with a small integration prototype before accepting their maintenance cost.

## Test strategy and traceability
Use unit tests for parsers, rule transitions and authorization; contract tests for each device family; integration tests for queue redelivery and state writes; load tests for nominal and burst traffic; and operator acceptance tests for useful alerts. Include threat-focused tests for cross-site access and credential leakage. Inject collector, queue, store and notification failures and rehearse restoration.

| Requirement | Component | Acceptance evidence |
| --- | --- | --- |
| FR1 and FR4 | Inventory and API | Authorized site filter and cross-site denial tests |
| FR2 and NFR1 | Collector and normalizer | Device fixtures and nominal/burst ingestion results |
| FR3 | Rules worker | Duplicate, recovery and stale-event scenarios |
| FR5 and NFR4 | API and audit store | Actor records and redaction/access tests |
| NFR2 and NFR5 | Whole system | Freshness measurements and restore rehearsal |

## Delivery risks and review gates
The highest initial risks are heterogeneous devices and noisy thresholds. Prototype two representative integrations first, then expand in small validated increments. Review each increment with operators weekly. Promote the twelve-week pilot only after agreed alert usefulness, authorization, recovery and load criteria pass. The service owner must still confirm operational responsibility and the support roster.

## References
No external reference material is used in this deterministic demonstration. The figures and design choices above are stated request facts or explicitly proposed assumptions. Production generation can use separately attributed corpus excerpts.
"""

DEMO_SDLC = """# Campus Network Observability Lifecycle Recommendation

## Recommendation
Use an iterative and incremental lifecycle with short, Scrum-style planning and review cycles. Begin with an integration prototype, then deliver tested slices of collection, incident correlation and dashboard capability. Add explicit security, operational readiness and acceptance gates before broad deployment. DevSecOps practices support this lifecycle; they are not a separate lifecycle model.

## Confidence and missing evidence
This is a provisional recommendation with moderate qualitative confidence. The design supplies weekly operator feedback, a four-person team, a twelve-week pilot and uncertain integrations and thresholds. It does not establish a compliance regime, contractual approval process, team experience or available device laboratory. A safety-critical requirement or mandatory independent verification could justify stronger V-model controls.

## Evidence from the design
The Stakeholders and requirements evidence section states weekly feedback and uncertain device behavior. These favor early working increments and revised acceptance examples. FR2 requires tests for each device family; a prototype can retire integration uncertainty before expansion. FR3 depends on useful alert thresholds, so operator review of observed alerts is more informative than freezing every rule at the outset. The Delivery risks and review gates section explicitly calls for a representative pilot and measured promotion criteria. NFR4 and NFR5 justify security and recovery gates in every release.

## Alternatives and tradeoffs
| Candidate | Fit to the evidence | Decision |
| --- | --- | --- |
| Iterative and incremental | Weekly feedback and uncertain alert behavior support repeated validated slices | Primary recommendation |
| Waterfall | Clear stage documentation helps procurement, but freezing unknown integrations early creates rework | Reconsider for stable externally fixed requirements |
| V-model | Strong verification traceability suits assurance needs, but no safety certification obligation is supplied | Borrow traceability and acceptance gates; reconsider if assurance obligations emerge |
| Risk-driven spiral | Explicit experiments suit integration risk, but extensive formal cycles can burden four developers | Use focused risk prototypes; reconsider for major novel or high-impact risks |

## Execution phases and exit criteria
Weeks 1 and 2 are discovery and risk reduction. Confirm owners, device access, pilot scope and proposed targets; prototype representative device families. Exit when adapters produce validated fixtures and operators approve measurable pilot acceptance criteria.
Weeks 3 through 8 contain three proposed two-week increments. Deliver inventory and ingestion first, then incident lifecycle and audit, then dashboard filters and operational visibility. Each increment exits with demonstrated acceptance tests, threat review, feedback and an updated backlog. Adapt the sequence if integration findings change the critical path.
Weeks 9 and 10 validate a representative pilot segment. Test nominal and burst load, queue redelivery, authorization boundaries and backup restoration. Exit only when agreed criteria and unresolved defect thresholds are met; missing targets are decisions to resolve, not automatic passes.
Weeks 11 and 12 stabilize and prepare handover. Rehearse rollback, confirm alert ownership, document runbooks and review pilot evidence. Expansion is conditional on operator acceptance and operational readiness. The schedule is a planning proposal and does not guarantee completion.

## Risk and validation plan
Maintain FR-to-test traceability from the design and attach actual integration, load, security and recovery results to each review. Track invalid event rate, device coverage, alert usefulness and time to restore with definitions agreed with operators. Automate tests and deployment checks, keep secrets out of logs, and require a reviewed promotion decision for each network segment.

## Reassessment triggers
Revisit the lifecycle if weekly feedback becomes unavailable, a fixed certification regime appears, device integrations remain unresolved after the prototype, or a major new safety impact is identified. Choose stronger V-model verification when assurance needs demand it; use more explicit spiral risk cycles if novel integration hazards dominate. Stable scope alone does not remove the need to test and review increments.
"""


class InferenceError(RuntimeError):
    pass


class Backend(Protocol):
    is_demo: bool

    def generate(self, task: str, messages: list[dict[str, str]]) -> str: ...
    def metadata(self, task: str) -> dict: ...


class DemoBackend:
    """An explicit fixed fixture, never a claim of training or arbitrary inference."""
    is_demo = True

    def generate(self, task: str, messages: list[dict[str, str]]) -> str:
        payload = json.loads(messages[-1]["content"])
        if task == "design" and payload.get("request") == DEMO_REQUEST:
            return DEMO_DESIGN
        if task == "sdlc" and "DEMONSTRATION FIXTURE" in payload.get("software_design", "") and \
                "Campus Network Observability" in payload["software_design"]:
            return DEMO_SDLC
        raise InferenceError("Demo mode supports the fixed campus observability fixture only. Use Kaggle adapters for your own inputs.")

    def metadata(self, task: str) -> dict:
        return {"backend": "deterministic_demo", "trained_model_output": False,
                "task": task, "fixture_id": "campus-observability-v1"}


def validate_adapter(path: str | Path, task: str) -> dict:
    directory = Path(path)
    manifest_path = directory / "model_manifest.json"
    if not manifest_path.is_file():
        raise InferenceError(f"Missing trained adapter manifest: {manifest_path}. Complete Kaggle training first.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        adapter_config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceError(f"Invalid adapter metadata at {directory}") from exc
    if manifest.get("task") != task or manifest.get("base_model") != BASE_MODEL:
        raise InferenceError(f"Adapter manifest task/base mismatch at {directory}")
    if manifest.get("training_completed") is not True:
        raise InferenceError(f"Adapter training is not recorded complete at {directory}")
    if adapter_config.get("base_model_name_or_path") != BASE_MODEL:
        raise InferenceError(f"PEFT adapter belongs to a different base model at {directory}")
    weights = directory / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise InferenceError(f"Missing safe adapter weights: {weights}")
    return manifest


class KaggleAdapterBackend:
    """Loads one quantized base and switches the independently trained task adapters.

    Heavy libraries are deliberately imported only after the Kaggle guard, so the local
    Streamlit app and all demo/document processing remain CPU-only without model downloads.
    """
    is_demo = False

    def __init__(self, model_root: str | Path = "models", *, max_new_tokens: int = 4096,
                 max_context_tokens: int = 8192):
        if not Path("/kaggle/working").is_dir() or os.name == "nt":
            raise InferenceError("Model inference is restricted to Kaggle GPU sessions. Local mode provides document tools and the explicit demo only.")
        if not 1 <= max_new_tokens < max_context_tokens:
            raise ValueError("max_new_tokens must be positive and smaller than max_context_tokens")
        self.paths = {task: Path(model_root) / f"{task}-adapter" for task in TASK_PROMPTS}
        self.manifests = {task: validate_adapter(path, task) for task, path in self.paths.items()}
        self.max_new_tokens, self.max_context_tokens = max_new_tokens, max_context_tokens
        self._model = self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise InferenceError("Enable a Kaggle GPU accelerator for adapter inference.")
        revisions = {m.get("base_revision") for m in self.manifests.values() if m.get("base_revision")}
        if len(revisions) > 1:
            raise InferenceError("Task adapters use different base model revisions; retrain or deploy matching artifacts.")
        revision = next(iter(revisions), None)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.paths["design"]), trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=revision, trust_remote_code=False, device_map={"": 0},
            torch_dtype=torch.float16, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16),
        )
        self._model = PeftModel.from_pretrained(base, str(self.paths["design"]), adapter_name="design")
        self._model.load_adapter(str(self.paths["sdlc"]), adapter_name="sdlc")
        self._model.eval()

    def generate(self, task: str, messages: list[dict[str, str]]) -> str:
        if task not in TASK_PROMPTS:
            raise ValueError("task must be design or sdlc")
        self._load()
        import torch

        encoded = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                      return_tensors="pt", return_dict=True)
        input_length = encoded["input_ids"].shape[-1]
        if input_length + self.max_new_tokens > self.max_context_tokens:
            raise InferenceError(f"Input has {input_length:,} tokens; limit is {self.max_context_tokens - self.max_new_tokens:,} with the output budget. Shorten the document explicitly or increase Kaggle context capacity.")
        self._model.set_adapter(task)
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            result = self._model.generate(**encoded, do_sample=False, max_new_tokens=self.max_new_tokens,
                                          pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id)
        output_ids = result[0, input_length:]
        eos = self._model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        if len(output_ids) >= self.max_new_tokens and int(output_ids[-1]) not in eos_ids:
            raise InferenceError("Generation reached its output token limit. No partial report was exported; increase the Kaggle output budget and retry.")
        response = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if not response:
            raise InferenceError("The trained adapter returned no report")
        return response

    def metadata(self, task: str) -> dict:
        return {"backend": "kaggle_qlora_adapter", "trained_model_output": True,
                "adapter_path": str(self.paths[task]), "model_manifest": self.manifests[task]}


@dataclass
class ReportResult:
    task: str
    markdown: str
    metadata: dict
    warnings: list[str] = field(default_factory=list)
    markdown_path: Path | None = None
    docx_path: Path | None = None
    metadata_path: Path | None = None


class SMSACPipeline:
    def __init__(self, backend: Backend, *, output_dir: str | Path = "outputs",
                 template: str | Path | None = None):
        self.backend, self.output_dir, self.template = backend, Path(output_dir), template

    def _run(self, task: str, payload: dict, title: str, stem: str,
             warnings: list[str] | None = None) -> ReportResult:
        messages = [{"role": "system", "content": TASK_PROMPTS[task]},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        markdown = self.backend.generate(task, messages)
        if not isinstance(markdown, str) or len(markdown.strip()) < 100:
            raise InferenceError("Model output is empty or too short to be a usable report")
        report_warnings = list(warnings or [])
        if self.backend.is_demo:
            report_warnings.insert(0, "DEMONSTRATION FIXTURE: fixed content; no model was trained or invoked.")
            markdown = "> DEMONSTRATION FIXTURE: fixed content, not output from a trained model.\n\n" + markdown
        backend_info = self.backend.metadata(task)
        manifest = backend_info.get("model_manifest", {})
        if manifest and (manifest.get("run_tier") != "full" or manifest.get("evaluation_status") != "passed"):
            report_warnings.append("Adapter training completion does not establish report accuracy; inspect its run tier, evaluation report and human review results.")
        metadata = {"task": task, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "input_sha256": hashlib.sha256(messages[1]["content"].encode()).hexdigest(),
                    "visual_review_status": "not_rendered", "warnings": report_warnings, **backend_info}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        md_path = self.output_dir / f"{stem}.md"
        docx_path = self.output_dir / f"{stem}.docx"
        meta_path = self.output_dir / f"{stem}.metadata.json"
        md_path.write_text(markdown, encoding="utf-8")
        markdown_to_docx(markdown, docx_path, title=title, demo=self.backend.is_demo, template=self.template)
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return ReportResult(task, markdown, metadata, report_warnings, md_path, docx_path, meta_path)

    def clarify(self, conversation: list[dict[str, str]]) -> str:
        """Use the design specialist for adaptive discussion; never export a chat as a report."""
        self._validate_conversation(conversation)
        if not conversation or conversation[-1]["role"] != "user":
            raise ValueError("Clarification requires a final user message")
        return self.backend.generate("design", [
            {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT}, *conversation])

    @staticmethod
    def _validate_conversation(conversation: list[dict[str, str]]) -> None:
        if not isinstance(conversation, list) or len(conversation) > 80:
            raise ValueError("Conversation must be a list of at most 80 turns")
        for index, turn in enumerate(conversation):
            expected = "user" if index % 2 == 0 else "assistant"
            if not isinstance(turn, dict) or turn.get("role") != expected:
                raise ValueError("Conversation must alternate user and assistant turns")
            content = turn.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Every conversation turn needs nonempty text")
        if sum(len(t["content"]) for t in conversation) > 60000:
            raise ValueError("Conversation exceeds 60,000 characters; explicitly summarize it first")

    def design(self, request: str, *, reference_material: str | None = None,
               conversation: list[dict[str, str]] | None = None,
               provisional: bool = False) -> ReportResult:
        if not request.strip() or len(request) > 25_000:
            raise ValueError("Request must contain 1 to 25,000 characters")
        conversation = conversation or []
        self._validate_conversation(conversation)
        if not self.backend.is_demo and not any(t['role']=='assistant' for t in conversation) and not provisional:
            raise ValueError("Discuss requirements first, or explicitly request a provisional draft")
        if reference_material is None:
            if self.backend.is_demo:
                reference_material = ""
            else:
                from .retrieval import retrieve_context
                retrieval_query = request + '\n' + '\n'.join(
                    turn['content'] for turn in conversation if turn['role'] == 'user')
                reference_material = retrieve_context(retrieval_query)
        return self._run("design", {"request": request, "reference_material": reference_material,
                         "clarification_transcript": conversation, "provisional": provisional},
                         "Planning, Requirements and Preliminary Design", "software_design",
                         ["Provisional draft: unresolved requirements require review."] if provisional else [])

    def recommend(self, design_docx: str | Path | bytes, *, filename: str | None = None,
                  reference_material: str | None = None) -> ReportResult:
        extracted = read_design_docx(design_docx, filename=filename)
        if reference_material is None:
            from .lifecycle_references import retrieve_lifecycle_context
            reference_material = '' if self.backend.is_demo else retrieve_lifecycle_context(extracted.text)
        return self._run("sdlc", {"software_design": extracted.text,
                                 "reference_material": reference_material,
                                 "extraction_warnings": list(extracted.warnings)},
                         "Software Lifecycle Recommendation", "sdlc_recommendation", list(extracted.warnings))

    def run_demo(self) -> tuple[ReportResult, ReportResult]:
        if not self.backend.is_demo:
            raise InferenceError("run_demo requires DemoBackend explicitly")
        design = self.design(DEMO_REQUEST)
        return design, self.recommend(design.docx_path)
