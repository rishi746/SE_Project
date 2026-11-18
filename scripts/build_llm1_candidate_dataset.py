"""Build LLM1 candidate data for networking/cybersecurity SRS report generation.

The output is intentionally marked as assistant-reviewed candidate data, not
expert-approved gold data. It is designed for LLM1 only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "llm1_candidate_v1"

SECTION_TITLES = [
    "Document Purpose And Control",
    "Problem Statement And Existing Environment",
    "Objectives And Success Criteria",
    "Scope And Boundaries",
    "Stakeholders And Responsibilities",
    "Role And Permission Model",
    "Discovery Record",
    "Network And Trust Context",
    "Business And Stakeholder Requirements",
    "Functional Requirements",
    "Nonfunctional Requirements",
    "Security And Privacy Requirements",
    "User Stories",
    "Use Cases",
    "Workflow And State Descriptions",
    "Preliminary Architecture",
    "Protocols And Interfaces",
    "Data Model And Lifecycle",
    "Threat And Risk Register",
    "Feasibility",
    "Priorities And Release Scope",
    "Proposed Work Phases And Timeline",
    "Verification And Acceptance Plan",
    "Deployment And Operations",
    "Requirements Traceability Matrix",
    "Applicable Standards And Policies",
    "Open Issues And Decision Register",
    "Glossary And References",
]

DOMAINS = [
    ("DNS resolver platform", "networking", "resolve internal and external names with TTL-aware caching", "DNS UDP/TCP, recursive resolver upstreams", "cache poisoning, stale answers, upstream timeout"),
    ("VPN access manager", "cybersecurity", "grant, monitor, and revoke remote access sessions", "identity provider, VPN gateway API", "stale access, stolen credentials, orphaned tunnels"),
    ("Firewall policy change manager", "cybersecurity", "review, simulate, approve, and stage firewall policies", "firewall controller API, policy repository", "lockout, shadowed rules, unauthorized change"),
    ("Network flow observability service", "networking", "collect flow summaries and expose traffic investigations", "NetFlow/IPFIX collectors, analytics store", "collector backlog, clock skew, dropped observations"),
    ("Campus Wi-Fi onboarding portal", "networking", "enroll users and issue scoped wireless sessions", "RADIUS/NAC controller, HTTPS portal", "session abuse, controller outage, inaccessible enrollment"),
    ("DDoS mitigation coordinator", "cybersecurity", "detect traffic anomalies and coordinate bounded mitigation", "monitoring feeds, provider mitigation API", "false positive blocking, delayed mitigation, recovery confusion"),
    ("Router configuration backup service", "networking", "schedule, verify, and restore network device backups", "NETCONF/SSH export, encrypted object storage", "unrestorable backup, leaked secrets, partial snapshot"),
    ("Zero-trust application gateway", "cybersecurity", "authorize application access based on identity, device, and policy", "identity provider, policy decision point, gateway", "policy bypass, token replay, dependency outage"),
    ("IoT telemetry gateway", "networking", "normalize and buffer device telemetry from constrained sites", "MQTT/TLS broker, device registry", "intermittent uplink, spoofed device, buffer overflow"),
    ("Service discovery registry", "networking", "register services and resolve healthy endpoints", "heartbeat API, lookup API, health checks", "stale endpoints, partitioned registry, bad health signal"),
    ("Webhook delivery platform", "networking", "deliver tenant events with retries, signatures, and replay", "HTTPS receivers, event queue", "SSRF, duplicate delivery, receiver outage"),
    ("Intrusion alert triage system", "cybersecurity", "correlate security signals into analyst cases", "SIEM feed, telemetry intake, case API", "alert fatigue, missed evidence, unauthorized case access"),
]

CONTEXTS = [
    ("campus", "a university campus with segmented student, staff, guest, and operations networks", 1200),
    ("branch", "a multi-branch enterprise with a small network operations team", 450),
    ("hybrid-cloud", "a hybrid cloud environment with private endpoints and on-premises dependencies", 2200),
    ("edge", "edge sites with intermittent uplinks and limited local support", 180),
    ("managed-service", "a managed service provider handling several tenant networks", 5000),
]

PROFILES = [
    ("unclear-workflow", "Operator workflow is partly unknown, so early clarification and disposable prototypes are needed before freezing the interface."),
    ("high-assurance", "Sensitive network-changing actions require independent approval and requirement-to-test evidence."),
    ("risk-first", "The main integration path is uncertain and must be proven before the team commits to a full build."),
    ("incremental-value", "Core capabilities can be delivered in independently useful releases after interface boundaries are confirmed."),
    ("stable-contract", "External interfaces are already baselined, while deployment still needs careful verification and rollback planning."),
    ("feedback-heavy", "Operators are available weekly and expect the report to capture changing priorities and unresolved decisions."),
]

SOURCE_EVIDENCE = [
    ("EVID-901", "NetConfEval policy translation/conflict dataset", "https://huggingface.co/datasets/NetConfEval/NetConfEval", "MIT", "Used only for network policy phrasing and conflict examples, not as full SRS gold data."),
    ("EVID-902", "OpenStack Neutron approved specs", "https://github.com/openstack/neutron-specs", "CC-BY-3.0", "Used as networking design reference material and evidence that real specs are not full SRS reports."),
    ("EVID-903", "Cilium design CFPs", "https://github.com/cilium/design-cfps", "Apache-2.0", "Used as networking/security design reference material."),
    ("EVID-904", "Kubernetes enhancement proposals", "https://github.com/kubernetes/enhancements", "Apache-2.0", "Used as an example of structured technical proposal and graduation criteria."),
    ("EVID-905", "NIST Secure Software Development Framework 1.1", "https://www.nist.gov/publications/secure-software-development-framework-ssdf-version-11-recommendations-mitigating-risk", "Public domain unless otherwise noted by NIST", "Used for secure development practice categories and review discipline."),
]

QUESTION_BANK = [
    "Who owns day-to-day operation, approval of sensitive changes, and final acceptance of the report?",
    "Which network segments, sites, devices, tenants, and trust boundaries are in scope?",
    "What action is the system allowed to perform automatically, and which action needs human approval?",
    "What uptime, response-time, retention, recovery, and audit targets are already agreed?",
    "Which integrations are confirmed, simulated, missing, or still blocked by vendor access?",
    "What budget, team capacity, milestone date, and support owner are known?",
    "What evidence would prove the riskiest assumption before implementation begins?",
    "Which roles must be separated to prevent unauthorized network changes?",
    "What data can be retained, redacted, exported, deleted, or restored?",
    "Which tests must pass before a pilot touches production network paths?",
]


def sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def split_for(group_id: str) -> str:
    bucket = int(sha(group_id, 8), 16) % 100
    if bucket < 82:
        return "train"
    if bucket < 91:
        return "validation"
    return "test"


def scenario(domain: tuple[str, str, str, str, str], context: tuple[str, str, int], profile: tuple[str, str], variant: int) -> dict:
    name, area, capability, integration, risk = domain
    ctx_key, environment, scale = context
    profile_key, profile_text = profile
    sid = f"{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}-{ctx_key}-{profile_key}-{variant:02d}"
    request = (
        f"Create a detailed software requirements and preliminary design Word report for a {name}. "
        f"The system must {capability}. Environment: {environment}, approximately {scale + variant * 37} managed endpoints. "
        f"Integrations: {integration}. Planning condition: {profile_text} Main risk: {risk}. "
        "Ask clarification first and keep unresolved items explicit."
    )
    facts = {
        "project": name,
        "domain": area,
        "capability": capability,
        "environment": environment,
        "scale": scale + variant * 37,
        "integration": integration,
        "main_risk": risk,
        "profile": profile_text,
        "variant": variant,
    }
    return {"id": sid, "group_id": sid, "request": request, "facts": facts, "split": split_for(sid)}


def evidence_rows(seed: int) -> list[dict]:
    rows = []
    for index, (eid, title, url, license_name, use) in enumerate(SOURCE_EVIDENCE, 1):
        rows.append({"id": f"EVID-{seed:03d}{index}", "title": title, "url": url, "license": license_name, "use": use})
    return rows


def req_ids(seed: int) -> dict[str, list[str]]:
    return {
        "fr": [f"FR-{seed:03d}{i}" for i in range(1, 7)],
        "nfr": [f"NFR-{seed:03d}{i}" for i in range(1, 5)],
        "sec": [f"SEC-{seed:03d}{i}" for i in range(1, 5)],
        "uc": [f"UC-{seed:03d}{i}" for i in range(1, 5)],
        "risk": [f"RISK-{seed:03d}{i}" for i in range(1, 5)],
        "test": [f"TEST-{seed:03d}{i}" for i in range(1, 7)],
        "dec": [f"DEC-{seed:03d}{i}" for i in range(1, 5)],
    }


def clarification_target(sc: dict) -> str:
    f = sc["facts"]
    picks = random.Random(sc["id"]).sample(QUESTION_BANK, 5)
    return (
        "Confirmed so far: this is a {project} in {environment}, with about {scale} managed endpoints and integration through {integration}. "
        "The primary capability is to {capability}. The main stated risk is {main_risk}.\n\n"
        "Before I draft the Word report, I need these clarifications:\n"
        "1. {q1}\n2. {q2}\n3. {q3}\n\n"
        "Useful follow-up if those are unknown: {q4} Also confirm: {q5}\n\n"
        "I can still prepare a provisional report, but unknown budget, ownership, acceptance authority, workload, retention, and production access must be labelled as open decisions."
    ).format(q1=picks[0], q2=picks[1], q3=picks[2], q4=picks[3], q5=picks[4], **f)


def section_body(title: str, sc: dict, ids: dict[str, list[str]], evid: list[dict]) -> str:
    f = sc["facts"]
    ev = evid[0]["id"]
    if "Document Purpose" in title:
        return "\n".join([
            "### Document Purpose And Control",
            f"This document is a planning, requirements definition, and preliminary design report for the {f['project']}. It is prepared before implementation to clarify scope, roles, feasibility, risk, and acceptance evidence for {f['environment']}.",
            "Draft status: candidate report content. It does not claim stakeholder sign-off, production approval, measured capacity, budget approval, or regulatory certification.",
            "Intended readers are the service owner, network operators, administrators, security reviewers, developers, testers, and support team. Revision control must record changes to requirements, assumptions, risks, and acceptance criteria.",
            f"Primary traceability anchors for this candidate are {ids['fr'][0]}, {ids['nfr'][0]}, {ids['sec'][0]}, {ids['uc'][0]}, {ids['risk'][0]}, and {ids['test'][0]}.",
        ])
    if "Problem Statement" in title:
        return "\n".join([
            "### Problem Statement And Existing Environment",
            f"The organization needs software support to {f['capability']} across {f['environment']}. About {f['scale']} managed endpoints are in the planning scope, but real traffic, user concurrency, payload size, and failure rates remain unmeasured.",
            f"The existing environment depends on {f['integration']}. Current shortcomings are assumed to include manual tracking, inconsistent approval evidence, weak retry visibility, and unclear ownership for failed operations.",
            f"The main operational risk is {f['main_risk']}. This risk can affect availability, correctness, security review, and trust in the produced network state.",
            "The report must separate confirmed facts from proposed targets. Missing facts become open decisions rather than invented requirements.",
        ])
    if "Objectives" in title:
        return "\n".join([
            "### Objectives And Success Criteria",
            f"Objective 1: enable authorized users to {f['capability']} with traceable approvals and observable outcomes.",
            f"Objective 2: reduce operational uncertainty around {f['main_risk']} by requiring explicit tests and review evidence.",
            "Objective 3: define role permissions, interfaces, data lifecycle, deployment boundaries, and acceptance criteria before construction begins.",
            f"Success criteria include passing {ids['test'][0]} authorization tests, {ids['test'][1]} validation tests, {ids['test'][2]} audit checks, and {ids['test'][4]} recovery rehearsal.",
            "Baselines for latency, throughput, recovery, cost, and schedule are unresolved until measured or approved by stakeholders.",
        ])
    if "Scope And Boundaries" in title:
        return "\n".join([
            "### Scope And Boundaries",
            f"In scope: management-plane workflows for the {f['project']}, role-based access, request validation, audit evidence, integration through {f['integration']}, and pilot deployment planning for {f['environment']}.",
            "Out of scope until approved: autonomous production changes outside the named capability, unsupported device families, unrelated user-management systems, cosmetic-only dashboards, and final SDLC selection.",
            "Deployment boundaries include user-facing administrative access, service components, integration credentials, storage, monitoring, and controlled egress to dependencies.",
            "Deferred items must include a reason, dependency, and decision owner. Deferred does not mean unimportant; it means not justified for the current release scope.",
        ])
    if "Stakeholders" in title:
        return "\n".join([
            "### Stakeholders And Responsibilities",
            "Service owner: approves objectives, release scope, residual risks, and operational handover.",
            "Network operator: performs daily scoped actions and reports workflow gaps.",
            "Administrator: proposes configuration, integration, and policy changes.",
            "Independent approver or reviewer: validates sensitive changes and evidence without performing conflicting operational actions.",
            "Development and test team: implement and verify requirement-linked behavior. Support team: owns incident response, restore rehearsal, patching, and user assistance after handover.",
            "Named individuals, budget authority, and acceptance authority remain open decisions unless supplied by the project owner.",
        ])
    if "Role And Permission" in title:
        return "\n".join([
            "### Role And Permission Model",
            "Operator: create and inspect approved operations within assigned site, tenant, or network segment. Denied outside scope.",
            "Administrator: manage templates, integration settings, and staged changes. Sensitive execution requires independent approval.",
            "Approver: approve or reject sensitive changes after reviewing validation and rollback evidence. Approver cannot secretly edit the submitted change.",
            "Reviewer: read audit, test, and risk evidence. Reviewer has no write permission to operational state.",
            "Service owner: accept release scope, residual risk, support model, and handover readiness.",
            f"Permission tests map to {ids['fr'][0]}, {ids['fr'][2]}, {ids['sec'][0]}, {ids['sec'][3]}, and {ids['test'][0]}.",
        ])
    if "Discovery Record" in title:
        return "\n".join([
            "### Discovery Record",
            f"Confirmed project facts: {f['project']}; environment: {f['environment']}; approximate endpoint count: {f['scale']}; integration context: {f['integration']}; primary capability: {f['capability']}.",
            f"Confirmed planning condition: {f['profile']}",
            "Assumptions: identity provider access, a test environment, persistent storage, and a representative integration fixture can be made available.",
            "Open questions: budget, team capacity, support owner, acceptance authority, retention, backup policy, target workload, rollout window, and exact production access path.",
            "Conflict handling: if a role is described as read-only and allowed to change policies, split it into reviewer and administrator responsibilities before drafting final requirements.",
        ])
    if "Network And Trust" in title:
        return "\n".join([
            "### Network And Trust Context",
            f"The candidate system operates in {f['environment']} and integrates with {f['integration']}. Network segments, trust boundaries, and allowed flows must be confirmed before deployment design is approved.",
            "Proposed trust zones: user access zone, application/control service zone, integration worker zone, storage and audit zone, monitoring zone, and external dependency boundary.",
            "Allowed flows should be explicit: user to API, API to identity provider, worker to approved network dependency, service to monitoring, and controlled administrative access for support.",
            "Unknown routes, privileged device access, unmanaged endpoints, and internet egress must be recorded as risks until reviewed.",
        ])
    if "Business And Stakeholder" in title:
        return "\n".join([
            "### Business And Stakeholder Requirements",
            f"BR-1: The system should make {f['capability']} more reliable and reviewable for the stated environment.",
            "BR-2: Stakeholders need evidence for approvals, failures, acceptance tests, and residual risks before pilot deployment.",
            "BR-3: Operators need clear workflows that do not require hidden manual recovery steps.",
            "BR-4: Security reviewers need separation of duties, scoped authorization, credential handling, and audit retention decisions.",
            f"Business requirements are linked to {ids['fr'][0]}, {ids['fr'][2]}, {ids['nfr'][3]}, {ids['sec'][0]}, and {ids['risk'][0]}.",
        ])
    if "Functional Requirements" in title:
        return "\n".join([
            f"### Functional Requirements",
            f"{ids['fr'][0]}: Authorized operators can {f['capability']} only inside assigned scope. Acceptance: {ids['test'][0]} proves allowed and denied operations.",
            f"{ids['fr'][1]}: The system validates requests against versioned schemas and rejects malformed or cross-scope input without durable side effects.",
            f"{ids['fr'][2]}: Administrators can review, approve, stage, and roll back sensitive actions with actor, timestamp, previous version, and outcome.",
            f"{ids['fr'][3]}: Users can inspect operation status, stale dependency state, and retryable failures without receiving secret values.",
            f"{ids['fr'][4]}: The service records source evidence and unresolved decisions so the final report does not present assumptions as approved facts.",
            f"{ids['fr'][5]}: Bulk operations require idempotency keys and bounded execution to prevent duplicate network changes after retries.",
            f"Evidence: {ev}. Review status: assistant candidate; domain expert must confirm device behavior and acceptance fixtures.",
        ])
    if "Nonfunctional" in title:
        return "\n".join([
            "### Nonfunctional Requirements",
            f"{ids['nfr'][0]}: Propose 95 percent of ordinary administrative reads within one second after workload measurement. The target is provisional because real event rate and payload size are unknown.",
            f"{ids['nfr'][1]}: Availability target is proposed at 99.5 percent monthly for the management plane unless stakeholders approve a stricter target.",
            f"{ids['nfr'][2]}: Recovery objectives must be agreed before sign-off; backup and restore rehearsal evidence is required by {ids['test'][4]}.",
            f"{ids['nfr'][3]}: Observability must expose dependency latency, queue depth, rejected requests, authorization denials, and stale data age.",
            f"Capacity note: about {f['scale']} endpoints is a sizing input, not a throughput guarantee. Measure traffic before cost and schedule commitments.",
        ])
    if "Security" in title:
        return "\n".join([
            "### Security And Privacy Requirements",
            f"{ids['sec'][0]}: Authenticate every administrative user through the approved identity provider and enforce scoped authorization per operation.",
            f"{ids['sec'][1]}: Store integration credentials only in a secret manager; business records store references, not raw secrets.",
            f"{ids['sec'][2]}: Encrypt supported control traffic and reject unauthenticated or downgrade-prone channels unless a documented legacy exception is approved.",
            f"{ids['sec'][3]}: Maintain immutable audit events for sensitive actions and read access to security evidence.",
            f"Threat focus: {f['main_risk']}. Relevant secure-development reference: {evid[-1]['id']} {evid[-1]['url']}. Applicability must be reviewed for the final project.",
        ])
    if "Use Cases" in title:
        return "\n".join([
            "### Use Cases",
            f"{ids['uc'][0]} Operator performs an approved scoped action. Trigger: authorized request. Main flow: authenticate, authorize, validate, commit, observe result. Exceptions: denied scope, malformed input, dependency timeout.",
            f"{ids['uc'][1]} Administrator stages a sensitive change. Main flow: create proposal, run validation, request approval, execute during permitted window, record evidence.",
            f"{ids['uc'][2]} Reviewer inspects audit and acceptance evidence. Reviewer is read-only and cannot change operational state.",
            f"{ids['uc'][3]} Service owner reviews pilot readiness. The owner checks unresolved decisions, risks, tests, deployment plan, and support handover.",
            f"Linked tests: {', '.join(ids['test'][:4])}. Linked risks: {', '.join(ids['risk'][:3])}.",
        ])
    if "User Stories" in title:
        return "\n".join([
            "### User Stories",
            f"As a network operator, I want to {f['capability']} inside my assigned scope so that routine work is traceable and recoverable.",
            "Acceptance: allowed scope succeeds, denied scope fails, and the user sees status without secret data.",
            "As an administrator, I want to stage and validate sensitive changes before execution so that review can happen before network impact.",
            "Acceptance: validation results, approval state, rollback plan, and actor history are visible.",
            "As a reviewer, I want read-only access to audit and test evidence so that I can approve readiness without changing operational state.",
            "As a service owner, I want unresolved assumptions and feasibility gaps separated from confirmed requirements so that I can make scope and schedule decisions.",
        ])
    if "Workflow" in title:
        return "\n".join([
            "### Workflow And State Descriptions",
            "Primary operation states: draft, validated, awaiting approval, approved, queued, executing, succeeded, failed-retryable, failed-terminal, rolled-back, and archived.",
            "Main flow: user submits request, service authenticates and authorizes, schema validation runs, approval is requested when needed, operation is durably recorded, integration worker executes, audit evidence is written, and the user receives status.",
            "Exception flow: denied authorization stops before durable side effects. Malformed input returns actionable validation errors. Dependency timeout marks status uncertain and schedules reconciliation.",
            f"Recovery flow for {f['main_risk']}: expose stale or uncertain status, prevent duplicate side effects with idempotency keys, and require operator review before repeated sensitive execution.",
        ])
    if "Architecture" in title:
        return "\n".join([
            "### Preliminary Architecture",
            f"Use a control API, authorization layer, domain adapter for {f['integration']}, durable operation store, worker, audit log, and read model. This decomposition supports safe retry and reviewable failure handling for {f['project']}.",
            "A single deployable service with a separate worker is enough for the first pilot unless measured load or ownership boundaries justify splitting components.",
            "The design separates administrative control-plane behavior from domain protocol behavior. Do not assume the network data plane is HTTP.",
            f"Tradeoff: asynchronous work improves recovery from {f['main_risk']} but adds queue monitoring and delayed feedback. Synchronous work is simpler for low-latency reads.",
        ])
    if "Protocols" in title or "Interfaces" in title:
        return "\n".join([
            "### Protocols And Interfaces",
            "Administrative API: authenticated HTTPS for create, validate, approve, status, audit, and report-export operations.",
            f"Domain integration: {f['integration']}. The adapter must preserve protocol-specific timeout, retry, versioning, and error semantics.",
            "Required API behavior: 400 malformed input, 401 missing authentication, 403 denied scope, 409 stale version or conflicting operation, 429 quota exceeded, 503 dependency unavailable.",
            "Mutable requests require expected version and idempotency key. Audit and status reads must be scoped by role and environment boundary.",
            "Compatibility assumptions, vendor versions, and protocol limitations remain open until verified against a representative fixture.",
        ])
    if "Data Model" in title:
        return "\n".join([
            "### Data Model And Lifecycle",
            "Core entities: Actor, RoleAssignment, Scope, Request, Approval, Operation, IntegrationTarget, AuditEvent, EvidenceRecord, RiskRecord, TestResult, and ReportVersion.",
            f"Domain entity: a {f['project']} record linked to scope, current version, requested change, and observed result.",
            "Lifecycle: draft data becomes validated request, then approved operation, then execution result, then archived evidence. Failed or uncertain operations retain enough context for reconciliation.",
            "Classification: credentials and tokens are secret references only; audit, status, and report content are restricted operational records. Retention, deletion, export, and backup periods require owner approval.",
            f"Traceability connects {ids['fr'][0]} through {ids['test'][0]} and evidence ledger entries.",
        ])
    if "Threat" in title or "Risk Register" in title:
        return "\n".join([
            "### Threat And Risk Register",
            f"{ids['risk'][0]}: {f['main_risk']}. Impact: incorrect or unsafe network state. Controls: validation, reconciliation, timeout handling, and operator-visible stale status.",
            f"{ids['risk'][1]}: Unauthorized scoped access. Impact: cross-segment or cross-tenant exposure. Controls: scoped authorization, denial tests, and audit review.",
            f"{ids['risk'][2]}: Credential exposure. Impact: dependency compromise. Controls: secret manager references, least privilege, rotation, and no raw secret storage.",
            f"{ids['risk'][3]}: Overload or backlog. Impact: delayed operations and stale evidence. Controls: quotas, queue depth alerts, backpressure, and capacity testing.",
            "Likelihood is qualitative until incident history and workload data are provided. Residual risk needs owner acceptance before pilot rollout.",
        ])
    if "Feasibility" in title:
        return "\n".join([
            "### Feasibility",
            f"Technical feasibility is conditional on a representative integration experiment using {f['integration']}. The experiment must test the riskiest behavior: {f['main_risk']}.",
            "Operational feasibility requires named owners for support, incident response, restore rehearsal, and permission review.",
            "Economic feasibility is unknown because budget, labor rates, hosting cost, tooling licenses, and support capacity were not supplied.",
            "Schedule feasibility is unknown until scope, capacity, test environment access, and acceptance authority are confirmed. Record unknown values as open decisions instead of inventing dates.",
            f"Decision links: {', '.join(ids['dec'])}. Evidence links: {', '.join(e['id'] for e in evid)}.",
        ])
    if "Priorities" in title:
        return "\n".join([
            "### Priorities And Release Scope",
            f"Essential: scoped authorization, core ability to {f['capability']}, validation, audit, status visibility, and failure handling.",
            "Desirable: saved filters, richer dashboards, notification preferences, and bulk reporting after essential controls pass.",
            "Optional: cosmetic theme customization and convenience exports that do not affect acceptance.",
            "Deferred: unsupported devices, autonomous sensitive changes, final lifecycle decision, and production-wide rollout. Each deferred item requires a reason and owner.",
            f"Priority decisions link to {ids['dec'][0]}, {ids['dec'][1]}, and risks {ids['risk'][0]} through {ids['risk'][3]}.",
        ])
    if "Phases" in title or "Timeline" in title:
        return "\n".join([
            "### Proposed Work Phases And Timeline",
            "Phase 1 discovery: confirm actors, scope, trust boundaries, success criteria, integrations, and unresolved decisions. Exit when acceptance examples are reviewed.",
            "Phase 2 feasibility: run a bounded integration and workload experiment. Exit when critical risks have evidence or an explicit mitigation plan.",
            "Phase 3 preliminary design: define APIs, data lifecycle, authorization model, deployment boundaries, and traceability. Exit when security and operational reviewers accept the baseline.",
            "Phase 4 construction: implement the approved pilot scope with automated checks. Exit when requirement-linked tests pass.",
            "Phase 5 validation and handover: rehearse deployment, rollback, restore, incident response, and support ownership. Calendar dates require confirmed team capacity and dependencies.",
        ])
    if "Verification" in title:
        return "\n".join([
            "### Verification And Acceptance Plan",
            f"{ids['test'][0]} authorization matrix test covers allowed role, denied role, cross-scope access, and reviewer write denial.",
            f"{ids['test'][1]} schema and contract test covers malformed input, stale version, duplicate idempotency key, and unsupported integration target.",
            f"{ids['test'][2]} audit test checks actor, timestamp, previous version, approval, result, and immutable evidence retrieval.",
            f"{ids['test'][3]} failure-injection test covers {f['main_risk']} and confirms visible recovery state.",
            f"{ids['test'][4]} restore rehearsal proves backup usability and recovery assumptions.",
            f"{ids['test'][5]} load and backpressure test measures latency, throughput, queue age, rejection behavior, and stale data reporting.",
        ])
    if "Deployment" in title:
        return "\n".join([
            "### Deployment And Operations",
            "Rollout begins in a test environment, then a limited pilot scope, then broader deployment only after acceptance review.",
            "Deployment prerequisites: approved trust boundaries, scoped credentials, monitoring, rollback path, backup/restore rehearsal, support owner, and incident contact path.",
            "Rollback must restore prior configuration or disable new control-plane actions without hiding already executed external effects.",
            "Operations require dashboards for health, dependency latency, authorization denials, queue age, stale data, failed operations, and audit export success.",
            "Handover includes runbooks, troubleshooting guide, permission review process, patch cadence, training, and service ownership.",
        ])
    if "Traceability" in title:
        return "\n".join([
            "### Requirements Traceability Matrix",
            f"{evid[0]['id']} -> {ids['fr'][0]} -> {ids['uc'][0]} -> {ids['test'][0]} -> pending expert review.",
            f"{evid[1]['id']} -> {ids['fr'][2]} -> {ids['uc'][1]} -> {ids['test'][1]} -> pending expert review.",
            f"{evid[-1]['id']} -> {ids['sec'][0]} and {ids['sec'][3]} -> {ids['uc'][2]} -> {ids['test'][2]} -> pending security review.",
            f"Every requirement ID in this report must resolve before training approval: {', '.join(ids['fr'] + ids['nfr'] + ids['sec'])}.",
        ])
    if "Standards" in title:
        return "\n".join([
            "### Applicable Standards And Policies",
            f"{evid[0]['id']} supports network policy examples but is not a full SRS source.",
            f"{evid[1]['id']} and {evid[2]['id']} provide public networking design proposal examples, not project-specific approvals.",
            f"{evid[3]['id']} provides structured technical proposal practice that can inform evidence and graduation criteria.",
            f"{evid[4]['id']} is relevant to secure software development practices such as review, testing, and risk reduction.",
            "No certification, legal compliance, or organization-specific policy is claimed unless supplied by the project owner.",
        ])
    if "Open Issues" in title:
        return "\n".join([
            "### Open Issues And Decision Register",
            f"{ids['dec'][0]}: Confirm acceptance authority and service owner. Impact: release cannot be approved without an owner.",
            f"{ids['dec'][1]}: Confirm workload, endpoint behavior, traffic rate, and retention. Impact: capacity and cost remain provisional.",
            f"{ids['dec'][2]}: Confirm test environment and integration fixture for {f['integration']}. Impact: feasibility remains unproven.",
            f"{ids['dec'][3]}: Confirm support model, rollback window, and incident process. Impact: pilot readiness is incomplete.",
            "Each decision needs owner, due date, evidence required, and downstream requirements affected.",
        ])
    if "Glossary" in title or "References" in title:
        refs = "\n".join(f"- {e['id']}: {e['title']} - {e['url']} ({e['license']}). {e['use']}" for e in evid)
        return "\n".join([
            "### Glossary And References",
            "Control plane: software and workflows used to manage or observe network behavior.",
            "Data plane: traffic forwarding or protocol behavior affected by the system but usually not implemented by the administrative application itself.",
            "Scope: the tenant, site, segment, device group, or role boundary that limits an action.",
            "Idempotency key: stable request identifier used to avoid duplicate durable operations during retries.",
            "Acceptance evidence: recorded proof that a requirement-linked test or review passed.",
            refs,
        ])
    generic = (
        f"### {title}\n"
        f"For the {f['project']}, this section records confirmed facts, proposed assumptions, and open decisions for {f['environment']}. "
        f"The project must {f['capability']} through {f['integration']}, while treating {f['main_risk']} as a risk requiring evidence. "
        f"Use {ids['fr'][0]}, {ids['nfr'][0]}, {ids['sec'][0]}, {ids['uc'][0]}, {ids['risk'][0]}, and {ids['test'][0]} as traceable anchors. "
        "Do not present stakeholder approval, budget, schedule, certification, or measured performance as known unless it is supplied. "
        f"Reference ledger entries {', '.join(e['id'] for e in evid)} give background only; final content needs domain review."
    )
    return generic


def full_report(sc: dict, seed: int) -> tuple[str, list[dict]]:
    evid = evidence_rows(seed)
    ids = req_ids(seed)
    parts = [f"# {sc['facts']['project']} Software Requirements And Preliminary Design Report"]
    for title in SECTION_TITLES:
        body = section_body(title, sc, ids, evid)
        if body.startswith("###"):
            parts.append("## " + title + "\n" + "\n".join(body.splitlines()[1:]))
        else:
            parts.append("## " + title + "\n" + body)
    parts.append("## Evidence Ledger")
    for e in evid:
        parts.append(f"- {e['id']}: {e['title']} ({e['url']}), license: {e['license']}. Use: {e['use']}")
    return "\n\n".join(parts), evid


def make_rows(target_rows: int) -> list[dict]:
    scenarios = [
        scenario(domain, context, profile, variant)
        for domain, context, profile, variant in itertools.product(DOMAINS, CONTEXTS, PROFILES, range(1, 5))
    ]
    rows = []
    seed = 100
    for sc in scenarios:
        seed += 1
        report, evid = full_report(sc, seed)
        ids = req_ids(seed)
        base_meta = {
            "supervision": "assistant_generated_source_guided_candidate",
            "review_status": "assistant_reviewed_candidate",
            "training_approved": False,
            "domain": sc["facts"]["domain"],
            "source_urls": [e["url"] for e in evid],
            "source_licenses": [e["license"] for e in evid],
            "generator": "build_llm1_candidate_dataset.py",
            "scenario_family": sc["group_id"],
        }
        rows.append({
            "id": "clarify-" + sc["id"],
            "kind": "llm1_clarification_turn",
            "task": "llm1_clarification",
            "group_id": sc["group_id"],
            "split": sc["split"],
            "messages": [
                {"role": "system", "content": "Ask adaptive clarification questions before a networking/cybersecurity SRS report. Do not invent missing answers."},
                {"role": "user", "content": sc["request"]},
                {"role": "assistant", "content": clarification_target(sc)},
            ],
            "metadata": base_meta,
        })
        rows.append({
            "id": "full-" + sc["id"],
            "kind": "complete_llm1_report_candidate",
            "task": "llm1_full_report",
            "group_id": sc["group_id"],
            "split": sc["split"],
            "prompt": sc["request"],
            "report_markdown": report,
            "evidence": evid,
            "metadata": base_meta,
        })
        for ix, title in enumerate(SECTION_TITLES, 1):
            rows.append({
                "id": f"section-{ix:02d}-" + sc["id"],
                "kind": "llm1_report_section_candidate",
                "task": "llm1_section_generation",
                "group_id": sc["group_id"],
                "split": sc["split"],
                "messages": [
                    {"role": "system", "content": "Generate one section of a detailed networking/cybersecurity SRS report using supplied facts, IDs, and evidence ledger. Keep assumptions explicit."},
                    {"role": "user", "content": json.dumps({"section": title, "project_request": sc["request"], "facts": sc["facts"], "ids": ids, "evidence": evid}, ensure_ascii=False)},
                    {"role": "assistant", "content": section_body(title, sc, ids, evid)},
                ],
                "metadata": {**base_meta, "section": title},
            })
        rows.append({
            "id": "consistency-" + sc["id"],
            "kind": "llm1_consistency_review_candidate",
            "task": "llm1_consistency_review",
            "group_id": sc["group_id"],
            "split": sc["split"],
            "messages": [
                {"role": "system", "content": "Review an SRS draft for unresolved assumptions, missing IDs, unsupported claims, and domain contamination."},
                {"role": "user", "content": report[:12000]},
                {"role": "assistant", "content": (
                    f"Review result for {sc['facts']['project']}: IDs resolve for the candidate set: "
                    f"{', '.join(ids['fr'][:2] + ids['nfr'][:1] + ids['sec'][:1] + ids['uc'][:1] + ids['risk'][:1] + ids['test'][:1])}. "
                    "No finance-domain contamination is expected. The draft still needs expert networking/security review, real workload measurements, budget, owner approvals, and confirmed acceptance authority before training approval."
                )},
            ],
            "metadata": base_meta,
        })
        if len(rows) >= target_rows:
            break
    return rows[:target_rows]


def write_rows(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        (out / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    (out / "all.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    counts = Counter(f"{row['task']}/{row['split']}" for row in rows)
    manifest = {
        "name": "sms-ac-llm1-candidate-v1",
        "row_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "unique_scenario_families": len({row["group_id"] for row in rows}),
        "status": "candidate_data_not_expert_approved",
        "training_approval": "withheld_until_expert_review",
        "purpose": "LLM1 clarification, detailed section generation, full report assembly, and consistency review for networking/cybersecurity SRS reports.",
        "limitations": [
            "Generated examples are diverse but not 7,000 independent expert-authored SRS reports.",
            "Internet sources are used as evidence context and style grounding, not copied as target reports.",
            "Full report candidates support assembly behavior; section examples carry most supervised signal for long Word reports.",
            "A technical reviewer must approve a sampled batch before another long Kaggle fine-tuning run.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-rows", type=int, default=7000)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    rows = make_rows(args.target_rows)
    write_rows(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
