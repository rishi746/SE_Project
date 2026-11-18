from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "generated" / "networking_requirements_sft.jsonl"

INSTRUCTION = (
    "Identify and classify software requirements for the given networking project scenario. "
    "Return structured JSON with functional, security, performance, availability, audit logging, "
    "integration, operational, ambiguity, clarification, traceability, and human approval details."
)


TOPICS = {
    "Network Intrusion Detection System": {
        "code": "IDS",
        "sources": ["NIST-SP-800-94", "NIST-SP-800-61R2", "RFC-5424"],
        "actors": ["SOC analyst", "network security engineer", "incident responder", "security administrator"],
        "contexts": [
            "campus network with separate student, faculty, and server VLANs",
            "enterprise network with branch offices and a central SOC",
            "data center east-west traffic environment",
            "hybrid cloud network with VPN and internet-facing services",
        ],
        "features": [
            "detect port scanning and brute-force activity",
            "correlate alerts with firewall and syslog events",
            "support custom detection rules",
            "assign severity to suspicious traffic",
            "generate incident tickets for high-confidence alerts",
        ],
        "security": [
            "protect rule changes using role-based access control",
            "preserve alert evidence with tamper-evident audit logs",
            "mask sensitive packet payload fields in analyst views",
            "separate detection administration from alert triage permissions",
        ],
    },
    "Firewall Rule Management System": {
        "code": "FWM",
        "sources": ["NIST-SP-800-41R1", "NIST-SP-800-61R2", "RFC-5424"],
        "actors": ["network administrator", "security architect", "change manager", "auditor"],
        "contexts": [
            "multi-zone enterprise firewall environment",
            "branch office perimeter network",
            "cloud firewall policy environment",
            "segmented production and development network",
        ],
        "features": [
            "submit and approve firewall rule change requests",
            "detect duplicate and shadowed rules",
            "validate rule syntax before deployment",
            "simulate traffic impact before a policy change",
            "rollback failed firewall policy deployments",
        ],
        "security": [
            "enforce least-privilege rule approval workflows",
            "block any-any rules unless a security exception is approved",
            "record who requested, approved, changed, and deployed each rule",
            "require emergency changes to be reviewed after deployment",
        ],
    },
    "VPN Access Management System": {
        "code": "VPN",
        "sources": ["NIST-SP-800-77R1", "NIST-SP-800-61R2", "RFC-5424"],
        "actors": ["remote employee", "identity administrator", "security analyst", "network administrator"],
        "contexts": [
            "remote workforce connecting to internal applications",
            "contractor access environment with limited network reachability",
            "site-to-site connectivity between branch and headquarters",
            "hybrid cloud private access environment",
        ],
        "features": [
            "authenticate users before VPN tunnel establishment",
            "assign network access based on user role and device posture",
            "log VPN session start, end, source IP, and assigned address",
            "revoke active sessions for disabled users",
            "report failed login bursts and impossible travel events",
        ],
        "security": [
            "require multi-factor authentication for privileged VPN users",
            "encrypt VPN tunnels using approved cryptographic settings",
            "rotate keys and certificates according to policy",
            "deny access from unmanaged or non-compliant devices",
        ],
    },
    "Network Traffic Monitoring System": {
        "code": "NTM",
        "sources": ["RFC-5424", "RFC-3411", "NIST-SP-800-94"],
        "actors": ["NOC operator", "network engineer", "capacity planner", "security analyst"],
        "contexts": [
            "enterprise WAN with multiple branch links",
            "campus network with wireless and wired access layers",
            "data center network carrying application traffic",
            "service provider edge network",
        ],
        "features": [
            "collect flow, interface, and syslog telemetry",
            "display bandwidth, packet loss, latency, and device health",
            "trigger alerts when thresholds are breached",
            "retain searchable historical traffic summaries",
            "export reports for capacity and incident review",
        ],
        "security": [
            "restrict telemetry access by operational role",
            "protect monitoring credentials in a secret store",
            "detect monitoring gaps caused by collector failure",
            "audit alert threshold and device inventory changes",
        ],
    },
    "DDoS Detection and Mitigation System": {
        "code": "DDS",
        "sources": ["CISA-DDOS-GUIDE", "NIST-SP-800-61R2", "RFC-5424"],
        "actors": ["SOC analyst", "network defender", "incident commander", "service owner"],
        "contexts": [
            "public-facing e-commerce network",
            "university web services environment",
            "cloud-hosted API platform",
            "internet-facing DNS and application services",
        ],
        "features": [
            "detect traffic floods and protocol anomalies",
            "classify suspected DDoS events by target, vector, and severity",
            "activate rate limiting or upstream mitigation workflows",
            "notify incident responders and service owners",
            "track service availability during mitigation",
        ],
        "security": [
            "require human approval before blocking broad traffic categories",
            "preserve attack traffic summaries for post-incident review",
            "ensure mitigation actions are reversible and logged",
            "protect mitigation credentials and API tokens",
        ],
    },
    "Router/Switch Configuration Management System": {
        "code": "RCM",
        "sources": ["RFC-6241", "RFC-3411", "RFC-5424"],
        "actors": ["network engineer", "configuration reviewer", "operations lead", "auditor"],
        "contexts": [
            "enterprise router and switch fleet",
            "campus access and distribution network",
            "data center spine-leaf fabric",
            "managed service provider customer network",
        ],
        "features": [
            "store approved configuration templates",
            "validate proposed device configuration before deployment",
            "detect drift from approved baseline",
            "support rollback to last known good configuration",
            "schedule configuration deployment windows",
        ],
        "security": [
            "require peer review for high-impact configuration changes",
            "lock device configuration during deployment",
            "audit every configuration read, change, validation, and rollback",
            "prevent storage of plaintext device credentials",
        ],
    },
}

SCALES = [
    ("small", "50 devices", "500 users", "99.5% availability"),
    ("medium", "500 devices", "5,000 users", "99.9% availability"),
    ("large", "5,000 devices", "50,000 users", "99.95% availability"),
]

CONSTRAINTS = [
    "must integrate with existing SIEM tooling",
    "must support role-based access control",
    "must retain audit logs for at least one year",
    "must support encrypted management connections",
    "must avoid exposing sensitive operational data to unauthorized users",
    "must provide human approval gates for high-risk changes",
]


def req(requirement_id: str, statement: str, category: str, priority: str, risk: str, criteria: list[str], sources: list[str]) -> dict:
    unique_sources = list(dict.fromkeys(sources))
    return {
        "requirement_id": requirement_id,
        "statement": statement,
        "category": category,
        "priority": priority,
        "risk_level": risk,
        "acceptance_criteria": criteria,
        "source_refs": unique_sources,
    }


def build_example(topic: str, index: int, rng: random.Random) -> dict:
    spec = TOPICS[topic]
    scale, device_count, user_count, availability = rng.choice(SCALES)
    context = rng.choice(spec["contexts"])
    actor = rng.choice(spec["actors"])
    selected_features = rng.sample(spec["features"], 3)
    selected_security = rng.sample(spec["security"], 3)
    constraints = rng.sample(CONSTRAINTS, 2)
    code = spec["code"]
    sources = spec["sources"]

    scenario = (
        f"A {scale} organization wants to build a {topic.lower()} for a {context}. "
        f"The main stakeholder is the {actor}. The system should {selected_features[0]}, "
        f"{selected_features[1]}, and {selected_features[2]}. It {constraints[0]} and {constraints[1]}. "
        f"The expected environment includes about {device_count} and {user_count}, with a target of {availability}."
    )

    requirements = [
        req(
            f"BR-{code}-{index:03d}",
            f"The system shall support the {actor} in managing the {topic.lower()} workflow for the {context}.",
            "business_requirements",
            "Must",
            "Medium",
            [f"The {actor} can complete the primary workflow without manual spreadsheet tracking."],
            sources[:1],
        ),
        req(
            f"FR-{code}-{index:03d}-01",
            f"The system shall {selected_features[0]}.",
            "functional_requirements",
            "Must",
            "High",
            [f"A test scenario proves that the system can {selected_features[0]}."],
            sources,
        ),
        req(
            f"FR-{code}-{index:03d}-02",
            f"The system shall {selected_features[1]}.",
            "functional_requirements",
            "Must",
            "High",
            [f"A test scenario proves that the system can {selected_features[1]}."],
            sources,
        ),
        req(
            f"FR-{code}-{index:03d}-03",
            f"The system shall {selected_features[2]}.",
            "functional_requirements",
            "Should",
            "Medium",
            [f"A test scenario proves that the system can {selected_features[2]}."],
            sources,
        ),
        req(
            f"SR-{code}-{index:03d}-01",
            f"The system shall {selected_security[0]}.",
            "security_requirements",
            "Must",
            "Critical",
            [f"Unauthorized attempts to bypass this control are denied and logged."],
            sources,
        ),
        req(
            f"SR-{code}-{index:03d}-02",
            f"The system shall {selected_security[1]}.",
            "security_requirements",
            "Must",
            "High",
            [f"Security test evidence confirms that the control is enforced consistently."],
            sources,
        ),
        req(
            f"SR-{code}-{index:03d}-03",
            f"The system shall {selected_security[2]}.",
            "security_requirements",
            "Should",
            "High",
            [f"Audit review confirms that the security action is traceable to a user or service account."],
            sources,
        ),
        req(
            f"PR-{code}-{index:03d}",
            "The system shall mask sensitive user, device, credential, and packet payload information in non-privileged views.",
            "privacy_requirements",
            "Must",
            "High",
            ["A non-privileged user cannot view sensitive fields in UI, exports, logs, or API responses."],
            sources,
        ),
        req(
            f"PERF-{code}-{index:03d}",
            f"The system shall operate at the planned scale of {device_count} and {user_count} without breaching documented response-time thresholds.",
            "performance_requirements",
            "Should",
            "Medium",
            ["Load testing demonstrates that dashboard, alert, and API operations remain within approved thresholds."],
            sources,
        ),
        req(
            f"AV-{code}-{index:03d}",
            f"The system shall meet the target service availability of {availability}.",
            "availability_reliability_requirements",
            "Must",
            "High",
            ["Availability reports show that planned uptime objectives are met during the evaluation window."],
            sources,
        ),
        req(
            f"AUD-{code}-{index:03d}",
            "The system shall record security-relevant events with timestamp, actor, source address, action, decision, and result.",
            "audit_logging_requirements",
            "Must",
            "Critical",
            ["An auditor can trace each privileged action from request through approval and execution."],
            ["RFC-5424", *sources],
        ),
        req(
            f"INT-{code}-{index:03d}",
            "The system shall integrate with existing identity, logging, ticketing, and monitoring tools through documented APIs or connectors.",
            "integration_requirements",
            "Should",
            "Medium",
            ["Integration tests demonstrate successful event exchange with at least one identity provider and one logging or ticketing system."],
            sources,
        ),
        req(
            f"OPS-{code}-{index:03d}",
            "The system shall support backup, restore, configuration export, health checks, and operator-visible failure states.",
            "operational_maintenance_requirements",
            "Should",
            "Medium",
            ["Operations testing confirms recovery from a failed deployment or service restart without data loss beyond the approved recovery point."],
            sources,
        ),
        req(
            f"CA-{code}-{index:03d}",
            f"The system is assumed to operate in a managed network where authorized administrators can supply device inventory, network ranges, and approval policies.",
            "constraints_assumptions",
            "Could",
            "Low",
            ["The project baseline identifies data owners for inventory, ranges, and approval policy inputs."],
            sources[:1],
        ),
    ]

    ambiguities = [
        "The exact alert severity thresholds are not fully defined.",
        "The approved retention period and jurisdiction-specific privacy constraints need confirmation.",
        "The expected integration endpoints and authentication methods are not fully specified.",
    ]
    clarification_questions = [
        "Which user roles are allowed to approve high-risk network or security changes?",
        "What are the maximum acceptable detection, alerting, or deployment latencies?",
        "Which SIEM, identity provider, ticketing tool, and monitoring systems must be integrated?",
        "What log retention period and masking policy should be applied to operational and security data?",
    ]
    human_approval = [
        "Final requirement baseline",
        "High-risk security controls",
        "Access-control policy",
        "Log retention and privacy policy",
        "Production deployment readiness",
    ]

    return {
        "id": f"{code}-{index:03d}",
        "instruction": INSTRUCTION,
        "input": scenario,
        "output": {
            "project_type": topic,
            "domain": "networking",
            "requirements": requirements,
            "ambiguities": ambiguities,
            "clarification_questions": clarification_questions,
            "traceability_notes": [
                f"Source-grounded using {', '.join(sources)}.",
                "Each requirement includes category, priority, risk level, acceptance criteria, and source references.",
            ],
            "human_approval_required": human_approval,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-per-topic", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    for topic in TOPICS:
        for index in range(1, args.examples_per_topic + 1):
            examples.append(build_example(topic, index, rng))

    rng.shuffle(examples)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"wrote={len(examples)}")
    print(f"path={out_path}")


if __name__ == "__main__":
    main()
