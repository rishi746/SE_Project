"""Create detailed, case-grounded SDLC teaching examples for LLM2.

Original synthetic cases only. Source references justify model definitions,
not the fictional project facts or a claim of expert adjudication.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("SMS_AC_LLM2_OUTPUT", str(ROOT / "data" / "llm2_sdlc_detailed")))
MODELS = ("Waterfall", "V-shaped", "RAD", "Prototype", "Spiral", "Iterative")

# Every field is used in the question and in at least one justification, plan,
# acceptance criterion, or model-specific advantage.
PROJECTS = [
    dict(name="DNS resolver management service", goal="centralize resolver policies and cache visibility", existing="a documented resolver fleet and known DNS query behavior", interface="DNS queries and resolver administration APIs", hazard="stale or poisoned answers", test="positive and negative TTL cases plus upstream timeout recovery", artifact="versioned resolver policy and replayable query fixtures", users="campus network operators", guard="compare pilot answers with a trusted resolver"),
    dict(name="remote-access VPN control service", goal="manage enrollment, session revocation, and access policies", existing="a deployed identity provider and a supported gateway API", interface="identity-provider and VPN gateway APIs", hazard="sessions remaining active after revocation", test="revocation of an active session within an agreed window", artifact="access-policy and session-state records", users="remote-access administrators", guard="retain an emergency access route and rollback plan"),
    dict(name="firewall rule review console", goal="simulate, approve, and stage network rules", existing="an approved firewall API and a catalog of existing rules", interface="firewall configuration and approval APIs", hazard="a rule disconnecting management access", test="rule-diff approval and failed-deployment rollback", artifact="versioned rule sets and simulation fixtures", users="firewall administrators", guard="stage changes on a canary device"),
    dict(name="network intrusion alert triage service", goal="turn authorized sensor events into reviewable cases", existing="a documented sensor feed and reviewed alert examples", interface="authenticated sensor ingestion and case APIs", hazard="missed events or overwhelming false positives", test="precision on reviewed fixtures and complete evidence linkage", artifact="correlation-rule versions and case evidence IDs", users="security analysts", guard="run in shadow mode before enabling actions"),
    dict(name="flow observability portal", goal="show traffic trends and anomalous utilization", existing="working NetFlow or IPFIX exporters and an approved dashboard stack", interface="flow collection and dashboard APIs", hazard="late or duplicate flows distorting metrics", test="aggregation of delayed and duplicated flow records", artifact="flow-window definitions and dashboard acceptance examples", users="NOC analysts", guard="mark stale flow windows explicitly"),
    dict(name="DDoS response coordinator", goal="recommend and stage approved traffic mitigations", existing="monitoring feeds and provider mitigation endpoints", interface="monitoring and mitigation-provider APIs", hazard="blocking legitimate customer traffic", test="false-alarm review and bounded mitigation rollback", artifact="incident thresholds and mitigation-request records", users="incident commanders", guard="require human approval for traffic-blocking actions"),
    dict(name="router configuration deployment service", goal="validate and stage router changes across sites", existing="known device models and an approved configuration baseline", interface="NETCONF and supported vendor configuration APIs", hazard="partial rollout or loss of connectivity", test="confirmed commit and failed-commit recovery", artifact="candidate configurations and rollback snapshots", users="routing engineers", guard="deploy a canary before fleet rollout"),
    dict(name="secure operations messaging service", goal="synchronize authorized incident-room messages", existing="an established identity provider and message transport", interface="HTTPS and WebSocket interfaces", hazard="cross-room disclosure or missed reconnect events", test="membership denial and reconnect-history consistency", artifact="room-membership and message-cursor contracts", users="incident-room participants", guard="authorize every message operation"),
    dict(name="branch configuration file sync service", goal="transfer approved artifacts between sites", existing="known file types and a storage API", interface="HTTPS chunk-upload and change-feed APIs", hazard="conflicting edits or interrupted transfers", test="checksum, resume, and conflict-version behavior", artifact="file-version and chunk manifests", users="branch support engineers", guard="retain the last verified artifact until replacement completes"),
    dict(name="tenant event webhook gateway", goal="deliver network events to subscribed tenants", existing="a published event schema and customer endpoints", interface="signed outbound HTTPS and subscription APIs", hazard="cross-tenant delivery or retry amplification", test="signature validation and bounded dead-letter replay", artifact="subscription and delivery-attempt records", users="integration owners", guard="restrict outbound destinations and retry budgets"),
    dict(name="IoT telemetry edge gateway", goal="buffer and normalize sensor observations", existing="documented device payloads and a supported MQTT broker", interface="MQTT intake and device-management interfaces", hazard="lost data during intermittent uplinks", test="buffer overflow and delayed-ack behavior", artifact="observation schemas and offline-queue rules", users="edge-device operators", guard="define explicit overflow and recovery behavior"),
    dict(name="zero-trust API gateway", goal="authorize and rate-limit service requests", existing="an approved identity provider and API routing inventory", interface="HTTPS upstream and identity-provider APIs", hazard="cross-tenant access or cascading retries", test="per-tenant denial and upstream failure isolation", artifact="route, quota, and identity-policy contracts", users="platform security engineers", guard="propagate deadlines and enforce retry budgets"),
    dict(name="remote-support video signaling service", goal="establish secure real-time support sessions", existing="a working signaling prototype and supported WebRTC stack", interface="HTTPS signaling and WebRTC negotiation", hazard="NAT traversal failure or session leakage", test="relay fallback, reconnect, and session teardown", artifact="room and relay-allocation contracts", users="remote-support staff", guard="separate signaling authorization from media transport"),
    dict(name="service-discovery registry", goal="publish healthy service endpoints", existing="documented registration and heartbeat behavior", interface="registration, heartbeat, and lookup APIs", hazard="stale endpoints during a network partition", test="lease expiry and partitioned lookup behavior", artifact="instance-lease and health-check rules", users="site reliability engineers", guard="distinguish suspected failure from confirmed removal"),
    dict(name="network device backup coordinator", goal="capture and restore approved device configurations", existing="supported device export APIs and protected storage", interface="device export APIs and object storage", hazard="unrestorable backups or exposed credentials", test="checksum verification and timed restore rehearsal", artifact="snapshot inventories and restore runbooks", users="recovery engineers", guard="encrypt snapshots and rehearse restores"),
    dict(name="Wi-Fi enrollment portal", goal="issue authorized network sessions after enrollment", existing="a supported access controller and identity provider", interface="HTTPS portal and network-access-controller APIs", hazard="unauthorized sessions or controller outage", test="session expiry and denied-enrollment behavior", artifact="enrollment, consent, and session workflows", users="IT staff and end users", guard="define controller-outage behavior before rollout"),
    dict(name="message-broker administration console", goal="manage queues, subscriptions, and access policies", existing="an operational broker with documented administration APIs", interface="broker administration and metrics APIs", hazard="destructive retention changes or topic access leaks", test="policy isolation and offset recovery", artifact="queue policies and offset records", users="messaging-platform operators", guard="preview retention changes before applying them"),
    dict(name="network asset inventory service", goal="reconcile approved discovery with owned device records", existing="a device inventory and authorized discovery adapters", interface="discovery and inventory APIs", hazard="duplicate assets or incorrect ownership inference", test="reviewed merge decisions and provenance retention", artifact="asset observations and reconciliation rules", users="inventory custodians", guard="bound discovery scope and scan rate"),
    dict(name="certificate rotation coordinator", goal="track and rotate certificates used by network services", existing="a certificate inventory and supported issuance endpoints", interface="certificate authority and service deployment APIs", hazard="expired certificates causing service outages", test="renewal, rollback, and expiry-alert behavior", artifact="certificate ownership and rotation records", users="PKI administrators", guard="pilot rotation on a noncritical service"),
    dict(name="network access control policy console", goal="review and publish endpoint access decisions", existing="documented NAC rules and an identity directory", interface="NAC controller and identity-directory APIs", hazard="legitimate devices being quarantined", test="policy simulation and exception rollback", artifact="access-policy versions and device exception records", users="NAC administrators", guard="retain a time-limited recovery exception path"),
    dict(name="WAN failover monitoring service", goal="detect circuit failure and show failover state", existing="known branch circuits and existing link telemetry", interface="router telemetry and alert APIs", hazard="false failover alarms or missed outages", test="failure-injection and recovery-time measurement", artifact="link-state timelines and escalation rules", users="WAN operations engineers", guard="test on a noncritical circuit before expansion"),
    dict(name="secure DNS filtering console", goal="manage domain-blocking policy and audit decisions", existing="a running filtering resolver and approved policy lists", interface="resolver policy and audit APIs", hazard="legitimate domains being blocked", test="allowlist precedence and policy rollback", artifact="blocklist versions and decision logs", users="security and help-desk staff", guard="rehearse emergency allowlist changes"),
    dict(name="cloud network segmentation planner", goal="review and deploy service-to-service network boundaries", existing="a cloud inventory and supported network-policy API", interface="cloud network-policy and asset APIs", hazard="unintended service isolation", test="reachability simulation and rollback validation", artifact="segment maps and policy-change records", users="cloud network engineers", guard="stage policy on a limited workload group"),
    dict(name="network maintenance scheduler", goal="coordinate approved device changes and outage windows", existing="a change calendar and device inventory", interface="change-ticket and device-management APIs", hazard="overlapping changes causing an outage", test="conflict detection and cancelled-window recovery", artifact="maintenance plans and dependency maps", users="change managers", guard="require approval before production change execution"),
]

# Seven genuinely different deployment contexts. The decision signal comes from
# the model-specific situation below, not the organization or technology name.
CONTEXTS = [
    ("a university IT organization", "three engineers", "one academic term", "a single campus", "operators at scheduled reviews"),
    ("a regional network operator", "four engineers", "two quarters", "one noncritical site", "operations and security reviewers"),
    ("a multi-site enterprise", "five engineers", "six months", "two representative sites", "network engineering reviewers"),
    ("a managed-service provider", "six engineers", "eight months", "one consenting tenant", "tenant and operations reviewers"),
    ("a public-sector IT department", "four engineers", "nine months", "a lab followed by one approved site", "change-board reviewers"),
    ("an industrial IT organization", "three engineers", "seven months", "a disconnected test segment", "operations and security reviewers"),
    ("a hybrid-cloud enterprise", "seven engineers", "five months", "one internal service group", "service owners and reliability engineers"),
]

SITUATIONS = {
    "Waterfall": (
        "The statement of work already specifies the functions, data contracts, and acceptance boundary. The team has implemented this type of integration before. Stakeholders can approve documents at scheduled phase gates but cannot attend weekly working sessions. One controlled production handover is required; partial feature releases add no business value.",
        "The signed scope and familiar integration make a stable baseline credible.",
        "The stated review schedule favors sequential requirements, design, implementation, and acceptance gates.",
        "The requested single handover does not reward frequent deployable increments.",
        "Testing still needs planning throughout; Waterfall is not permission to defer defect detection until the last week.",
    ),
    "V-shaped": (
        "The approved requirements and interfaces are stable. Release policy requires every requirement to have a corresponding verification method and independent acceptance evidence. Existing behavior or an approved baseline can be used as an oracle. The cost of an unnoticed error is high, so the team must specify failure, recovery, and authorization tests before implementation.",
        "A known baseline allows requirements to be paired with planned acceptance evidence.",
        "The explicit independent verification obligation makes testing an organizing principle, not merely a final phase.",
        "The existing behavior provides an oracle for regression and equivalence checks.",
        "The operational hazard makes early test design and traceable failure cases valuable.",
    ),
    "RAD": (
        "The underlying network service and security boundaries are already supported; the new work is primarily an operator-facing application built from approved APIs and reusable UI components. A first usable release is needed within twelve weeks. Named users can attend twice-weekly design workshops and immediately test each working slice. The production pilot still requires access-control and integration checks.",
        "The first-release deadline and reusable components make rapid construction feasible.",
        "Available users can give immediate feedback on working screens and workflows.",
        "The known backend reduces the need for a long technology-feasibility program.",
        "The goal is a usable application, not a disposable proof of concept.",
    ),
    "Prototype": (
        "The sponsor wants a working demonstration before deciding whether to fund production development. Users cannot yet specify the correct workflow, controls, or display. They can evaluate candidate interactions in a safe simulator. The first milestone is to learn and refine requirements, not to deploy a secure production service.",
        "Users need to see and use a candidate interface before they can state reliable requirements.",
        "The requested deliverable is a learning demonstration rather than an accepted production release.",
        "Simulation permits cheap, safe experiments before production architecture is fixed.",
        "Feedback should become a validated specification; demonstration code is not automatically production code.",
    ),
    "Spiral": (
        "A consequential integration behavior is unproven and committing to full deployment would be expensive. Budget is released after risk-reduction gates. The team can run measured testbed experiments, compare alternatives, and stop or redesign at each go/no-go review. A quick UI demonstration would not resolve the dominant technical and operational risks.",
        "The decision is governed by unresolved risk, not merely by changing feature requests.",
        "Experiments must retire the dominant integration or failure uncertainty before full commitment.",
        "Funding gates align each new investment with evidence from the previous loop.",
        "Repeated risk assessment can change the design or terminate an unsafe approach.",
    ),
    "Iterative": (
        "The core interfaces and safety boundary are understood. The service can be divided into independently useful capabilities. Operators will review each real release and may change priorities as usage data arrives. The sponsor wants a first deployable slice followed by tested refinements and additional capabilities, not a one-off demonstration.",
        "A useful first capability can be deployed before every feature is known.",
        "Real use, rather than speculation, can reprioritize later capabilities.",
        "Known architecture makes each cycle a delivery cycle rather than a feasibility experiment.",
        "Regression testing must protect accepted behavior as subsequent slices are integrated.",
    ),
}

def cap(value: str) -> str:
    return value[0].upper() + value[1:]

def article(value: str) -> str:
    return "an" if value[0].lower() in "aeiou" else "a"

def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def question(model: str, p: dict, c: tuple[str, ...]) -> str:
    org, team, duration, pilot, reviewers = c
    acceptance = (f"any later production release must demonstrate {p['test']}"
                  if model == "Prototype" else f"acceptance must demonstrate {p['test']}")
    rollout = (f"If production is later funded, a rollout would start with {pilot} after review by {reviewers}; the initial demonstration remains simulated."
               if model == "Prototype" else
               f"It intends to start with {pilot}, with {reviewers} before wider use.")
    return (
        f"{cap(org)} is planning {article(p['name'])} {p['name']} to {p['goal']}. "
        f"The current environment has {p['existing']}, and the proposed service uses {p['interface']}. "
        f"{cap(p['users'])} will operate it. The main failure concern is {p['hazard']}; {acceptance}. "
        f"The team has {team} and a planning window of {duration}. {rollout} "
        f"The work products include {p['artifact']}. {SITUATIONS[model][0]} "
        "Choose the most suitable primary software development life-cycle model from Waterfall, V-shaped, RAD, Prototype, Spiral, and Iterative. Explain exactly why it fits this case, how its phases would be applied, its advantages for this project, and why close alternatives are weaker."
    )

def model_plan(model: str, p: dict, c: tuple[str, ...]) -> list[str]:
    pilot = c[3]
    if model == "Waterfall":
        return [
            f"Requirements: freeze the approved functions, {p['interface']}, and measurable acceptance for {p['test']}.",
            f"Design: review {p['artifact']}, authorization, failure behavior, deployment sequence, and a rollback design.",
            f"Implementation: build the approved design; use code review and unit and integration tests covering the ability to {p['goal']}.",
            f"Acceptance and handover: demonstrate {p['test']} at {pilot}, verify the safeguard ({p['guard']}), and obtain the scheduled sign-off.",
        ]
    if model == "V-shaped":
        return [
            f"User requirements ↔ acceptance tests: specify that {p['users']} can {p['goal']} and define accepted results for {p['test']}.",
            f"System design ↔ system tests: define the end-to-end behavior of {p['interface']}, including {p['hazard']} and recovery.",
            f"Component design ↔ integration and unit tests: map {p['artifact']} to interface, authorization, and failure-case tests.",
            f"Validation: use {p['existing']} as reference context for equivalent behavior; have independent reviewers sign off the evidence before rollout expands beyond {pilot}.",
        ]
    if model == "RAD":
        return [
            f"Requirements workshop: agree on the first operator journey for {p['users']} and the non-negotiable security constraints of {p['interface']}.",
            f"Rapid construction: assemble a first usable slice from approved components and expose {p['artifact']} in a reviewable form.",
            f"User design cycles: demonstrate the slice twice weekly; revise the workflow while checking {p['test']}.",
            f"Cutover: run access-control, integration, and rollback checks, then pilot at {pilot}; speed does not waive these gates.",
        ]
    if model == "Prototype":
        return [
            f"Learning goals: ask which controls and displays {p['users']} need when they try to {p['goal']} safely.",
            f"Build a disposable simulator-backed model around {p['artifact']}; do not connect it to live production changes.",
            f"Observe user trials, noting misunderstandings, missing information, and responses to {p['hazard']}.",
            f"Revise and evaluate the prototype; only then baseline production requirements, including tests for {p['test']} and a plan to {p['guard']}.",
        ]
    if model == "Spiral":
        return [
            f"Loop 1 objectives and alternatives: define feasible designs for {p['interface']} and rank {p['hazard']} as a candidate top risk.",
            f"Risk experiment: use a controlled testbed to measure {p['test']} and document the result in {p['artifact']}.",
            f"Go/no-go review: evaluate observed failure behavior, cost, and the safeguard ({p['guard']}); redesign or stop if evidence is insufficient.",
            f"Next loops: address the next dominant risk, then pilot at {pilot} only after the funding gate and operational review approve expansion.",
        ]
    return [
        f"Iteration 1: deliver the smallest useful capability for {p['users']} to {p['goal']}, with measurable acceptance for {p['test']}.",
        f"Review actual use at {pilot}; refine the backlog based on operator feedback and observed {p['hazard']} cases.",
        f"Later iterations: add and improve independently useful functions, updating {p['artifact']} and regression tests each cycle.",
        f"Expansion: retain the safeguard ({p['guard']}) and require acceptance of each integrated release before broader rollout.",
    ]

def benefits(model: str, p: dict, c: tuple[str, ...]) -> list[tuple[str, str]]:
    org, team, duration, pilot, reviewers = c
    common = {
        "scope": f"For this {p['name']}, the team can tie the agreed scope to {p['goal']} and prevent unrelated changes from displacing the work.",
        "interface": f"The documented {p['interface']} allow checks against actual integration behavior rather than an assumed interface.",
        "hazard": f"The named failure mode, {p['hazard']}, becomes an explicit review and test case instead of an afterthought.",
        "test": f"Acceptance can use {p['test']}, giving reviewers observable evidence rather than a claim that the service 'works'.",
        "artifact": f"The {p['artifact']} provide a concrete record that {reviewers} can inspect and approve.",
        "pilot": f"Starting at {pilot} contains rollout exposure while the team measures the real behavior.",
        "guard": f"The team can verify the production safeguard ({p['guard']}) before applying changes more widely.",
        "staff": f"With {team} over {duration}, the delivery plan can assign explicit ownership and avoid inventing unlimited capacity.",
        "users": f"{cap(p['users'])} can judge whether the result supports their real tasks before broader acceptance.",
        "unknown": f"If the stated starting point—{p['existing']}—proves less reliable than described, the team has a clear trigger to revisit the baseline and model choice.",
        "recovery": f"The team can test the safeguard ({p['guard']}) before wider use to limit the impact of a failed change.",
        "exit": f"Release criteria can require documented passes for {p['test']} and reviewed handling of failed cases.",
        "feedback": f"At each review, {p['users']} can point out missing steps in the real workflow before the team invests in wider rollout.",
        "record": f"Versioned records such as {p['artifact']} can show exactly what changed and what evidence supported approval.",
        "regression": f"Tests for {p['test']} can be compared with baselined behavior or fixtures associated with {p['existing']}.",
        "safe_experiment": f"A simulator can present {p['artifact']} and expose questions about {p['hazard']} without changing production network state.",
    }
    mapping = {
        "Waterfall": [
            ("Clear requirements baseline", "scope"), ("Predictable phase gates", "staff"),
            ("Design before build", "interface"), ("Controlled change requests", "artifact"),
            ("One planned handover", "pilot"), ("Defined responsibilities", "users"),
            ("Auditable approval", "record"), ("Visible failure risk", "hazard"),
            ("Explicit operational safety", "recovery"), ("Measurable final acceptance", "exit"),
        ],
        "V-shaped": [
            ("Testing planned from requirements", "test"), ("Requirement-to-test traceability", "record"),
            ("Stable baseline fit", "scope"), ("Interface verification", "interface"),
            ("Early defect discovery", "hazard"), ("Regression against existing behavior", "regression"),
            ("Independent acceptance", "users"), ("Recovery validation", "recovery"),
            ("Controlled pilot evidence", "pilot"), ("Clear exit criteria", "exit"),
        ],
        "RAD": [
            ("Fast usable release", "staff"), ("Reuse of approved components", "interface"),
            ("Frequent operator feedback", "users"), ("Visible working slices", "artifact"),
            ("Early workflow corrections", "feedback"), ("Time-boxed scope decisions", "scope"),
            ("Integrated user design", "test"), ("Pilot before scale", "pilot"),
            ("Security preserved", "guard"), ("Assumptions exposed early", "unknown"),
        ],
        "Prototype": [
            ("Visible concept", "artifact"), ("Clarifies unknown requirements", "users"),
            ("Safe experimentation", "safe_experiment"), ("Early usability findings", "feedback"),
            ("Cheap revision before build", "scope"), ("Feasibility demonstration", "interface"),
            ("Stakeholder alignment", "record"), ("Failure behavior explored", "hazard"),
            ("Better production acceptance criteria", "test"), ("Bounded first milestone", "staff"),
        ],
        "Spiral": [
            ("Risk-driven spending", "staff"), ("Tests the top uncertainty", "hazard"),
            ("Evidence at each gate", "test"), ("Alternatives remain open", "interface"),
            ("Early stop or redesign option", "unknown"), ("Incremental commitment", "pilot"),
            ("Operational risk visibility", "scope"), ("Recorded experiment results", "record"),
            ("Stakeholder go/no-go reviews", "users"), ("Safer scale-up", "recovery"),
        ],
        "Iterative": [
            ("Useful early capability", "users"), ("Feedback from real use", "feedback"),
            ("Flexible prioritization", "scope"), ("Smaller integrated changes", "interface"),
            ("Regression evidence each cycle", "test"), ("Operational learning", "hazard"),
            ("Progress through accepted slices", "artifact"), ("Controlled rollout", "pilot"),
            ("Continuous safety checks", "guard"), ("Capacity-aware planning", "staff"),
        ],
    }
    return [(title, common[key]) for title, key in mapping[model]]

def alternatives(model: str, p: dict) -> list[str]:
    pairs = {
        "Waterfall": [f"V-shaped would be stronger if independent requirement-by-requirement verification were the governing demand. Here the defining constraint is the stable contract and sequential handover; testing of {p['test']} is still mandatory.", "Iterative would add repeated release and review overhead without a need for useful intermediate deployments or frequent reprioritization."],
        "V-shaped": [f"Waterfall fits the stable scope but does not foreground the required pairing of each requirement with a verification method for {p['test']}.", "Prototype would help if user needs were unknown; the approved baseline makes equivalence and independent evidence the central challenge."],
        "RAD": ["Prototype would stop at an evaluated concept. The sponsor needs an operational first release assembled from supported components.", "Iterative also uses feedback, but RAD better reflects the unusually short first-release deadline, intensive user workshops, and explicit component reuse."],
        "Prototype": ["RAD assumes a usable application can be rapidly assembled from sufficiently understood requirements and services. Here the first milestone exists to discover the workflow itself.", f"Spiral would be warranted by large, formal risk-gated investment; the immediate uncertainty is what {p['users']} need to see and do."],
        "Spiral": ["Prototype may supply one experiment, but it does not by itself provide repeated risk ranking, evidence review, and investment gates.", f"Iterative releases could be useful later; they do not first resolve the unproven behavior around {p['hazard']}."],
        "Iterative": ["RAD demands an exceptional first-release deadline and intensive rapid workshops, neither of which is the primary constraint here.", f"Spiral is excessive without a dominant unproven technical risk or funding gates; known interfaces let the team deliver and improve useful slices of the {p['name']}."],
    }
    return pairs[model]

def answer(model: str, p: dict, c: tuple[str, ...]) -> str:
    org, team, duration, pilot, reviewers = c
    situation = SITUATIONS[model]
    lines = [
        f"# Most Appropriate Life Cycle Model: {model}",
        f"The **{model} model** is the most suitable primary life cycle for this {p['name']}. "
        f"The decision rests on the delivery facts in the problem, not on the network technology name: {situation[1]} "
        f"{situation[2]} The team must still prove {p['test']} and control {p['hazard']}.",
        "## (i) Why this model fits the stated project",
        f"1. **Nature of the work.** The requested service must {p['goal']}; the scenario says the environment already has {p['existing']}. {situation[1]}",
        f"2. **Requirements and uncertainty.** {situation[3]} This is a direct reason to choose {model} for the stated delivery conditions.",
        f"3. **Verification need.** The acceptance question is not merely whether a screen appears. The team must demonstrate {p['test']} and record the result against {p['artifact']}.",
        f"4. **Failure consequence.** {cap(p['hazard'])} could affect users or operations. The plan therefore includes failure cases and the safeguard ({p['guard']}).",
        f"5. **People and timing.** {cap(org)} has {team} and a planning window of {duration}. {cap(p['users'])} and {reviewers} have the roles stated; the answer does not assume unlimited stakeholder time.",
        f"6. **Delivery and rollout.** The initial deployment will cover {pilot}. {situation[4]}",
        "## (ii) How the model would be used here",
    ]
    lines += [f"{i}. {step}" for i, step in enumerate(model_plan(model, p, c), 1)]
    lines += ["## (iii) Ten advantages and their relevance to this project"]
    lines += [f"{i}. **{title}.** {detail}" for i, (title, detail) in enumerate(benefits(model, p, c), 1)]
    lines += ["## (iv) Why two close alternatives are weaker"]
    lines += [f"{i}. {item}" for i, item in enumerate(alternatives(model, p), 1)]
    lines += [
        "## (v) Acceptance, limits, and conclusion",
        f"**Acceptance evidence:** Include tests for {p['test']}, denied or invalid operations, {p['hazard']}, and evidence that the safeguard is effective: {p['guard']}. {cap(p['users'])} should assess task fit, and the {reviewers} should inspect the evidence before expansion beyond {pilot}.",
        f"**Limits and assumptions:** Confirm the availability of {p['interface']}, the accuracy of {p['existing']}, the actual workload, and the real capacity of {team}. The planning window of {duration} is a constraint, not proof of feasibility or a guaranteed date. Security, backup, and rollback responsibilities remain necessary whatever model is chosen.",
        f"**Conclusion:** {model} is preferred because {situation[1].lower()} {situation[2]} "
        f"For this {p['name']}, success means measured evidence of {p['test']} without leaving {p['hazard']} unmanaged. "
        "If the stated delivery facts change materially, reassess the primary life-cycle choice rather than forcing the project into this label.",
    ]
    return "\n\n".join(lines)

def reasons_v3(model: str, p: dict, c: tuple[str, ...]) -> list[tuple[str, str]]:
    org, team, duration, pilot, reviewers = c
    if model == "Waterfall":
        return [
            ("Defined scope", f"The statement of work fixes what the {p['name']} must do: {p['goal']}. This gives the team a real requirements baseline instead of an evolving wish list."),
            ("Familiar technology", f"The organization already has {p['existing']}. The planned {p['interface']} are known, so there is no stated need for a feasibility experiment before design."),
            ("Phase-gate access", f"Stakeholders can approve documents at scheduled gates but cannot join frequent working sessions. Sequential requirements, design, build, and acceptance reviews match that availability."),
            ("Single handover", "The sponsor wants one controlled production handover. The case does not describe independently useful partial releases that would justify an iterative delivery program."),
            ("Testable acceptance", f"The team can specify {p['test']} before construction and use {p['artifact']} to record what was tested. Waterfall must still include early test planning."),
            ("Manageable rollout", f"With {team} and a planning window of {duration}, the team can sequence work and pilot at {pilot}. It must {p['guard']} before broader use."),
        ]
    if model == "V-shaped":
        return [
            ("Known reference", f"The environment has {p['existing']}. Approved behavior and fixtures can serve as the reference for the new {p['name']} where functions must remain equivalent."),
            ("Stable requirements", f"The functions are approved, as are the interfaces ({p['interface']}). The team can write precise, testable requirements before implementing changes."),
            ("Mandatory test mapping", f"Release policy asks for a verification method and independent evidence for every requirement. The V-shaped model pairs specifications with their corresponding tests."),
            ("High cost of an error", f"The failure concern is {p['hazard']}. A successful demonstration alone would miss the failed, unauthorized, and recovery paths that must be checked."),
            ("Early test design", f"Tests for {p['test']} can be designed alongside requirements and system design. Defects in assumptions or interfaces can be found before full integration."),
            ("Independent acceptance", f"{cap(p['users'])} and {reviewers} can inspect traceable results before rollout expands beyond {pilot}; the planning window of {duration} has a clear release gate."),
        ]
    if model == "RAD":
        return [
            ("Existing service foundation", f"The project already has {p['existing']}. It also has supported interfaces ({p['interface']}). The immediate work is an operator application, not invention of a new network protocol."),
            ("Short first-release target", f"A usable slice is required within twelve weeks, inside the larger planning window of {duration}. Rapid construction and strict prioritization matter more than exhaustive upfront screen specifications."),
            ("Reusable parts", "Approved APIs and UI components allow the team to assemble working screens quickly instead of building every layer from scratch."),
            ("Available users", f"{cap(p['users'])} can test slices in twice-weekly workshops. Their feedback can directly change the next screen or workflow."),
            ("Working application goal", f"The first release must support the goal to {p['goal']}. This is more than a throwaway demonstration for deciding whether the project should exist."),
            ("Quality still gates release", f"Before the pilot at {pilot}, the team must prove {p['test']}, access control, and integration behavior. Speed does not justify silently accepting {p['hazard']}."),
        ]
    if model == "Prototype":
        return [
            ("Unclear user needs", f"{cap(p['users'])} cannot yet specify the right controls or display for a service that will {p['goal']}. Written assumptions alone are therefore a weak production baseline."),
            ("Demonstration is the milestone", "The sponsor asks for a working model before deciding whether to fund production development. The first output is learning, not an accepted live release."),
            ("Safe user trials", f"A simulator can show candidate {p['artifact']} without changing live network state or exposing users to {p['hazard']}."),
            ("Observable feedback", "Watching users complete tasks can reveal confusing steps, missing information, and error behavior that interviews might not uncover."),
            ("Low-cost revision", "The team can alter the interaction flow while it is cheap to change, then turn findings into approved, testable requirements."),
            ("Production boundary", f"Even a persuasive demonstration does not prove {p['test']}, authorization, or recovery. A later production build must also verify the safeguard ({p['guard']})."),
        ]
    if model == "Spiral":
        return [
            ("Unproven behavior", f"Important behavior across {p['interface']} has not been shown to remain safe under failure. Full construction before checking it would be a substantial gamble."),
            ("Serious consequence", f"The dominant risk is {p['hazard']}. It can harm operations even if the ordinary success path works."),
            ("Expensive commitment", "The sponsor releases funding only after measured risk-reduction reviews. The team must avoid committing the whole budget on an untested assumption."),
            ("Experiments are possible", f"A controlled testbed can measure {p['test']} and record results in {p['artifact']} before production exposure."),
            ("Go/no-go gates", "Each loop can compare alternatives, review the evidence, and decide to proceed, redesign, or stop. The order of work follows the most consequential remaining risk."),
            ("Bounded expansion", f"Only after the leading risks are reduced should the service pilot at {pilot}, with a verified safeguard to {p['guard']}."),
        ]
    return [
        ("Known technical core", f"The team already has {p['existing']}, and the basic {p['interface']} boundary is understood. A useful release can be built without waiting for a feasibility program."),
        ("Divisible capability", f"The service can deliver a first part of the capability needed to {p['goal']}, then add independently useful functions in later cycles."),
        ("Early operational value", f"{cap(p['users'])} can use an accepted first slice at {pilot} rather than waiting for every future feature."),
        ("Real feedback", "Operators will review each release and can change priorities after seeing actual use and defects."),
        ("Regression discipline", f"Later increments must continue to pass {p['test']} and must not reintroduce {p['hazard']}."),
        ("Controlled adoption", f"A team of {team} has a planning window of {duration}. Small accepted releases allow evidence to accumulate before wider deployment."),
    ]

def advantages_v3(model: str, p: dict, c: tuple[str, ...]) -> list[tuple[str, str]]:
    org, team, duration, pilot, reviewers = c
    if model == "Waterfall":
        return [
            ("Clear requirements baseline", f"The team can approve the specific aim to {p['goal']} and reject unrelated scope changes before design starts."),
            ("Predictable reviews", f"A team of {team} can prepare requirements, design, and acceptance evidence for scheduled reviewer meetings during the planning window of {duration}."),
            ("Design before implementation", f"The known {p['interface']} can be checked against the current environment before code changes affect it."),
            ("Visible change control", f"A request to alter {p['artifact']} can be assessed against the signed statement of work, rather than quietly added halfway through construction."),
            ("Planned integration", f"The team can define how the new service uses {p['interface']} and identify dependencies before the single cutover."),
            ("Defined ownership", f"{cap(p['users'])} can own operational acceptance while {reviewers} approve the planned design and rollout evidence."),
            ("Auditable approvals", f"Records such as {p['artifact']} can show which baseline was implemented and what reviewers accepted."),
            ("Explicit failure checks", f"Tests for {p['hazard']} can be named in the acceptance plan instead of appearing as a surprise during cutover."),
            ("Controlled deployment", f"The pilot at {pilot} allows the team to {p['guard']} before one wider handover."),
            ("Objective acceptance", f"The final gate can require documented results for {p['test']}, not just a statement that the system is finished."),
        ]
    if model == "V-shaped":
        return [
            ("Testing begins with requirements", f"Acceptance cases for {p['test']} can be written before implementation, so the team knows what proof is needed."),
            ("Requirement-to-test traceability", f"Each approved function can be linked to a verification case and its result; {p['artifact']} preserve the evidence."),
            ("Stable-baseline fit", f"The starting point—{p['existing']}—gives reviewers a defensible reference for expected behavior."),
            ("Interface verification", f"Contract and integration tests can exercise the actual {p['interface']}, including rejected or failed operations."),
            ("Earlier defect discovery", f"Reviewing test designs can reveal missing cases for {p['hazard']} before the service is assembled."),
            ("Regression comparison", f"Observed results for {p['test']} can be checked against approved fixtures and existing behavior where equivalent functions are preserved."),
            ("Independent review", f"{cap(p['users'])} assess task fit while {reviewers} inspect objective evidence before sign-off."),
            ("Recovery validation", f"Failure tests can verify the safeguard ({p['guard']}) before the team risks a fleet-wide rollout."),
            ("Contained pilot", f"Starting at {pilot} limits the damage from an unexpected failure while test evidence is collected."),
            ("Unambiguous exit criteria", f"Release can depend on traceable passes for {p['test']}, authorization, failure recovery, and reviewer approval."),
        ]
    if model == "RAD":
        return [
            ("Faster usable result", f"The first time box can deliver a real operator journey for {p['users']} within twelve weeks."),
            ("Component reuse", f"Supported {p['interface']} and approved UI components avoid rebuilding infrastructure that already exists."),
            ("Frequent user feedback", f"Twice-weekly sessions let {p['users']} say whether the working application supports their tasks."),
            ("Visible progress", f"Reviewers can try a functioning slice around {p['artifact']} instead of approving only screen drawings."),
            ("Early workflow correction", "If navigation, filters, or error messages confuse users, the next construction cycle can correct them promptly."),
            ("Focused first scope", f"The team can deliver the most useful part of the work needed to {p['goal']} first and defer optional presentation features."),
            ("Integrated feedback and testing", f"User changes can be checked against {p['test']} so a nicer screen does not conceal wrong network data."),
            ("Pilot before expansion", f"A release at {pilot} exposes real use while the affected network remains limited."),
            ("Production safeguards remain", f"The team still has to prove access control and the safeguard to {p['guard']} before wider rollout."),
            ("Fast assumption checks", f"Early slices will show whether the claimed starting point—{p['existing']}—is as reusable as expected."),
        ]
    if model == "Prototype":
        return [
            ("Visible concept", f"Users can see and try candidate {p['artifact']} rather than interpreting an abstract workflow description."),
            ("Clearer requirements", f"{cap(p['users'])} can identify the controls and information needed to {p['goal']} safely."),
            ("Safe experimentation", f"A simulator lets the team explore {p['hazard']} without connecting the model to production changes."),
            ("Early usability findings", "Watching task attempts exposes confusing navigation, missing context, and unclear error recovery."),
            ("Cheap changes", "The team can revise the interaction before production integration and security hardening make changes expensive."),
            ("Feasibility demonstration", f"The sponsor can see how the proposed workflow relates to {p['interface']} before approving a build."),
            ("Stakeholder alignment", f"Successive versions of {p['artifact']} give users and funders a common object to discuss."),
            ("Failure behavior explored", f"Trials can ask how the interface should explain and recover from {p['hazard']}."),
            ("Better later acceptance criteria", f"What the team learns can become testable production requirements for {p['test']}."),
            ("Bounded first milestone", f"A team of {team} can demonstrate learning inside the planning window of {duration} without falsely calling the prototype a secure release."),
        ]
    if model == "Spiral":
        return [
            ("Risk-driven funding", "The sponsor commits more budget only after the previous loop supplies measured evidence."),
            ("Top risk addressed first", f"The team investigates {p['hazard']} before spending heavily on lower-risk features."),
            ("Evidence at every gate", f"Measured {p['test']} results inform whether the current design is safe enough to continue."),
            ("Alternatives stay open", f"Different approaches to {p['interface']} can be compared while redesign is still affordable."),
            ("Early stop or redesign", "If a testbed result is unacceptable, the team can change direction instead of defending a sunk implementation."),
            ("Incremental commitment", f"The eventual pilot at {pilot} happens only after the preceding risk reviews justify it."),
            ("Operational risk visibility", f"Every loop states whether {p['hazard']} has been reduced, transferred, or remains unresolved."),
            ("Recorded experiments", f"Versioned {p['artifact']} can preserve test conditions, measured outcomes, and decisions."),
            ("Stakeholder go/no-go decisions", f"{cap(p['users'])} and {reviewers} can judge evidence before the next investment."),
            ("Safer scale-up", f"The team proves the safeguard to {p['guard']} before exposing a larger part of the network."),
        ]
    return [
        ("Early useful capability", f"{cap(p['users'])} can use an accepted first part of the capability needed to {p['goal']} before the whole feature list is complete."),
        ("Feedback from real use", "Priorities can change in response to observed sessions, defects, and operator reports rather than predictions alone."),
        ("Flexible ordering", f"Later capabilities of the {p['name']} can be reordered without discarding an accepted first release."),
        ("Smaller integrated changes", f"Each increment can exercise the relevant interfaces ({p['interface']}) without bundling every future feature into one cutover."),
        ("Regression evidence", f"Tests for {p['test']} run again after each change, protecting earlier accepted behavior."),
        ("Operational learning", f"Reports of {p['hazard']} can guide the next iteration's engineering priorities."),
        ("Visible progress", f"Accepted increments and updated {p['artifact']} show what has actually been delivered."),
        ("Controlled adoption", f"Starting at {pilot} gives the team use data before a larger rollout."),
        ("Repeated safety checks", f"Each integrated release must preserve the safeguard to {p['guard']}."),
        ("Capacity-aware planning", f"A team of {team} can commit to a bounded slice within its planning window of {duration}, then replan using evidence."),
    ]

def steps_v3(model: str, p: dict, c: tuple[str, ...]) -> list[str]:
    pilot = c[3]
    if model == "Waterfall":
        return [
            f"Requirements: approve the functions, {p['interface']}, and measurable acceptance for {p['test']}.",
            f"Design: review {p['artifact']}, security boundaries, failure handling, and rollback before implementation.",
            f"Build and verify: implement the approved design with unit and integration tests; include scenarios involving {p['hazard']}.",
            f"Acceptance: pilot at {pilot}, verify the safeguard to {p['guard']}, and obtain the scheduled handover approval.",
        ]
    if model == "V-shaped":
        return [
            f"User requirements ↔ acceptance tests: define acceptance criteria for the ability to {p['goal']}, including {p['test']}.",
            f"System design ↔ system tests: specify end-to-end behavior across {p['interface']} under success, denial, and {p['hazard']}.",
            f"Component design ↔ integration and unit tests: map {p['artifact']} to interface, authorization, and rollback checks.",
            f"Validation: compare with approved behavior where applicable and obtain independent evidence before expansion beyond {pilot}.",
        ]
    if model == "RAD":
        return [
            f"Workshop: choose the first operator journey for {p['users']} and fix the non-negotiable security boundary.",
            f"Rapid construction: assemble a working slice from approved components and {p['interface']}.",
            f"User design cycles: demonstrate it twice weekly; revise screens while checking {p['test']}.",
            f"Cutover: run security, integration, and rollback checks before the controlled pilot at {pilot}.",
        ]
    if model == "Prototype":
        return [
            f"Learning questions: identify what {p['users']} must see and control to {p['goal']}.",
            f"Model: build a disposable simulator-backed journey around {p['artifact']}; avoid live changes.",
            f"Trials: observe users, unclear decisions, and responses to {p['hazard']}; revise the model.",
            f"Requirements: document the validated workflow and later production tests for {p['test']}.",
        ]
    if model == "Spiral":
        return [
            f"Set loop objectives: rank {p['hazard']} and compare plausible designs for {p['interface']}.",
            f"Run the highest-value testbed experiment for {p['test']}; record conditions in {p['artifact']}.",
            "At the funding gate, proceed, redesign, or stop based on measured risk reduction.",
            f"Repeat for the next dominant risk; pilot at {pilot} only after an approved go/no-go review.",
        ]
    return [
        f"Deliver a small, accepted first capability that lets {p['users']} begin to {p['goal']}.",
        f"Review real use at {pilot}; reprioritize based on feedback and {p['hazard']} reports.",
        f"Add functions in later cycles while updating {p['artifact']} and rerunning {p['test']} checks.",
        "Accept each integrated increment before wider rollout; keep earlier functions working.",
    ]

def answer_v3(model: str, p: dict, c: tuple[str, ...]) -> str:
    reason_items = reasons_v3(model, p, c)
    advantage_items = advantages_v3(model, p, c)
    assert len(reason_items) == 6 and len(advantage_items) == 10
    decision_summary = {
        "Waterfall": "the scope is signed, integration behavior is familiar, stakeholders review at phase gates, and one controlled handover is required",
        "V-shaped": "approved behavior and stable interfaces allow early test design, while every requirement needs independent verification evidence",
        "RAD": "the first usable release is urgent, approved components can be reused, and users can review working slices twice weekly",
        "Prototype": "the first milestone is to learn the workflow through a safe working model before committing to production development",
        "Spiral": "unproven integration behavior creates a serious risk, and further funding depends on measured risk-reduction gates",
        "Iterative": "the technical foundation is known, useful capabilities can be released in stages, and real use will guide later priorities",
    }[model]
    decision = (
        f"**Decision:** The Prototype model is the best primary life cycle for the first milestone of this {p['name']}. "
        f"A working model should clarify the user workflow before production requirements are fixed. "
        f"A later release will still have to prove {p['test']} and address {p['hazard']}."
        if model == "Prototype" else
        f"**Decision:** The {model} model is the best primary life cycle for this {p['name']}. "
        f"The choice follows the stated delivery conditions and the need to prove {p['test']}; "
        f"it is not based merely on the fact that the application uses {p['interface']}."
    )
    acceptance = (
        f"**Acceptance:** For the prototype milestone, observe {p['users']} using the simulated workflow and record which controls, information, and error paths need revision. "
        f"Turn the findings into later production criteria for {p['test']}, authorization, recovery, and the safeguard ({p['guard']})."
        if model == "Prototype" else
        f"**Acceptance:** Record results for {p['test']}, denied and failed operations, {p['hazard']}, and the safeguard to {p['guard']}. "
        f"{cap(p['users'])} should assess task fit; {c[4]} should review evidence before expansion beyond {c[3]}."
    )
    outcome = (
        f"For this {p['name']}, immediate success means a tested, revised workflow and an agreed production requirements baseline; "
        f"it does not mean the live service has already passed {p['test']}."
        if model == "Prototype" else
        f"For the {p['name']}, success requires measured evidence for {p['test']} without leaving {p['hazard']} unmanaged."
    )
    lines = [
        f"# Most Appropriate Life Cycle Model: {model}",
        decision,
        "## Why this model is appropriate",
    ]
    lines.extend(f"{n}. **{title}.** {explanation}" for n, (title, explanation) in enumerate(reason_items, 1))
    lines.append(f"## Applying {model} to this project")
    lines.extend(f"{n}. {step}" for n, step in enumerate(steps_v3(model, p, c), 1))
    lines.append("## Ten advantages and their relevance to this project")
    lines.extend(f"{n}. **{title}.** {explanation}" for n, (title, explanation) in enumerate(advantage_items, 1))
    lines.append("## Why close alternatives are less suitable")
    lines.extend(f"{n}. {comparison}" for n, comparison in enumerate(alternatives(model, p), 1))
    lines.extend([
        "## Acceptance and conclusion",
        acceptance,
        f"**Conclusion:** {model} is preferred because {decision_summary}. {outcome} If those assumptions change, revisit the model choice. "
        f"The planning window of {c[2]} is a constraint, not a guaranteed delivery date.",
    ])
    return "\n\n".join(lines)

def build() -> list[dict]:
    # Split by system, not by cosmetic organization. Every model occurs in each split.
    order = sorted(range(len(PROJECTS)), key=lambda i: digest(PROJECTS[i]["name"]))
    split = {i: ("train" if rank < 18 else "validation" if rank < 21 else "test")
             for rank, i in enumerate(order)}
    rows = []
    for pi, p in enumerate(PROJECTS):
        for ci, c in enumerate(CONTEXTS):
            for model in MODELS:
                case_id = f"llm2-v3-{digest(f'{pi}:{ci}:{model}')[:16]}"
                q, a = question(model, p, c), answer_v3(model, p, c)
                rows.append({
                    "id": case_id, "task": "sdlc_selection", "label": model,
                    "split": split[pi], "question": q, "answer": a,
                    "messages": [
                        {"role": "system", "content": "Select exactly one primary SDLC model from Waterfall, V-shaped, RAD, Prototype, Spiral, and Iterative. Justify it in detail from the case facts, explain its project phases, map ten advantages to the project, compare close alternatives, and give acceptance evidence and limits. Do not invent unstated facts."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "metadata": {
                        "domain": "networking/cybersecurity", "project_family": p["name"],
                        "project_index": pi, "context_index": ci,
                        "provenance": "original_synthetic_case",
                        "review_status": "rule_and_structure_validated_not_independent_expert_gold",
                        "references_for_model_definitions": [
                            "https://www.ibm.com/think/topics/sdlc",
                            "https://www.ibm.com/think/topics/software-development",
                            "https://www.ibm.com/think/topics/rapid-application-development",
                            "https://www.cs.hmc.edu/~markk/SWE_copies/boehmspiral.pdf",
                            "https://swehb.nasa.gov/spaces/SWEHBVB/pages/32604465/SWE-028%2B-%2BVerification%2BPlanning",
                        ],
                    },
                })
    return rows

def validate(rows: list[dict]) -> dict:
    assert len(rows) == 1008
    assert Counter(r["label"] for r in rows) == {model: 168 for model in MODELS}
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["question"] for r in rows}) == len(rows)
    assert len({r["answer"] for r in rows}) == len(rows)
    project_splits = {}
    for row in rows:
        i = row["metadata"]["project_index"]
        assert project_splits.setdefault(i, row["split"]) == row["split"]
        assert row["answer"].startswith(f"# Most Appropriate Life Cycle Model: {row['label']}")
        assert row["answer"].count("## ") == 5
        assert row["answer"].count("**") >= 32
        assert all(marker in row["answer"] for marker in ("1. **", "10. **", "**Conclusion:**", "**Acceptance:**"))
        advantages = row["answer"].split("## Ten advantages and their relevance to this project\n\n", 1)[1].split("\n\n## Why close alternatives", 1)[0].split("\n\n")
        assert len(advantages) == 10 and len(set(advantages)) == 10
        assert len(row["question"].split()) >= 145
        assert len(row["answer"].split()) >= 450
    return {
        "valid": True, "total": len(rows),
        "per_model": dict(Counter(r["label"] for r in rows)),
        "per_split": dict(Counter(r["split"] for r in rows)),
        "project_families": len(project_splits),
        "min_question_words": min(len(r["question"].split()) for r in rows),
        "min_answer_words": min(len(r["answer"].split()) for r in rows),
        "validation_scope": "all records checked for structure, case grounding, uniqueness, balance and family isolation; independent expert semantic review remains pending",
    }

def main() -> None:
    rows = build()
    report = validate(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "all.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for split in ("train", "validation", "test"):
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["split"] == split), encoding="utf-8")
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sample_families = {"Waterfall": 0, "V-shaped": 6, "RAD": 4,
                       "Prototype": 15, "Spiral": 5, "Iterative": 12}
    picks = {model: next(r for r in rows if r["label"] == model
                         and r["metadata"]["project_index"] == sample_families[model]
                         and r["metadata"]["context_index"] == 2)
             for model in MODELS}
    sample = ["# One complete input and output per SDLC model", "These are exact records from the rebuilt dataset. They are synthetic and have not been independently expert adjudicated."]
    for model, row in picks.items():
        sample.extend((f"# {model} — {row['id']}", "## Input", row["question"], "## Output", row["answer"]))
    (OUT / "six_complete_examples.md").write_text("\n\n".join(sample) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
