"""Build paired, explicitly weak-supervision datasets; never label templates expert data."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sms_ac.prompts import TASK_PROMPTS, CLARIFICATION_SYSTEM_PROMPT
from planning_examples import discussion, planning_sections
from sms_ac.retrieval import CorpusIndex

# Domain-specific contracts, entities and failure modes; not just project-name substitutions.
DOMAINS = [
 ('DNS resolver', 'resolve names with TTL-aware positive and negative caching', 'DNS UDP/TCP queries', 'Query, CacheEntry, Upstream', 'expired or poisoned answers', 'validate response identity; expire entries at TTL; test negative caching and upstream timeout'),
 ('VPN access manager', 'authorize remote sessions and revoke access', 'identity provider and VPN gateway control API', 'User, Device, AccessPolicy, Session', 'stale authorization or orphaned tunnels', 'test revoked identities and active-session termination; scope gateway credentials'),
 ('Firewall policy manager', 'review, simulate and stage policy changes', 'device configuration API', 'Rule, ChangeRequest, Approval, Deployment', 'rules that disconnect management access', 'dry run rules; preserve an out-of-band recovery path; rehearse rollback'),
 ('Intrusion alert service', 'correlate authorized network telemetry into analyst alerts', 'authenticated telemetry intake', 'Signal, RuleVersion, Alert, Case', 'false positives and missed observations', 'retain evidence IDs; measure precision on reviewed fixtures; never claim perfect detection'),
 ('Traffic observability', 'aggregate flow summaries and expose traffic trends', 'flow exporter intake and dashboard API', 'Device, FlowWindow, Aggregate, Alert', 'collector backlog and clock skew', 'track event and ingestion time; mark stale windows; test duplicate flow reports'),
 ('DDoS response coordinator', 'detect traffic anomalies and coordinate approved mitigation', 'monitoring feeds and provider mitigation API', 'Baseline, Incident, MitigationRequest', 'blocking legitimate traffic during mitigation', 'require bounded mitigation duration; test false alarms and rollback with simulated traffic'),
 ('Router configuration service', 'validate, version and stage router configurations', 'NETCONF or confirmed vendor API', 'Device, CandidateConfig, Commit, Rollback', 'partial fleet rollout or loss of connectivity', 'canary representative devices; compare intended and observed state; test failed commit recovery'),
 ('Group chat service', 'deliver messages and synchronize reconnecting clients', 'HTTPS API and WebSocket sessions', 'User, Room, Membership, Message, Cursor', 'duplicate messages and missed reconnect events', 'authorize room membership per operation; use stable message IDs and bounded history synchronization'),
 ('File synchronization service', 'synchronize authorized file versions across clients', 'HTTPS chunk upload and change-feed API', 'File, Chunk, Version, ChangeCursor', 'conflicting edits or interrupted uploads', 'checksum chunks; expose conflict versions; resume uploads without replacing unverified content'),
 ('Webhook delivery service', 'deliver tenant events with retries and operator replay', 'outbound HTTPS and administration API', 'Tenant, Subscription, Event, Attempt', 'unreachable receivers and cross-tenant disclosure', 'enforce SSRF destination restrictions; sign deliveries; bound retries and dead-letter failed events'),
 ('IoT telemetry gateway', 'buffer and normalize sensor observations', 'MQTT or validated device transport', 'Sensor, Observation, Buffer, Ack', 'intermittent links and full device buffers', 'size buffers from rate times outage duration times record size; document overflow policy'),
 ('API gateway', 'route authenticated client requests with per-tenant limits', 'HTTPS upstream APIs', 'Route, Tenant, Quota, RequestTrace', 'retry amplification and cascading upstream failures', 'apply deadline propagation and retry budgets; test per-tenant isolation and circuit breaking'),
 ('Video session signaling', 'establish authorized real-time media sessions', 'HTTPS signaling and negotiated WebRTC media', 'Participant, Room, Session, RelayAllocation', 'NAT traversal failure and degraded connectivity', 'test relay fallback and reconnect; separate signaling state from media transport assumptions'),
 ('Service discovery registry', 'register services and resolve healthy endpoints', 'registration heartbeat and lookup API', 'Service, Instance, Lease, HealthCheck', 'stale endpoints during partitions', 'expire leases explicitly; distinguish failure suspicion from proof; document consistency choices'),
 ('Network backup coordinator', 'schedule and verify authorized device backup snapshots', 'device export API and object storage', 'Device, Snapshot, Checksum, RestoreJob', 'unrestorable backups or exposed credentials', 'encrypt snapshots; verify checksums and restore rehearsals; restrict export scope'),
 ('Campus Wi-Fi portal', 'issue authorized network sessions after enrollment', 'HTTPS portal and network-access controller', 'Identity, Enrollment, Session, Consent', 'session abuse and controller outage', 'limit session scope and duration; test expiration and accessible login; define outage behavior'),
 ('Message broker administration', 'manage queues, subscriptions and access policies', 'broker administration and metrics APIs', 'Queue, Subscription, Policy, Offset', 'unauthorized topic access or destructive retention changes', 'preview retention changes; isolate tenant policies; test redelivery and offset recovery'),
 ('Network inventory service', 'reconcile authorized device discovery with owned inventory', 'approved discovery adapters and inventory API', 'Asset, Observation, Owner, Reconciliation', 'duplicate assets or incorrect ownership inference', 'record provenance and confidence; require reviewed merges; bound discovery scope and rate'),
]
PROFILES = [
 ('feedback', 'Requirements will evolve after weekly operator demonstrations. Users are available throughout development. Useful slices can be released independently.', 'Iterative and incremental with Agile reviews', 'Frequent feedback and independently useful slices justify repeated delivery and revised requirements.', 'review a working slice with users every iteration', 'if users become unavailable or contractual baselines change'),
 ('assurance', 'Interfaces are baselined. Every approved requirement needs a mapped verification test and independent acceptance evidence before deployment.', 'V-model with controlled increments', 'Stable interfaces and explicit verification obligations justify paired specification and test activities.', 'pair each requirement with verification and acceptance evidence', 'if interfaces become uncertain enough to require feasibility prototypes'),
 ('risk', 'Core integration feasibility is unresolved. Budget is released at risk-reduction milestones. Experiments must test uncertain technical assumptions before scope expands.', 'Risk-driven spiral', 'Technical uncertainty and risk-linked milestones favor explicit experiments before commitment.', 'identify top risks, run bounded experiments and review exit evidence', 'if feasibility is demonstrated and ordinary incremental delivery becomes sufficient'),
 ('stable', 'Requirements and acceptance criteria are signed and stable. The team has implemented this integration before. Stakeholders can attend only formal stage reviews.', 'Waterfall with explicit verification and change control', 'A stable known integration and stage-based stakeholder access support planned sequential baselines.', 'baseline requirements and review design, implementation and acceptance at agreed gates', 'if material requirements change or integration risk invalidates the baseline'),
 ('discovery', 'Workflows are unclear and must be explored with disposable prototypes. Users can test interface concepts before committing to production requirements.', 'Prototyping followed by iterative development', 'Unknown workflows require learning before production architecture and requirements are committed.', 'test disposable concepts, record findings, then rebuild validated production increments', 'if prototypes resolve workflow uncertainty and the team can use normal release planning'),
 ('phased', 'Core interfaces are understood. Four independently useful capabilities must be released in priority order. Feedback is available at the end of each capability.', 'Incremental development', 'Independent capabilities and staged value favor integrated capability releases with verified interfaces.', 'deliver one complete capability at a time with regression tests across earlier increments', 'if cross-component dependencies prevent independently useful releases'),
 ('mixed', 'The operator interface changes after user trials, while device-changing actions require a frozen contract and independent verification before release.', 'Iterative primary lifecycle with V-model verification gates', 'Different uncertainty and assurance constraints require iterative discovery with explicit verification of device-changing interfaces.', 'iterate on workflows and gate device-changing functions on independent test evidence', 'if all components acquire uniform assurance or uncertainty constraints'),
]
DEPLOYMENTS = ['a centrally managed campus', 'a multi-branch enterprise', 'a hybrid cloud environment', 'an edge environment with intermittent uplinks']
SCALES = [50, 200, 1000, 5000]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def split_for(group):
    bucket = int(digest(group)[:8], 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def design_report(domain, profile, deployment, scale, variant, evidence):
    name, capability, interface, entities, failure, control = domain
    _, conditions, _, _, _, _ = profile
    cadence = [15, 30, 60][variant]
    return f'''# {name}: software design

## Scope and decision context
Build a {name.lower()} for {deployment} serving {scale} managed endpoints. The primary capability is to {capability}. This is a proposed design, not an implemented or certified system. Endpoint count is supplied; real request rate, payload distribution, team experience and hosting capacity remain unknown.
Delivery evidence: {conditions}

## Assumptions and questions
Assume an identity provider and persistent storage can be provisioned. Confirm device capabilities, stakeholders, budget, retention obligations, support ownership and acceptable outage windows. Proposed polling or refresh interval is {cadence} seconds where this fits the domain; validate it with operators instead of treating it as a protocol requirement. Ask which users approve acceptance tests and which interfaces already exist.

## Functional requirements and acceptance
FR1: Authorized users can {capability}. Test a representative success scenario and denied-access scenario for every supported role.
FR2: The system validates input against versioned contracts before committing state. Malformed records receive actionable errors and do not alter accepted state.
FR3: Operators can inspect processing status and safely retry failed work. Repeating an operation with the same stable identifier must not create duplicate durable state.
FR4: Administrative changes retain actor, timestamp, prior version and outcome. Test audit retrieval and prevention of unauthorized changes.

## Nonfunctional requirements and capacity
NFR1: Preserve tenant/site access boundaries across storage, API and background jobs. Verify cross-scope denial with integration tests.
NFR2: Propose 95 percent of administrative queries within one second under an agreed load profile, and 99.5 percent monthly service availability. These are review targets, not measured performance.
NFR3: Define recovery time, recovery point, event rate and payload size before capacity sign-off. For periodic observations, estimated events per second are endpoint count divided by interval; event throughput alone does not establish byte or CPU capacity.
NFR4: Record dependency failure and stale data explicitly. No silent success or unbounded buffers.

## Architecture and responsibilities
Use a modular control API, domain adapter, bounded durable work queue, processing worker and transactional state store. A small team can deploy the API as one service; separate workers isolate slow dependencies. The adapter implements {interface}. The worker validates and deduplicates work; the state store enforces unique operation IDs. A read view serves authorized operational queries. Reassess this decomposition if latency or protocol constraints require a different data plane.

## Data flow and interface contract
An authenticated actor submits a domain operation. The API authorizes scope, validates schema and persists its identifier before asynchronous work. The worker invokes the allowed integration, commits the result and updates observable status. Acknowledgement follows durable acceptance; external side effects need reconciliation after ambiguous timeouts.
Use GET /v1/operations/{{id}} for status and POST /v1/operations for administrative requests where applicable. Require an idempotency key and expected version for mutable operations. Return 400 for malformed input, 401 for missing authentication, 403 for denied scope, 409 for version conflict, 429 for quota and 503 for dependency failure. Domain traffic uses {interface}, rather than assuming every protocol operation is HTTP.

## Data model and retention
Primary domain records: {entities}. Each record has stable ID, scope, version and lifecycle timestamps where appropriate. Operation records link requested changes to results and audit evidence. Use uniqueness constraints to deduplicate acknowledged work; provide reconciliation for external effects. Confirm retention and deletion policies per entity; do not store raw secrets in business records. Backups require restoration tests and access controls.

## Network deployment and security
Place integration workers in approved network segments. Restrict ingress to authenticated users and egress to approved dependencies. Use encrypted authenticated transport where supported, scoped credentials, input validation and secret references. Threats include forged requests, stolen integration credentials, cross-scope access and resource exhaustion. Apply rate limits and least privilege. Domain-specific control: {control}.

## Failure behavior and tradeoffs
The main domain hazard is {failure}. Set bounded deadlines, exponential backoff with jitter and retry budgets; quarantine persistent failures for review. At-least-once delivery requires idempotent state transitions. It does not prove exactly-once external delivery. Display stale or partial status and reconcile uncertain results. Size outage buffers from measured rate, record size and outage duration; define overflow rather than assuming infinite storage.
A synchronous integration is simpler but couples user latency to dependency failures. A queue isolates outages but adds delay and operations cost. Validate the proposed choice against domain timing constraints before implementation. Avoid unnecessary microservices until independent scaling or ownership justifies them.

## Validation and traceability
FR1 maps to domain-adapter acceptance and authorization tests. FR2 maps to schema and invalid-input tests. FR3 maps to duplicate/retry and failure-injection tests. FR4 maps to immutable audit and actor-trace tests. NFR1 maps to cross-scope access tests; NFR2 and NFR3 require measured load and recovery rehearsals; NFR4 requires overload and stale-data checks. Include domain-specific tests: {control}.

## Delivery risks and review
{conditions} Begin by confirming assumptions and representative integration fixtures. Promote a pilot only after acceptance, security, failure and recovery evidence is reviewed. Revisit targets when real workload or stakeholder availability differs from assumptions. No schedule, certification or stakeholder approval is implied.

## Reference context and limits
{evidence}
The supplied excerpt is background context; it does not validate every proposed design choice. Verify applicability and current implementation status before adopting a referenced proposal.
'''


def lifecycle_report(domain, profile, deployment, scale):
    key, conditions, recommendation, reason, action, trigger = profile
    return f'''# {domain[0]}: lifecycle recommendation

## Recommendation and confidence
Use **{recommendation}** for this design, provisionally. {reason} Confidence is qualitative and conditional; no empirical optimality score is available. The project concerns {scale} managed endpoints in {deployment}, but scale alone does not determine its lifecycle.

## Evidence from the design
The Scope and decision context section states: {conditions}
The Delivery risks and review section repeats these delivery constraints. FR1 through FR4 require functional, authorization, retry and audit evidence. NFR1 through NFR4 require access-boundary, measured load, recovery and overload checks. The Failure behavior section identifies {domain[4]}; this hazard needs specific experiments and verification regardless of lifecycle choice.

## Alternatives and tradeoffs
Waterfall fits stable known requirements and formal stage reviews, but late discovery can cause expensive rework. V-model makes requirement-to-test correspondence explicit, but uncertain interfaces need exploratory work before a baseline is credible. Iterative and incremental development handles feedback and staged value, but requires a stable review cadence and disciplined regression testing. Risk-driven spiral prioritizes uncertainty and risk experiments, but its formal review overhead may exceed the needs of routine integrations. Prototyping discovers workflows, but a disposable prototype should not silently become a production system.
For the supplied evidence, {reason.lower()} Reconsider these alternatives if the evidence changes rather than treating them as universally inferior. A hybrid must name its primary lifecycle and the verification gates it borrows.

## Execution plan and exit criteria
1. Discovery: confirm scope, stakeholders, integration capabilities, critical failure modes and unknown targets. Exit when the owner approves testable acceptance examples and unresolved assumptions have named owners.
2. Design and risk reduction: prototype the uncertain integration or workflow and specify data, API and security boundaries. Exit when the prototype addresses its stated hypothesis and critical interfaces have executable tests. Do not report a risk retired without evidence.
3. Construction: {action}. Deliver domain functionality with automated tests, audit and observable failure states. Exit each planned unit only after its acceptance examples pass and remaining defects are reviewed.
4. System validation: run contract, authorization, load, retry, overload and restore tests mapped to FR/NFR IDs. Compare results to agreed targets, not to invented benchmarks. Independent acceptance is required where the design says so.
5. Deployment and maintenance: rehearse rollback, pilot a representative segment and confirm the operational owner. Expand only after the agreed acceptance review. Track incidents and update requirements and risk records.

## Supporting engineering practices
Use version control, peer review, continuous integration, secret scanning, automated tests and controlled delivery throughout. DevOps and DevSecOps support the lifecycle; they are not mutually exclusive standalone lifecycle alternatives. Security and operational gates belong in every increment or stage that affects them.

## Missing evidence and reassessment
Team experience, budget, contractual details, exact assurance obligations and workload measurements remain unspecified. Ask stakeholders to clarify those before committing effort estimates or delivery dates. Reassess {trigger}. Novel hazards can justify spiral risk cycles; mandatory independent verification can strengthen V-model gates. This recommendation is a reasoned proposal, not a guarantee.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/dual')
    args = parser.parse_args()
    corpus = ROOT/'data/corpus/chunks.jsonl'
    rows = [json.loads(line) for line in corpus.open(encoding='utf-8') if line.strip()]
    index = CorpusIndex(rows)
    references = {domain[0]: index.search(domain[0]+' '+domain[1], limit=1, split='train') for domain in DOMAINS}
    examples = {'design': [], 'sdlc': []}
    for domain, profile, deployment, scale, variant in itertools.product(DOMAINS, PROFILES, DEPLOYMENTS, SCALES, range(3)):
        group = f'scenario:{domain[0]}:{profile[0]}'
        split = split_for(group)
        hits = references[domain[0]]
        ref = hits[0] if hits else None
        evidence = ('['+ref['id']+'] '+ref['source_url']+'\nExcerpt: '+' '.join(ref['text'].split()[:75])) if ref else 'No applicable source excerpt was retrieved; verify the proposed design independently.'
        request = f'Design a {domain[0].lower()} for {deployment} with {scale} managed endpoints. It must {domain[1]}. {profile[1]} Focus review variant {variant+1}: '+['failure recovery and audit','interfaces and access boundaries','capacity assumptions and verification'][variant]+'.'
        transcript = discussion(request, domain, profile, variant)
        design = design_report(domain, profile, deployment, scale, variant, evidence) + planning_sections(domain, variant)
        recommendation = lifecycle_report(domain, profile, deployment, scale)
        identity = digest(request)[:20]
        metadata = {'supervision':'synthetic_template_weak_supervision','review_status':'requires_expert_review','domain':domain[0], 'decision_profile':profile[0], 'source_ids':[ref['id']] if ref else [], 'source_urls':[ref['source_url']] if ref else [], 'source_licenses':[ref['license']] if ref else [], 'attribution':[ref['attribution']] if ref else [], 'generator_version':'1.0','note':'Rule-authored targets; structural variations are not independent expert judgments.'}
        metadata['generator_version'] = '2.0'
        for stage_index in range((len(transcript)-1)//2):
            history = transcript[:2 + stage_index * 2]
            examples['design'].append({'id':f'clarify-{identity}-{stage_index}', 'task':'design', 'group_id':group,
                'messages':[{'role':'system','content':CLARIFICATION_SYSTEM_PROMPT}, *history],
                'metadata':{**metadata,'stage':'clarification'}, '_split':split})
        for task, payload, target in [('design', {'request':request,'reference_material':evidence,'clarification_transcript':transcript,'provisional':True}, design), ('sdlc', {'software_design':design,'reference_material':''}, recommendation)]:
            examples[task].append({'id':task+'-'+identity,'task':task,'group_id':group,'messages':[{'role':'system','content':TASK_PROMPTS[task]},{'role':'user','content':json.dumps(payload,ensure_ascii=False)},{'role':'assistant','content':target}], 'metadata':metadata, '_split':split})
    summary = {'kind':'weak_supervision_not_expert_dataset','unique_scenario_families':len(DOMAINS)*len(PROFILES),'domains':len(DOMAINS),'counts':{},'limitations':['Targets are rule-authored templates, not gold labels.','Family splitting does not remove shared report skeletons; synthetic test loss is not real-world accuracy.','Reference snippets use training corpus only and are reused; references do not validate all generated claims.','Independent expert review and diverse natural documents are required before quality claims.']}
    for task, records in examples.items():
        random.Random(42).shuffle(records)
        for split in ['train','validation','test']:
            folder=args.output/task
            folder.mkdir(parents=True,exist_ok=True)
            selected=[{k:v for k,v in row.items() if k!='_split'} for row in records if row['_split']==split]
            (folder/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected),encoding='utf-8')
            summary['counts'][task+'/'+split]=len(selected)
    summary['total_rows']=sum(summary['counts'].values())
    (args.output/'dataset_manifest.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
