"""Transparent synthetic planning examples; these are not expert-reviewed gold."""

def discussion(request, domain, profile, variant):
    followups = {
        'feedback': 'Who can accept each demonstrated increment, and which device capability is still unverified?',
        'assurance': 'Who independently reviews acceptance evidence, and which verification obligations need clarification?',
        'risk': f'What evidence would reduce the risk of {domain[4]}, and who decides whether the next funding milestone is met?',
        'stable': 'Who approves changes to the baseline, and which deployment window has been confirmed?',
        'discovery': 'Which operator workflow should a disposable prototype test first, and how will users judge it?',
        'phased': 'Who decides capability priority, and which dependency could prevent an independently useful release?',
        'mixed': 'Which device-changing operations need independent acceptance, and who owns the interface baseline?',
    }
    questions = [
        'Who operates and approves this system, and what actions may each role perform?',
        'Which deadline, team capacity and budget are confirmed? Unknown is a valid answer.',
        followups[profile[0]],
    ]
    answers = [
        'Propose operators for scoped daily actions, administrators for policy changes, and independent reviewers for audit. These role assignments still need approval.',
        ('We have four engineers allocated and want a twelve-week pilot. Budget approval and effective weekly availability are still unknown. Treat the deadline as a goal, not a feasibility result.' if variant==1 else
         'The budget and team capacity are unknown. Please use a conditional timeline and identify what must be estimated before committing a delivery date.'),
        profile[1] + ' Device capabilities and representative workload measurements still need confirmation.',
    ]
    turns = [{'role':'user','content':request}]
    for question, answer in zip(questions, answers):
        turns.extend([{'role':'assistant','content':question}, {'role':'user','content':answer}])
    if variant==2:
        turns[-1]['content'] += ' Reviewers must be read-only, but reviewers should also edit operational policies.'
        turns.extend([
            {'role':'assistant','content':'Those reviewer permissions conflict: read-only access excludes editing operational policies. Should policy editing belong to a separate administrator role, while reviewers inspect evidence?'},
            {'role':'user','content':'Yes. Keep reviewers read-only; administrators propose changes and an independent approver authorizes sensitive changes. This is a proposed role design pending stakeholder review.'},
        ])
    return turns


def planning_sections(domain, variant):
    emphasis = ['failure recovery and audit', 'interfaces and access boundaries', 'capacity assumptions and verification'][variant]
    schedule = ('''The discussion supplies four allocated engineers and a twelve-week pilot goal. Proposed calendar: weeks 1–2 clarify scope and acceptance; weeks 3–4 test feasibility and review preliminary interfaces; weeks 5–8 construct a narrow pilot capability; weeks 9–10 validate security, load and recovery; weeks 11–12 stabilize and hand over. This is an illustrative allocation, not an effort-derived forecast. Critical dependencies are test-device access, accepted interfaces and timely operator review. Effective engineer availability, task estimates and contingency are unresolved; reduce pilot scope or renegotiate the goal if estimates exceed capacity.''' if variant==1 else
                '''A numerical timeline is currently unsupported because neither a deadline nor effective team capacity was supplied. Obtain these inputs and estimate task ranges before assigning calendar dates.''')
    return f'''
## Role-based access and ownership proposals
These roles are proposals from the discussion, not approved staffing assignments. An operator may perform authorized daily actions only in assigned sites or tenants. An administrator may propose policy and integration changes; sensitive changes require an independent approver. A reviewer may read scoped audit evidence but may not alter operational state. A service owner accepts priorities and residual risks. Confirm whether staffing permits separation of duties; document compensating review if it does not.
Role tests: reject cross-site operator requests, prevent reviewer writes, and require recorded approval before a sensitive change. Map these tests to FR1, FR4 and NFR1. The current review emphasis is {emphasis}.

## Objectives, priorities and exclusions
Success means demonstrating the primary capability, {domain[1]}, on representative authorized fixtures, with FR1–FR4 acceptance evidence and agreed NFR targets. Essential: scoped authorization, correct primary behavior, observable failures and audit for sensitive operations. Desirable: saved views and convenient operator filtering after essential acceptance. Optional: cosmetic themes. Deferred: autonomous actions outside the agreed capability and unverified device families. These are proposed priorities; the owner must resolve conflicting stakeholder needs. Deferred does not mean intrinsically unimportant.

## Feasibility and decision gates
Technical feasibility is conditional on a representative {domain[2]} integration experiment and a measured workload; {domain[4]} is a specific risk to investigate. Operational feasibility requires named support ownership, access to a test environment, runbooks and restore rehearsal. Economic feasibility is undetermined because budget, labor rates, hosting/storage costs and support costs are unknown. Estimate cost as effort by role times agreed rates plus infrastructure, licenses, migration, training and support; do not invent a total. Schedule feasibility is undetermined until capacity, interface access and acceptance obligations are confirmed. Record each assumption with an owner and review date; do not treat missing cost evidence as feasibility approval.

## Provisional phases, milestones and timeline
The following are dependency-based planning work packages, not a selected SDLC or promised calendar schedule. Discovery produces agreed scope, role permissions and an unknowns register; exit requires review of acceptance examples. Feasibility work produces integration and workload evidence; exit requires explicit treatment of critical risks. Preliminary design produces interface contracts and requirement-to-test mapping; exit requires review of security and recovery assumptions. Construction produces testable capabilities; exit requires their acceptance tests. Validation and pilot produce measured functional, security, load and restore evidence. Handover produces runbooks, training and a named operational owner.
Estimate each package with the responsible team using optimistic, likely and pessimistic effort ranges. Convert effort to elapsed time only after availability and parallel dependencies are known. Integration access precedes credible capacity estimates; accepted interfaces precede dependable implementation estimates. Report the critical dependency path, contingency assumptions and milestone dates once those inputs exist. {schedule} The second specialist selects how these packages repeat or progress through a lifecycle.

## Resources, maintenance and open decisions
Required responsibilities include requirements facilitation, networking integration, application engineering, security review, test/acceptance and operations. These are responsibilities, not invented headcount. Confirm role assignments, budget owner, delivery deadline, service targets, retention, approved devices and stakeholder acceptance authority. Maintain requirement IDs through implementation and tests. After pilot acceptance, assign incident response, patch review, backup verification and periodic permission review; set their cadence with the service owner.
'''
