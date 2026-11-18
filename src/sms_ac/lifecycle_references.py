"""Short attributed research notes, not copied source documents or expert labels."""
from .retrieval import CorpusIndex

REFERENCES = [
    {'id':'method-agile-principles', 'title':'Agile principles: feedback and changing requirements',
     'source_url':'https://agilemanifesto.org/principles.html',
     'text':'Research summary: Agile principles emphasize frequent useful software delivery, responsiveness to requirement changes, close business/developer collaboration, sustainable work and regular process reflection. These are principles, not proof that a specific project should use Scrum.'},
    {'id':'method-spiral-sei', 'title':'Spiral: uncertainty, risk reduction and incremental commitment',
     'source_url':'https://www.sei.cmu.edu/library/spiral-development-experience-principles-and-refinements-spiral-development-workshop-february-9-2000/',
     'text':'Research summary: The SEI report describes spiral as repeated development activities with active risk reduction. It discusses incremental funding commitments and anchor-point milestones. Repeating development phases alone does not establish a risk-driven spiral process.'},
    {'id':'method-sequential-sebok', 'title':'Sequential and Vee development: stable requirements, verification and validation',
     'source_url':'https://sebokwiki.org/wiki/Sequential_Development_Approach',
     'text':'Research summary: Sequential development suits well-defined stable requirements and explicit deliverables; late changes can be expensive. The Vee connects definition activities with corresponding verification and stakeholder validation. SEBoK distinguishes the development-stage Vee from a complete lifecycle, so operation, maintenance and retirement still need planning.'},
]


def retrieve_lifecycle_context(document: str) -> str:
    """Rank the small reference library, retaining alternatives to avoid one-model bias."""
    hits = CorpusIndex(REFERENCES).search(document, limit=len(REFERENCES), split=None)
    ordered = hits + [r for r in REFERENCES if r['id'] not in {h['id'] for h in hits}]
    return ('Reference notes checked 2026-09-10; summaries are not full standards or project facts. '
            'Coverage is limited; infer applicability from document evidence.\n\n' +
            '\n\n'.join(f"[{r['id']}] {r['title']}\nSource: {r['source_url']}\n{r['text']}" for r in ordered))
