"""A tiny RAG pipeline for the fictional "Acme Workspace" app: 30 in-memory chunks, a lexical
retriever with scores and a template generator. It exists so the demo and tests have something real
to monitor, and it has three knobs to break it on purpose."""

from __future__ import annotations

from rag_quality_monitor.judge.fake import content_words, overlap, split_sentences
from rag_quality_monitor.models import Chunk, RetrievalResult

CORPUS: list[tuple[str, str, str]] = [
    ("h1", "projects", "To create a project, open the Projects page and press New project. Every project has an owner and a color."),
    ("h2", "projects", "To archive a project, open its settings and choose Archive. Archived projects become read-only and can be restored later."),
    ("h3", "tasks", "To add a task, open a project and press New task. A task has a title, an assignee and a due date."),
    ("h4", "tasks", "To assign a task, open the task and pick a teammate in the Assignee field. The teammate receives a notification."),
    ("h5", "tasks", "Recurring tasks repeat on a daily, weekly or monthly schedule. Set the schedule in the task's Repeat menu."),
    ("h6", "documents", "To share a document, press Share and enter a teammate's email address. You can give view or edit access."),
    ("h7", "documents", "Document versions are saved automatically every few minutes. Open version history to restore an earlier version."),
    ("h8", "documents", "To export a document as PDF, open the menu and choose Export. The PDF keeps headings and tables."),
    ("h9", "meetings", "To schedule a meeting, press New meeting and pick a date and attendees. Attendees receive a calendar invite."),
    ("h10", "meetings", "Meeting notes are created from a template when the meeting starts. Action items in the notes become tasks."),
    ("h11", "meetings", "To record a meeting summary, press Summarize after the meeting ends. The summary lists decisions and action items."),
    ("h12", "workflows", "A workflow runs steps when a trigger happens, for example when a task is marked done. Build one from the Workflows page."),
    ("h13", "workflows", "Workflow steps can send a message, create a task or update a field. Steps run in the order shown."),
    ("h14", "workflows", "To pause a workflow, open it and switch off the Active toggle. Paused workflows ignore new triggers."),
    ("h15", "search", "Use the search bar to find projects, tasks and documents. Filters narrow results by owner or date."),
    ("c1", "retention", "Deleted documents stay in the trash for 30 days before permanent removal. Admins can restore items during that period."),
    ("c2", "retention", "Audit logs are retained for 365 days. Older log entries are removed automatically."),
    ("c3", "access", "Workspace admins manage member roles on the Members page. Roles are Owner, Admin, Member and Guest."),
    ("c4", "access", "Guests can only see projects they are invited to. Guests cannot create workflows or export documents."),
    ("c5", "security", "Two-factor authentication can be required for all members. Admins turn it on under Security settings."),
    ("c6", "security", "Sessions expire after 12 hours of inactivity. Members must sign in again after a session expires."),
    ("c7", "data", "Workspace data is encrypted at rest and in transit. Encryption keys are rotated every 90 days."),
    ("c8", "data", "Data export requests are handled by admins from the Privacy page. The export contains projects, tasks and documents in JSON format."),
    ("c9", "privacy", "A member can ask for their personal data to be erased. Admins review the request and confirm erasure within 30 days."),
    ("c10", "privacy", "Meeting recordings are stored for 90 days by default. Admins can shorten the period under Privacy settings."),
    ("c11", "sso", "Single sign-on supports SAML identity providers. Members who sign in through SSO skip the password form."),
    ("c12", "sharing", "External sharing links can be disabled for the whole workspace. When disabled, existing public links stop working."),
    ("c13", "billing", "Invoices are issued monthly and sent to the billing contact. Billing contacts are set by the Owner."),
    ("c14", "backup", "Backups are taken every night and kept for 35 days. Restoring a backup is a request to support."),
    ("c15", "incident", "Security incidents are announced to admins by email within 72 hours. The notice lists affected data and next steps."),
]  # fmt: skip

HALLUCINATION = "Every plan includes unlimited phone support for premium customers."


class AcmeRag:
    """Knobs (all off by default):
    - flatten_scores: squeeze scores into a narrow band. Ranking is unchanged, so the right chunk is
      still in the top k, but the retriever can no longer tell good from bad. Hook 1 sees it.
    - drop_chunk_ids: remove chunks from the index, as a bad re-indexing might. Hook 3 sees it.
    - hallucinate: the generator adds one sentence that no chunk supports. Hook 2 sees it.
    """

    def __init__(
        self,
        flatten_scores: bool = False,
        drop_chunk_ids: frozenset[str] | set[str] = frozenset(),
        hallucinate: bool = False,
        latency_ms: float = 12.0,
    ) -> None:
        self.flatten_scores = flatten_scores
        self.hallucinate = hallucinate
        self.latency_ms = latency_ms
        self.corpus = [row for row in CORPUS if row[0] not in drop_chunk_ids]

    async def retrieve(self, tenant_id: str, query: str, k: int = 5) -> RetrievalResult:
        """Score = share of the query's content words found in the chunk. Zero-score chunks are not returned."""
        scored = [(overlap(query, text), cid, doc, text) for cid, doc, text in self.corpus if content_words(text)]
        scored = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:k]
        chunks = [
            Chunk(id=cid, doc_id=doc, text=text, score=0.5 + 0.02 * s if self.flatten_scores else s)
            for s, cid, doc, text in scored
        ]
        return RetrievalResult(query=query, chunks=chunks, k=k, latency_ms=self.latency_ms)

    async def generate(self, question: str, chunks: list[Chunk]) -> str:
        """Template answer: the first sentence of each of the top two chunks."""
        sentences = [split_sentences(c.text)[0] for c in chunks[:2]]
        if self.hallucinate:
            sentences.append(HALLUCINATION)
        return " ".join(sentences) if sentences else "I could not find anything about that."


def build_pipeline():
    """Used by `python -m rag_quality_monitor.suite`."""
    rag = AcmeRag()
    return rag.retrieve, rag.generate
