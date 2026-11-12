"""
Deterministic OKF Knowledge Composer.
Filters, resolves, and stitches modular OKF documents into clean prefix contexts
for Cache-Augmented Generation prefilling.
"""

from typing import List, Optional, Set
import hashlib
from cag_okf.models import OKFBundle, OKFDocument


DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an expert AI assistant with embedded organizational knowledge.\n"
    "Strictly adhere to the corporate guidelines, standards, and exceptions defined in the "
    "preloaded knowledge base below when reviewing code, architecture, or answering queries.\n"
)


class OKFComposer:
    """
    Composes selected OKF documents into an optimized prompt prefix.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_INSTRUCTION,
        include_manifest: bool = True
    ):
        self.system_prompt = system_prompt
        self.include_manifest = include_manifest

    def filter_documents(
        self,
        bundle: OKFBundle,
        target_services: Optional[List[str]] = None,
        target_languages: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        current_date: Optional[str] = None
    ) -> List[OKFDocument]:
        """
        Deterministically filters documents from the bundle matching specified criteria.
        Excludes inactive, draft, or expired rules.
        """
        filtered: List[OKFDocument] = []
        svc_set: Set[str] = set(s.lower() for s in (target_services or []))
        lang_set: Set[str] = set(l.lower() for l in (target_languages or []))
        tag_set: Set[str] = set(t.lower() for t in (tags or []))
        type_set: Set[str] = set(tp.lower() for tp in (types or []))

        for doc_id, doc in bundle.documents.items():
            # Check status and expiration
            if not doc.is_active(current_date):
                continue

            # Don't duplicate index in rule list
            if bundle.index_doc and doc.metadata.id == bundle.index_doc.metadata.id:
                continue

            meta = doc.metadata

            # Type check
            if type_set and meta.type.lower() not in type_set:
                continue

            # Service check
            if svc_set:
                doc_svcs = set(s.lower() for s in meta.target_services)
                # If document defines target services, it must match at least one
                if doc_svcs and not (doc_svcs & svc_set):
                    continue

            # Language check
            if lang_set:
                doc_langs = set(l.lower() for l in meta.target_languages)
                if doc_langs and not (doc_langs & lang_set):
                    continue

            # Tag check
            if tag_set:
                doc_tags = set(t.lower() for t in meta.tags)
                if doc_tags and not (doc_tags & tag_set):
                    continue

            filtered.append(doc)

        # Sort deterministically by id for reproducible token sequences
        filtered.sort(key=lambda d: d.metadata.id or "")
        return filtered

    def compose_prefix(
        self,
        bundle: OKFBundle,
        selected_docs: Optional[List[OKFDocument]] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Constructs the structured context prefix string to be prefilled into the KV cache.
        """
        docs = selected_docs if selected_docs is not None else list(bundle.documents.values())
        docs = [d for d in docs if not (bundle.index_doc and d.metadata.id == bundle.index_doc.metadata.id)]
        docs.sort(key=lambda d: d.metadata.id or "")

        lines: List[str] = []
        sys_text = custom_instructions or self.system_prompt
        lines.append(sys_text.strip())
        lines.append("\n<organizational_knowledge_base format=\"OKF-v0.2\">")

        if self.include_manifest:
            lines.append("  <bundle_manifest>")
            lines.append(f"    <bundle_name>{bundle.name}</bundle_name>")
            lines.append(f"    <active_documents_count>{len(docs)}</active_documents_count>")
            lines.append("    <table_of_contents>")
            for d in docs:
                m = d.metadata
                lines.append(
                    f"      - {m.id}: {m.title or 'Untitled'} (type: {m.type}, status: {m.status})"
                )
            lines.append("    </table_of_contents>")
            lines.append("  </bundle_manifest>\n")

        # Document bodies
        for d in docs:
            m = d.metadata
            tags_str = ", ".join(m.tags) if m.tags else "none"
            lines.append(
                f"  <document id=\"{m.id}\" type=\"{m.type}\" title=\"{m.title}\" tags=\"{tags_str}\">"
            )
            for line in d.body.splitlines():
                lines.append(f"    {line}")
            lines.append("  </document>\n")

        lines.append("</organizational_knowledge_base>\n")
        return "\n".join(lines)

    @staticmethod
    def compute_prefix_hash(prefix_text: str) -> str:
        """Computes deterministic hash of composed prefix."""
        return hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()
