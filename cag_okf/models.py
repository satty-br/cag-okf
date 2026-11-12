"""
Data models and schemas for OKF (Open Knowledge Format) and CAG (Cache-Augmented Generation).
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class OKFMetadata(BaseModel):
    """
    Metadata representation following the Open Knowledge Format (OKF) specification.
    Includes extensions for fine-grained LLM rule targeting and token pointers.
    """
    id: Optional[str] = None
    type: str = "concept"  # Mandatory in OKF specification (e.g., 'concept', 'rule', 'playbook')
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    target_services: List[str] = Field(default_factory=list)
    target_languages: List[str] = Field(default_factory=list)
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    valid_until: Optional[str] = None
    status: str = "APPROVED"
    token_pointer: Optional[str] = None
    content_hash: Optional[str] = None
    extra_fields: Dict[str, Any] = Field(default_factory=dict)


class OKFDocument(BaseModel):
    """
    Represents an atomic OKF document consisting of YAML frontmatter and Markdown body.
    """
    metadata: OKFMetadata
    body: str
    raw_content: str
    file_path: Optional[str] = None
    token_count: Optional[int] = None
    token_ids: Optional[List[int]] = None

    def is_active(self, current_date: Optional[str] = None) -> bool:
        """Checks if the document is active and not expired."""
        if self.metadata.status.upper() in ["DEPRECATED", "DRAFT", "ARCHIVED", "REJECTED"]:
            return False
        if self.metadata.valid_until and current_date:
            try:
                cur = datetime.strptime(current_date, "%Y-%m-%d")
                exp = datetime.strptime(self.metadata.valid_until, "%Y-%m-%d")
                if cur > exp:
                    return False
            except ValueError:
                pass
        return True


class OKFBundle(BaseModel):
    """
    A collection of OKF documents organized within a directory (LLM-wiki bundle).
    """
    name: str
    root_path: str
    index_doc: Optional[OKFDocument] = None
    documents: Dict[str, OKFDocument] = Field(default_factory=dict)

    def total_docs(self) -> int:
        return len(self.documents)


class CacheProfile(BaseModel):
    """
    Metadata manifest saved alongside the precomputed DynamicCache on disk.
    """
    model_id: str
    total_tokens: int
    document_ids: List[str]
    created_at: str
    content_hash: str
    prefix_prompt_preview: str


class BenchmarkResult(BaseModel):
    """
    Performance and token savings metrics for comparative benchmarks.
    """
    paradigm: str  # 'No-Cache Baseline', 'RAG Simulation', 'Monolithic CAG', 'OKF-CAG'
    query: str
    prefill_tokens_evaluated: int
    output_tokens_generated: int
    time_to_first_token_ms: float
    total_time_s: float
    tokens_per_second: float
    token_savings_percent: float
    response_text: str
