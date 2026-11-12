"""
Fast In-Memory Cache Store for OKF Documents and Pre-tokenized Token Pointers.
Provides sub-microsecond O(1) in-memory retrieval, inverted indexes by service/tag,
thread-safe concurrent updates, and dynamic dirty-tracking for KV cache compilation.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import torch

from cag_okf.models import OKFDocument, OKFMetadata, CacheProfile
from cag_okf.okf_parser import OKFParser
from cag_okf.cache_manager import CAGCacheManager


class FastInMemoryCache:
    """
    Ultra-low latency in-memory document repository with indexed lookup
    and integrated DynamicCache lifecycle management.
    """

    def __init__(self, tokenizer: Optional[Any] = None):
        self._lock = threading.RLock()
        self.tokenizer = tokenizer

        # Document storage: ID -> OKFDocument
        self._documents: Dict[str, OKFDocument] = {}

        # Inverted index mappings for O(1) multi-attribute queries
        self._index_by_service: Dict[str, Set[str]] = {}
        self._index_by_language: Dict[str, Set[str]] = {}
        self._index_by_tag: Dict[str, Set[str]] = {}
        self._index_by_status: Dict[str, Set[str]] = {}
        self._index_by_type: Dict[str, Set[str]] = {}

        # Compiled Key-Value Attention states resident in memory (Tier 2)
        self._kv_cache: Optional[Any] = None
        self._kv_prefix_ids: Optional[torch.Tensor] = None
        self._kv_profile: Optional[CacheProfile] = None
        self._is_dirty: bool = True

        # Telemetry & performance metrics
        self._read_count: int = 0
        self._write_count: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._last_rebuild_timestamp: float = 0.0
        self._last_rebuild_duration_ms: float = 0.0

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Assigns tokenizer for background token counting."""
        with self._lock:
            self.tokenizer = tokenizer
            # Re-tokenize existing documents if needed
            for doc in self._documents.values():
                if doc.token_ids is None and self.tokenizer is not None:
                    self._tokenize_doc(doc)

    def _tokenize_doc(self, doc: OKFDocument) -> None:
        """Precomputes token IDs and token count for an OKF document."""
        if self.tokenizer is None:
            return
        formatted = f"--- Document: {doc.metadata.id} ({doc.metadata.title}) ---\n{doc.body}\n"
        token_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        doc.token_count = len(token_ids)
        doc.token_ids = token_ids

    def _update_indexes_for_doc(self, doc_id: str, meta: OKFMetadata) -> None:
        """Inserts document ID into inverted indexes."""
        # Services
        for svc in meta.target_services:
            self._index_by_service.setdefault(svc.lower(), set()).add(doc_id)

        # Languages
        for lang in meta.target_languages:
            self._index_by_language.setdefault(lang.lower(), set()).add(doc_id)

        # Tags
        for tag in meta.tags:
            self._index_by_tag.setdefault(tag.lower(), set()).add(doc_id)

        # Status
        status_key = (meta.status or "APPROVED").upper()
        self._index_by_status.setdefault(status_key, set()).add(doc_id)

        # Type
        type_key = (meta.type or "concept").lower()
        self._index_by_type.setdefault(type_key, set()).add(doc_id)

    def _remove_from_indexes(self, doc_id: str) -> None:
        """Removes document ID from all inverted indexes."""
        for idx in [
            self._index_by_service,
            self._index_by_language,
            self._index_by_tag,
            self._index_by_status,
            self._index_by_type,
        ]:
            for entries in idx.values():
                entries.discard(doc_id)

    def put(
        self,
        doc: OKFDocument,
        persist_to_disk: bool = False,
        file_path: Optional[str] = None
    ) -> OKFDocument:
        """
        Stores or updates a document in memory with sub-microsecond latency.
        Updates inverted indexes and marks KV cache as dirty.
        """
        doc_id = doc.metadata.id or f"doc_{int(time.time() * 1000)}"
        doc.metadata.id = doc_id

        # Compute hash if missing
        if not doc.metadata.content_hash:
            doc.metadata.content_hash = OKFParser.compute_hash(doc.raw_content or doc.body)

        with self._lock:
            # Tokenize if tokenizer is available
            if doc.token_ids is None and self.tokenizer is not None:
                self._tokenize_doc(doc)

            # Clean previous index entries if updating
            if doc_id in self._documents:
                self._remove_from_indexes(doc_id)

            self._documents[doc_id] = doc
            self._update_indexes_for_doc(doc_id, doc.metadata)
            self._is_dirty = True
            self._write_count += 1

            target_path = file_path or doc.file_path
            if persist_to_disk and target_path:
                doc.file_path = str(Path(target_path).resolve())
                Path(target_path).parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(doc.raw_content)

            return doc

    def get(self, doc_id: str) -> Optional[OKFDocument]:
        """Sub-microsecond document retrieval by ID."""
        with self._lock:
            self._read_count += 1
            doc = self._documents.get(doc_id)
            if doc:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            return doc

    def delete(self, doc_id: str, remove_from_disk: bool = False) -> bool:
        """Evicts a document from in-memory cache and optional disk removal."""
        with self._lock:
            doc = self._documents.pop(doc_id, None)
            if not doc:
                return False

            self._remove_from_indexes(doc_id)
            self._is_dirty = True
            self._write_count += 1

            if remove_from_disk and doc.file_path:
                p = Path(doc.file_path)
                if p.exists():
                    p.unlink()

            return True

    def list_all(self) -> List[OKFDocument]:
        """Returns all documents currently cached in memory."""
        with self._lock:
            self._read_count += 1
            return list(self._documents.values())

    def filter_docs(
        self,
        service: Optional[str] = None,
        language: Optional[str] = None,
        tag: Optional[str] = None,
        status: Optional[str] = None,
        doc_type: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[OKFDocument]:
        """
        Fast indexed query using set intersections in memory.
        """
        with self._lock:
            self._read_count += 1
            candidate_sets: List[Set[str]] = []

            if service:
                candidate_sets.append(self._index_by_service.get(service.lower(), set()))
            if language:
                candidate_sets.append(self._index_by_language.get(language.lower(), set()))
            if tag:
                candidate_sets.append(self._index_by_tag.get(tag.lower(), set()))
            if status:
                candidate_sets.append(self._index_by_status.get(status.upper(), set()))
            if doc_type:
                candidate_sets.append(self._index_by_type.get(doc_type.lower(), set()))

            if candidate_sets:
                matching_ids = set.intersection(*candidate_sets)
                docs = [self._documents[d_id] for d_id in matching_ids if d_id in self._documents]
            else:
                docs = list(self._documents.values())

            # Text search filter
            if search_query:
                q = search_query.lower()
                docs = [
                    d for d in docs
                    if q in (d.metadata.title or "").lower()
                    or q in (d.metadata.id or "").lower()
                    or q in d.body.lower()
                    or any(q in t.lower() for t in d.metadata.tags)
                ]

            return docs

    def load_from_directory(self, bundle_dir: str) -> int:
        """
        Loads an entire directory of OKF Markdown files into the in-memory cache.
        """
        bundle = OKFParser.parse_bundle(bundle_dir)
        count = 0
        with self._lock:
            for doc_id, doc in bundle.documents.items():
                self.put(doc, persist_to_disk=False)
                count += 1
            print(f"[FastInMemoryCache] Loaded {count} documents into RAM from {bundle_dir}.", file=sys.stderr)
        return count

    def set_kv_cache(
        self,
        cache: Any,
        prefix_ids: torch.Tensor,
        profile: CacheProfile,
        duration_ms: float = 0.0
    ) -> None:
        """Stores precomputed Key-Value attention states resident in RAM."""
        with self._lock:
            self._kv_cache = cache
            self._kv_prefix_ids = prefix_ids
            self._kv_profile = profile
            self._is_dirty = False
            self._last_rebuild_timestamp = time.time()
            self._last_rebuild_duration_ms = duration_ms

    def get_kv_cache(self) -> Tuple[Optional[Any], Optional[torch.Tensor], Optional[CacheProfile]]:
        """Returns the precomputed Key-Value attention cache and metadata."""
        with self._lock:
            return self._kv_cache, self._kv_prefix_ids, self._kv_profile

    def is_dirty(self) -> bool:
        """Checks if memory documents have changed since last KV cache compilation."""
        with self._lock:
            return self._is_dirty

    def get_stats(self) -> Dict[str, Any]:
        """Provides real-time telemetry on cache health, memory, and hit rates."""
        with self._lock:
            total_docs = len(self._documents)
            total_tokens = sum(d.token_count or 0 for d in self._documents.values())
            kv_tokens = self._kv_prefix_ids.shape[1] if self._kv_prefix_ids is not None else 0

            # Approximation of memory footprint
            body_bytes = sum(len(d.raw_content.encode("utf-8")) for d in self._documents.values())
            tensor_bytes = 0
            if self._kv_prefix_ids is not None:
                tensor_bytes += self._kv_prefix_ids.nelement() * self._kv_prefix_ids.element_size()

            total_ops = self._read_count + self._write_count
            hit_ratio = (self._cache_hits / (self._cache_hits + self._cache_misses)) if (self._cache_hits + self._cache_misses) > 0 else 1.0

            return {
                "total_documents": total_docs,
                "total_pretokenized_tokens": total_tokens,
                "kv_cache_tokens": kv_tokens,
                "kv_cache_active": self._kv_cache is not None,
                "is_dirty": self._is_dirty,
                "read_operations": self._read_count,
                "write_operations": self._write_count,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_ratio_percent": round(hit_ratio * 100, 2),
                "estimated_ram_kb": round((body_bytes + tensor_bytes) / 1024, 2),
                "last_rebuild_duration_ms": round(self._last_rebuild_duration_ms, 2),
                "services_indexed": list(self._index_by_service.keys()),
                "tags_indexed": list(self._index_by_tag.keys()),
            }
