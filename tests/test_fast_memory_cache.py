"""
Unit tests for FastInMemoryCache.
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cag_okf.models import OKFDocument, OKFMetadata, CacheProfile
from cag_okf.fast_memory_cache import FastInMemoryCache


class TestFastInMemoryCache(unittest.TestCase):

    def setUp(self):
        self.cache = FastInMemoryCache()

    def test_put_get_and_indexes(self):
        meta = OKFMetadata(
            id="WIKI-TEST-001",
            type="rule",
            title="Database Pool Standard",
            status="APPROVED",
            target_services=["pagamentos-service", "checkout-api"],
            target_languages=["golang"],
            tags=["database", "vault"]
        )
        doc = OKFDocument(
            metadata=meta,
            body="Use internal dbpool package.",
            raw_content="---\nid: WIKI-TEST-001\n---\nUse internal dbpool package."
        )

        self.cache.put(doc)
        self.assertTrue(self.cache.is_dirty())

        # Direct retrieval O(1)
        retrieved = self.cache.get("WIKI-TEST-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.metadata.title, "Database Pool Standard")

        # Indexed filter by service
        docs_by_svc = self.cache.filter_docs(service="pagamentos-service")
        self.assertEqual(len(docs_by_svc), 1)
        self.assertEqual(docs_by_svc[0].metadata.id, "WIKI-TEST-001")

        # Filter by nonexistent service
        docs_empty = self.cache.filter_docs(service="unknown-service")
        self.assertEqual(len(docs_empty), 0)

        # Filter by tag
        docs_by_tag = self.cache.filter_docs(tag="vault")
        self.assertEqual(len(docs_by_tag), 1)

    def test_delete_and_stats(self):
        meta = OKFMetadata(id="DOC-TO-DELETE", title="Temporary Rule")
        doc = OKFDocument(metadata=meta, body="Temp", raw_content="Temp")
        self.cache.put(doc)

        self.assertEqual(len(self.cache.list_all()), 1)
        deleted = self.cache.delete("DOC-TO-DELETE")
        self.assertTrue(deleted)
        self.assertEqual(len(self.cache.list_all()), 0)

        stats = self.cache.get_stats()
        self.assertEqual(stats["total_documents"], 0)
        self.assertGreaterEqual(stats["write_operations"], 2)


if __name__ == "__main__":
    unittest.main()
