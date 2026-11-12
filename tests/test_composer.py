"""
Unit tests for OKFComposer.
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cag_okf.okf_parser import OKFParser
from cag_okf.composer import OKFComposer


class TestOKFComposer(unittest.TestCase):

    def setUp(self):
        sample_dir = Path(__file__).parent.parent / "examples" / "sample_okf_bundle"
        self.bundle = OKFParser.parse_bundle(str(sample_dir))
        self.composer = OKFComposer()

    def test_filter_by_service(self):
        # Only pagamentos-service rules
        docs = self.composer.filter_documents(
            self.bundle,
            target_services=["pagamentos-service"]
        )
        doc_ids = [d.metadata.id for d in docs]
        self.assertIn("WIKI-RULE-001", doc_ids)
        self.assertIn("WIKI-RULE-003", doc_ids)
        self.assertNotIn("WIKI-RULE-002", doc_ids)

    def test_filter_by_language(self):
        docs = self.composer.filter_documents(
            self.bundle,
            target_languages=["typescript"]
        )
        doc_ids = [d.metadata.id for d in docs]
        self.assertIn("WIKI-RULE-002", doc_ids)
        self.assertNotIn("WIKI-RULE-001", doc_ids)

    def test_compose_prefix(self):
        docs = self.composer.filter_documents(self.bundle, target_services=["pagamentos-service"])
        prefix = self.composer.compose_prefix(self.bundle, selected_docs=docs)
        self.assertIn("<organizational_knowledge_base", prefix)
        self.assertIn("WIKI-RULE-001", prefix)
        self.assertIn("</organizational_knowledge_base>", prefix)


if __name__ == "__main__":
    unittest.main()
