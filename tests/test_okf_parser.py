"""
Unit tests for OKFParser.
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cag_okf.okf_parser import OKFParser
from cag_okf.models import OKFDocument


SAMPLE_DOC_TEXT = """---
id: "TEST-001"
type: "rule"
title: "Test Security Standard"
category: "security"
target_services: ["auth-svc", "gateway"]
target_languages: ["golang", "python"]
status: "APPROVED"
tags: ["auth", "jwt"]
---

# Test Heading
This is the document body text for testing OKF parsing.
"""


class TestOKFParser(unittest.TestCase):

    def test_parse_text(self):
        doc = OKFParser.parse_text(SAMPLE_DOC_TEXT, file_path="test.md")
        self.assertEqual(doc.metadata.id, "TEST-001")
        self.assertEqual(doc.metadata.type, "rule")
        self.assertEqual(doc.metadata.title, "Test Security Standard")
        self.assertIn("auth-svc", doc.metadata.target_services)
        self.assertIn("jwt", doc.metadata.tags)
        self.assertEqual(doc.metadata.status, "APPROVED")
        self.assertIn("This is the document body text", doc.body)
        self.assertIsNotNone(doc.metadata.content_hash)

    def test_parse_bundle(self):
        sample_dir = Path(__file__).parent.parent / "examples" / "sample_okf_bundle"
        bundle = OKFParser.parse_bundle(str(sample_dir))
        self.assertGreaterEqual(bundle.total_docs(), 4)
        self.assertIn("WIKI-RULE-001", bundle.documents)
        self.assertIsNotNone(bundle.index_doc)


if __name__ == "__main__":
    unittest.main()
