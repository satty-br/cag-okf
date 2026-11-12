"""
Unit tests for AI Integration Endpoints and MCP tools.
"""

import unittest
from pathlib import Path
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import app


class TestAIIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_v1_context_endpoint(self):
        # Request context for pagamentos-service in Go
        payload = {
            "service": "pagamentos-service",
            "language": "golang",
            "tags": ["database"]
        }
        res = self.client.post("/v1/context", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("WIKI-RULE-001", data["document_ids"])
        self.assertIn("<organizational_knowledge_base", data["context"])

    def test_v1_chat_completions_structure(self):
        # Test OpenAI compatible payload structure
        payload = {
            "model": "cag-okf",
            "messages": [
                {"role": "system", "content": "You are a code reviewer."},
                {"role": "user", "content": "Is direct sql.Open permitted?"}
            ],
            "max_tokens": 15
        }
        res = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertIn("choices", data)
        self.assertIn("usage", data)
        self.assertIn("prompt_tokens", data["usage"])
        # Should report low prompt tokens due to CAG
        self.assertLess(data["usage"]["prompt_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
