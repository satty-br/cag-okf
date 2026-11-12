"""
Python SDK Client for AI Agents and External Applications.
Allows any AI script, LangChain tool, CrewAI agent, or bot to connect to the
CAG-OKF in-memory server and retrieve organizational context and rules.
"""

import json
from typing import Optional, List, Dict, Any
import urllib.request
import urllib.error
import urllib.parse


class CAGOKFClient:
    """
    Lightweight, zero-dependency client to connect AI agents to CAG-OKF.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        req_data = json.dumps(data).encode("utf-8") if data else None

        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Could not connect to CAG-OKF server at {self.base_url}: {e}")

    def get_context(
        self,
        service: Optional[str] = None,
        language: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Retrieves pre-composed XML/Markdown context for prompt injection into external LLMs.
        Example: Inject directly into <organizational_knowledge_base> in system prompt.
        """
        payload = {
            "service": service,
            "language": language,
            "tags": tags or []
        }
        res = self._request("POST", "/v1/context", payload)
        return res.get("context", "")

    def search_rules(
        self,
        query: Optional[str] = None,
        service: Optional[str] = None,
        tag: Optional[str] = None,
        status: Optional[str] = "APPROVED"
    ) -> List[Dict[str, Any]]:
        """
        Searches active rules in the in-memory cache.
        """
        params = []
        if query:
            params.append(f"search={urllib.parse.quote(query)}")
        if service:
            params.append(f"service={urllib.parse.quote(service)}")
        if tag:
            params.append(f"tag={urllib.parse.quote(tag)}")
        if status:
            params.append(f"status={urllib.parse.quote(status)}")

        qs = f"?{'&'.join(params)}" if params else ""
        return self._request("GET", f"/api/documents{qs}")

    def get_rule(self, rule_id: str) -> Dict[str, Any]:
        """
        Retrieves a single rule and its full body and frontmatter by ID.
        """
        return self._request("GET", f"/api/documents/{urllib.parse.quote(rule_id)}")

    def ask(self, question: str, max_tokens: int = 40) -> Dict[str, Any]:
        """
        Queries the in-memory CAG Key-Value cache directly with zero prefill overhead.
        """
        payload = {"query": question, "max_tokens": max_tokens, "use_cache": True}
        return self._request("POST", "/api/query", payload)
