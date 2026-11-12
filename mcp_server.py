"""
Model Context Protocol (MCP) Server for CAG-OKF.
Enables AI assistants (Cursor, Claude Desktop, Antigravity, Windsurf) to connect
via stdio and fetch organizational context, active rules, and zero-prefill CAG answers.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure cag-okf is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from cag_okf.fast_memory_cache import FastInMemoryCache
from cag_okf.okf_parser import OKFParser
from cag_okf.composer import OKFComposer
from cag_okf.engine import OKFCAGEngine

# Global in-memory cache
cache = FastInMemoryCache()
bundle_dir = Path(__file__).parent / "examples" / "sample_okf_bundle"
cache.load_from_directory(str(bundle_dir))
engine: Optional[OKFCAGEngine] = None


def get_engine() -> OKFCAGEngine:
    global engine
    if engine is None:
        engine = OKFCAGEngine(model_id="HuggingFaceTB/SmolLM2-135M-Instruct", lazy_load=False)
        cache.set_tokenizer(engine.tokenizer)
        engine.load_bundle(str(bundle_dir))
        engine.build_cache()
    return engine


# --- MCP Tool Handlers ---

def tool_search_rules(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = arguments.get("query")
    service = arguments.get("service")
    tag = arguments.get("tag")

    docs = cache.filter_docs(service=service, tag=tag, search_query=query, status="APPROVED")
    results = [
        {
            "id": d.metadata.id,
            "type": d.metadata.type,
            "title": d.metadata.title,
            "target_services": d.metadata.target_services,
            "tags": d.metadata.tags,
            "token_count": d.token_count or 0,
        }
        for d in docs
    ]
    return [{"type": "text", "text": json.dumps(results, indent=2, ensure_ascii=False)}]


def tool_get_rule(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    rule_id = arguments.get("rule_id", "")
    doc = cache.get(rule_id)
    if not doc:
        return [{"type": "text", "text": f"Error: Rule '{rule_id}' not found in in-memory knowledge base."}]

    out = {
        "metadata": doc.metadata.model_dump(),
        "body": doc.body,
        "token_count": doc.token_count or 0
    }
    return [{"type": "text", "text": json.dumps(out, indent=2, ensure_ascii=False)}]


def tool_get_context(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    service = arguments.get("service")
    language = arguments.get("language")
    tags = arguments.get("tags") or []

    docs = cache.list_all()
    bundle = OKFParser.parse_bundle(str(bundle_dir))
    for d in docs:
        if d.metadata.id:
            bundle.documents[d.metadata.id] = d

    composer = OKFComposer()
    selected = composer.filter_documents(
        bundle,
        target_services=[service] if service else None,
        target_languages=[language] if language else None,
        tags=tags if tags else None
    )

    prefix_text = composer.compose_prefix(bundle, selected_docs=selected)
    return [{
        "type": "text",
        "text": f"Matched {len(selected)} rules for service='{service}', lang='{language}':\n\n{prefix_text}"
    }]


def tool_ask_cag(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    question = arguments.get("question", "")
    eng = get_engine()
    answer, metrics = eng.generate(question, max_new_tokens= arguments.get("max_tokens", 40), use_cache=True)

    result_text = (
        f"Answer:\n{answer}\n\n"
        f"--- Token Performance Metrics ---\n"
        f"Prefill Tokens Evaluated: {metrics.prefill_tokens_evaluated}\n"
        f"Token Savings:           {metrics.token_savings_percent:.1f}%\n"
        f"Latency:                 {metrics.total_time_s:.2f}s"
    )
    return [{"type": "text", "text": result_text}]


def tool_create_or_update_rule(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    rule_id = arguments.get("rule_id")
    title = arguments.get("title", "Untitled Rule")
    body = arguments.get("body", "")
    target_services = arguments.get("target_services", [])
    target_languages = arguments.get("target_languages", [])
    tags = arguments.get("tags", [])
    status = arguments.get("status", "APPROVED")
    valid_until = arguments.get("valid_until")

    yaml_lines = [
        "---",
        f"id: \"{rule_id}\"",
        f"type: \"rule\"",
        f"title: \"{title}\"",
        f"status: \"{status}\"",
    ]
    if target_services:
        yaml_lines.append(f"target_services: {target_services}")
    if target_languages:
        yaml_lines.append(f"target_languages: {target_languages}")
    if tags:
        yaml_lines.append(f"tags: {tags}")
    if valid_until:
        yaml_lines.append(f"valid_until: \"{valid_until}\"")
    yaml_lines.append("---\n")

    raw_content = "\n".join(yaml_lines) + body
    target_path = str(bundle_dir / "rules" / f"{rule_id}.md")
    doc = OKFParser.parse_text(raw_content, file_path=target_path)
    stored = cache.put(doc, persist_to_disk=True, file_path=target_path)

    return [{"type": "text", "text": f"Success: Rule '{stored.metadata.id}' saved in memory and disk ({stored.token_count or 0} tokens)."}]


def tool_append_rule_exception(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    rule_id = arguments.get("rule_id", "")
    exception_text = arguments.get("exception_text", "")
    approved_by = arguments.get("approved_by", "ai-assistant")

    existing = cache.get(rule_id)
    if not existing:
        return [{"type": "text", "text": f"Error: Rule '{rule_id}' not found."}]

    appended_body = (
        existing.body +
        f"\n\n## Exceção Aprovada ({approved_by})\n"
        f"> Registrado via MCP em {existing.metadata.valid_until or 'vigente'}\n\n"
        f"{exception_text}\n"
    )

    yaml_lines = [
        "---",
        f"id: \"{rule_id}\"",
        f"type: \"{existing.metadata.type}\"",
        f"title: \"{existing.metadata.title or ''}\"",
        f"status: \"{existing.metadata.status}\"",
    ]
    if existing.metadata.target_services:
        yaml_lines.append(f"target_services: {existing.metadata.target_services}")
    if existing.metadata.target_languages:
        yaml_lines.append(f"target_languages: {existing.metadata.target_languages}")
    if existing.metadata.tags:
        yaml_lines.append(f"tags: {existing.metadata.tags}")
    if existing.metadata.valid_until:
        yaml_lines.append(f"valid_until: \"{existing.metadata.valid_until}\"")
    yaml_lines.append("---\n")

    raw_content = "\n".join(yaml_lines) + appended_body
    target_path = existing.file_path or str(bundle_dir / "rules" / f"{rule_id}.md")
    updated = OKFParser.parse_text(raw_content, file_path=target_path)
    stored = cache.put(updated, persist_to_disk=True, file_path=target_path)

    return [{"type": "text", "text": f"Success: Exception appended to rule '{rule_id}'. Total tokens: {stored.token_count}."}]


def tool_delete_rule(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    rule_id = arguments.get("rule_id", "")
    deleted = cache.delete(rule_id, remove_from_disk=True)
    if not deleted:
        return [{"type": "text", "text": f"Error: Rule '{rule_id}' not found."}]
    return [{"type": "text", "text": f"Success: Rule '{rule_id}' deleted from memory and disk."}]


TOOLS = [
    {
        "name": "okf_search_rules",
        "description": "Searches corporate security rules, exceptions, and playbooks in the in-memory OKF knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search in title, body, or tags"},
                "service": {"type": "string", "description": "Target service name (e.g. 'pagamentos-service')"},
                "tag": {"type": "string", "description": "Specific tag (e.g. 'database', 'vault', 'rate-limit')"}
            }
        }
    },
    {
        "name": "okf_get_rule",
        "description": "Retrieves the full markdown specification, approved exceptions, and LLM directives of an OKF rule by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string", "description": "The unique rule ID (e.g., 'WIKI-RULE-001')"}
            },
            "required": ["rule_id"]
        }
    },
    {
        "name": "okf_get_context_for_target",
        "description": "Composes a tailored, filtered OKF organizational context block ready for prompt injection into code reviews or PR checks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Microservice name being inspected (e.g. 'pagamentos-service')"},
                "language": {"type": "string", "description": "Programming language (e.g. 'golang')"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Specific tags to match"}
            }
        }
    },
    {
        "name": "okf_ask_knowledge_base",
        "description": "Queries the in-memory Cache-Augmented Generation (CAG) engine directly with 97%+ token compute savings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural language question about organizational rules or exceptions"},
                "max_tokens": {"type": "integer", "description": "Max new tokens to generate (default: 40)"}
            },
            "required": ["question"]
        }
    },
    {
        "name": "okf_create_or_update_rule",
        "description": "Creates a new organizational security rule or updates an existing one in memory and disk.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string", "description": "Unique identifier (e.g. 'WIKI-RULE-005')"},
                "title": {"type": "string", "description": "Human-readable title"},
                "body": {"type": "string", "description": "Full markdown specification"},
                "target_services": {"type": "array", "items": {"type": "string"}},
                "target_languages": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["APPROVED", "DRAFT", "DEPRECATED"]},
                "valid_until": {"type": "string", "description": "YYYY-MM-DD expiration date"}
            },
            "required": ["rule_id", "title", "body"]
        }
    },
    {
        "name": "okf_append_rule_exception",
        "description": "Appends an approved exception or security waiver directly to an existing rule.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string", "description": "ID of rule to modify"},
                "exception_text": {"type": "string", "description": "Description of the exception or waiver"},
                "approved_by": {"type": "string", "description": "Identity of the approver / ticket reference"}
            },
            "required": ["rule_id", "exception_text"]
        }
    },
    {
        "name": "okf_delete_rule",
        "description": "Deletes an obsolete rule from memory and disk.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string", "description": "ID of rule to delete"}
            },
            "required": ["rule_id"]
        }
    }
]


# --- JSON-RPC 2.0 Dispatcher ---

def handle_request(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "cag-okf-mcp-server",
                    "version": "0.2.0"
                }
            }
        }

    elif method == "notifications/initialized":
        return None  # No response needed for notifications

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "okf_search_rules":
            content = tool_search_rules(arguments)
        elif tool_name == "okf_get_rule":
            content = tool_get_rule(arguments)
        elif tool_name == "okf_get_context_for_target":
            content = tool_get_context(arguments)
        elif tool_name == "okf_ask_knowledge_base":
            content = tool_ask_cag(arguments)
        elif tool_name == "okf_create_or_update_rule":
            content = tool_create_or_update_rule(arguments)
        elif tool_name == "okf_append_rule_exception":
            content = tool_append_rule_exception(arguments)
        elif tool_name == "okf_delete_rule":
            content = tool_delete_rule(arguments)
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": content
            }
        }

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not supported: {method}"}
        }


def main():
    """Stdio loop for MCP clients."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            if res is not None:
                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()
        except Exception as e:
            err_res = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
