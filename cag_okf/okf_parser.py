"""
OKF (Open Knowledge Format) Parser and Pre-tokenizer.
Parses Markdown files with YAML frontmatter into validated OKF structures,
computes deterministic cryptographic hashes, and manages token pointers.
"""

import os
import re
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import yaml

from cag_okf.models import OKFMetadata, OKFDocument, OKFBundle


FRONTMATTER_REGEX = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


class OKFParser:
    """
    Parser for Open Knowledge Format documents and bundles.
    """

    @staticmethod
    def compute_hash(text: str) -> str:
        """Computes SHA-256 hash of text for caching and invalidation checks."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def parse_text(cls, content: str, file_path: Optional[str] = None) -> OKFDocument:
        """
        Parses raw text containing optional YAML frontmatter and Markdown body.
        """
        content_hash = cls.compute_hash(content)
        match = FRONTMATTER_REGEX.match(content)

        if match:
            yaml_block, body = match.groups()
            try:
                data = yaml.safe_load(yaml_block) or {}
            except Exception as e:
                data = {"title": "Parse Error", "extra_fields": {"yaml_error": str(e)}}
        else:
            # Document without frontmatter: treat as generic OKF concept
            yaml_block = ""
            body = content
            first_line = body.strip().split("\n")[0] if body.strip() else "Untitled"
            clean_title = re.sub(r"^#+\s*", "", first_line).strip()
            data = {"type": "concept", "title": clean_title}

        if not isinstance(data, dict):
            data = {"type": "concept", "title": "Malformed Frontmatter"}

        # Extract standard known fields
        known_keys = {
            "id", "type", "title", "description", "tags", "category",
            "target_services", "target_languages", "approved_by",
            "approved_at", "valid_until", "status", "token_pointer"
        }
        meta_dict = {k: v for k, v in data.items() if k in known_keys}
        extra_fields = {k: v for k, v in data.items() if k not in known_keys}

        # Normalize list fields
        for list_field in ["tags", "target_services", "target_languages"]:
            val = meta_dict.get(list_field)
            if isinstance(val, str):
                meta_dict[list_field] = [x.strip() for x in val.split(",") if x.strip()]
            elif not isinstance(val, list):
                meta_dict[list_field] = []

        if not meta_dict.get("id") and file_path:
            meta_dict["id"] = Path(file_path).stem

        meta_dict["content_hash"] = content_hash
        meta_dict["extra_fields"] = extra_fields

        metadata = OKFMetadata(**meta_dict)
        return OKFDocument(
            metadata=metadata,
            body=body.strip(),
            raw_content=content,
            file_path=file_path,
        )

    @classmethod
    def parse_file(cls, file_path: str) -> OKFDocument:
        """
        Reads and parses a single file into an OKFDocument.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return cls.parse_text(content, file_path=str(path.resolve()))

    @classmethod
    def parse_bundle(cls, directory_path: str, bundle_name: Optional[str] = None) -> OKFBundle:
        """
        Recursively scans a directory of OKF Markdown files and constructs an OKFBundle.
        """
        root = Path(directory_path)
        if not root.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        name = bundle_name or root.name
        bundle = OKFBundle(name=name, root_path=str(root.resolve()))

        index_doc = None
        for file in root.rglob("*.md"):
            try:
                doc = cls.parse_file(str(file))
                doc_id = doc.metadata.id or file.stem

                # Check if this file serves as the bundle index
                if file.name.lower() in ["index.md", "readme.md"] and file.parent == root:
                    index_doc = doc

                bundle.documents[doc_id] = doc
            except Exception as e:
                print(f"[OKFParser Warning] Skipping {file}: {e}")

        bundle.index_doc = index_doc
        return bundle

    @classmethod
    def pretokenize_bundle(
        cls,
        bundle: OKFBundle,
        tokenizer: Any,
        cache_dir: Optional[str] = None
    ) -> OKFBundle:
        """
        Pre-computes token IDs for all documents in the bundle.
        Enables token pointers for rapid zero-tokenizer handoffs.
        """
        token_manifest: Dict[str, Any] = {}

        for doc_id, doc in bundle.documents.items():
            formatted_text = f"--- Document: {doc_id} ({doc.metadata.title}) ---\n{doc.body}\n"
            token_ids = tokenizer.encode(formatted_text, add_special_tokens=False)
            doc.token_count = len(token_ids)
            doc.token_ids = token_ids

            if cache_dir:
                token_manifest[doc_id] = {
                    "token_count": len(token_ids),
                    "token_ids": token_ids,
                    "content_hash": doc.metadata.content_hash,
                }

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            manifest_path = Path(cache_dir) / f"{bundle.name}_token_pointers.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(token_manifest, f)

        return bundle
