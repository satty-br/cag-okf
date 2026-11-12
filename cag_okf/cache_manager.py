"""
CAG (Cache-Augmented Generation) Key-Value Cache Manager.
Manages precomputation of Key-Value states into HuggingFace DynamicCache,
disk serialization/deserialization, atomic cloning, and rollback slicing.
"""

import os
import copy
from pathlib import Path
from typing import Tuple, Optional, Any, Dict
import torch
from transformers import DynamicCache

from cag_okf.models import CacheProfile


class CAGCacheManager:
    """
    Orchestrates the lifecycle of Key-Value (KV) caches for Cache-Augmented Generation.
    """

    @staticmethod
    def precompute_cache(
        model: Any,
        tokenizer: Any,
        prefix_text: str,
        prefix_ids: Optional[torch.Tensor] = None,
        device: Optional[str] = None
    ) -> Tuple[DynamicCache, torch.Tensor, int]:
        """
        Runs a forward pass on the prefix text once to compute and populate the Key-Value cache.
        Returns the populated DynamicCache, prefix token IDs tensor, and sequence length.
        """
        if device is None:
            device = next(model.parameters()).device

        if prefix_ids is None:
            tokens = tokenizer(prefix_text, return_tensors="pt")
            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device) if hasattr(tokens, "attention_mask") else None
        else:
            input_ids = prefix_ids.to(device)
            attention_mask = torch.ones_like(input_ids)

        seq_len = input_ids.shape[1]
        cache = DynamicCache()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True
            )

        # DynamicCache is mutated in place and returned inside outputs.past_key_values
        resolved_cache = outputs.past_key_values or cache
        return resolved_cache, input_ids, seq_len

    @staticmethod
    def clone_cache(cache: DynamicCache) -> DynamicCache:
        """
        Creates an independent deep-copy of the DynamicCache.
        Ensures thread safety and allows concurrent query evaluations without cache contamination.
        """
        return copy.deepcopy(cache)

    @staticmethod
    def rollback_cache(cache: DynamicCache, target_length: int) -> None:
        """
        Crops the Key-Value cache back to target_length, discarding tokens added during generation.
        """
        if hasattr(cache, "crop"):
            cache.crop(target_length)
        else:
            # Fallback for older transformers if crop is absent
            for layer in getattr(cache, "layers", []):
                if hasattr(layer, "key_cache") and layer.key_cache is not None:
                    layer.key_cache = layer.key_cache[..., :target_length, :]
                if hasattr(layer, "value_cache") and layer.value_cache is not None:
                    layer.value_cache = layer.value_cache[..., :target_length, :]

    @staticmethod
    def save_cache(
        cache: DynamicCache,
        prefix_ids: torch.Tensor,
        profile: CacheProfile,
        file_path: str
    ) -> str:
        """
        Serializes the Key-Value cache, prefix token IDs, and metadata manifest to disk.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "cache": cache,
            "prefix_ids": prefix_ids.cpu(),
            "profile": profile.model_dump(),
        }

        torch.save(payload, str(path))
        return str(path.resolve())

    @staticmethod
    def load_cache(
        file_path: str,
        device: Optional[str] = None
    ) -> Tuple[DynamicCache, torch.Tensor, CacheProfile]:
        """
        Deserializes a cached KV state and profile from disk.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {file_path}")

        payload = torch.load(str(path), weights_only=False)
        cache: DynamicCache = payload["cache"]
        prefix_ids: torch.Tensor = payload["prefix_ids"]
        profile = CacheProfile(**payload["profile"])

        if device:
            target_device = torch.device(device)
            prefix_ids = prefix_ids.to(target_device)
            # Move internal cache tensors to target device if needed
            for layer in getattr(cache, "layers", []):
                if hasattr(layer, "key_cache") and layer.key_cache is not None:
                    layer.key_cache = layer.key_cache.to(target_device)
                if hasattr(layer, "value_cache") and layer.value_cache is not None:
                    layer.value_cache = layer.value_cache.to(target_device)

        return cache, prefix_ids, profile
