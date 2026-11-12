"""
Unified OKF-CAG Inference and Caching Engine.
Binds the Open Knowledge Format parser, deterministic composer,
and DynamicCache manager into an end-to-end token-optimized generation engine.
"""

import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from cag_okf.models import OKFBundle, CacheProfile, BenchmarkResult
from cag_okf.okf_parser import OKFParser
from cag_okf.composer import OKFComposer
from cag_okf.cache_manager import CAGCacheManager


class OKFCAGEngine:
    """
    Main engine orchestrating Cache-Augmented Generation over Open Knowledge Format bundles.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        device: Optional[str] = None,
        dtype: str = "float32",
        lazy_load: bool = False
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_str = dtype

        if dtype == "float16" and torch.cuda.is_available():
            self.torch_dtype = torch.float16
        elif dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        self.model = None
        self.tokenizer = None
        self.bundle: Optional[OKFBundle] = None
        self.composer = OKFComposer()

        # Cache state
        self.active_cache: Optional[DynamicCache] = None
        self.active_prefix_ids: Optional[torch.Tensor] = None
        self.active_prefix_len: int = 0
        self.active_prefix_text: str = ""
        self.cache_profile: Optional[CacheProfile] = None

        if not lazy_load:
            self.load_model()

    def load_model(self) -> None:
        """Loads model and tokenizer into memory."""
        if self.model is None:
            print(f"[OKFCAGEngine] Loading model '{self.model_id}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.torch_dtype
            )
            self.model.to(self.device)
            self.model.eval()
            print("[OKFCAGEngine] Model successfully loaded.")

    def load_bundle(self, bundle_dir: str, bundle_name: Optional[str] = None) -> OKFBundle:
        """Parses an OKF knowledge base directory."""
        self.bundle = OKFParser.parse_bundle(bundle_dir, bundle_name=bundle_name)
        return self.bundle

    def build_cache(
        self,
        target_services: Optional[List[str]] = None,
        target_languages: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        custom_system_prompt: Optional[str] = None
    ) -> CacheProfile:
        """
        Filters the OKF bundle deterministically, composes the prefix,
        and precomputes the DynamicCache Key-Value states.
        """
        if self.bundle is None:
            raise ValueError("No OKF bundle loaded. Call load_bundle() first.")
        if self.model is None:
            self.load_model()

        # 1. Deterministic filter
        selected_docs = self.composer.filter_documents(
            self.bundle,
            target_services=target_services,
            target_languages=target_languages,
            tags=tags,
            types=types
        )

        # 2. Composition
        prefix_text = self.composer.compose_prefix(
            self.bundle,
            selected_docs=selected_docs,
            custom_instructions=custom_system_prompt
        )
        self.active_prefix_text = prefix_text
        prefix_hash = self.composer.compute_prefix_hash(prefix_text)

        # 3. DynamicCache Precomputation (The CAG prefill step)
        start_time = time.perf_counter()
        cache, prefix_ids, seq_len = CAGCacheManager.precompute_cache(
            self.model,
            self.tokenizer,
            prefix_text=prefix_text,
            device=self.device
        )
        elapsed = time.perf_counter() - start_time

        self.active_cache = cache
        self.active_prefix_ids = prefix_ids
        self.active_prefix_len = seq_len

        doc_ids = [d.metadata.id or "unknown" for d in selected_docs]
        self.cache_profile = CacheProfile(
            model_id=self.model_id,
            total_tokens=seq_len,
            document_ids=doc_ids,
            created_at=datetime.now(timezone.utc).isoformat(),
            content_hash=prefix_hash,
            prefix_prompt_preview=prefix_text[:200] + "..." if len(prefix_text) > 200 else prefix_text
        )

        print(
            f"[OKFCAGEngine] Precomputed KV cache for {len(doc_ids)} OKF documents: "
            f"{seq_len} tokens prefilled once in {elapsed:.3f}s."
        )
        return self.cache_profile

    def save_cache(self, file_path: str) -> str:
        """Saves active precomputed KV cache to disk."""
        if self.active_cache is None or self.cache_profile is None or self.active_prefix_ids is None:
            raise ValueError("No active cache to save. Call build_cache() first.")

        return CAGCacheManager.save_cache(
            self.active_cache,
            self.active_prefix_ids,
            self.cache_profile,
            file_path
        )

    def load_cache(self, file_path: str) -> CacheProfile:
        """Restores a precomputed KV cache from disk."""
        if self.model is None:
            self.load_model()

        cache, prefix_ids, profile = CAGCacheManager.load_cache(file_path, device=self.device)
        self.active_cache = cache
        self.active_prefix_ids = prefix_ids
        self.active_prefix_len = prefix_ids.shape[1]
        self.cache_profile = profile
        print(f"[OKFCAGEngine] Restored KV cache from {file_path} ({self.active_prefix_len} tokens).")
        return profile

    def generate(
        self,
        query: str,
        max_new_tokens: int = 60,
        temperature: float = 0.3,
        top_p: float = 0.9,
        use_cache: bool = True
    ) -> Tuple[str, BenchmarkResult]:
        """
        Generates response using either the precomputed KV cache (CAG mode)
        or from scratch (baseline zero-cache comparison).
        """
        if self.model is None:
            self.load_model()

        formatted_query = f"\nUser Query: {query}\nResponse:"
        query_tokens = self.tokenizer(formatted_query, return_tensors="pt")
        query_ids = query_tokens.input_ids.to(self.device)
        q_len = query_ids.shape[1]

        if use_cache:
            if self.active_cache is None or self.active_prefix_ids is None:
                raise ValueError("Active cache not initialized. Call build_cache() or load_cache().")

            # In CAG mode: We pass full_ids = [prefix_ids, query_ids]
            # Since past_key_values has prefix_len entries, transformers ONLY evaluates q_len tokens!
            full_ids = torch.cat([self.active_prefix_ids, query_ids], dim=-1)
            q_cache = CAGCacheManager.clone_cache(self.active_cache)

            # Measure TTFT and overall generation time
            t0 = time.perf_counter()
            with torch.no_grad():
                gen_output = self.model.generate(
                    input_ids=full_ids,
                    past_key_values=q_cache,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    top_p=top_p if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            t1 = time.perf_counter()

            # Slicing out newly generated tokens
            generated_ids = gen_output[0][full_ids.shape[1]:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            out_count = len(generated_ids)
            total_duration = t1 - t0
            tps = out_count / total_duration if total_duration > 0 else 0.0

            # TTFT in CAG is essentially the prefill time of JUST the query tokens
            prefill_tokens_evaluated = q_len
            total_context_tokens = self.active_prefix_len + q_len
            token_savings = ((total_context_tokens - prefill_tokens_evaluated) / total_context_tokens) * 100.0

            result = BenchmarkResult(
                paradigm="OKF-CAG (Cache-Augmented Generation)",
                query=query,
                prefill_tokens_evaluated=prefill_tokens_evaluated,
                output_tokens_generated=out_count,
                time_to_first_token_ms=round((total_duration / (out_count or 1)) * 1000, 2),
                total_time_s=round(total_duration, 4),
                tokens_per_second=round(tps, 2),
                token_savings_percent=round(token_savings, 2),
                response_text=response_text
            )
            return response_text, result

        else:
            # Baseline Zero-Cache mode: Re-evaluates entire prefix + query from scratch
            full_text = f"{self.active_prefix_text}{formatted_query}"
            full_tokens = self.tokenizer(full_text, return_tensors="pt")
            full_input_ids = full_tokens.input_ids.to(self.device)
            total_input_len = full_input_ids.shape[1]

            t0 = time.perf_counter()
            with torch.no_grad():
                gen_output = self.model.generate(
                    input_ids=full_input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    top_p=top_p if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            t1 = time.perf_counter()

            generated_ids = gen_output[0][total_input_len:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            out_count = len(generated_ids)
            total_duration = t1 - t0
            tps = out_count / total_duration if total_duration > 0 else 0.0

            result = BenchmarkResult(
                paradigm="Zero-Cache Baseline (Full Re-prefill)",
                query=query,
                prefill_tokens_evaluated=total_input_len,
                output_tokens_generated=out_count,
                time_to_first_token_ms=round((total_duration / (out_count or 1)) * 1000, 2),
                total_time_s=round(total_duration, 4),
                tokens_per_second=round(tps, 2),
                token_savings_percent=0.0,
                response_text=response_text
            )
            return response_text, result
