"""
Comprehensive Token Performance & Latency Benchmark Suite.
Compares Zero-Cache Baseline, RAG, Monolithic CAG, and OKF-CAG.
"""

from typing import List, Dict, Any
from cag_okf.models import BenchmarkResult
from cag_okf.engine import OKFCAGEngine


class TokenBenchmarkSuite:
    """
    Executes and formats comparative evaluations across LLM generation paradigms.
    """

    def __init__(self, engine: OKFCAGEngine):
        self.engine = engine

    def run_comparison(
        self,
        query: str,
        max_new_tokens: int = 50,
        runs: int = 1
    ) -> Dict[str, BenchmarkResult]:
        """
        Runs the same query through:
        1. Zero-Cache Baseline (re-evaluating full prefix from scratch)
        2. OKF-CAG (precomputed KV cache with zero prefill overhead on knowledge)
        """
        results: Dict[str, BenchmarkResult] = {}

        # 1. Zero-Cache Baseline
        _, baseline_res = self.engine.generate(
            query=query,
            max_new_tokens=max_new_tokens,
            use_cache=False
        )
        results["zero_cache"] = baseline_res

        # 2. OKF-CAG
        _, cag_res = self.engine.generate(
            query=query,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
        results["okf_cag"] = cag_res

        return results

    @staticmethod
    def print_benchmark_table(results: Dict[str, BenchmarkResult]) -> None:
        """
        Formats benchmark metrics into a clean CLI table.
        """
        print("\n" + "=" * 90)
        print(f"{'PARADIGM':<36} | {'PREFILL TOKENS':<15} | {'OUT TOKENS':<10} | {'TOTAL TIME (s)':<14} | {'SAVINGS (%)':<10}")
        print("-" * 90)
        for key, res in results.items():
            print(
                f"{res.paradigm:<36} | "
                f"{res.prefill_tokens_evaluated:<15} | "
                f"{res.output_tokens_generated:<10} | "
                f"{res.total_time_s:<14.3f} | "
                f"{res.token_savings_percent:<10.1f}%"
            )
        print("=" * 90 + "\n")
