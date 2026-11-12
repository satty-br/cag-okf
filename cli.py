"""
Command-Line Interface for OKF-CAG (Cache-Augmented Generation + Open Knowledge Format).
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from cag_okf.engine import OKFCAGEngine
from cag_okf.okf_parser import OKFParser
from cag_okf.benchmark import TokenBenchmarkSuite


def cmd_serve(args):
    """Starts the FastAPI web server and UI studio."""
    from server import start
    start(port=args.port, host=args.host)


def cmd_validate_bundle(args):
    """Validates an OKF directory."""
    print(f"\n[OKF Validator] Scanning bundle at: {args.bundle_dir}")
    bundle = OKFParser.parse_bundle(args.bundle_dir)
    print(f"Bundle Name: {bundle.name}")
    print(f"Total Documents: {bundle.total_docs()}")
    if bundle.index_doc:
        print(f"Index File: {bundle.index_doc.metadata.title or 'index.md'}")

    print("\n--- Document Inventory ---")
    for doc_id, doc in bundle.documents.items():
        m = doc.metadata
        print(f"[{m.status:<8}] ID: {m.id:<16} Type: {m.type:<10} Title: {m.title or 'Untitled'}")
        if m.target_services:
            print(f"           Target Services: {', '.join(m.target_services)}")
        if m.target_languages:
            print(f"           Languages:       {', '.join(m.target_languages)}")
        if m.tags:
            print(f"           Tags:            {', '.join(m.tags)}")
    print("-" * 50)
    print("Bundle validation complete.\n")


def cmd_build_cache(args):
    """Compiles OKF documents into a serialized Key-Value cache."""
    engine = OKFCAGEngine(model_id=args.model_id, lazy_load=False)
    engine.load_bundle(args.bundle_dir)

    target_services = args.services.split(",") if args.services else None
    target_languages = args.languages.split(",") if args.languages else None
    tags = args.tags.split(",") if args.tags else None

    profile = engine.build_cache(
        target_services=target_services,
        target_languages=target_languages,
        tags=tags
    )

    saved_path = engine.save_cache(args.output_cache)
    print(f"\n[OKF-CAG] Key-Value cache saved to: {saved_path}")
    print(f"Precomputed Tokens: {profile.total_tokens}")
    print(f"Documents Included: {len(profile.document_ids)}")


def cmd_query(args):
    """Answers a query using precomputed Key-Value cache."""
    engine = OKFCAGEngine(model_id=args.model_id, lazy_load=False)
    engine.load_cache(args.cache)

    print(f"\n[Query] {args.query}")
    answer, metrics = engine.generate(args.query, max_new_tokens=args.max_tokens, use_cache=True)
    print(f"\n[Answer]\n{answer}\n")
    print("-" * 50)
    print(f"Prefill Tokens Evaluated: {metrics.prefill_tokens_evaluated} (vs {engine.active_prefix_len + metrics.prefill_tokens_evaluated} total context)")
    print(f"Token Savings:           {metrics.token_savings_percent:.1f}%")
    print(f"Generation Latency:      {metrics.total_time_s:.3f}s")
    print(f"Throughput:              {metrics.tokens_per_second:.1f} tokens/s")
    print("-" * 50 + "\n")


def cmd_benchmark(args):
    """Runs a direct comparison between Zero-Cache Baseline and OKF-CAG."""
    engine = OKFCAGEngine(model_id=args.model_id, lazy_load=False)
    engine.load_bundle(args.bundle_dir)

    target_services = args.services.split(",") if args.services else None
    target_languages = args.languages.split(",") if args.languages else None
    tags = args.tags.split(",") if args.tags else None

    engine.build_cache(
        target_services=target_services,
        target_languages=target_languages,
        tags=tags
    )

    suite = TokenBenchmarkSuite(engine)
    print(f"\nRunning benchmark for query: '{args.query}'")
    results = suite.run_comparison(args.query, max_new_tokens=args.max_tokens)
    suite.print_benchmark_table(results)

    print("\n--- Detailed Responses ---")
    for key, res in results.items():
        print(f"\n[{res.paradigm}]")
        print(res.response_text)
    print("-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="OKF-CAG: Cache-Augmented Generation on Open Knowledge Format"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-bundle
    p_val = subparsers.add_parser("validate-bundle", help="Validate an OKF directory")
    p_val.add_argument("--bundle-dir", required=True, help="Path to OKF bundle directory")
    p_val.set_defaults(func=cmd_validate_bundle)

    # build-cache
    p_build = subparsers.add_parser("build-cache", help="Precompute and serialize KV cache")
    p_build.add_argument("--bundle-dir", required=True, help="Path to OKF bundle directory")
    p_build.add_argument("--output-cache", required=True, help="Path to save output .pt cache")
    p_build.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p_build.add_argument("--services", help="Comma-separated target services")
    p_build.add_argument("--languages", help="Comma-separated target languages")
    p_build.add_argument("--tags", help="Comma-separated tags")
    p_build.set_defaults(func=cmd_build_cache)

    # query
    p_q = subparsers.add_parser("query", help="Query model using preloaded KV cache")
    p_q.add_argument("--cache", required=True, help="Path to cached .pt file")
    p_q.add_argument("--query", required=True, help="User question / prompt")
    p_q.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p_q.add_argument("--max-tokens", type=int, default=40)
    p_q.set_defaults(func=cmd_query)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run token & latency benchmark")
    p_bench.add_argument("--bundle-dir", required=True, help="Path to OKF bundle directory")
    p_bench.add_argument("--query", default="Is direct sql.Open permitted in Go?", help="Benchmark query")
    p_bench.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p_bench.add_argument("--max-tokens", type=int, default=40)
    p_bench.add_argument("--services", help="Comma-separated target services")
    p_bench.add_argument("--languages", help="Comma-separated target languages")
    p_bench.add_argument("--tags", help="Comma-separated tags")
    p_bench.set_defaults(func=cmd_benchmark)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch the Web Frontend Studio and REST API")
    p_serve.add_argument("--port", type=int, default=8000, help="HTTP server port")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
