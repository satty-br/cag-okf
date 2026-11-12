import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Cache-Augmented Generation (CAG) + Open Knowledge Format (OKF)\n",
                "### Ultra-Efficient Token Performance, Zero-Prefill Latency & Modular LLM Memory\n",
                "\n",
                "Based on the foundational work of **Ronan Takizawa** ([cacheaugmentedgeneration](https://github.com/ronantakizawa/cacheaugmentedgeneration))\n",
                "and Google's **Open Knowledge Format (OKF)** specification for structured agentic memory.\n",
                "\n",
                "---\n",
                "\n",
                "## Why Combine CAG with OKF?\n",
                "| Paradigm | Knowledge Loading | Query Prefill Cost | Modularity & Governance |\n",
                "| :--- | :--- | :--- | :--- |\n",
                "| **Standard RAG** | Vector DB chunk lookup on every query | Evaluates query + chunks every time (high TTFT) | Context shredded into arbitrary chunks |\n",
                "| **Naive CAG** | Preloads monolithic raw text into KV cache | O(\\Delta) query cost (0 knowledge prefill) | Unstructured; modifying 1 doc invalidates entire cache |\n",
                "| **OKF-CAG** | Modular YAML frontmatter rules preloaded into KV cache | **O(\\Delta) query cost (95%+ token savings)** | **Deterministic filtering, hash-based cache updates, zero cache pollution** |\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys, os, time, copy, torch\n",
                "from pathlib import Path\n",
                "\n",
                "# Add cag-okf to python path\n",
                "sys.path.insert(0, os.path.abspath('.'))\n",
                "\n",
                "from cag_okf.models import OKFMetadata, OKFDocument\n",
                "from cag_okf.okf_parser import OKFParser\n",
                "from cag_okf.composer import OKFComposer\n",
                "from cag_okf.cache_manager import CAGCacheManager\n",
                "from cag_okf.engine import OKFCAGEngine\n",
                "from cag_okf.benchmark import TokenBenchmarkSuite\n",
                "\n",
                "print('PyTorch Version:', torch.__version__)\n",
                "print('CUDA Available:', torch.cuda.is_available())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Parse and Inspect the OKF Bundle\n",
                "OKF documents contain YAML frontmatter with explicit attributes (`target_services`, `status`, `tags`, `valid_until`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "bundle_dir = 'examples/sample_okf_bundle'\n",
                "bundle = OKFParser.parse_bundle(bundle_dir)\n",
                "print(f'Loaded OKF Bundle: {bundle.name} ({bundle.total_docs()} documents)')\n",
                "\n",
                "for doc_id, doc in bundle.documents.items():\n",
                "    m = doc.metadata\n",
                "    print(f'-> [{m.status}] {m.id:<14} | Type: {m.type:<10} | Title: {m.title}')\n",
                "    if m.target_services:\n",
                "        print(f'   Targets: {m.target_services}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Deterministic Context Filtering\n",
                "Instead of bloating the KV cache with irrelevant documents, OKF enables selective compilation based on the target microservice or domain."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "composer = OKFComposer()\n",
                "# Filter specifically for rules affecting payments services\n",
                "selected_docs = composer.filter_documents(bundle, target_services=['pagamentos-service'])\n",
                "print(f'Selected {len(selected_docs)} relevant rules out of {bundle.total_docs()}:')\n",
                "for d in selected_docs:\n",
                "    print(f'  - {d.metadata.id}: {d.metadata.title}')\n",
                "\n",
                "prefix_prompt = composer.compose_prefix(bundle, selected_docs=selected_docs)\n",
                "print('\\n--- Composed Context Prefix Preview ---')\n",
                "print(prefix_prompt[:400] + '...')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Initialize Engine & Precompute Key-Value Cache (CAG Prefill)\n",
                "The model performs a single forward pass over the composed OKF knowledge base and captures the Key-Value attention tensors in a `DynamicCache`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Using an open-source instruct model (runs on CPU or CUDA)\n",
                "engine = OKFCAGEngine(model_id='HuggingFaceTB/SmolLM2-135M-Instruct')\n",
                "engine.load_bundle(bundle_dir)\n",
                "\n",
                "profile = engine.build_cache(target_services=['pagamentos-service'])\n",
                "print(f'Precomputed KV Cache Tokens: {profile.total_tokens}')\n",
                "print(f'Active Documents: {profile.document_ids}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Zero-Prefill Question Answering\n",
                "When queries arrive, the model bypasses re-evaluating the knowledge tokens. Only the query tokens are processed during prefill!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "query = 'Is direct database sql.Open permitted in Go?'\n",
                "answer, metrics = engine.generate(query, max_new_tokens=30, use_cache=True)\n",
                "\n",
                "print(f'Question: {query}')\n",
                "print(f'Answer:\\n{answer}\\n')\n",
                "print(f'Prefill Tokens Evaluated: {metrics.prefill_tokens_evaluated} (vs {engine.active_prefix_len + metrics.prefill_tokens_evaluated} total context)')\n",
                "print(f'Token Compute Savings:    {metrics.token_savings_percent:.1f}%')\n",
                "print(f'Throughput:               {metrics.tokens_per_second:.1f} tokens/sec')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Cache Persistence (Save & Reload from Disk)\n",
                "Just like Ronan Takizawa demonstrated, we can serialize the precomputed Key-Value states to disk so production workers can warm-boot in milliseconds without re-evaluating the knowledge base."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cache_file = '.cache/cag_okf_demo.pt'\n",
                "engine.save_cache(cache_file)\n",
                "print(f'Serialized KV cache saved to {cache_file}')\n",
                "\n",
                "# Spin up a second fresh engine instance and restore\n",
                "fresh_engine = OKFCAGEngine(model_id='HuggingFaceTB/SmolLM2-135M-Instruct')\n",
                "restored_profile = fresh_engine.load_cache(cache_file)\n",
                "print(f'Restored from disk! Tokens in cache: {restored_profile.total_tokens}')\n",
                "\n",
                "# Querying the freshly restored engine\n",
                "ans, _ = fresh_engine.generate('What library should Go services use for database connections?', max_new_tokens=25)\n",
                "print(f'Restored Engine Response: {ans}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Empirical Benchmark (Zero-Cache Baseline vs OKF-CAG)\n",
                "Here we measure the actual prefill token counts, latencies, and computational savings."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "suite = TokenBenchmarkSuite(engine)\n",
                "bench_results = suite.run_comparison('Can billing-worker exceed 20 database connections?', max_new_tokens=30)\n",
                "suite.print_benchmark_table(bench_results)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.6"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_file = Path("cag-okf/cag_demo.ipynb")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook cag_demo.ipynb generated successfully!")
