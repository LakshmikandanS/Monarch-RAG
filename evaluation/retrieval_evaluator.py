import numpy as np
from typing import Dict, List, Any

def _get_query_vector(query: str, model) -> np.ndarray:
    """Helper to apply asymmetric instruction and embed the query."""
    bge_instruction = "Represent this sentence for searching relevant passages: "
    asymmetric_query = f"{bge_instruction}{query}"
    # Wrap in list() to resolve the fastembed generator
    return np.array(list(model.embed([asymmetric_query])))[0]


def benchmark_chunking_strategies(
    query: str, model, strategy_vault: Dict[str, Any], top_k: int = 2
) -> str:
    """
    Evaluates multiple chunking strategies against a single query.
    Returns the name of the winning strategy (highest Cosine Similarity).
    """
    print(f"\n🔍 QUERY: 「 {query} 」")
    print("=" * 85)

    query_vec = _get_query_vector(query, model)
    scoreboard = {}

    for strat_name, vault in strategy_vault.items():
        chunks = vault["chunks"]
        matrix = vault["matrix"]  # Shape: (N, 768)

        # Instant Cosine Similarity via Dot Product
        scores = np.dot(matrix, query_vec)
        best_indices = np.argsort(scores)[::-1][:top_k]
        
        peak_score = float(scores[best_indices[0]])
        scoreboard[strat_name] = peak_score

        print(f"\n📦 STRATEGY: 【 {strat_name} 】")
        for rank, idx in enumerate(best_indices, 1):
            doc = chunks[idx]
            score = scores[idx]
            meta = doc.get("metadata", {})

            file_src = meta.get("file_name", "doc").replace(".html", "")
            header = meta.get("section") or meta.get("specific_header") or "Root"

            # Flatten newlines so the notebook printout stays clean
            snippet = doc["content"].replace("\n", " ⏎ ")
            if len(snippet) > 180:
                snippet = snippet[:180] + "..."

            print(f"  #{rank} [Sim: {score:.4f}] ── [{file_src} > {header}]")
            print(f'      "{snippet}"')

    champion = max(scoreboard, key=scoreboard.get)
    print("-" * 85)
    print(f"🏆 ROUND WINNER: {champion}  (Peak Score: {scoreboard[champion]:.4f})\n")
    
    return champion


def execute_rrf_fusion(
    query: str, 
    model, 
    vault: Dict[str, Any], 
    strategies_to_fuse: List[str] = ["Header Aware", "Code Aware Sliding"],
    top_k: int = 3, 
    rrf_k: int = 60,
    verbose: bool = True
) -> List[Dict]:
    """
    Production-grade Reciprocal Rank Fusion.
    Fuses multiple retrievers based on anchor consensus and returns the actual document payloads.
    """
    query_vec = _get_query_vector(query, model)
    rank_lists = {}
    
    # 1. Gather Top 15 rankings from each requested strategy
    for strat in strategies_to_fuse:
        if strat not in vault:
            continue
        matrix = vault[strat]["matrix"]
        sims = np.dot(matrix, query_vec)
        rank_lists[strat] = np.argsort(sims)[::-1][:15]

    rrf_scoreboard = {}
    chunk_payload_lookup = {}

    # 2. Calculate Reciprocal Rank Consensus
    for strat, top_indices in rank_lists.items():
        chunks = vault[strat]["chunks"]
        matrix = vault[strat]["matrix"]
        
        for rank, idx in enumerate(top_indices, start=1):
            doc = chunks[idx]
            meta = doc["metadata"]
            
            # Universal Anchor (e.g., "pytorch_workflow::Training Loop")
            file_key = meta.get("file_name", "").replace(".html", "")
            sec_key = meta.get("section") or meta.get("specific_header") or "Root"
            universal_anchor = f"{file_key}::{sec_key}"
            
            # Add RRF points
            points = 1.0 / (rrf_k + rank)
            rrf_scoreboard[universal_anchor] = rrf_scoreboard.get(universal_anchor, 0.0) + points
            
            # Store the highest-quality text payload for this anchor
            # (Prefers Header Aware if available, otherwise just keeps the first seen)
            if strat == "Header Aware" or universal_anchor not in chunk_payload_lookup:
                chunk_payload_lookup[universal_anchor] = {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "source_anchor": universal_anchor,
                    "rrf_score": 0.0 # Will update below
                }

    # 3. Sort anchors by consensus score
    fused_anchors = sorted(rrf_scoreboard, key=rrf_scoreboard.get, reverse=True)[:top_k]
    
    final_results = []
    
    if verbose:
        print(f"\n🧬 FUSED RETRIEVAL FOR: 「 {query} 」")
        print("=" * 85)
        
    for pos, anchor in enumerate(fused_anchors, 1):
        payload = chunk_payload_lookup[anchor]
        payload["rrf_score"] = rrf_scoreboard[anchor]
        final_results.append(payload)
        
        if verbose:
            snippet = payload["content"].replace("\n", " ⏎ ")[:160] + "..."
            print(f" #{pos} [RRF Score: {payload['rrf_score']:.5f}] ── {anchor}")
            print(f"     \"{snippet}\"\n")
            
    return final_results


def run_stress_test_suite(model, vault: Dict[str, Any],test_queries: List[str]):
    """
    Runs the 8 hardcore PyTorch queries to benchmark all indexing strategies blindly.
    
    test_queries = [
        "What is the exact code to manually set the random seed for CPU and CUDA?",
        "Show me the code to check which device a model's parameters are sitting on.",
        "Why am I getting the RuntimeError: Expected all tensors to be on the same device?",
        "How do I fix a shape mismatch error inside nn.Linear forward pass?",
        "Does the Zero to Mastery PyTorch course cover PyTorch version 2.0?",
        "What specific data science bootcamp is recommended as a prerequisite before taking this course?",
        "What is the overarching computer vision project built throughout the milestone chapters called?",
        "What is the difference between torch.rand and torch.randn?"
    ]
    """
    win_counts = {strat: 0 for strat in vault.keys()}

    print("\n🚀 INITIATING BLIND STRESS TEST SUITE...")
    
    for q in test_queries:
        # Run silently and just grab the winner
        winner = benchmark_chunking_strategies(q, model, vault, top_k=1)
        win_counts[winner] += 1

    print("\n🏁 OVERALL SERIES CHAMPIONSHIP:")
    print("=" * 40)
    for strat, wins in sorted(win_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {strat.ljust(25)}: {wins} / {len(test_queries)} rounds won")