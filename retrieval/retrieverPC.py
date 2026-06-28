import numpy as np
from typing import Dict, List, Any

def _get_query_vector(query: str, model) -> np.ndarray:
    """Helper to apply asymmetric instruction and embed the query."""
    bge_instruction = "Represent this sentence for searching relevant passages: "
    asymmetric_query = f"{bge_instruction}{query}"
    return np.array(list(model.embed([asymmetric_query])))[0]

def build_parent_child_vault(
    parent_chunks: List[Dict], 
    child_chunks: List[Dict], 
    model
) -> Dict[str, Any]:
    print(f"Building Vault: {len(parent_chunks)} Parents, {len(child_chunks)} Children...")
    
    # 1. Parent Lookups (Lightweight)
    parent_lookup = {
        f"{p['metadata'].get('file_name', '').replace('.html', '')}::{p['metadata'].get('section') or 'Root'}": {
            "content": p["content"], "metadata": p["metadata"]
        } for p in parent_chunks
    }
        
    # 2. Optimized Embedding (Pre-allocate memory to avoid list bloat)
    child_texts = [child["content"] for child in child_chunks]
    
    # Use a generator approach to prevent crashing system RAM
    embeddings = []
    for i in range(0, len(child_texts), 32): # Batching manually
        batch = child_texts[i:i+32]
        embeddings.extend(list(model.embed(batch)))
        
    child_matrix = np.array(embeddings, dtype=np.float32) # Force float32 to save 50% VRAM
    
    return {
        "parent_lookup": parent_lookup,
        "children_data": child_chunks,
        "children_matrix": child_matrix
    }

def retrieve_context_for_agent(
    query: str, 
    model, 
    vault: Dict[str, Any], 
    top_k_parents: int = 2,
    verbose: bool = True
) -> List[Dict]:
    """
    Small-to-Big Retrieval.
    Searches the tiny child vectors, but returns the massive parent documents.
    Perfect for LLM Agents needing full context.
    """
    query_vec = _get_query_vector(query, model)
    
    children_data = vault["children_data"]
    children_matrix = vault["children_matrix"]
    parent_lookup = vault["parent_lookup"]
    
    # 1. GPU Dot Product against the highly-accurate child vectors
    scores = np.dot(children_matrix, query_vec)
    
    # Sort children by highest similarity
    best_child_indices = np.argsort(scores)[::-1]
    
    retrieved_parents = []
    seen_anchors = set()
    
    if verbose:
        print(f"\nAGENT RETRIEVAL FOR: 「 {query} 」")
        print("=" * 85)
        
    # 2. Walk down the winning children, but extract their Parents
    for idx in best_child_indices:
        if len(retrieved_parents) >= top_k_parents:
            break
            
        child = children_data[idx]
        child_score = scores[idx]
        meta = child["metadata"]
        
        # Determine which Parent this child belongs to
        file_key = meta.get("file_name", "").replace(".html", "")
        sec_key = meta.get("section") or meta.get("specific_header") or "Root"
        parent_anchor = f"{file_key}::{sec_key}"
        
        # Deduplication: If we already grabbed this Parent because another 
        # child inside it also matched, skip it so we don't spam the LLM context.
        if parent_anchor in seen_anchors:
            continue
            
        seen_anchors.add(parent_anchor)
        
        # Lookup the full Parent text
        parent_doc = parent_lookup.get(parent_anchor)
        
        if parent_doc:
            parent_doc["triggering_child_score"] = float(child_score)
            retrieved_parents.append(parent_doc)
            
            if verbose:
                print(f" HIT: Child chunk triggered Parent [{parent_anchor}] (Child Sim: {child_score:.4f})")
                snippet = parent_doc["content"].replace("\n", " ⏎ ")[:150] + "..."
                print(f" PARENT PAYLOAD: \"{snippet}\"\n")
                
    return retrieved_parents