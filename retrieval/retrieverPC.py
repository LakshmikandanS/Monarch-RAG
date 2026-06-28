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
    """
    Builds a highly efficient, single-matrix database.
    - Parents (Header-Aware) are stored only as text lookups (Zero VRAM).
    - Children (Code-Aware Sliding/Semantic) are embedded for search.
    """
    print(f"Building Multi-Vector Vault: {len(parent_chunks)} Parents, {len(child_chunks)} Children...")
    
    # 1. Create a fast O(1) lookup dictionary for Parents based on their anchor
    parent_lookup = {}
    for parent in parent_chunks:
        meta = parent["metadata"]
        file_key = meta.get("file_name", "").replace(".html", "")
        sec_key = meta.get("section") or meta.get("specific_header") or "Root"
        anchor = f"{file_key}::{sec_key}"
        
        parent_lookup[anchor] = {
            "content": parent["content"],
            "metadata": meta,
            "anchor": anchor
        }
        
    # 2. Embed ONLY the children (Saves massive GPU memory & compute)
    child_texts = [child["content"] for child in child_chunks]
    print(f"Vectorizing {len(child_texts)} child chunks on RTX 5060...")
    child_matrix = np.array(list(model.embed(child_texts, batch_size=128)))
    
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