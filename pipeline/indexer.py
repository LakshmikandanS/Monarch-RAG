import json
from typing import List, Dict, Any

def link_prechunked_parents_and_children(
    raw_parents: List[Dict], 
    raw_children: List[Dict]
) -> Dict[str, List[Dict]]:
    """
    Takes pre-chunked JSON data (e.g., Header-Aware for parents, Semantic for children)
    and firmly links them together using a Universal Anchor.
    """
    parent_chunks = []
    child_chunks = []

    print(f"🔗 Linking {len(raw_parents)} Parents and {len(raw_children)} Children via Universal Anchors...")

    # 1. PROCESS PARENTS (The Large Narrative Text from Header-Aware chunking)
    for doc in raw_parents:
        meta = doc.get("metadata", {})
        file_name = meta.get("file_name", "unknown_file").replace(".html", "")
        # Handle variations in metadata keys between older and newer chunkers
        section_path = meta.get("section") or meta.get("specific_header") or "Root"
        
        # ⭐️ THE LINK: Create the Universal Anchor
        anchor = f"{file_name}::{section_path}"
        
        # Reconstruct to ensure a strict schema for the vault
        parent_chunks.append({
            "anchor": anchor,  # Parent holds the anchor as its primary key
            "content": doc["content"],
            "metadata": meta
        })

    # 2. PROCESS CHILDREN (The Small, Searchable Vectors from Code-Aware/Semantic)
    for child in raw_children:
        meta = child.get("metadata", {})
        file_name = meta.get("file_name", "unknown_file").replace(".html", "")
        section_path = meta.get("section") or meta.get("specific_header") or "Root"
        
        # ⭐️ THE LINK: Recreate the EXACT same Universal Anchor
        anchor = f"{file_name}::{section_path}"
        
        # Assign the pointer so the Child knows who its Parent is
        child["metadata"]["parent_anchor"] = anchor
        child_chunks.append(child)
        
    print(f"✅ Successfully linked {len(parent_chunks)} Parents and {len(child_chunks)} Children.")
    
    return {
        "parents": parent_chunks,
        "children": child_chunks
    }

def load_json_chunks(filepath: str) -> List[Dict]:
    """Utility to load your saved chunk JSONs."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)