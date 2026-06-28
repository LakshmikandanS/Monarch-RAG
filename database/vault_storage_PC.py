import json
import os
import numpy as np
from typing import Dict, Any

def save_vault(vault: Dict[str, Any], base_path: str = "data/vault"):
    """
    Saves the Parent-Child vault to disk, creating the directory if it doesn't exist.
    """
    # Extract the directory portion of the base_path and create it if missing
    directory = os.path.dirname(base_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # 1. Save Parent dictionary to JSON
    with open(f"{base_path}_parents.json", "w", encoding="utf-8") as f:
        json.dump(vault["parent_lookup"], f, indent=2)
    
    # 2. Save Child metadata (list) to JSON
    with open(f"{base_path}_children_meta.json", "w", encoding="utf-8") as f:
        json.dump(vault["children_data"], f, indent=2)
        
    # 3. Save Child vectors to NPY (High-performance binary)
    np.save(f"{base_path}_children_matrix.npy", vault["children_matrix"])
    
    print(f"Vault saved to {base_path}*")

def load_vault(base_path: str = "data/vault") -> Dict[str, Any]:
    """
    Loads the Parent-Child vault from disk back into memory.
    """
    with open(f"{base_path}_parents.json", "r", encoding="utf-8") as f:
        parent_lookup = json.load(f)
        
    with open(f"{base_path}_children_meta.json", "r", encoding="utf-8") as f:
        children_data = json.load(f)
        
    children_matrix = np.load(f"{base_path}_children_matrix.npy")
    
    return {
        "parent_lookup": parent_lookup,
        "children_data": children_data,
        "children_matrix": children_matrix
    }