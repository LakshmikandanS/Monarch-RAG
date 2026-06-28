import os
import json
import numpy as np
from pathlib import Path
from typing import Dict

def save_vault(vault: Dict, save_dir: str = "./local_vector_vault"):
    """
    Saves the multi-strategy chunk vault and numpy matrices to disk.
    """
    base_path = Path(save_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for strategy_name, data in vault.items():
        # Create a subfolder for each strategy (e.g., "./local_vector_vault/Header Aware/")
        safe_name = strategy_name.replace(" ", "_").lower()
        strategy_path = base_path / safe_name
        strategy_path.mkdir(exist_ok=True)
        
        # 1. Save the metadata and text as JSON
        chunks_path = strategy_path / "chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(data["chunks"], f, indent=2)
            
        # 2. Save the dense GPU embeddings as a highly compressed .npy binary
        matrix_path = strategy_path / "matrix.npy"
        np.save(matrix_path, data["matrix"])
        
    print(f"Vault securely persisted to disk at: {base_path.resolve()}")


def load_vault(save_dir: str = "./local_vector_vault") -> Dict:
    """
    Loads the vault from disk back into RAM. Returns None if it doesn't exist.
    """
    base_path = Path(save_dir)
    if not base_path.exists():
        return None
        
    restored_vault = {}
    
    # Iterate through all saved strategy folders
    for strategy_dir in base_path.iterdir():
        if strategy_dir.is_dir():
            chunks_path = strategy_dir / "chunks.json"
            matrix_path = strategy_dir / "matrix.npy"
            
            if chunks_path.exists() and matrix_path.exists():
                # Reconstruct the original display name from the folder name
                strategy_name = strategy_dir.name.replace("_", " ").title()
                if strategy_name == "Recursive (Control)": # Special case handling if needed
                    pass 
                
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                    
                matrix = np.load(matrix_path)
                
                restored_vault[strategy_name] = {
                    "chunks": chunks,
                    "matrix": matrix
                }
                
    if restored_vault:
        print(f"Vault loaded from disk! Restored {len(restored_vault)} indexing strategies.")
        return restored_vault
        
    return None