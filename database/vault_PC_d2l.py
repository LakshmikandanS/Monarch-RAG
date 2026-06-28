import json
from typing import List, Dict, Any
from pathlib import Path
from retrieval.retrieverPC import build_parent_child_vault
from database.vector_storage import save_vault

def assemble_d2l_vault(
    parent_json_path: str, 
    child_json_paths: List[str], 
    model, 
    output_vault_path: str
):
    """
    Orchestrates the D2L Parent-Child vault assembly from pre-saved chunks.
    """
    print(f"📂 Loading Parent data from: {parent_json_path}")
    with open(parent_json_path, "r", encoding="utf-8") as f:
        parents = json.load(f)

    all_children = []
    for c_path in child_json_paths:
        print(f"📂 Loading Child data from: {c_path}")
        with open(c_path, "r", encoding="utf-8") as f:
            all_children.extend(json.load(f))

    # The Logic:
    # 1. build_parent_child_vault maps parents to anchors (File::Section)
    # 2. It embeds all children using the provided BGE model
    # 3. It returns the dictionary structure required by your database/vector_storage.py
    
    print(f"🚀 Building Vault: {len(parents)} Parents, {len(all_children)} Children...")
    vault = build_parent_child_vault(parents, all_children, model)
    
    # Save using the updated storage logic
    save_vault(vault, output_vault_path)
    print(f"🎉 Vault successfully assembled and saved to {output_vault_path}")

# --- Example Usage ---
# assemble_d2l_vault(
#     parent_json_path="data/header_aware.json",
#     child_json_paths=["data/code_aware.json", "data/semantic.json"],
#     model=my_bge_model,
#     output_vault_path="data/d2l_pc_vault/vault"
# )