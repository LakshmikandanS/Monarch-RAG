import bisect
import numpy as np
from typing import Dict, List
from chunking.utils import get_sections_with_hierarchy

def semantic_chunk_documents(
    documents: List[Dict], model, min_chars: int = 250, max_chars: int = 1200
) -> List[Dict]:
    """
    True Semantic Chunker engineered for technical Markdown.
    Replaces blind sentence regex with AST-Protected Atomization.
    """
    all_semantic_chunks = []

    for doc_idx, doc in enumerate(documents):
        content = doc["content"].replace('\r\n', '\n')
        meta = doc.get("metadata", {})
        file_name = meta.get("file_name") or meta.get("title") or "document"

        header_info = get_sections_with_hierarchy(content)
        header_starts = [h["start_char"] for h in header_info]

        # 1. ATOMIZE: Break into discrete paragraphs or whole code blocks
        atoms = _atomize_markdown(content)
        if not atoms:
            continue

        if len(atoms) == 1:
            all_semantic_chunks.append(_package(atoms[0]["text"], file_name, header_info, header_starts, atoms[0]["start"], 0))
            continue

        # 2. BATCH VECTORIZE ATOMS ON GPU
        atom_texts = [a["text"] for a in atoms]
        raw_vecs = np.array(list(model.embed(atom_texts, batch_size=128)))
        
        # L2 Normalize
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        unit_vecs = raw_vecs / np.maximum(norms, 1e-9)

        # 3. CALCULATE ADJACENT DISTANCES WITH SYNTAX GRAVITY
        distances = []
        for i in range(len(unit_vecs) - 1):
            # SYNTAX GRAVITY OVERRIDE: 
            # If the next atom is a code block, force distance to 0.0 so it welds to the prose above it.
            if atoms[i+1]["is_code"]:
                distances.append(0.0)
            else:
                dist = 1.0 - np.dot(unit_vecs[i], unit_vecs[i+1])
                distances.append(float(dist))

        # 4. DYNAMIC THRESHOLD (Mean + 1.1 * Standard Deviation)
        # Ignores arbitrary percentiles; adapts to the document's native variance.
        mu, sigma = np.mean(distances), np.std(distances)
        dynamic_threshold = mu + (1.1 * sigma)

        # 5. ASSEMBLE CHUNKS
        compiled_chunks = []
        curr_text = ""
        curr_start = atoms[0]["start"]

        for i, atom in enumerate(atoms):
            atom_len = len(atom["text"])

            # Guardrail A: If chunk is getting too big, force cut
            if len(curr_text) + atom_len > max_chars and curr_text:
                compiled_chunks.append(_package(curr_text, file_name, header_info, header_starts, curr_start, len(compiled_chunks)))
                curr_text = atom["text"]
                curr_start = atom["start"]
                continue

            curr_text += ("\n\n" if curr_text else "") + atom["text"]

            # Guardrail B: Check if semantic shift triggers a cut
            if i < len(distances) and distances[i] > dynamic_threshold:
                # Don't sever if the chunk is still tiny (< min_chars)
                if len(curr_text) >= min_chars:
                    compiled_chunks.append(_package(curr_text, file_name, header_info, header_starts, curr_start, len(compiled_chunks)))
                    curr_text = ""
                    curr_start = atoms[i+1]["start"]

        if curr_text.strip():
            compiled_chunks.append(_package(curr_text, file_name, header_info, header_starts, curr_start, len(compiled_chunks)))

        all_semantic_chunks.extend(compiled_chunks)

    return all_semantic_chunks


def _atomize_markdown(text: str) -> List[Dict]:
    """Slices text into unbreakable Paragraphs and Code Blocks."""
    lines = text.split('\n')
    atoms = []
    buffer = []
    buffer_start = 0
    cursor = 0
    in_code = False

    for line in lines:
        line_len = len(line) + 1 # +1 for \n

        if line.strip().startswith('```'):
            if not in_code:
                # Entering code: flush whatever prose was in the buffer
                if buffer:
                    atoms.append({"text": "\n".join(buffer).strip(), "start": buffer_start, "is_code": False})
                    buffer = []
                in_code = True
                buffer_start = cursor
                buffer.append(line)
            else:
                # Exiting code: flush the entire code block as one atom
                buffer.append(line)
                atoms.append({"text": "\n".join(buffer).strip(), "start": buffer_start, "is_code": True})
                buffer = []
                in_code = False
            cursor += line_len
            continue

        if in_code:
            buffer.append(line)
            cursor += line_len
            continue

        # Outside code: split prose on double newlines or headers
        if not line.strip() or line.strip().startswith('#'):
            if buffer:
                atoms.append({"text": "\n".join(buffer).strip(), "start": buffer_start, "is_code": False})
                buffer = []
            if line.strip().startswith('#'):
                buffer.append(line)
                buffer_start = cursor
            else:
                buffer_start = cursor + line_len
        else:
            if not buffer: buffer_start = cursor
            buffer.append(line)

        cursor += line_len

    if buffer:
        atoms.append({"text": "\n".join(buffer).strip(), "start": buffer_start, "is_code": in_code})

    return [a for a in atoms if a["text"]]


def _package(text, fname, h_info, h_starts, start_pos, c_idx):
    pos = bisect.bisect_right(h_starts, start_pos)
    sec = h_info[pos - 1]["section_path"] if h_info and pos > 0 else ""
    clean = text.strip()
    return {
        "content": clean,
        "metadata": {
            "file_name": fname, "chunking_strategy": "semantic_upgraded",
            "chunk_index": c_idx, "section": sec, "characters": len(clean),
            "words": len(clean.split()), "lines": len(clean.splitlines())
        }
    }
