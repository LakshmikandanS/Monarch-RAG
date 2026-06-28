import bisect
from typing import Dict, List
from chunking.utils import get_sections_with_hierarchy

def code_aware_chunk_documents(
    documents: List[Dict], max_chars: int = 1000, overlap: int = 200
) -> List[Dict]:
    """
    Sliding window chunker that enforces Abstract Syntax Tree (AST) integrity.
    Uses Absolute Character Tracking to guarantee perfect header alignment with RRF.
    """
    chunks = []
    
    for doc in documents:
        # Standardize newlines before doing any length math
        content = doc["content"].replace('\r\n', '\n')
        metadata = doc.get("metadata", {})
        
        file_name = (
            metadata.get("file_name")
            or metadata.get("name")
            or metadata.get("title")
            or "document"
        )
        
        # Build the hierarchical mapping for this document
        header_info = get_sections_with_hierarchy(content)
        header_starts = [h["start_char"] for h in header_info]
        
        paragraphs = content.split('\n')
        current_chunk_lines = []
        
        # Absolute tracking variables
        current_char_pos = 0     # The parser's current position in the document
        chunk_start_char = 0     # Where the current chunk began
        current_length = 0       # Length of the current chunk
        chunk_idx = 0
        
        for line in paragraphs:
            current_chunk_lines.append(line)
            line_len = len(line) + 1  # +1 to account for the '\n' we split on
            current_length += line_len
            current_char_pos += line_len
            
            # Check if we should sever the chunk
            if current_length >= max_chars:
                temp_text = "\n".join(current_chunk_lines)
                open_ticks = temp_text.count("```")
                
                # AST Shield: If ticks are even, we are safely OUTSIDE a code block
                if open_ticks % 2 == 0:
                    clean_content = temp_text.strip()
                    
                    if clean_content:
                        # 1. Use the mathematical absolute position to map the header
                        pos = bisect.bisect_right(header_starts, chunk_start_char)
                        section_path = (
                            header_info[pos - 1]["section_path"]
                            if header_info and pos > 0
                            else ""
                        )
                        
                        # 2. Package with identical metadata schema
                        chunks.append({
                            "content": clean_content,
                            "metadata": {
                                "file_name": file_name,
                                "chunking_strategy": "code_aware_sliding",
                                "chunk_index": chunk_idx,
                                "section": section_path,
                                "characters": len(clean_content),
                                "words": len(clean_content.split()),
                                "lines": len(clean_content.splitlines())
                            }
                        })
                        chunk_idx += 1
                    
                    # 3. Slide the window back for the overlap
                    overlap_lines = []
                    overlap_length = 0
                    for rev_line in reversed(current_chunk_lines):
                        # Don't let overlap exceed our defined limit
                        if overlap_length + len(rev_line) + 1 > overlap:
                            break
                        overlap_lines.insert(0, rev_line)
                        overlap_length += len(rev_line) + 1
                    
                    # AST Overlap Shield: Drop overlap if it contains partial code blocks
                    overlap_check = "\n".join(overlap_lines)
                    if overlap_check.count("```") % 2 != 0:
                        overlap_lines = []
                        overlap_length = 0
                    
                    current_chunk_lines = overlap_lines
                    current_length = overlap_length
                    
                    # Track where the *next* chunk starts mathematically
                    chunk_start_char = current_char_pos - overlap_length
        
        # Flush the remaining text in the document at the very end
        if current_chunk_lines:
            temp_text = "\n".join(current_chunk_lines).strip()
            if temp_text:
                pos = bisect.bisect_right(header_starts, chunk_start_char)
                section_path = (
                    header_info[pos - 1]["section_path"]
                    if header_info and pos > 0
                    else ""
                )
                
                chunks.append({
                    "content": temp_text,
                    "metadata": {
                        "file_name": file_name,
                        "chunking_strategy": "code_aware_sliding",
                        "chunk_index": chunk_idx,
                        "section": section_path,
                        "characters": len(temp_text),
                        "words": len(temp_text.split()),
                        "lines": len(temp_text.splitlines())
                    }
                })

    return chunks