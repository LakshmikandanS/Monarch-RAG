import bisect
from chunking.utils import get_sections_with_hierarchy


def recursive_chunk_document(documents, n=0, chunk_size=512, overlap=128):
    chunks = []
    content = documents[n]["content"]
    
    # Metadata setup matching your style
    metadata = documents[n].get("metadata", {})
    file_name = (
        metadata.get("file_name")
        or metadata.get("name")
        or metadata.get("title")
        or "document"
    )
    
    header_info = get_sections_with_hierarchy(content)
    header_starts = [h["start_char"] for h in header_info]
    
    # Internal recursive text splitter
    def split_text(text, start_idx, separators=["\n\n", "\n", " ", ""]):
        if len(text) <= chunk_size:
            return [(text, start_idx)]
        
        # Pick the best separator
        separator = separators[-1]
        for s in separators:
            if s in text:
                separator = s
                break
                
        splits = []
        parts = text.split(separator) if separator != "" else list(text)
        
        current_piece = ""
        current_start = start_idx
        
        for part in parts:
            # Reconstruct the text with its original separator
            join_str = separator if current_piece else ""
            test_piece = current_piece + join_str + part
            
            if len(test_piece) <= chunk_size:
                current_piece = test_piece
            else:
                if current_piece:
                    splits.append((current_piece, current_start))
                    # Handle overlap for the next chunk
                    overlap_size = min(overlap, len(current_piece))
                    if overlap_size > 0:
                        current_piece = current_piece[-overlap_size:] + join_str + part
                        current_start = current_start + (len(splits[-1][0]) - overlap_size)
                    else:
                        current_piece = part
                        current_start = current_start + len(splits[-1][0]) + len(join_str)
                else:
                    # Single part is larger than chunk_size, split recursively with finer separators
                    finer_splits = split_text(part, current_start, separators[separators.index(separator)+1:])
                    splits.extend(finer_splits)
                    
        if current_piece:
            splits.append((current_piece, current_start))
            
        return splits

    # Run the recursive partitioner
    text_splits = split_text(content, 0)
    
    for idx, (chunk_text, start_char) in enumerate(text_splits):
        if not chunk_text.strip():
            continue
            
        # Dynamically find hierarchy section path via bisect
        pos = bisect.bisect_right(header_starts, start_char)
        section_path = (
            header_info[pos - 1]["section_path"]
            if header_info and pos > 0
            else ""
        )
        
        chunks.append({
            "content": chunk_text,
            "metadata": {
                "file_name": file_name,
                "chunking_strategy": "recursive",
                "chunk_index": idx,
                "section": section_path,
                "start_char": start_char,
                "end_char": start_char + len(chunk_text),
                "characters": len(chunk_text),
                "words": len(chunk_text.split()),
                "lines": len(chunk_text.splitlines())
            }
        })
        
    return chunks

def recursive_chunk_documents(documents, chunk_size=512, overlap=128):
    all_chunks = []
    for n in range(len(documents)):
        all_chunks.extend(recursive_chunk_document(documents, n=n, chunk_size=chunk_size, overlap=overlap))
    return all_chunks