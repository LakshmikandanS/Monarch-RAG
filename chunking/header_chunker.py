from chunking.utils import get_sections_with_hierarchy
import re
import bisect

def header_aware_chunk_documents(documents):
    chunks=[]
    for i, doc in enumerate(documents):
        sections = re.split(r'\n(?=#)', doc["content"])
        file_name=doc["metadata"]["file_name"]
        header_info=get_sections_with_hierarchy(doc["content"])
        header_starts=[h["start_char"] for h in header_info]
        current_char_pos = 0
        for idx, section in enumerate(sections):
            if section.strip():
                section_start = doc["content"].find(section, current_char_pos)
                pos = bisect.bisect_right(header_starts, section_start)
                section_path = (
                    header_info[pos - 1]["section_path"]
                    if header_info and pos > 0
                    else ""
                )
                clean_content = re.sub(r'^\s*#+.*(?:\n|$)', '', section).strip()
                chunks.append({
                    "content": clean_content,
                    "metadata": {
                        "file_name": file_name,
                        "chunking_strategy": "header_aware",
                        "chunk_index": idx,
                        "section": section_path,
                        "characters": len(clean_content),
                        "words": len(clean_content.split()),
                        "lines": len(clean_content.splitlines())
                    }
                })
                current_char_pos = section_start + len(section)
    return chunks
