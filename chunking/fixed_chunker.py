from chunking.utils import get_sections_with_hierarchy
import bisect

def fixed_chunk_document(documents,n=0, chunk_size=512):
    chunks=[]
    content=documents[n]["content"]
    metadata = documents[n].get("metadata", {})
    file_name = (
        metadata.get("file_name")
        or metadata.get("name")
        or metadata.get("title")
        or "document"
    )
    header_info=get_sections_with_hierarchy(content)
    header_starts=[h["start_char"] for h in header_info]
    for i in range(0, len(content), chunk_size):
        chunk=content[i:i+chunk_size]
        pos = bisect.bisect_right(header_starts, i)

        section_path = (
            header_info[pos - 1]["section_path"]
            if header_info and pos > 0
            else ""
        )
        if chunk:
            chunks.append({"content": chunk,
                          "metadata": {
                                "file_name": file_name,
                                "chunking_strategy": "fixed",
                                "chunk_index": i//chunk_size,
                                "start_char": i,
                                "section": section_path,
                                "words": len(chunk.split()),
                                "lines": len(chunk.splitlines())
                          }}
                          )
    return chunks

def fixed_chunk_documents(documents, chunk_size=512):
    all_chunks=[]
    for n in range(len(documents)):
        all_chunks.extend(fixed_chunk_document(documents, n=n, chunk_size=chunk_size))
    return all_chunks