from chunking.utils import get_sections_with_hierarchy
import bisect

def sliding_chunk_document(documents,n=0, chunk_size=512, overlap=128):
    chunks=[]
    metadata = documents[n].get("metadata", {})
    file_name = (
        metadata.get("file_name")
        or metadata.get("name")
        or metadata.get("title")
        or "document"
    )
    step=chunk_size-overlap
    header_info=get_sections_with_hierarchy(documents[n]["content"])
    header_starts=[h["start_char"] for h in header_info]
    for i in range(0, len(documents[n]["content"]), step):
        chunk=documents[n]["content"][i:i+chunk_size]
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
                                "chunking_strategy": "sliding",
                                "chunk_index": i//chunk_size,
                                "section": section_path,
                                "start_char": i,
                                "end_char": i+len(chunk),
                                "characters": len(chunk),
                                "words": len(chunk.split()),
                                "lines": len(chunk.splitlines())
                          }}
                          )
    return chunks

def sliding_chunk_documents(documents, chunk_size=512, overlap=128):
    all_chunks=[]
    for n in range(len(documents)):
        all_chunks.extend(sliding_chunk_document(documents, n=n, chunk_size=chunk_size, overlap=overlap))
    return all_chunks