from typing import Dict, List


# -----------------------------
# Chunking configuration
# -----------------------------
CHUNK_SIZE_CHARS = 3500
OVERLAP_CHARS = 700


def chunk_novel(
    story_id: str,
    full_text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> List[dict]:
    """
    Splits a novel into overlapping chunks with metadata.
    """

    if overlap >= chunk_size:
        raise ValueError("OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS")

    chunks = []
    text_length = len(full_text)

    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = full_text[start:end].strip()

        # Skip empty / whitespace-only chunks
        if not chunk_text:
            start += step
            continue

        chunk = {
            "chunk_id": f"{story_id}_{chunk_index:05d}",
            "story_id": story_id,
            "text": chunk_text,
            "start_char": start,
            "end_char": end,
            "position": start / text_length,
        }

        chunks.append(chunk)

        start += step
        chunk_index += 1

    return chunks


def chunk_all_novels(novels: Dict[str, str]) -> List[dict]:
    """
    Chunks all novels.

    Input:
        { story_id: full_text }

    Output:
        List of chunk dicts
    """
    all_chunks = []

    for story_id, full_text in novels.items():
        novel_chunks = chunk_novel(story_id, full_text)
        all_chunks.extend(novel_chunks)

    return all_chunks


if __name__ == "__main__":
    from ingestion.data_ingestion import load_novels

    novels = load_novels("data/novels")
    chunks = chunk_all_novels(novels)

    print(f"Total chunks: {len(chunks)}")

    sample = chunks[0]
    print("\nSample chunk:")
    for k, v in sample.items():
        if k == "text":
            print(f"{k}: {v[:300]}...")
        else:
            print(f"{k}: {v}")
