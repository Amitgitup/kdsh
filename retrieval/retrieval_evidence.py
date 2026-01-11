from typing import List, Dict


def normalize_story_id(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def retrieve_evidence(
    claim: str,
    story_id: str,
    vector_index,
    character_name: str = None,
    top_k: int = 10,
    min_similarity: float = 0.03,
) -> List[Dict]:
    """
    Robust dual-query retrieval with fallback.
    """

    if not claim or not claim.strip():
        return []

    story_id_norm = normalize_story_id(story_id)

    queries = [claim]
    if character_name and character_name.strip():
        queries.append(f"{character_name} {claim}")

    all_results = []

    # -------------------------------
    # Stage 1: strict (story filter)
    # -------------------------------
    for q in queries:
        results = vector_index.query(
            query_text=q,
            story_id=story_id_norm,
            top_k=top_k * 3,
            return_scores=True,
        )
        all_results.extend(results)

    # -------------------------------
    # Stage 2: fallback (no story filter)
    # -------------------------------
    if not all_results:
        for q in queries:
            results = vector_index.query(
                query_text=q,
                story_id=None,  # <-- fallback
                top_k=top_k * 3,
                return_scores=True,
            )
            all_results.extend(results)

    if not all_results:
        return []

    # -------------------------------
    # Deduplicate by chunk_id
    # -------------------------------
    merged = {}
    for r in all_results:
        cid = r["chunk_id"]
        score = float(r.get("score", 0.0))
        r = dict(r)
        r["score"] = score

        if cid not in merged or score > merged[cid]["score"]:
            merged[cid] = r

    final = [
        v for v in merged.values()
        if v["score"] >= min_similarity
    ]
    final.sort(key=lambda x: x["score"], reverse=True)

    return final[:top_k]


if __name__ == "__main__":
    from ingestion.data_ingestion import load_novels
    from indexing.chunking import chunk_all_novels
    from indexing.local_vector_index import LocalVectorIndex

    novels = load_novels("data/novels")
    chunks = chunk_all_novels(novels)

    index = LocalVectorIndex()
    index.index_chunks(chunks)

    claim = (
        "Thalcave's people faded as colonists advanced; his father was the last "
        "of the tribal guides and knew the pampas geography and animal ways."
    )

    evidence = retrieve_evidence(
        claim=claim,
        story_id="In Search of the Castaways",
        vector_index=index,
        character_name="Thalcave",
        top_k=8,
    )

    print(f"Retrieved {len(evidence)} chunks")
    for e in evidence:
        print(round(e["score"], 4), e["text"][:200])
