from typing import List, Dict


def retrieve_evidence(
    claim: str,
    story_id: str,
    vector_index,
    character_name: str = None,
    top_k: int = 10,
    min_similarity: float = 0.05,
) -> List[Dict]:
    """
    Dual-query retrieval:
    1) claim
    2) character_name + claim (if provided)
    """

    if not claim or not claim.strip():
        return []

    queries = [claim]
    if character_name:
        queries.append(f"{character_name} {claim}")

    all_results = []

    for q in queries:
        results = vector_index.query(
            query_text=q,
            story_id=story_id,
            top_k=top_k * 2,
            return_scores=True,
        )
        all_results.extend(results)

    # Deduplicate by chunk_id, keep best score
    merged = {}

    for r in all_results:
        cid = r["chunk_id"]
        score = float(r.get("score", 0.0))

        # Always store score explicitly
        r_with_score = dict(r)
        r_with_score["score"] = score

        if cid not in merged or score > merged[cid]["score"]:
            merged[cid] = r_with_score


    # Filter + sort
    final = [
        v for v in merged.values()
        if v.get("score", 0.0) >= min_similarity
    ]
    final.sort(key=lambda x: x["score"], reverse=True)

    return final[:top_k]


if __name__ == "__main__":
    from ingestion.data_ingestion import load_novels
    from indexing.chunking import chunk_all_novels
    from indexing.local_vector_index import LocalVectorIndex


    novels = load_novels("data/novels")
    chunks = chunk_all_novels(novels)

    # Build vector index
    index = LocalVectorIndex()
    index.index_chunks(chunks)


    claim = (
        "Thalcave's people faded as colonists advanced; his father was the last "
        "of the tribal guides and knew the pampas geography and animal ways."
    )

    story_id = "in_search_of_the_castaways"
    character_name = "Thalcave"

    evidence = retrieve_evidence(
        claim=claim,
        story_id=story_id,
        vector_index=index,
        character_name=character_name,
        top_k=8,
    )


    print("\n=== RETRIEVAL TEST OUTPUT ===")
    print(f"Claim: {claim}")
    print(f"Story ID: {story_id}")
    print(f"Character: {character_name}")
    print(f"Retrieved chunks: {len(evidence)}\n")

    for i, e in enumerate(evidence, 1):
        print(f"{'-'*80}")
        print(f"Evidence #{i}")
        print(f"Chunk ID   : {e['chunk_id']}")
        print(f"Position   : {round(e['position'], 3)}")
        print(f"Similarity : {round(e.get('score', 0.0), 4)}")
        print("Text preview:")
        print(e["text"][:400].strip(), "...")
