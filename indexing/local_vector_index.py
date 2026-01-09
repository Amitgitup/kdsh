from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class LocalVectorIndex:
    """
    Local FAISS-based vector index for narrative chunks.
    Uses cosine similarity via normalized inner product.
    """

    def __init__(self, embedding_model: str = "BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(embedding_model)

        self.index = None                 # FAISS index
        self.chunks: List[Dict] = []      # chunk metadata
        self.story_ids: List[str] = []    # parallel list for filtering


    # --------------------------------------------------
    # Indexing
    # --------------------------------------------------
    def index_chunks(self, chunks: List[Dict]) -> None:
        """
        Build FAISS index from chunk dictionaries.
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        dim = embeddings.shape[1]

        # Exact cosine similarity search
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.chunks = chunks
        self.story_ids = [chunk["story_id"] for chunk in chunks]

        print(f"✅ Indexed {len(chunks)} chunks (dim={dim})")

    # --------------------------------------------------
    # Querying (Layer 3 primitive)
    # --------------------------------------------------
    def query(
        self,
        query_text: str,
        story_id: str,
        top_k: int = 50,
        return_scores: bool = True,
    ) -> List[Dict]:
        """
        Retrieve candidate chunks for a query.
        This function OVER-FETCHES and DOES NOT truncate to top_k.
        Layer 4 decides how many to keep.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call index_chunks() first.")

        query_vec = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        search_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, search_k)

        results = []
        seen = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx in seen:
                continue
            if self.story_ids[idx] != story_id:
                continue

            seen.add(idx)

            chunk = dict(self.chunks[idx])
            if return_scores:
                # 🔑 STANDARDIZED KEY NAME
                chunk["score"] = float(score)

            results.append(chunk)

        return results


if __name__ == "__main__":
    from ingestion.data_ingestion import load_novels
    from indexing.chunking import chunk_all_novels

    novels = load_novels("data/novels")
    chunks = chunk_all_novels(novels)

    index = LocalVectorIndex()
    index.index_chunks(chunks)

    query = "Thalcave's people faded as colonists advanced; his father was the last of the tribal guides and knew the pampas geography and animal ways."
    story_id = "in_search_of_the_castaways"

    results = index.query(query, story_id, top_k=50)

    print("\nQuery results:")
    for r in results[:5]:
        print("-" * 40)
        print(
            r["chunk_id"],
            "| pos:", round(r["position"], 3),
            "| score:", round(r["score"], 3),
        )
        print(r["text"][:300], "...")
