from typing import Dict, List

from ingestion.data_ingestion import load_novels, load_dataset
from indexing.chunking import chunk_all_novels
from indexing.local_vector_index import LocalVectorIndex
from retrieval.retrieval_evidence import retrieve_evidence
from reasoning.claim_reasoner import ClaimReasoner
from config.llm_config import GeminiLLM


class NarrativeConsistencyPipeline:
    def __init__(self):
        # Load data
        novels = load_novels("data/novels")
        chunks = chunk_all_novels(novels)

        # Build vector index (once)
        self.index = LocalVectorIndex()
        self.index.index_chunks(chunks)

        # LLM
        self.llm = GeminiLLM(
            model_name="models/gemini-flash-latest",
            temperature=0.0,
            max_output_tokens=1536,
        )

        self.reasoner = ClaimReasoner(self.llm)

    def predict(
        self,
        claim: str,
        story_id: str,
        character_name: str = None,
        top_k: int = 8,
    ) -> Dict:
        """
        Runs full pipeline for a single claim.
        """

        evidence = retrieve_evidence(
            claim=claim,
            story_id=story_id,
            vector_index=self.index,
            character_name=character_name,
            top_k=top_k,
        )

        result = self.reasoner.verify_claim(claim, evidence)

        return {
            "label": result["label"],
            "explanation": result["explanation"],
            "num_evidence": len(evidence),
        }



