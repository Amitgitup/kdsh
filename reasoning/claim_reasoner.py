import re
from typing import List, Dict

from config.prompt_templates import CLAIM_VERIFICATION_PROMPT


class ClaimReasoner:
    """
    Layer 5: Reasoning / Claim Verification
    """

    def __init__(self, llm_client):
        self.llm = llm_client


    def verify_claim(
        self,
        claim: str,
        evidence_chunks: List[Dict],
    ) -> Dict:

        if not claim or not claim.strip():
            return {
                "label": "unclear",
                "explanation": "Empty claim provided."
            }

        if not evidence_chunks:
            return {
                "label": "unclear",
                "explanation": "No relevant evidence was retrieved."
            }

        evidence_blocks = self._format_evidence(evidence_chunks)

        prompt = CLAIM_VERIFICATION_PROMPT.format(
            claim=claim,
            evidence_blocks=evidence_blocks
        )

        raw_output = self.llm.generate(prompt) or ""

        print("\n----- RAW LLM OUTPUT -----")
        print(raw_output)
        print("----- END RAW OUTPUT -----\n")

        return self._parse_llm_output(raw_output)


    # --------------------------------------------------
    # Evidence formatting
    # --------------------------------------------------
    def _format_evidence(self, evidence_chunks: List[Dict]) -> str:
        blocks = []
        for i, chunk in enumerate(evidence_chunks, 1):
            # 🔑 truncate to avoid token overflow
            text = chunk["text"].strip()[:800]

            blocks.append(
                f"[Evidence {i}]\n{text}"
            )

        return "\n\n".join(blocks)


    # --------------------------------------------------
    # Output parsing
    # --------------------------------------------------
    def _parse_llm_output(self, text: str) -> Dict:
        text = text.strip()
        text = re.sub(r"[*_`]", "", text)

        label_match = re.search(
            r"Final Label\s*:\s*(CONSISTENT|CONTRADICT|UNCLEAR)",
            text,
            re.IGNORECASE,
        )

        explanation_match = re.search(
            r"Final Explanation\s*:\s*(.*)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if not label_match:
            return {
                "label": "unclear",
                "explanation": "Model output could not be parsed reliably."
            }

        label = label_match.group(1).lower()

        explanation = (
            explanation_match.group(1).strip()
            if explanation_match
            else "No explanation provided."
        )

        return {
            "label": label,
            "explanation": explanation
        }



# Local test
if __name__ == "__main__":
    from ingestion.data_ingestion import load_novels
    from indexing.chunking import chunk_all_novels
    from indexing.local_vector_index import LocalVectorIndex
    from retrieval.retrieval_evidence import retrieve_evidence
    from config.llm_config import GeminiLLM

    novels = load_novels("data/novels")
    chunks = chunk_all_novels(novels)

    index = LocalVectorIndex()
    index.index_chunks(chunks)

    claim = (
        "Thalcave's people faded as colonists advanced; "
        "his father was the last of the tribal guides and knew "
        "the pampas geography and animal ways."
    )

    story_id = "in_search_of_the_castaways"

    evidence = retrieve_evidence(
        claim=claim,
        story_id=story_id,
        vector_index=index,
        character_name="Thalcave",
        top_k=8,
    )

    llm = GeminiLLM(model_name="models/gemini-flash-latest")
    reasoner = ClaimReasoner(llm)

    result = reasoner.verify_claim(claim, evidence)

    print("\n=== REASONING OUTPUT ===")
    print("Label:", result["label"])
    print("Explanation:", result["explanation"])
