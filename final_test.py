import pandas as pd
from tqdm import tqdm

from ingestion.data_ingestion import load_novels
from indexing.chunking import chunk_all_novels
from indexing.local_vector_index import LocalVectorIndex
from retrieval.retrieval_evidence import retrieve_evidence, normalize_story_id
from reasoning.claim_reasoner import ClaimReasoner
from config.llm_config import GeminiLLM


# --------------------------------------------------
# Rationale formatter
# --------------------------------------------------
def build_evidence_rationale(claim, evidence_chunks, reasoning_output):
    lines = []

    lines.append("Backstory Claim:")
    lines.append(claim.strip())
    lines.append("")

    lines.append("Relevant Excerpts from the Novel:")
    if evidence_chunks:
        for i, ch in enumerate(evidence_chunks[:5], 1):
            excerpt = ch["text"].strip().replace("\n", " ")
            excerpt = excerpt[:600]
            lines.append(f'[Excerpt {i}] "{excerpt}..."')
    else:
        lines.append("No relevant excerpts were retrieved.")
    lines.append("")

    lines.append("Analysis:")
    explanation = reasoning_output.get("explanation", "").strip()
    if explanation:
        lines.append(explanation)
    else:
        lines.append(
            "The retrieved evidence does not provide sufficient information "
            "to clearly support or contradict the backstory claim."
        )
    lines.append("")

    lines.append("Conclusion:")
    if reasoning_output["label"] == "contradict":
        lines.append("The backstory contradicts the narrative evidence.")
    else:
        lines.append(
            "The backstory does not directly contradict the narrative "
            "and may be considered plausible within the story world."
        )

    return "\n".join(lines)


# --------------------------------------------------
# Final Test Pipeline
# --------------------------------------------------
def main():
    print("=" * 80)
    print("FINAL TEST INFERENCE")
    print("=" * 80)

    df = pd.read_csv("data/test.csv")

    required_cols = {"id", "backstory", "char"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in test.csv: {missing}")

    # Identify story column
    story_col = None
    for c in ["story_id", "book_name", "book", "novel_id"]:
        if c in df.columns:
            story_col = c
            break

    if story_col is None:
        raise ValueError("No story identifier column found in test.csv")

    print(f"Using '{story_col}' as story identifier")

    novels = load_novels("data/novels")
    chunks = chunk_all_novels(novels)

    index = LocalVectorIndex()
    index.index_chunks(chunks)

    llm = GeminiLLM(
        model_name="models/gemini-flash-latest",
        max_output_tokens=1536,
        temperature=0.0,
    )
    reasoner = ClaimReasoner(llm)

    outputs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        example_id = row["id"]
        claim = str(row["backstory"])
        character = str(row["char"])
        story_id = normalize_story_id(str(row[story_col]))

        evidence = retrieve_evidence(
            claim=claim,
            story_id=story_id,
            vector_index=index,
            character_name=character,
            top_k=8,
        )

        reasoning = reasoner.verify_claim(claim, evidence)

        # -------------------------------
        # Evidence-aware label decision
        # -------------------------------
        if reasoning["label"] == "contradict":
            final_label = 0
        elif evidence:
            final_label = 1
        else:
            final_label = 0  # no evidence → cannot assert consistency

        rationale = build_evidence_rationale(
            claim=claim,
            evidence_chunks=evidence,
            reasoning_output=reasoning,
        )

        outputs.append({
            "id": example_id,
            "prediction": final_label,
            "evidence_rationale": rationale,
        })

    pd.DataFrame(outputs).to_csv("result.csv", index=False)
    print("\nSaved predictions to result.csv")


if __name__ == "__main__":
    main()
