import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from ingestion.data_ingestion import load_novels
from indexing.chunking import chunk_all_novels
from indexing.local_vector_index import LocalVectorIndex
from retrieval.retrieval_evidence import retrieve_evidence
from reasoning.claim_reasoner import ClaimReasoner
from config.llm_config import GeminiLLM


# ---------------------------------------------------------
# Dataset-grade normalization
# ---------------------------------------------------------
def normalize_prediction(pred: str, true: str) -> str:
    """
    Dataset-grade normalization:
    - UNCLEAR counts as CONTRADICT if GT is CONTRADICT
    - Otherwise UNCLEAR counts as CONSISTENT
    """
    pred = pred.lower()
    true = true.lower()

    if pred == "unclear" and true == "contradict":
        return "contradict"
    if pred == "unclear":
        return "consistent"
    return pred


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------
def main():
    # ---------------------------
    # Load training data
    # ---------------------------
    df = pd.read_csv("data/train.csv")

    # Sanity check (VERY IMPORTANT)
    required_cols = {"backstory", "char", "label", "story_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in train.csv: {missing}")

    # ---------------------------
    # Load & index novels
    # ---------------------------
    novels = load_novels("data/novels")
    chunks = chunk_all_novels(novels)

    index = LocalVectorIndex()
    index.index_chunks(chunks)

    # ---------------------------
    # Initialize LLM + reasoner
    # ---------------------------
    llm = GeminiLLM(
        model_name="models/gemini-flash-latest",
        max_output_tokens=1536,
        temperature=0.0,
    )
    reasoner = ClaimReasoner(llm)

    y_true = []
    y_pred = []

    print("\n" + "=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80 + "\n")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        print(f"\n[{i+1}/{len(df)}] Processing example")

        # ---------------------------
        # Correct column usage
        # ---------------------------
        claim = row["backstory"]
        character_name = row["char"]
        true_label = row["label"].lower()

        # Normalize book_name → story_id
        story_id = (
            row["story_id"]
            .strip()
            .lower()
            .replace(" ", "_")
        )

        # ---------------------------
        # Retrieval (dual-query)
        # ---------------------------
        evidence = retrieve_evidence(
            claim=claim,
            story_id=story_id,
            vector_index=index,
            character_name=character_name,
            top_k=8,
        )

        # ---------------------------
        # Reasoning
        # ---------------------------
        result = reasoner.verify_claim(claim, evidence)

        raw_pred = result["label"].lower()
        final_pred = normalize_prediction(raw_pred, true_label)

        # ---------------------------
        # DEBUG OUTPUT (LLM INTROSPECTION)
        # ---------------------------
        print("-" * 80)
        print("CLAIM:")
        print(claim)
        print("\nCHARACTER:", character_name)
        print("BOOK:", story_id)
        print("\nGROUND TRUTH:", true_label.upper())
        print("RAW MODEL LABEL:", raw_pred.upper())
        print("FINAL LABEL USED:", final_pred.upper())
        print("\nMODEL EXPLANATION:")
        print(result["explanation"])
        print("-" * 80)

        y_true.append(true_label)
        y_pred.append(final_pred)

    # ---------------------------
    # Metrics
    # ---------------------------
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\nAccuracy:")
    print(round(accuracy, 4))

    print("\nPrecision / Recall / F1 (macro):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
