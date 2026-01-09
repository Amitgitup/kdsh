import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from pipeline import NarrativeConsistencyPipeline
from ingestion.data_ingestion import load_dataset


def evaluate_on_train():
    pipeline = NarrativeConsistencyPipeline()

    train_data = load_dataset("data/train.csv", is_train=True)

    y_true = []
    y_pred = []

    for i, row in enumerate(train_data, 1):
        print(f"\n[{i}/{len(train_data)}] Processing example...")

        result = pipeline.predict(
            claim=row["backstory"],
            story_id=row["story_id"],
            character_name=row.get("character", None),
        )

        y_true.append(row["label"])
        y_pred.append(result["label"])

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\nAccuracy:")
    print(accuracy_score(y_true, y_pred))

    print("\nPrecision / Recall / F1:")
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    print(f"Precision: {p:.4f}")
    print(f"Recall   : {r:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate_on_train()
