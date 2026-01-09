import os
import csv
from typing import Dict, List
from ingestion.text_cleaning import strip_gutenberg_text

# Load full novels
def load_novels(novels_dir: str) -> Dict[str, str]:
    """
    Loads all novels from a directory.

    Returns:
        {
            story_id: full_text
        }
    """
    novels = {}

    for filename in os.listdir(novels_dir):
        if not filename.endswith(".txt"):
            continue

        story_id = filename.replace(".txt", "").strip().lower().replace(" ", "_")

        file_path = os.path.join(novels_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            full_text = strip_gutenberg_text(full_text)

        if len(full_text) < 10000:
            raise ValueError(
                f"Novel {filename} seems too short. Possible read error."
            )

        novels[story_id] = full_text

    if not novels:
        raise ValueError("No novels loaded. Check novels directory.")

    return novels


# Load train / test CSV
def load_dataset(csv_path: str, is_train: bool = True) -> List[dict]:
    """
    Loads train or test CSV.

    Returns:
        [
            {
                "story_id": str,
                "backstory": str,
                "label": int (only for train)
            }
        ]
    """
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {"story_id", "backstory"}
        if is_train:
            required_cols.add("label")

        if not required_cols.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV {csv_path} missing required columns: {required_cols}"
            )

        for row in reader:
            example = {
                "story_id": row["story_id"].strip().lower().replace(" ", "_"),
                "backstory": row["backstory"].strip(),
            }

            if is_train:
                label_str = row["label"].strip().lower()
                if label_str not in {"consistent", "contradict"}:
                    raise ValueError(f"Unknown label: {row['label']}")
                example["label"] = label_str

            rows.append(example)

    if not rows:
        raise ValueError(f"No rows loaded from {csv_path}")

    return rows


if __name__ == "__main__":
    novels = load_novels("data/novels")
    train_data = load_dataset("data/train.csv", is_train=True)
    test_data = load_dataset("data/test.csv", is_train=False)

    print("INGESTION SUMMARY")
    print(f"Loaded novels: {len(novels)}")
    for sid, text in novels.items():
        print(f"  {sid}: {len(text)} characters")

    print(f"\nLoaded train rows: {len(train_data)}")
    print(f"Loaded test rows: {len(test_data)}")

    # Mapping sanity check
    missing = [
        row["story_id"] for row in train_data
        if row["story_id"] not in novels
    ]

    if missing:
        raise ValueError(
            f"Train rows reference missing novels: {set(missing)}"
        )

    print("\n Story ID mapping verified")
