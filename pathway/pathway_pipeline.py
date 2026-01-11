import pathway as pw
from pathlib import Path


# ----------------------------
# Define schema (IMMUTABLE)
# ----------------------------
class Novel(pw.Schema):
    story_id: str
    text: str


# ----------------------------
# Load novels from mounted volume
# ----------------------------
def load_novels(folder: str):
    data = []

    for file in Path(folder).glob("*.txt"):
        data.append(
            (
                file.stem.lower().replace(" ", "_"),
                file.read_text(encoding="utf-8"),
            )
        )

    if not data:
        raise RuntimeError(f"No novels found in {folder}")

    return data


# ----------------------------
# Main Pathway pipeline
# ----------------------------
def main():
    novels_dir = "/data/novels"

    rows = load_novels(novels_dir)

    # ✅ OFFICIAL Pathway constructor
    novels = pw.Table.from_rows(
        rows,
        schema=Novel,   # ← keyword ONLY
    )

    pw.io.jsonlines.write(novels, "/data/pathway_novels.json")

    print(f"✅ Pathway ingested {len(rows)} novels")

    pw.run()


if __name__ == "__main__":
    main()
