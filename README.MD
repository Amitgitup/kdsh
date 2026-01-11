# 📚 Narrative Backstory Consistency Verification

**Kharagpur Data Science Hackathon — Track A**

---

## 🔍 Problem Overview
This project addresses the challenging task of verifying whether a hypothetical character backstory is **consistent** with the full text of a long-form literary narrative.

**Each input example consists of:**
- A complete novel (no truncation or summaries)
- A hypothetical backstory describing a character’s past, beliefs, or motivations

**The task:**  
Determine if the backstory:
- **Contradicts** the narrative
- or is **Consistent/Plausible** in the story world

Optionally, the system provides a *comprehensive evidence rationale* directly grounded in textual excerpts from the novel.

---

## 🧠 High-Level System Design

The system is a **modular reasoning pipeline** emphasizing interpretability and robustness:
- **Ingestion:** Load and preprocess entire novels
- **Chunking:** Split narratives into overlapping, coherent semantic chunks
- **Vector Indexing:** Embed all chunks for fast local semantic search
- **Claim-driven Retrieval:** Retrieve evidence passages directly relevant to each backstory
- **LLM-Based Reasoning:** Compare claim and retrieved text, explicitly seeking contradictions or plausibility
- **Decision Logic:** Generate evidence-grounded verdicts & rationales

```
┌──────────────────┐
│   Test Example   │
│ (Backstory, ID)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│ Long Narrative Ingestion │
│ + Semantic Chunking      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Vector Index (Embeddings)│
│  • Local semantic search │
│  • Story-aware filtering │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Evidence Retrieval       │
│  • Claim-based queries   │
│  • Character-aware       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Claim Reasoning (LLM)    │
│  • Literary reasoning    │
│  • Not fact checking     │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Final Decision +         │
│ Evidence Rationale       │
└──────────────────────────┘
```

---

## 📖 Narrative Handling Strategy

### 1. Narrative Chunking
- Novels are split into overlapping, locally-coherent chunks.
- Each chunk includes metadata: `story_id`, novel-relative position, and raw text.

### 2. Vector Indexing
- Chunks are embedded using `sentence-transformers` and stored in a local FAISS vector index.
- **Supports efficient, story-specific semantic retrieval.**

### 3. Claim-driven Retrieval
- For each backstory, the system issues two queries:
  1. The backstory itself
  2. The backstory with the character's name for additional context
- Results are deduplicated and filtered by a relevance threshold.

### 4. LLM Reasoning & Decision
- LLM is prompted to *reason* about consistency—distinct from fact-checking.
- It labels each claim as **CONTRADICT**, **CONSISTENT**, or **UNCLEAR** based on direct textual evidence.
- Conservative bias: Only mark “consistent” with supporting context.

### 5. Evidence Rationale Generation
- For every test case, the system produces:
  - The backstory
  - Direct textual excerpts
  - Short analytic commentary
  - An explicit, evidence-based conclusion

---

## 🏗️ Project Structure

```
.
├── ingestion/        # Novel loading and preprocessing
├── indexing/         # Chunking and vector indexing
├── retrieval/        # Evidence retrieval logic
├── reasoning/        # Claim reasoning module
├── config/           # LLM configuration and prompting
├── data/
│   ├── novels/       # Full novel texts
│   ├── train.csv
│   └── test.csv
├── evaluate.py       # Training-time evaluation
├── final_test.py     # Final test inference
├── result.csv        # Submission output
├── requirements.txt
└── setup.py
```

---

## 🚀 Setup & Usage

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd KDSH

# [Optional] Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
- Place all uncompressed novel `.txt` files in `data/novels/`
- Place `train.csv` and `test.csv` in `data/`

### 3. Running Evaluation & Inference

**Training-time Evaluation:**
```bash
python evaluate.py
```

**Final Test Inference:**
```bash
python final_test.py
```
Outputs will be written to `result.csv`.

---

## 📝 Submission Output

Each row in `result.csv` contains:
- `id`: Example identifier
- `prediction`: 0 (contradict) or 1 (consistent/unclear with evidence)
- `evidence_rationale`: Transparent, evidence-based rationale

---

## ⚙️ Requirements

Main dependencies:
- numpy, pandas, scikit-learn
- sentence-transformers
- faiss-cpu
- pathway, python-dotenv
- google-genai

Full list in `requirements.txt`.

---

## 💡 Design Choices

- **Separation of retrieval and reasoning** ensures interpretability.
- **Chunked semantic retrieval** makes handling long novels tractable and efficient.
- **Conservative, evidence-backed decisions** are prioritized over risky speculation.
- **Modular design** allows easy adaptation to new reasoning models, retrieval strategies, or data formats.

---

## 🧩 Final Notes

- The system favors **robust reasoning over aggressive prediction**.
- **Outputs are always evidence-backed and transparent**.
- Principles of transparency, rigor, and literary sensitivity guide all design decisions.

---
