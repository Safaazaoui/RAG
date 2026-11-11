# Enterprise Knowledge RAG Assistant & Evaluation Suite

A production-grade Retrieval-Augmented Generation (RAG) pipeline designed to ingest multi-format unstructured documents, perform dense vector similarity search, rerank context candidates, and synthesize grounded, hallucination-free answers with verifiable source attribution.

Includes an automated **RAG Triad Evaluation Suite** to continuously benchmark answer faithfulness and retrieval accuracy.

---

##  System Architecture & Execution Flow

```text
User Query / Document Ingestion
          │
          ▼
┌──────────────────┐
│  Data Loaders    │  (Parses PDF, TXT, CSV, DOCX, JSON via LangChain)
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Context Chunking │  (1,000-char sliding windows w/ 200-char overlap)
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Vector Embedding │  (Sentence-Transformers `all-MiniLM-L6-v2` -> 384-dim)
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│   FAISS Index    │  (Persisted vector store on disk for instant retrieval)
└─────────┬────────┘
          │  (Top-K Similarity Search)
          ▼
┌──────────────────┐
│  Cross-Encoder   │  (Reranks top context candidates via SentenceTransformers)
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│  LLM Synthesis   │  (OpenAI GPT-4o / gpt-4o-mini with Grounded Prompts)
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Automated Eval   │  (Pydantic Structured Evaluation for Faithfulness)
└──────────────────┘
```

---

##  Key Engineering Features

* **Multi-Format Ingestion:** Ingests and processes `.pdf`, `.txt`, `.csv`, `.xlsx`, `.docx`, and `.json` files seamlessly.
* **Optimized Chunking Strategy:** Implements 1,000-character sliding-window chunking with 200-character overlap to retain context across sentence boundaries.
* **Dense Vector Search:** Encodes text into 384-dimensional embeddings using Sentence-Transformers (`all-MiniLM-L6-v2`) and indexes vectors locally using FAISS.
* **Cross-Encoder Reranking (`rerank.py`):** Filters raw vector retrieval results using a second-pass cross-encoder model to eliminate context noise and lower token costs.
* **Automated Evaluation Benchmarking (`evaluate.py`):** Uses OpenAI Structured Outputs via Pydantic schemas to calculate hallucination risk and score response faithfulness (0.0 to 1.0).
* **Interactive UI (`app.py`):** Streamlit web UI featuring document upload controls, real-time query handling, and expandable citation trees.

---

## Project Structure

```text
RAG/
├── app.py                # Interactive Streamlit Web Interface
├── evaluate.py           # Automated RAG Faithfulness & Evaluation Suite
├── rerank.py             # Cross-encoder reranking & candidate filtering
├── src/
│   ├── data_loader.py    # Multi-format document parser (PDF, TXT, CSV, DOCX, JSON)
│   ├── embedding.py      # Chunking strategy & Sentence-Transformer pipeline
│   ├── vectorstore.py    # FAISS index builder, disk persister, and query engine
│   └── search.py         # End-to-end retrieval + LLM synthesis engine (RAGSearch)
├── notebook/
│   ├── document.ipynb    # Ingestion exploration walkthrough
│   └── pdf_loader.ipynb  # ChromaDB prototype with confidence scoring
├── data/
│   ├── pdf/              # Document intake (PDFs)
│   └── text_files/       # Document intake (Text files)
├── faiss_store/          # Auto-generated local FAISS vector store
├── pyproject.toml        # Dependencies & package specification
└── uv.lock               # Lockfile for reproducible builds
```

---

## Getting Started

### Prerequisites
* Python 3.12+
* [`uv`](https://github.com/astral-sh/uv) (Fast Python package installer)
* OpenAI API Key

### 1. Installation
```bash
git clone https://github.com/Safaazaoui/RAG.git
cd RAG
uv sync
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```
*Note: `.env` is ignored by git. Never commit live API keys.*

### 3. Add Documents
Place your raw source files inside the `data/` directory (subfolders supported).

### 4. Running the System

* **Launch Interactive Web UI:**
  ```bash
  uv run streamlit run app.py
  ```

* **Run RAG Faithfulness Evaluation Benchmarks:**
  ```bash
  uv run evaluate.py
  ```

* **Run Command-Line Entrypoint:**
  ```bash
  uv run app.py
  ```
  *Note: The first run auto-indexes documents into `faiss_store/`. To re-index after adding new files, delete `faiss_store/` and re-run.*

---

## Programmatic Usage

Integrate `RAGSearch` directly into backend API routes or worker jobs:

```python
from src.search import RAGSearch

# Initialize search engine
rag = RAGSearch()

# Query index and return synthesized response with top-k context
answer = rag.search_and_summarize("What is Prolog?", top_k=3)
print(answer)
```

---

## Tech Stack & Dependencies

| Layer | Technology / Framework |
| :--- | :--- |
| **Orchestration** | LangChain (`langchain-core`, `langchain-community`) |
| **Document Processing** | PyPDF, PyMuPDF, Unstructured |
| **Embeddings & Reranking**| Sentence-Transformers (`all-MiniLM-L6-v2`), CrossEncoder |
| **Vector Indexing** | FAISS (`faiss-cpu`), ChromaDB |
| **LLM Interface** | OpenAI API (`langchain-openai`), Pydantic |
| **Frontend & UI** | Streamlit |
| **Package Management** | `uv`, `python-dotenv` |

---

## Roadmap

- [x] Multi-format document ingestion pipeline
- [x] Local vector persistence with FAISS
- [x] Cross-encoder context reranking module (`rerank.py`)
- [x] Automated faithfulness evaluation benchmark (`evaluate.py`)
- [x] Interactive Streamlit application interface (`app.py`)
- [ ] Hybrid retrieval (BM25 sparse search + FAISS dense search with RRF)
- [ ] Docker containerization for deployment

---

## Author

**Safaa Zaoui**  
*Computer Science (Co-op), Toronto Metropolitan University*  

