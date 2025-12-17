# 🧠 Local RAG: Hybrid Search & Summarization with Streamlit + Evaluation

This repository contains a fully local **Retrieval-Augmented Generation (RAG)** system designed to perform intelligent document search and summarization. It processes a Wikipedia-based corpus, utilizes a **Hybrid Search Architecture** (Dense Vector + Sparse Keyword), and generates grounded summaries using a local **Ollama LLM**.

Key features include an interactive **Streamlit UI** (bonus) and a rigorous **Evaluation Script** that benchmarks performance using Recall@K and ROUGE metrics.

---

## 🚀 Features

* **Corpus Builder:** Streams `wikimedia/wikipedia` (Simple English), cleans text, chunks it into manageable pieces, and saves a JSON corpus.
* **Hybrid Retrieval Engine:**
    * **Dense Retrieval:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) + **FAISS** for semantic understanding.
    * **Sparse Retrieval:** Uses **BM25** for precise keyword matching.
    * **Ensemble:** Combines both methods for maximum accuracy.
* **Local Summarization:** Powered by **Ollama** (e.g., `llama3.1:8b`). Generates grounded summaries derived *only* from retrieved chunks.
* **Adjustable Summaries:** User options for **short**, **medium**, or **long** outputs.
* **Streamlit App:** A user-friendly interface with query suggestions, pagination, and history management.
* **Evaluation Suite:** Automatically generates synthetic queries from mid-document chunks to test **Recall@K** and **ROUGE-1/ROUGE-L** scores.

---

## 📂 Repository Structure

```plaintext
.
├── data/
│   └── processed/
│       └── corpus.json              # Generated corpus (after running preprocess.py)
├── engine.py                        # Core RAG Logic (FAISS + BM25 + Ollama connection)
├── preprocess.py                    # Corpus builder (Loader + Cleaner + Chunker)
├── app.py                           # Streamlit Web UI
├── evaluate.py                      # Evaluation script (Recall & ROUGE metrics)
├── evaluation_results.csv           # Log file generated after evaluation
└── README.md                        # Project documentation
