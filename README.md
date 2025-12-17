# Local RAG (Hybrid Search + Summarization) with Streamlit + Evaluation

## 📋 Project Overview
This project implements a **fully local Retrieval-Augmented Generation (RAG)** system that can search a large text corpus and generate **grounded summaries** using a local LLM.

The pipeline includes:
- **Data preparation** (Wikipedia → clean → chunk → JSON corpus)
- **Hybrid retrieval** (Dense FAISS + Sparse BM25)
- **Local summarization** (Ollama LLM, context-only)
- **Streamlit UI** (bonus)
- **Evaluation** using **Recall@K** and **ROUGE**

---

## 💡 The Problem
Given a sizeable text corpus, a user should be able to type a query like:

> “What is the history of Artificial Intelligence?”

And the system should:
1. Retrieve the **Top-N most relevant excerpts** from the corpus.
2. Generate a **coherent summary** based **only** on retrieved context.
3. Allow the user to select **summary length** (`short / medium / long`).
4. Provide **quantitative evaluation** of retrieval and summarization quality.

---

## 🧱 What is RAG?
**RAG = Retrieval + Generation**

Instead of letting an LLM answer from memory, we:
1. **Retrieve** relevant document chunks from the corpus.
2. Provide those chunks to the LLM as **context**.
3. The LLM generates an answer **grounded in the retrieved sources**.

This improves factuality and reduces hallucination.

---

## 📚 Corpus Details
This project uses the official Wikimedia Wikipedia dataset:

- **Dataset:** `wikimedia/wikipedia`
- **Version:** `20231101.simple` (Simple English)
- **Mode:** streaming
- **Indexed subset:** `NUM_DOCUMENTS = 1000` (configurable)

After preprocessing, the corpus is stored as:
- `data/processed/corpus.json`

Each entry contains:
- `id`, `doc_id`, `title`, `url`, `content`, `chunk_index`

---

## 🧠 Solution Approach
The system is built in four steps: **Ingest → Retrieve → Summarize → Evaluate**

### 1) Data Preparation (Chunking)
We clean and split each article into overlapping chunks:
- `CHUNK_SIZE = 500`
- `CHUNK_OVERLAP = 50`

Chunking improves retrieval granularity and keeps context sizes manageable for the LLM.

### 2) Hybrid Retrieval (Dense + Sparse)
We use **two retrievers**:

| Retriever Type | Method | Strength |
|---|---|---|
| Dense | Embeddings + FAISS | Semantic similarity (paraphrases) |
| Sparse | BM25 | Keyword matches (names, dates, exact terms) |

Then combine them using an **Ensemble Retriever** (default weights 0.5 / 0.5).

### 3) Grounded Summarization (Local LLM)
We summarize using **Ollama**:
- Model: `llama3.1:8b`
- Temperature: `0` (stable outputs)

The prompt explicitly enforces:
- “Use ONLY the provided context”
- “Do not introduce outside information”

### 4) Evaluation (Recall@K + ROUGE)
We evaluate:
- **Recall@K** for retrieval performance
- **ROUGE-1 / ROUGE-L** for summary quality

The evaluation script uses a “hardened” approach by generating queries from **middle chunks** of articles to avoid intro bias.

---

## ✨ Features
- **Corpus Builder:** streams Wikipedia, cleans text, chunks it, saves `corpus.json`
- **Hybrid Retrieval:**
  - Dense retrieval via HuggingFace embeddings + FAISS
  - Sparse retrieval via BM25
  - Combined via Ensemble Retriever
- **Local Summarization:** Ollama LLM generates grounded summaries from retrieved chunks only
- **Adjustable Summary Length:** `short / medium / long`
- **Streamlit App (Bonus):**
  - Query input
  - Suggestion pills
  - Pagination
  - Clear history button
- **Evaluation Script:**
  - Synthetic query generation from mid-document chunks
  - Recall@K
  - ROUGE-1 / ROUGE-L
  - Logs saved to CSV

---

## 📦 Repository Structure
```text
.
├── data/
│   └── processed/
│       └── corpus.json              # generated after preprocessing
├── engine.py                        # LocalRAGEngine (FAISS + BM25 + Ollama summary)
├── app.py                           # Streamlit UI
├── evaluate.py                      # Recall@K + ROUGE evaluation
├── evaluation_results.csv           # generated after running evaluation
└── README.md
