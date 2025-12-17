import re
import json
import os
from typing import List, Dict
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = "corpus.json"
# Use the official Parquet-based dataset
DATASET_NAME = "wikimedia/wikipedia" 
# 'simple' english contains ~200k articles, good for testing
DATASET_VERSION = "20231101.simple" 

NUM_DOCUMENTS = 1000  
CHUNK_SIZE = 500      
CHUNK_OVERLAP = 50    

def clean_text(text: str) -> str:
    # Remove citations like [1]
    text = re.sub(r'\[\d+\]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_and_process_data():
    print(f"Loading {NUM_DOCUMENTS} documents from {DATASET_NAME} ({DATASET_VERSION})...")
    
    # Load the official dataset (Parquet format)
    dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split="train", streaming=True)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split by paragraph first, then sentence, then words
        separators=["\n\n", "\n", ".", " ", ""]
    )

    processed_corpus: List[Dict] = []
    
    count = 0
    for doc in tqdm(dataset, total=NUM_DOCUMENTS, desc="Processing Articles"):
        if count >= NUM_DOCUMENTS:
            break
            
        # The official dataset uses 'title', 'text', 'url', and 'id'
        title = doc.get("title", "No Title")
        raw_text = doc.get("text", "")
        url = doc.get("url", "https://wikipedia.org")

        if not raw_text:
            continue

        clean_content = clean_text(raw_text)
        chunks = text_splitter.split_text(clean_content)

        # Skip articles that are too short (optional check)
        if not chunks:
            continue

        for i, chunk in enumerate(chunks):
            entry = {
                "id": f"doc_{count}_chunk_{i}",
                "doc_id": doc.get("id", str(count)), # Keep original Wiki ID
                "title": title,
                "url": url,
                "content": chunk,
                "chunk_index": i
            }
            processed_corpus.append(entry)
        
        count += 1

    return processed_corpus

def save_data(data: List[Dict]):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    print(f"Saving {len(data)} chunks to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print("Data preparation complete.")

if __name__ == "__main__":
    try:
        data = fetch_and_process_data()
        save_data(data)
        
        # Automatic sanity check at the end
        print("\n--- SANITY CHECK (First Entry) ---")
        print(json.dumps(data[0], indent=2))
        
    except Exception as e:
        print(f"An error occurred: {e}")

import json
import os
from typing import List, Dict

# --- IMPORTS ---
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# Free Open Source Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Import EnsembleRetriever
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever

# --- CONFIGURATION ---
DATA_PATH = "data/processed/corpus.json"

# CONFIG: Local Free Models
# "all-MiniLM-L6-v2" is a small, fast, and powerful embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

LLM_MODEL_NAME = "llama3.1:8b" 

TOP_K = 5

class LocalRAGEngine:
    def __init__(self, data_path: str = DATA_PATH):
        print("Initializing LOCAL RAG Engine...")
        
        # 1. Load Data
        self.documents = self._load_documents(data_path)
        
        # 2. Setup Embeddings (Local & Free)
        print(f"Loading local embeddings ({EMBEDDING_MODEL_NAME})...")

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # 3. Dense Retriever (Vector Search)
        print("Building Vector Index (Dense)...")
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K})
        
        # 4. Sparse Retriever (BM25)
        print("Building BM25 Index (Sparse)...")
        self.sparse_retriever = BM25Retriever.from_documents(self.documents)
        self.sparse_retriever.k = TOP_K
        
        # 5. Hybrid Ensemble Retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.5, 0.5]
        )
        
        # 6. Initialize LLM (Ollama)
        print(f"Connecting to Ollama ({LLM_MODEL_NAME})...")
        self.llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)
        
        print("RAG Engine Ready.")

    def _load_documents(self, path: str) -> List[Document]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus file not found at {path}.")
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        docs = []
        for entry in data:
            metadata = {
                "title": entry.get("title"),
                "url": entry.get("url"),
                "chunk_id": entry.get("id"),
                "doc_id": entry.get("doc_id")
            }
            doc = Document(page_content=entry.get("content"), metadata=metadata)
            docs.append(doc)
        return docs

    def search(self, query: str) -> List[Document]:
        print(f"Searching for: '{query}'")
        return self.ensemble_retriever.invoke(query)

    def summarize(self, query: str, docs: List[Document], length: str = "medium") -> str:
        length_guidelines = {
            "short": "a concise 2-3 sentence summary",
            "medium": "a detailed paragraph capturing key points",
            "long": "a comprehensive report with bullet points and sections"
        }
        
        instruction = length_guidelines.get(length, length_guidelines["medium"])
        context_text = "\n\n".join([f"Source ({d.metadata['title']}): {d.page_content}" for d in docs])
        
        template = """
        You are an intelligent research assistant. 
        User Query: {query}
        
        Based ONLY on the provided context below, write a summary that directly answers the query.
        Do not introduce outside information.
        
        Format Requirement: {instruction}
        
        --- CONTEXT ---
        {context}
        ----------------
        
        Summary:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "instruction", "context"]
        )
        
        chain = prompt | self.llm
        
        print("Generating summary with Ollama (this may take a moment)...")
        response = chain.invoke({
            "query": query,
            "instruction": instruction,
            "context": context_text
        })
        
        return response.content

# --- Example Usage ---
if __name__ == "__main__":
    try:
      
        engine = LocalRAGEngine()
        
        test_query = "What is the history of Artificial Intelligence?"
        
        # 1. Retrieve
        results = engine.search(test_query)
        print(f"\nFound {len(results)} documents.")
        
        # 2. Summarize
        summary = engine.summarize(test_query, results, length="short")
        print(f"\nGenerated Summary:\n{summary}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Make sure Ollama is installed and running (run `ollama serve` in a terminal).")