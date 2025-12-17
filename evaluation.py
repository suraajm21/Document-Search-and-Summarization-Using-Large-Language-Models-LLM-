import json
import random
import statistics
import csv
import os
from typing import List, Dict
from tqdm import tqdm
from rouge_score import rouge_scorer
from collections import defaultdict

# Import your existing engine
from engine import LocalRAGEngine

# --- CONFIGURATION ---
DATA_PATH = "data/processed/corpus.json"
OUTPUT_LOG = "evaluation_results.csv"
TEST_SET_SIZE = 20      
TOP_K_CHECK = 5         
RANDOM_SEED = 42        

def set_seed(seed: int):
    random.seed(seed)

def load_and_group_documents(path: str) -> Dict[str, List[Dict]]:
    """Groups chunks by parent Document ID, ensuring string IDs."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    grouped_docs = defaultdict(list)
    for chunk in data:
        # Force string casting for ID safety
        d_id = str(chunk.get('doc_id'))
        if d_id:
            grouped_docs[d_id].append(chunk)
            
    return {k: v for k, v in grouped_docs.items() if len(v) >= 3} # Only test docs with sufficient content

def generate_synthetic_query(llm, doc_chunks: List[Dict]) -> tuple[str, str]:
    """
    Selects a chunk from the MIDDLE of the article to generate a harder query.
    Returns: (generated_query, source_chunk_content)
    """
    # Target middle 50% of article to avoid intro-bias
    n = len(doc_chunks)
    if n > 1:
        start = n // 4
        end = max(start + 1, (3 * n) // 4)
        target_chunk = random.choice(doc_chunks[start:end])
    else:
        target_chunk = doc_chunks[0]

    text_content = target_chunk['content']
    
    prompt = f"""
    You are an evaluator. Read the following text and write ONE specific search query 
    that a user would type to find this specific information.
    
    Constraint: Do not explicitly mention the document title.
    Output ONLY the query. Do not add quotes.

    --- Text ---
    {text_content[:800]}
    
    Query:
    """
    response = llm.invoke(prompt)
    return response.content.strip().replace('"', ''), text_content

def evaluate_system():
    print(f"--- 📊 Starting Hardened Evaluation (Seed: {RANDOM_SEED}) ---")
    set_seed(RANDOM_SEED)
    
    # 1. Initialize Engine
    try:
        engine = LocalRAGEngine()
    except Exception as e:
        print(f"CRITICAL: Engine failed to load. {e}")
        return

    # 2. Prepare Data
    grouped_corpus = load_and_group_documents(DATA_PATH)
    doc_ids = list(grouped_corpus.keys())
    
    # Sample IDs
    if len(doc_ids) < TEST_SET_SIZE:
        sampled_ids = doc_ids
    else:
        sampled_ids = random.sample(doc_ids, TEST_SET_SIZE)

    # Metrics containers
    retrieval_hits = 0
    rouge_lead_scores = []   # Compare vs Intro
    rouge_source_scores = [] # Compare vs Source Chunk (Answer Accuracy)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    results_log = []

    print(f"\n--- 🧪 Testing {len(sampled_ids)} Documents ---")
    
    for doc_id in tqdm(sampled_ids, desc="Evaluating"):
        chunks = grouped_corpus[doc_id]
        
        # A. Generate Query (Hard Mode: Middle Chunk)
        query, source_content = generate_synthetic_query(engine.llm, chunks)
        
        # B. Retrieval (Strict Top-K)
        # Explicit slicing ensures we test strict Recall@K
        retrieved_docs = engine.search(query)[:TOP_K_CHECK]
        
        # String casting for comparison
        found_doc_ids = [str(d.metadata.get('doc_id')) for d in retrieved_docs if 'doc_id' in d.metadata]
        is_hit = str(doc_id) in found_doc_ids
        
        if is_hit:
            retrieval_hits += 1

        # C. Summarization
        # Reference 1: The Article Lead (Contextual Summary)
        sorted_chunks = sorted(chunks, key=lambda x: x['chunk_index'])
        lead_content = sorted_chunks[0]['content']
        
        # Generate Summary
        generated_summary = engine.summarize(query, retrieved_docs, length="short")
        
        # Score 1: Vs Lead (Did we capture the main topic?)
        score_lead = scorer.score(lead_content, generated_summary)
        rouge_lead_scores.append(score_lead['rouge1'].fmeasure)
        
        # Score 2: Vs Source (Did we answer the specific question?)
        # Align reference with the query source
        score_source = scorer.score(source_content, generated_summary)
        rouge_source_scores.append(score_source['rouge1'].fmeasure)

        # Log individual result
        results_log.append({
            "doc_id": doc_id,
            "query": query,
            "hit": is_hit,
            "rouge_lead": round(score_lead['rouge1'].fmeasure, 3),
            "rouge_source": round(score_source['rouge1'].fmeasure, 3)
        })

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("   📈 FINAL EVALUATION REPORT")
    print("="*40)
    
    hit_rate = (retrieval_hits / len(sampled_ids)) * 100
    avg_r_lead = statistics.mean(rouge_lead_scores)
    avg_r_source = statistics.mean(rouge_source_scores)
    
    print(f"\n Retrieval (Recall@{TOP_K_CHECK}):")
    print(f"   Hit Rate: {hit_rate:.1f}%")

    print(f"\n Summarization Quality:")
    print(f"   ROUGE-1 (vs Intro):  {avg_r_lead:.4f} (Topic Coverage)")
    print(f"   ROUGE-1 (vs Source): {avg_r_source:.4f} (Answer Accuracy)")
    
    # Save CSV
    keys = results_log[0].keys()
    with open(OUTPUT_LOG, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_log)
        
    print(f"\n Detailed logs saved to: {OUTPUT_LOG}")
    print("="*40)

if __name__ == "__main__":
    evaluate_system()