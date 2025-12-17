import streamlit as st
import time


from engine import LocalRAGEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ever Quint Local RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
# Adds a cleaner look to the expanders and buttons
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
    }
    .search-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED RESOURCE ---
@st.cache_resource(show_spinner=False)
def load_engine():
    """
    Loads the LocalRAGEngine only once. 
    Returns None if models (Ollama/HF) fail to load.
    """
    try:
        return LocalRAGEngine()
    except Exception as e:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Hugging_Face_icon.svg/480px-Hugging_Face_icon.svg.png", width=50)
    st.header("⚙️ Local Settings")
    
    st.markdown("Running on GPU")
    st.caption("Models: Llama 3.1 & MiniLM-L6")
    
    st.markdown("---")
    
    # Requirement: Adjustable summary lengths
    summary_length = st.select_slider(
        "Summary Detail",
        options=["short", "medium", "long"],
        value="medium",
        help="Controls how verbose the AI answer will be."
    )
    
    st.markdown("---")
    if st.button("🧹 Clear History"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- INITIALIZATION ---
# Load Engine (Display spinner only on first load)
if "engine_loaded" not in st.session_state:
    with st.spinner("🚀 Booting up Local AI on GPU..."):
        engine = load_engine()
        if engine:
            st.session_state.engine_loaded = True
            st.success("AI Ready!")
        else:
            st.error("Failed to load engine. Check terminal for errors.")
            st.stop()
else:
    engine = load_engine()

# Initialize Session State Variables
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "page_index" not in st.session_state:
    st.session_state.page_index = 0
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# --- MAIN INTERFACE ---
st.title("🧠 Ever Quint Intelligent Search")
st.markdown("##### Offline Neural Search powered by Llama 3.1")

# --- SEARCH CONTAINER ---
with st.container():
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Search Input
        query_input = st.text_input(
            "Search Query", 
            placeholder="Ask a question about your documents...",
            label_visibility="collapsed"
        )
    
    with col2:
        # Search Button
        search_clicked = st.button("🔍 Search", type="primary")

# --- SUGGESTIONS (Pills) ---
st.markdown("Try these:")
suggestions = ["History of AI", "What is Machine Learning?", "Neural Networks", "Turing Test"]
cols = st.columns(len(suggestions))

selected_suggestion = None
for i, suggestion in enumerate(suggestions):
    if cols[i].button(suggestion, key=f"sug_{i}"):
        selected_suggestion = suggestion

# --- LOGIC CONTROLLER ---
# Determine if we should run a search (Button Click, Enter Key, or Suggestion Click)
final_query = selected_suggestion if selected_suggestion else (query_input if search_clicked else None)

if final_query:
    # Reset pagination for new search
    st.session_state.page_index = 0
    st.session_state.last_query = final_query
    
    with st.spinner(f"Reading documents for: '{final_query}'..."):
        # 1. Search (Vector + Keyword)
        results = engine.search(final_query)
        st.session_state.search_results = results
        
        # 2. Summarize (LLM)
        summary_placeholder = st.empty()
        summary_placeholder.markdown("✍️ *Generating summary...*")
        
        summary_text = engine.summarize(final_query, results, length=summary_length)
        st.session_state.summary = summary_text
        summary_placeholder.empty()

# --- DISPLAY RESULTS ---
# 1. Summary Section
if st.session_state.summary:
    st.markdown("### 📝 AI Summary")
    st.info(st.session_state.summary)

# 2. Document Results (Paginated)
if st.session_state.search_results:
    st.markdown("---")
    st.subheader(f"📚 Found {len(st.session_state.search_results)} Relevant Sources")
    
    # Pagination Logic
    results = st.session_state.search_results
    items_per_page = 3
    total_docs = len(results)
    
    # Calculate indices
    start_idx = st.session_state.page_index * items_per_page
    end_idx = min(start_idx + items_per_page, total_docs)
    current_batch = results[start_idx:end_idx]
    
    # Render Documents
    for i, doc in enumerate(current_batch):
        with st.expander(f"📄 Result {start_idx + i + 1}: {doc.metadata.get('title', 'Unknown Source')}", expanded=True):
            st.markdown(f"**Relevance Rank:** #{start_idx + i + 1}")
            st.caption(f"Source ID: {doc.metadata.get('chunk_id', 'N/A')}")
            st.markdown(f"> {doc.page_content}")
            
    # Pagination Controls
    if total_docs > items_per_page:
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if st.session_state.page_index > 0:
                if st.button("⬅️ Previous"):
                    st.session_state.page_index -= 1
                    st.rerun()
        
        with col_info:
            st.markdown(
                f"<div style='text-align: center; color: gray; padding-top: 10px;'>"
                f"Page {st.session_state.page_index + 1} of {-(-total_docs // items_per_page)}"
                f"</div>", 
                unsafe_allow_html=True
            )
            
        with col_next:
            if end_idx < total_docs:
                if st.button("Next ➡️"):
                    st.session_state.page_index += 1
                    st.rerun()